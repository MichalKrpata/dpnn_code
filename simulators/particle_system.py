import torch
import torch.nn as nn
from typing import Literal, Tuple, Union, List

from simulators.base_simulator import BaseSimulator


class ParticleSystem(BaseSimulator):
    """A general Hamiltonian particle system simulator"""
    
    def __init__(self,
                 n_particles: int,
                 dim_spatial: int,
                 potential_fn: nn.Module,
                 m: Union[float, torch.Tensor, List[float]] = 1.0,
                 device=None):
        """
        n_particles (int): Number of particles (N)
        dim_spatial (int): Spatial dimension (D)
        potential_fn (nn.Module): A module that takes q [B, N, D] and returns V [B]
        m (float): Mass of each particle
        device: Torch device
        """
        super().__init__(dim=2*n_particles*dim_spatial, device=device)
        self.n_particles = n_particles
        self.dim_spatial = dim_spatial

        self.potential_fn = potential_fn

        m_tensor = torch.as_tensor(m, dtype=torch.float32, device=self.device)
        if m_tensor.ndim == 0:
            m_tensor = m_tensor.expand(n_particles)
        
        self.register_buffer('m', m_tensor)

    def __str__(self):
        return f"ParticleSystem(N={self.n_particles}, D={self.dim_spatial})"

    def _reshape_z(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reshapes flat state z [B, 2*N*D] into (q, p)"""
        z = z.view(-1, 2, self.n_particles, self.dim_spatial)
        q = z[:, 0]  # [B, N, D]
        p = z[:, 1]  # [B, N, D]
        return q, p

    def _flatten_qp(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Flattens (q, p) into state z [B, 2*N*D]"""
        return torch.stack([q, p], dim=1).view(-1, self.dim)

    def hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        """H = T + V"""
        q, p = self._reshape_z(z)
        
        m_broad = self.m.view(1, self.n_particles, 1)
        
        kinetic = torch.sum(p**2 / (2 * m_broad), dim=[-1, -2]) 
        potential = self.potential_fn(q)
        
        return kinetic + potential

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute time derivative dz/dt using Hamilton's equations."""
        q, p = self._reshape_z(z)
        
        m_broad = self.m.view(1, self.n_particles, 1)
        dq_dt = p / m_broad

        if hasattr(self.potential_fn, 'compute_force'):
            dp_dt = self.potential_fn.compute_force(q)
        else:
            with torch.enable_grad(): 
                q_grad = q.clone().requires_grad_(True)
                V = self.potential_fn(q_grad)
                dp_dt = -torch.autograd.grad(
                    outputs=V.sum(), 
                    inputs=q_grad,
                    create_graph=True
                )[0]
        
        return self._flatten_qp(dq_dt, dp_dt)

    def random_initial_conditions(self, n_traj: int, seed=None) -> torch.Tensor:
        """Creates a batch of random initial conditions"""
        if seed is not None:
            torch.manual_seed(seed)

        # sample in an interval
        q0 = (torch.rand(n_traj, self.n_particles, self.dim_spatial, device=self.device) * 2 - 1) * 1.0
        p0 = (torch.rand(n_traj, self.n_particles, self.dim_spatial, device=self.device) * 2 - 1) * 0.5
        
        return self._flatten_qp(q0, p0)


class GravitationalPotential(nn.Module):
    """
    N-body gravitational potential
    V(q) = sum_{i<j} - G * m_i * m_j / ||q_i - q_j||
    """
    def __init__(self, n_particles: int, G: float = 1e-3, m: Union[float, torch.Tensor, List[float]] = 1.0):
        super().__init__()
        self.register_buffer('G', torch.as_tensor(G, dtype=torch.float32))
        self.eps = 1e-6
        
        m_tensor = torch.as_tensor(m, dtype=torch.float32)
        if m_tensor.ndim == 0:
            m_tensor = m_tensor.expand(n_particles)
        self.register_buffer('m', m_tensor)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Computes the potential for the given input"""
        diff = q.unsqueeze(2) - q.unsqueeze(1) # [B, N, N, D]
        dist = torch.sqrt(torch.sum(diff**2, dim=-1) + self.eps) # [B, N, N]
        
        mass_matrix = self.m.unsqueeze(1) * self.m.unsqueeze(0) # [N, N]
        
        V_matrix = -self.G * mass_matrix / dist
        
        # Zero out diagonal
        V_matrix.diagonal(dim1=-2, dim2=-1).fill_(0.0)
        
        return torch.sum(V_matrix, dim=[-1, -2]) / 2.0
    
    def compute_force(self, q: torch.Tensor) -> torch.Tensor:
        """Analytically computes the negative gradient of the potential (force)"""
        diff = q.unsqueeze(2) - q.unsqueeze(1)
        dist_sq = torch.sum(diff**2, dim=-1)
        
        dist_sq.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))
        
        dist = torch.sqrt(dist_sq)
        
        mass_matrix = self.m.unsqueeze(1) * self.m.unsqueeze(0) # [N, N]
        force_mag = -self.G * mass_matrix / (dist_sq * dist) # [B, N, N]
        
        force = torch.sum(force_mag.unsqueeze(-1) * diff, dim=2) # [B, N, D]
        
        return force
    

class HarmonicPotential(nn.Module):
    """
    A simple harmonic potential
    V(q) = sum_i (0.5 * k_i * ||q_i||^2)
    """
    def __init__(self, n_particles: int, k: Union[float, torch.Tensor, List[float]] = 1.0):
        super().__init__()

        k_tensor = torch.as_tensor(k, dtype=torch.float32)
        if k_tensor.ndim == 0:
            k_tensor = k_tensor.expand(n_particles)

        self.register_buffer('k', k_tensor)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Computes the potential for the given input"""
        k_broad = self.k.view(1, -1, 1)

        q_norm_sq = torch.sum(q**2, dim=-1)

        V_per_particle = 0.5 * k_broad.squeeze(-1) * q_norm_sq

        return torch.sum(V_per_particle, dim=-1)

    def compute_force(self, q: torch.Tensor) -> torch.Tensor:
        """Analytically computes the negative gradient of the potential (force)"""
        k_broad = self.k.view(1, -1, 1)

        return -k_broad * q


class LennardJonesPotential(nn.Module):
    """
    A Lennard-Jones potential
    V(q) = sum{i<j} 
    """
    def __init__(self, epsilon: float = 0.19, sigma: float = 3.19):
        super().__init__()
        self.register_buffer('epsilon', torch.tensor(epsilon, dtype=torch.float32))
        self.register_buffer('sigma', torch.tensor(sigma, dtype=torch.float32))
        self.eps = 1e-8

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Computes the potential for the given input"""
        diff = q.unsqueeze(2) - q.unsqueeze(1)
        dist_sq = torch.sum(diff**2, dim=-1)
        
        dist_sq.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))

        r6 = dist_sq ** 3
        r12 = dist_sq ** 6
        
        s6 = self.sigma ** 6
        s12 = self.sigma ** 12
        
        V_matrix = 4 * self.epsilon * (s12 / (r12 + self.eps) - (s6 / (r6+self.eps)))
        
        V_scalar = torch.sum(V_matrix, dim=[-1, -2]) / 2.0
        
        return V_scalar
        
    def compute_force(self, q: torch.Tensor) -> torch.Tensor:
        """Analytically computes the negative gradient of the potential (force)"""
        diff = q.unsqueeze(2) - q.unsqueeze(1)
        dist_sq = torch.sum(diff**2, dim=-1)
        
        dist_sq.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))
        
        s6 = self.sigma ** 6
        s12 = self.sigma ** 12
        
        r8 = dist_sq ** 4
        r14 = dist_sq ** 7
        
        force_mag = 24 * self.epsilon * (2 * s12 / r14 - s6 / r8)
        force = torch.sum(force_mag.unsqueeze(-1) * diff, dim=2)
        
        return force    
