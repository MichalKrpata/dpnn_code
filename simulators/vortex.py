import torch
import torch.nn as nn
from typing import Union, List
from simulators.base_simulator import BaseSimulator
import matplotlib.pyplot as plt


class NPointVortex(BaseSimulator):
    """2D point vortex dynamics"""
    def __init__(self,
                 n_vortices: int = 3,
                 Gamma: Union[float, List[float], torch.Tensor] = 1.0,
                 device=None):
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device) if isinstance(device, str) else device

        self.n_vortices = n_vortices
        super().__init__(dim=2 * n_vortices, device=device)

        # accommodate for both scalar and iterable input 
        if isinstance(Gamma, (float, int)):
            Gamma_tensor = torch.full((n_vortices,), float(Gamma), dtype=torch.float32, device=device)
        else:
            Gamma_tensor = torch.tensor(Gamma, dtype=torch.float32, device=device)
            assert Gamma_tensor.shape[0] == n_vortices, f"Expected {n_vortices} Gamma values, got {Gamma_tensor.shape[0]}"

        self.register_buffer('Gamma', Gamma_tensor)
        self.eps = 1e-8

    def __str__(self):
        return f"NPointVortex(N={self.n_vortices})"

    def hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the Hamiltonian"""
        z_reshaped = z.view(-1, self.n_vortices, 2)
        
        diff = z_reshaped.unsqueeze(2) - z_reshaped.unsqueeze(1)
        r_sq = torch.sum(diff**2, dim=-1)
        
        mask = torch.eye(self.n_vortices, device=self.device, dtype=torch.bool)
        r_sq = r_sq.masked_fill(mask, 1.0)
        
        # H = - (1/4pi) * sum_{i<j} Gamma_i * Gamma_j * ln(r_ij^2) / 2
        Gamma_prod = self.Gamma.unsqueeze(1) * self.Gamma.unsqueeze(0)
        
        H_matrix = - (1.0 / (8.0 * torch.pi)) * Gamma_prod * torch.log(r_sq)
        
        H_matrix = H_matrix.masked_fill(mask, 0.0)
        
        H = torch.sum(H_matrix, dim=[-1, -2]) / 2.0
        
        return H

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute dz/dt using Hamilton's equations"""
        z_reshaped = z.view(-1, self.n_vortices, 2)
        
        x = z_reshaped[..., 0]
        y = z_reshaped[..., 1]
        
        dx = x.unsqueeze(2) - x.unsqueeze(1)
        dy = y.unsqueeze(2) - y.unsqueeze(1)
        
        r_sq = dx**2 + dy**2
        
        mask = torch.eye(self.n_vortices, device=self.device, dtype=torch.bool)
        r_sq = r_sq.masked_fill(mask, float('inf')) + self.eps
        Gamma_j = self.Gamma.view(1, 1, self.n_vortices)
        
        dx_dt = - (1.0 / (2.0 * torch.pi)) * torch.sum(Gamma_j * dy / r_sq, dim=-1)
        dy_dt =   (1.0 / (2.0 * torch.pi)) * torch.sum(Gamma_j * dx / r_sq, dim=-1)
        
        dz_dt = torch.stack([dx_dt, dy_dt], dim=-1)
        return dz_dt.view(-1, 2 * self.n_vortices)

    def random_initial_conditions(self, n_traj: int, seed=None) -> torch.Tensor:
        """Creates a batch of random initial conditions"""
        if seed is not None:
            torch.manual_seed(seed)

        # sample randomly on a grid (at least max_dist apart) to avoid chaotic behavior
        pos_max = 2.0
        max_dist = 0.5

        coords = torch.arange(-pos_max, pos_max + 1e-6, max_dist, device=self.device)
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing='ij')
        grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)

        n_grid = grid_points.shape[0]
        assert n_grid >= self.n_vortices, "Grid too small for number of vortices"

        z0 = []
        for _ in range(n_traj):
            idx = torch.randperm(n_grid)[:self.n_vortices]
            points = grid_points[idx]

            jitter = 0.1 * max_dist * (torch.rand_like(points) - 0.5)
            points = points + jitter

            z0.append(points)

        z0 = torch.stack(z0)
        return z0.view(n_traj, -1)
    

