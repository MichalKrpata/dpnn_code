import torch
import torch.nn as nn
from typing import Literal, Tuple


class BaseSimulator(nn.Module):
    """Base class for simulators"""
    
    def __init__(self, dim: int, device=None):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dim = dim

    def hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        """Compute Hamiltonian"""
        raise NotImplementedError
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute time derivative dz/dt"""
        raise NotImplementedError
    
    def step_euler(self, z: torch.Tensor, dt: float) -> torch.Tensor:
        """Single Euler step: z_{n+1} = z_n + dt * f(z_n)"""
        return z + dt * self.forward(z)

    def step_rk4(self, z: torch.Tensor, dt: float) -> torch.Tensor:
        """Single RK4 step"""
        f = self.forward
        k1 = f(z)
        k2 = f(z + 0.5 * dt * k1)
        k3 = f(z + 0.5 * dt * k2)
        k4 = f(z + dt * k3)
        return z + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def step_implicit_midpoint(self, z: torch.Tensor, dt: float, max_iters: int = 10, tol: float = 1e-6) -> torch.Tensor:
        """Single IMR step"""
        z_next = z.clone()
        
        # simple fixed-point iteration
        for _ in range(max_iters):
            z_mid = (z + z_next) / 2.0
            z_next_new = z + dt * self.forward(z_mid)
            
            if torch.max(torch.abs(z_next_new - z_next)) < tol:
                z_next = z_next_new
                break
                
            z_next = z_next_new
            
        return z_next
    
    def simulate_batch(
        self,
        z0: torch.Tensor,
        dt: float,
        n_steps: int,
        method: Literal['rk4', 'euler', 'midpoint'] = 'rk4'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulates a given number of steps from the initial state z0 for a batch"""
        n_traj = z0.shape[0]
        t = torch.arange(n_steps, device=self.device) * dt

        z_traj = torch.zeros(n_traj, n_steps, self.dim, device=self.device)
        dz_traj = torch.zeros_like(z_traj)
        
        z_traj[:, 0] = z0
        dz_traj[:, 0] = self.forward(z0).detach()
        if method == 'rk4':
            step_fn = self.step_rk4
        elif method == 'euler':
            step_fn = self.step_euler
        elif method == 'midpoint':
            step_fn = self.step_implicit_midpoint
        else:
            raise ValueError(f"Unknown integration method: {method}")
        z = z0

        # evolves the initial conditions for a given number of steps
        for i in range(1, n_steps):
            z = step_fn(z, dt).detach()
            z_traj[:, i] = z
            dz_traj[:, i] = self.forward(z).detach()

        return t, z_traj, dz_traj

    def random_initial_conditions(self, n_traj: int, seed=None) -> torch.Tensor:
        """Creates a batch of random initial conditions"""
        raise NotImplementedError
