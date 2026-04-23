import torch
import torch.nn as nn
from typing import Union, List

from simulators.base_simulator import BaseSimulator


class SingleFlywheel(BaseSimulator):
    """Single flywheel simulator"""
    def __init__(self, 
                 I: float = 1.0, 
                 k: float = 1.0, 
                 device=None):
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device) if isinstance(device, str) else device
        
        super().__init__(2, device)

        self.register_buffer('I', torch.tensor(I, dtype=torch.float32))
        self.register_buffer('k', torch.tensor(k, dtype=torch.float32))

    def __str__(self):
        return "SingleFlywheel"
    
    def hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the Hamiltonian"""
        theta = z[..., 0]
        p = z[..., 1]

        kinetic = p**2 / (2 * self.I)
        potential = 0.5 * self.k * theta**2

        return kinetic + potential
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute dz/dt using Hamilton's equations"""
        theta = z[..., 0]
        p = z[..., 1]

        dtheta = p / self.I
        dp = -self.k * theta

        return torch.stack([dtheta, dp], dim=-1)
        
    def random_initial_conditions(self, n_traj, seed=None):

        if seed is not None:
            torch.manual_seed(seed)

        max_energy = 5.0

        theta_max = torch.sqrt(2 * max_energy / self.k)
        theta0 = (torch.rand(n_traj, device=self.device)*2 - 1) * theta_max

        potential_energy = 0.5 * self.k * theta0**2
        p_max = torch.sqrt(torch.clamp(max_energy - potential_energy, min = 0.0) * 2 * self.I)

        p0 = (torch.rand(n_traj, device=self.device) * 2 - 1) * p_max

        return torch.stack([theta0, p0], dim=-1)


class TorsionalFlywheelChain(BaseSimulator):
    """1D chain of N coupled flywheels connected by torsional springs."""

    def __init__(self,
                 n_flywheels: int = 3,
                 I: Union[float, List[float], torch.Tensor] = 1.0,
                 k: Union[float, List[float], torch.Tensor] = 1.0,
                 device=None):
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device) if isinstance(device, str) else device

        self.n_flywheels = n_flywheels
        super().__init__(dim=2 * n_flywheels, device=device)

        # accommodate for both scalar and iterable input 
        if isinstance(I, (float, int)):
            I_tensor = torch.full((n_flywheels,), float(I), dtype=torch.float32)
        else:
            I_tensor = torch.tensor(I, dtype=torch.float32)
            assert I_tensor.shape[0] == n_flywheels, f"Expected {n_flywheels} I values, got {I_tensor.shape[0]}"
        
        if isinstance(k, (float, int)):
            k_tensor = torch.full((n_flywheels - 1,), float(k), dtype=torch.float32)
        else:
            k_tensor = torch.tensor(k, dtype=torch.float32)
            assert k_tensor.shape[0] == n_flywheels - 1 , f"Expected {n_flywheels - 1} k values, got {k_tensor.shape[0]}"

        self.register_buffer('I', I_tensor)
        self.register_buffer('k', k_tensor)

    def __str__(self):
        return f"TorsionalFlywheelChain(N={self.n_flywheels})"
        
    def hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the Hamiltonian"""
        z_reshaped = z.view(-1, self.n_flywheels, 2)
        theta = z_reshaped[..., 0]
        p = z_reshaped[..., 1]

        kinetic_per = p**2 / (2 * self.I)
        kinetic = torch.sum(kinetic_per, dim=-1)

        theta_dif = theta[..., 1:] - theta[..., :-1]
        potential_per = 0.5 * self.k * theta_dif**2
        potential = torch.sum(potential_per, dim=-1)

        return kinetic + potential
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute dz/dt using Hamilton's equations"""
        z_reshaped = z.view(-1, self.n_flywheels, 2)
        theta = z_reshaped[..., 0]
        p = z_reshaped[..., 1]

        dtheta = p / self.I

        spring_torques = self.k * (theta[..., 1:] - theta[..., :-1])

        dp = torch.zeros_like(p)
        dp[..., 0] = spring_torques[..., 0]
        dp[..., -1] = -spring_torques[..., -1]

        if self.n_flywheels > 2:
            dp[..., 1:-1] = spring_torques[..., 1:] - spring_torques[..., :-1]

        dz = torch.stack([dtheta, dp], dim=-1)

        return dz.view(z.shape[0], -1)

    def random_initial_conditions(self, n_traj, seed=None):
        """Creates a batch of random initial conditions"""
        if seed is not None:
            torch.manual_seed(seed)

        # sample in an interval
        theta_max = 1.0
        theta0 = (torch.rand(n_traj, self.n_flywheels, device=self.device)*2 - 1) * theta_max

        p_max = 0.5
        p0 = (torch.rand(n_traj, self.n_flywheels, device=self.device) * 2 - 1) * p_max

        return torch.stack([theta0, p0], dim=-1).view(n_traj, -1)
        