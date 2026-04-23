import torch
import torch.nn as nn
from typing import Tuple, Optional, Union, Literal

from simulators.base_simulator import BaseSimulator


class DoublePendulum(BaseSimulator):
    """Double pendulum simulator"""
    
    def __init__(
        self, 
        m1: float = 1.0, 
        m2: float = 1.0,
        l1: float = 1.0, 
        l2: float = 1.0, 
        g: float = 9.81,
        device: Optional[Union[str, torch.device]] = None
    ):

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device) if isinstance(device, str) else device

        super().__init__(dim=4, device=device)
        
        self.register_buffer('m1', torch.tensor(m1, device=self.device, dtype=torch.float32))
        self.register_buffer('m2', torch.tensor(m2, device=self.device, dtype=torch.float32))
        self.register_buffer('l1', torch.tensor(l1, device=self.device, dtype=torch.float32))
        self.register_buffer('l2', torch.tensor(l2, device=self.device, dtype=torch.float32))
        self.register_buffer('g', torch.tensor(g, device=self.device, dtype=torch.float32))

    def __str__(self):
        return "DoublePendulum"
        
    def hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        """Compute Hamiltonian"""
        theta1 = z[..., 0]
        theta2 = z[..., 1]
        p1 = z[..., 2]
        p2 = z[..., 3]

        m1, m2 = self.m1, self.m2
        l1, l2 = self.l1, self.l2
        g = self.g
        
        M = self.m1 + self.m2
        cos_diff = torch.cos(theta1 - theta2)
        sin_diff_sq = torch.sin(theta1 - theta2)**2
        
        # Kinetic energy
        T_num = (m2 * l2**2 * p1**2 + M * l1**2 * p2**2 - 2 * m2 * l1 * l2 * p1 * p2 * cos_diff)
        T_den = (2 * l1**2 * l2**2 * m2 * (m1 + m2 * sin_diff_sq))
        
        T = T_num / T_den
    
        # Potential energy 
        V = -M * g * l1 * torch.cos(theta1) - m2 * g * l2 * torch.cos(theta2)
            
        return T + V
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute time derivative dz/dt"""
        theta1 = z[..., 0]
        theta2 = z[..., 1]
        p1 = z[..., 2]
        p2 = z[..., 3]
        
        m1, m2 = self.m1, self.m2
        l1, l2 = self.l1, self.l2
        g = self.g
        
        M = m1 + m2
        
        cos_diff = torch.cos(theta1 - theta2)
        sin_diff = torch.sin(theta1 - theta2)
        sin_theta1 = torch.sin(theta1)
        sin_theta2 = torch.sin(theta2)
        
        denom = l1**2 * l2**2 * m2 * (m1 + m2 * sin_diff**2)

        dtheta1 = (m2 * l2**2 * p1 - m2 * l1 * l2 * p2 * cos_diff) / denom
        dtheta2 = (M * l1**2 * p2 - m2 * l1 * l2 * p1 * cos_diff) / denom
        
        H_num_term = (m2 * l2**2 * p1**2 + M * l1**2 * p2**2 - 2 * m2 * l1 * l2 * p1 * p2 * cos_diff)

        dH_dtheta1 = (
            (m2 * l1 * l2 * p1 * p2 * sin_diff) / denom -
            (H_num_term * m2**2 * l1**2 * l2**2 * sin_diff * cos_diff) / (denom**2)
        ) - M * g * l1 * sin_theta1

        dH_dtheta2 = (
                (-m2 * l1 * l2 * p1 * p2 * sin_diff) / denom +
                (H_num_term * m2**2 * l1**2 * l2**2 * sin_diff * cos_diff) / (denom**2)
            ) - m2 * g * l2 * sin_theta2
        
        dp1 = -dH_dtheta1
        dp2 = -dH_dtheta2
        
        return torch.stack([dtheta1, dtheta2, dp1, dp2], dim=-1)
    
    def random_initial_conditions(self, n_traj: int, energy_scale: float = 1.0, 
                                  seed: Optional[int] = None) -> torch.Tensor:
        """Creates a batch of random initial conditions"""
        if seed is not None:
            torch.manual_seed(seed)
        
        # sample in an interval
        theta1 = torch.rand(n_traj, device=self.device) * 2 * torch.pi - torch.pi
        theta2 = torch.rand(n_traj, device=self.device) * 2 * torch.pi - torch.pi
        
        p_scale = torch.sqrt(energy_scale * self.m1 * self.l1**2)
        p1 = (torch.rand(n_traj, device=self.device) * 2 - 1) * p_scale
        p2 = (torch.rand(n_traj, device=self.device) * 2 - 1) * p_scale
        
        return torch.stack([theta1, theta2, p1, p2], dim=-1)
