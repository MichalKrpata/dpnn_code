import torch

from simulators.base_simulator import BaseSimulator


class SinglePendulum(BaseSimulator):
    """Single pendulum simulator"""

    def __init__(self,
                 m: float = 1,
                 l: float = 1,
                 g: float = 9.81,
                 device=None):
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device) if isinstance(device, str) else device

        super().__init__(dim=2, device=device)

        self.register_buffer('m', torch.tensor(m, dtype=torch.float32))
        self.register_buffer('l', torch.tensor(l, dtype=torch.float32))
        self.register_buffer('g', torch.tensor(g, dtype=torch.float32))

    def __str__(self):
        return "SinglePendulum"

    def hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        """Compute Hamiltonian"""
        theta = z[..., 0]
        p = z[..., 1]
        kinetic = p**2 / (2 * self.m * self.l**2)
        potential = self.m * self.l * self.g * (1 - torch.cos(theta))
        return kinetic + potential

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute time derivative dz/dt"""
        theta = z[..., 0]
        p = z[..., 1]
        dtheta = p / (self.m * self.l**2)
        dp = -self.m * self.g * self.l * torch.sin(theta)
        return torch.stack([dtheta, dp], dim=-1)

    def random_initial_conditions(self, n_traj: int, seed=None) -> torch.Tensor:
        """Creates a batch of random initial conditions"""

        if seed is not None:
            torch.manual_seed(seed)

        # bound the maximum energy
        max_energy = 4 * self.m * self.g * self.l

        theta0 = (torch.rand(n_traj, device=self.device) * 2 - 1) * torch.pi

        potential_energy = self.m * self.g * self.l * (1 - torch.cos(theta0))

        p_max = torch.sqrt((max_energy - potential_energy) * 2 * self.m)

        p0 = (torch.rand(n_traj, device=self.device) * 2 - 1) * p_max

        return torch.stack([theta0, p0], dim=-1)
    