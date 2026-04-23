import torch
from simulators.base_simulator import BaseSimulator

def hat_map(v: torch.Tensor) -> torch.Tensor:
    """Converts a batched 3D vector to a 3x3 skew-symmetric matrix"""
    zero = torch.zeros_like(v[..., 0])
    v1, v2, v3 = v[..., 0], v[..., 1], v[..., 2]
    
    row1 = torch.stack([zero, -v3, v2], dim=-1)
    row2 = torch.stack([v3, zero, -v1], dim=-1)
    row3 = torch.stack([-v2, v1, zero], dim=-1)
    
    return torch.stack([row1, row2, row3], dim=-2)

class CoupledRigidBodies3D(BaseSimulator):
    """Two coupled rigid bodies with a ball-and-socket joint simulator"""

    def __init__(self, m1: float = 1.0, m2: float = 1.0, J1_diag: list = [1.0, 2.0, 3.0], 
                 J2_diag: list = [2.0, 1.0, 3.0], S1_0: list = [0.0, 0.0, 1.0], 
                 S2_0: list = [0.0, 0.0, -1.0], device=None):
        
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        super().__init__(dim=9, device=self.device)
        
        self.m1 = torch.tensor(m1, dtype=torch.float32, device=self.device)
        self.m2 = torch.tensor(m2, dtype=torch.float32, device=self.device)
        self.m = m1 + m2
        self.epsilon = (self.m1 * self.m2) / self.m
        
        J1 = torch.diag(torch.tensor(J1_diag, dtype=torch.float32, device=self.device))
        J2 = torch.diag(torch.tensor(J2_diag, dtype=torch.float32, device=self.device))
        
        S1 = torch.tensor(S1_0, dtype=torch.float32, device=self.device)
        S2 = torch.tensor(S2_0, dtype=torch.float32, device=self.device)
        
        self.S1_hat = hat_map(S1)
        self.S2_hat = hat_map(S2)
        
        eye = torch.eye(3, device=self.device)
        self.J1_bar = J1 + (self.m1**2 / self.m) * (torch.dot(S1, S1) * eye - torch.outer(S1, S1))
        self.J2_bar = J2 + (self.m2**2 / self.m) * (torch.dot(S2, S2) * eye - torch.outer(S2, S2))

    def angles_to_matrix(self, angles: torch.Tensor) -> torch.Tensor:
        """Convert Euler angles to a 3x3 rotation matrix A"""
        phi, theta, psi = angles[..., 0], angles[..., 1], angles[..., 2]
        
        c1, s1 = torch.cos(phi), torch.sin(phi)
        c2, s2 = torch.cos(theta), torch.sin(theta)
        c3, s3 = torch.cos(psi), torch.sin(psi)
        
        r00 = c1 * c3 - s1 * c2 * s3
        r01 = -c1 * s3 - s1 * c2 * c3
        r02 = s1 * s2
        
        r10 = s1 * c3 + c1 * c2 * s3
        r11 = -s1 * s3 + c1 * c2 * c3
        r12 = -c1 * s2
        
        r20 = s2 * s3
        r21 = s2 * c3
        r22 = c2
        
        row0 = torch.stack([r00, r01, r02], dim=-1)
        row1 = torch.stack([r10, r11, r12], dim=-1)
        row2 = torch.stack([r20, r21, r22], dim=-1)
        
        return torch.stack([row0, row1, row2], dim=-2)

    def build_euler(self, angles: torch.Tensor) -> torch.Tensor:
        """Constructs T^{-1} for the Euler Angles"""
        phi, theta, psi = angles[..., 0], angles[..., 1], angles[..., 2]
        
        s_th, c_th = torch.sin(theta), torch.cos(theta)
        s_psi, c_psi = torch.sin(psi), torch.cos(psi)
        
        csc_th = 1.0 / s_th
        cot_th = c_th / s_th
        
        zeros = torch.zeros_like(phi)
        ones = torch.ones_like(phi)
        
        row0 = torch.stack([s_psi * csc_th, c_psi * csc_th, zeros], dim=-1)
        row1 = torch.stack([c_psi, -s_psi, zeros], dim=-1)
        row2 = torch.stack([-s_psi * cot_th, -c_psi * cot_th, ones], dim=-1)
        
        return torch.stack([row0, row1, row2], dim=-2)

    def build_inertia_matrix(self, A: torch.Tensor) -> torch.Tensor:
        """Constructs the 6x6 inertia matrix J"""
        batch_dims = A.shape[:-2]
        
        S1_hat_T = -self.S1_hat 
        Lambda = torch.matmul(S1_hat_T, torch.matmul(A, self.S2_hat))
        
        eps_Lambda = self.epsilon * Lambda
        eps_Lambda_T = eps_Lambda.transpose(-2, -1)
        
        J1_exp = self.J1_bar.expand(*batch_dims, 3, 3)
        J2_exp = self.J2_bar.expand(*batch_dims, 3, 3)
        
        top = torch.cat([J1_exp, eps_Lambda], dim=-1)
        bottom = torch.cat([eps_Lambda_T, J2_exp], dim=-1)
        return torch.cat([top, bottom], dim=-2)

    def build_poisson_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """Constructs the 9x9 Poisson matrix P(z)"""
        batch_dims = z.shape[:-1]
        Pi1 = z[..., 0:3]
        Pi2 = z[..., 3:6]
        angles = z[..., 6:9]
        
        Pi1_hat = hat_map(Pi1)
        Pi2_hat = hat_map(Pi2)
        
        T_inv = self.build_euler(angles) 
        A = self.angles_to_matrix(angles)
        A_T = A.transpose(-2, -1)
        
        P = torch.zeros((*batch_dims, 9, 9), device=self.device)
        
        P[..., 0:3, 0:3] = Pi1_hat
        P[..., 3:6, 3:6] = Pi2_hat
        
        bottom_left_1 = -torch.matmul(T_inv, A_T)
        bottom_left_2 = T_inv
        
        P[..., 6:9, 0:3] = bottom_left_1
        P[..., 6:9, 3:6] = bottom_left_2
        
        P[..., 0:3, 6:9] = -bottom_left_1.transpose(-2, -1)
        P[..., 3:6, 6:9] = -bottom_left_2.transpose(-2, -1)
        
        return P

    def hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        """Compute Total Kinetic Energy H = 1/2 * Pi^T * J^{-1} * Pi"""
        Pi1 = z[..., 0:3]
        Pi2 = z[..., 3:6]
        angles = z[..., 6:9]
        
        A = self.angles_to_matrix(angles)
        J_matrix = self.build_inertia_matrix(A)
        
        Pi = torch.cat([Pi1, Pi2], dim=-1).unsqueeze(-1) 
        Omega = torch.linalg.solve(J_matrix, Pi) 
        
        H = 0.5 * torch.matmul(Pi.transpose(-2, -1), Omega).squeeze(-1).squeeze(-1)
        return H

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute time derivative dz/dt = P(z) * nabla_H(z)"""
        with torch.enable_grad():
            z.requires_grad_(True)
            H = self.hamiltonian(z)
            dH_dz = torch.autograd.grad(H.sum(), z, create_graph=True)[0]
        
        P_z = self.build_poisson_matrix(z)
        dz_dt = torch.matmul(P_z, dH_dz.unsqueeze(-1)).squeeze(-1)
        return dz_dt
    
    def random_initial_conditions(self, n_traj: int, seed=None) -> torch.Tensor:
        """Creates a batch of random initial conditions"""
        if seed is not None:
            torch.manual_seed(seed)

        # sample in an interval
        p_max = 0.5
        
        Pi1 = (torch.rand(n_traj, 3, device=self.device) * 2 - 1) * p_max
        Pi2 = (torch.rand(n_traj, 3, device=self.device) * 2 - 1) * p_max

        phi = (torch.rand(n_traj, 1, device=self.device) * 2 - 1) * torch.pi
        theta = (torch.rand(n_traj, 1, device=self.device) * (torch.pi / 2.0)) + (torch.pi / 4.0)
        psi = (torch.rand(n_traj, 1, device=self.device) * 2 - 1) * torch.pi
        
        angles = torch.cat([phi, theta, psi], dim=-1)

        return torch.cat([Pi1, Pi2, angles], dim=-1)
    