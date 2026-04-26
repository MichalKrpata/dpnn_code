import torch
import torch.nn as nn


class BaseL(nn.Module):
    """Base class for L matrix generators"""
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TrainableConstantL(BaseL):
    """A trainable, constant, antisymmetric matrix L = A - A^T"""
    def __init__(self, dim: int):
        super().__init__()
        self.A_param = nn.Parameter(torch.randn(dim, dim), requires_grad=True)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the antisymmetric matrix L, expanded to the batch size"""
        batch_size = z.shape[0]
        
        L_base = self.A_param - self.A_param.T
        
        return L_base.expand(batch_size, -1, -1)
    

class CanonicalL(BaseL):
    """A fixed, non-trainable, canonical L matrix for (q, p) systems"""
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        spatial_dim = dim//2
        
        L = torch.zeros(dim, dim)
        L[:spatial_dim, spatial_dim:] = torch.eye(spatial_dim)
        L[spatial_dim:, :spatial_dim] = -torch.eye(spatial_dim)
        
        self.register_buffer('L_matrix', L)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return the fixed L matrix, expanded to the batch size"""
        batch_size = z.shape[0]
        return self.L_matrix.expand(batch_size, -1, -1)


class NeuralL(BaseL):
    """A neural network that outputs an antisymmetric L matrix L(z)"""
    def __init__(self, input_dim: int, hidden_dim: int = 128, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        
        model_layers = [nn.Linear(input_dim, hidden_dim), nn.Softplus()]
        if dropout > 0:
            model_layers.append(nn.Dropout(dropout))

        for _ in range(layers - 1):
            model_layers.append(nn.Linear(hidden_dim, hidden_dim))
            model_layers.append(nn.Softplus())
            if dropout > 0:
                model_layers.append(nn.Dropout(dropout))

        self.output_dim = (input_dim * (input_dim - 1)) // 2

        model_layers.append(nn.Linear(hidden_dim, self.output_dim))

        self.mlp = nn.Sequential(*model_layers)
        self.input_dim = input_dim

        row_idx, col_idx = torch.triu_indices(input_dim, input_dim, offset=1)
        flat_idx = row_idx * input_dim + col_idx
        self.register_buffer('flat_idx', flat_idx)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        
        L_flat = self.mlp(z)

        zeros_flat = torch.zeros(batch_size, self.input_dim * self.input_dim, device=z.device, dtype=z.dtype)

        expanded_idx = self.flat_idx.unsqueeze(0).expand(batch_size, -1)

        L_full_flat = zeros_flat.scatter(dim=1, index=expanded_idx, src=L_flat)

        L = L_full_flat.view(batch_size, self.input_dim, self.input_dim)
        
        L_antisym = L - L.transpose(1, 2)
        
        return L_antisym
    

class LinearL(BaseL):
    """Creates an L matrix with linear terms"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.L_constant_part = TrainableConstantL(dim)
        
        self.T_param = nn.Parameter(torch.randn(dim, dim, dim))

    def get_linear_constants(self) -> torch.Tensor:
        return self.T_param - self.T_param.transpose(0, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        L_c_batch = self.L_constant_part(z)

        c = self.get_linear_constants()
        L_l_batch = torch.einsum('ijk,bk->bij', c, z)
        
        return L_c_batch + L_l_batch
