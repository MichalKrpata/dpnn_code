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

        model_layers.append(nn.Linear(hidden_dim, input_dim * input_dim))

        self.mlp = nn.Sequential(*model_layers)
        self.input_dim = input_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        
        L_flat = self.mlp(z)  # [batch, dim*dim]
        L = L_flat.view(batch_size, self.input_dim, self.input_dim)
        
        L_antisym = L - L.transpose(1, 2)
        
        return L_antisym
    
    def forward_with_jacobian(self, z: torch.Tensor):
        """Also computes the Jacobian matrix of the network manually (used in jacobi_loss_manual)"""
        B, D = z.shape
        x = z
        
        J = torch.eye(D, device=z.device).unsqueeze(0).expand(B, -1, -1)
        
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                x = layer(x)
                J = torch.einsum('oi,bij->boj', layer.weight, J)
                
            elif isinstance(layer, nn.Softplus):
                d_act = torch.sigmoid(x)
                x = layer(x)
                J = d_act.unsqueeze(-1) * J
                
            elif isinstance(layer, nn.Dropout):
                x_new = layer(x)
                if self.training and layer.p > 0:
                    mask = (x_new != 0.0).float() * (1.0 / (1.0 - layer.p))
                    J = mask.unsqueeze(-1) * J
                x = x_new
                
            else:
                print("Warning in forward_with_jacobian(): layer not recognized")
                x = layer(x)

        L_flat = x
        L = L_flat.view(B, D, D)
        
        J_L = J.view(B, D, D, D)
        
        L_antisym = L - L.transpose(1, 2)
        J_antisym = J_L - J_L.transpose(1, 2)
        
        return L_antisym, J_antisym
    

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
