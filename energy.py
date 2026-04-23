import torch
import torch.nn as nn


class EnergyNet(nn.Module):
    """A neural network that learns the Hamiltonian H(z)"""
    def __init__(self, input_dim: int, hidden_dim: int = 128, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        
        self.dim = input_dim

        model_layers = [nn.Linear(input_dim, hidden_dim), nn.Softplus()]
        if dropout > 0:
            model_layers.append(nn.Dropout(dropout))

        for _ in range(layers - 1):
            model_layers.append(nn.Linear(hidden_dim, hidden_dim))
            model_layers.append(nn.Softplus())
            if dropout > 0:
                model_layers.append(nn.Dropout(dropout))

        model_layers.append(nn.Linear(hidden_dim, 1))

        self.mlp = nn.Sequential(*model_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)
