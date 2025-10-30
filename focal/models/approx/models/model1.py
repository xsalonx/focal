import torch
import torch.nn.functional as F
from torch import nn


import os
script_path = os.path.abspath(__file__)

class ParticleShowerNet(nn.Module):
    def __init__(self, hidden_dims=[64, 64]):
        super(ParticleShowerNet, self).__init__()

        # Shared trunk
        layers = []
        input_dim = 3
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.shared_trunk = nn.Sequential(*layers)

        self.energy_head = nn.Linear(hidden_dims[-1], 148)

    def forward(self, x):
        x = self.shared_trunk(x)

        energy_deltas = F.softplus(self.energy_head(x))  # Ensures positive deltas
        energy_out = torch.cumsum(energy_deltas, dim=1)  # Strictly increasing output
        return energy_out
    

def get_optimizer(model, lr):
    """
        Get optimizer and lr_scheduler

        Return: 
            optimizer, lr_scheduler
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.87, 0.82), eps=2.5e-10, weight_decay=5.5e-4)
    return optimizer