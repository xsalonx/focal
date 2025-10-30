import torch
import torch.nn.functional as F
from torch import nn


class ParticleShowerNet(nn.Module):
    def __init__(self, latent_dim=16, input_dim=3, output_dim=148):
        super(ParticleShowerNet, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        self.cond_mlp = nn.Sequential(
            nn.Linear(input_dim, 8),
        )

        self.latent_mlp = nn.Sequential(
            nn.Linear(latent_dim, 56),
            nn.LeakyReLU(0.2),
        )

        self.fc_mu = nn.Linear(64, 64)
        self.fc_logvar = nn.Linear(64, 64)

        self.energy_head = nn.Linear(64, output_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Sample from standard normal
        return mu + eps * std

    def forward(self, z, cond):

        hidden = torch.cat([self.latent_mlp(z), self.cond_mlp(cond)], dim=1)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        x = self.reparameterize(mu, logvar)

        return torch.cumsum(F.softplus(self.energy_head(x)), dim=1)
    

def get_optimizer(model, lr):
    """
        Get optimizer and lr_scheduler

        Return: 
            optimizer, lr_scheduler
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.87, 0.82), eps=2.5e-10, weight_decay=5.5e-4)
    return optimizer