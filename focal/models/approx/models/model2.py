import torch
import torch.nn.functional as F
from torch import nn


import os
script_path = os.path.abspath(__file__)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ParticleShowerNet(nn.Module):
    def __init__(self, input_dim=3, latent_dim=64, output_dim=148):
        super(ParticleShowerNet, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, output_dim)
        )

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Sample from standard normal
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed


def get_optimizer(model, lr):
    """
        Get optimizer and lr_scheduler

        Return: 
            optimizer, lr_scheduler
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.87, 0.82), eps=2.5e-10, weight_decay=5.5e-4)
    return optimizer