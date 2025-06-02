import torch
import torch.nn as nn
import torch.nn.functional as F

class Planner(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super(Planner, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, latent):
        return torch.tanh(self.mlp(latent))
