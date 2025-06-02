import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, obs_shape, latent_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_shape, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )

    def forward(self, obs):
        return self.encoder(obs)
