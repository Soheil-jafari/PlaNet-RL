import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_dim, obs_shape):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, obs_shape)
        )

    def forward(self, latent):
        return self.decoder(latent)
