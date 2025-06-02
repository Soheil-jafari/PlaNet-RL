import torch
import torch.nn as nn
from encoder import Encoder
from rssm import RSSM
from decoder import Decoder
from planner import Planner

class Agent(nn.Module):
    def __init__(self, obs_shape, latent_dim, hidden_dim, action_dim):
        super(Agent, self).__init__()
        self.encoder = Encoder(obs_shape, latent_dim)
        self.rssm = RSSM(latent_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, obs_shape)
        self.planner = Planner(latent_dim, action_dim)
        self.hidden_dim = hidden_dim

    def forward(self, obs, prev_state):
        embedded = self.encoder(obs)
        hidden, prior, posterior = self.rssm(prev_state, embedded)
        recon = self.decoder(posterior)
        action = self.planner(posterior)
        return recon, action, hidden
