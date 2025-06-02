import torch
import torch.nn as nn

class RSSM(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(RSSM, self).__init__()
        self.rnn = nn.GRUCell(latent_dim, hidden_dim)
        self.fc_prior = nn.Linear(hidden_dim, latent_dim)
        self.fc_posterior = nn.Linear(hidden_dim + latent_dim, latent_dim)

    def forward(self, prev_state, embedded_obs=None):
        hidden = self.rnn(embedded_obs, prev_state)
        prior = self.fc_prior(hidden)
        if embedded_obs is not None:
            posterior = self.fc_posterior(torch.cat([hidden, embedded_obs], dim=-1))
        else:
            posterior = prior
        return hidden, prior, posterior
