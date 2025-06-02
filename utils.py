import os
import random
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, obs_shape, action_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=torch.tensor(self.obs[idx]),
            actions=torch.tensor(self.actions[idx]),
            rewards=torch.tensor(self.rewards[idx]),
            next_obs=torch.tensor(self.next_obs[idx]),
            dones=torch.tensor(self.dones[idx])
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model
