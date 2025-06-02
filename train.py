import torch
import torch.nn as nn
import torch.optim as optim
from envs.make_env import make_env
from model.encoder import Encoder
from model.rssm import RSSM
from model.decoder import Decoder
from agent.agent import Agent
from utils import ReplayBuffer, set_seed
import os
import numpy as np
import matplotlib.pyplot as plt

# Configuration
ENV_NAME = 'CartPole-v1'
SEED = 42
NUM_EPISODES = 100
MAX_STEPS = 500
BATCH_SIZE = 32
LATENT_DIM = 32
REPLAY_CAPACITY = 10000

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment and Models
env = make_env(ENV_NAME, SEED)
obs_shape = env.observation_space.shape
action_dim = env.action_space.n

encoder = Encoder(input_dim=obs_shape[0], latent_dim=LATENT_DIM).to(device)
rssm = RSSM(latent_dim=LATENT_DIM, action_dim=action_dim).to(device)
decoder = Decoder(latent_dim=LATENT_DIM, output_dim=obs_shape[0]).to(device)
agent = Agent(encoder, rssm, decoder, action_dim, LATENT_DIM).to(device)

optimizer = optim.Adam(agent.parameters(), lr=1e-3)
buffer = ReplayBuffer(REPLAY_CAPACITY, obs_shape, action_dim)

all_rewards = []

for episode in range(NUM_EPISODES):
    obs = env.reset()
    episode_reward = 0

    for step in range(MAX_STEPS):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        action = agent.act(obs_tensor)

        next_obs, reward, done, _ = env.step(action)
        buffer.add(obs, action, reward, next_obs, done)

        if buffer.size >= BATCH_SIZE:
            batch = buffer.sample(BATCH_SIZE)
            loss = agent.learn(batch, optimizer)

        obs = next_obs
        episode_reward += reward
        if done:
            break

    all_rewards.append(episode_reward)
    print(f"Episode {episode+1}, Reward: {episode_reward:.2f}")

# Save results
os.makedirs("results", exist_ok=True)
plt.plot(all_rewards)
plt.title("Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig("results/rewards_plot.png")
plt.close()

print("Training finished.")
