import torch
import numpy as np
import imageio
import os
from envs.make_env import make_env
from model.encoder import Encoder
from model.rssm import RSSM
from model.decoder import Decoder
from agent.agent import Agent
from utils import set_seed

ENV_NAME = 'CartPole-v1'
SEED = 42
EPISODE_STEPS = 500
LATENT_DIM = 32

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = make_env(ENV_NAME, SEED)
obs_shape = env.observation_space.shape
action_dim = env.action_space.n

encoder = Encoder(input_dim=obs_shape[0], latent_dim=LATENT_DIM).to(device)
rssm = RSSM(latent_dim=LATENT_DIM, action_dim=action_dim).to(device)
decoder = Decoder(latent_dim=LATENT_DIM, output_dim=obs_shape[0]).to(device)
agent = Agent(encoder, rssm, decoder, action_dim, LATENT_DIM).to(device)

obs = env.reset()
frames = []
reward_total = 0

for _ in range(EPISODE_STEPS):
    frame = env.render(mode='rgb_array')
    frames.append(frame)

    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    action = agent.act(obs_tensor)
    obs, reward, done, _ = env.step(action)
    reward_total += reward

    if done:
        break

env.close()

os.makedirs("results", exist_ok=True)
gif_path = "results/sample_episode.gif"
imageio.mimsave(gif_path, frames, fps=30)
print(f"Test episode finished. Total reward: {reward_total:.2f}")
