import numpy as np
import gym
import torch

from models import MLP
from tools import do_vpg_training, get_sample_trajectory
from tools import do_episode

policy = MLP([4,10,2], activation=torch.tanh)
env = gym.make('CartPole-v0')
max_timesteps=200
batch_size = 50
num_batches = 10

# states, actions, rewards, log_probs = do_episode(policy, env, max_timesteps)

# optimizer = torch.optim.Adam(policy.parameters(),lr=.1)

# avg_returns = do_vpg_training(policy, env, max_timesteps, 
#                             optimizer=optimizer,batch_size=batch_size, num_batches=num_batches)

s, a, r, lp = do_episode(policy, env, max_timesteps=1000, stop_on_done=False, render=True)