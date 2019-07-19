import numpy as np
import gym
import torch

from rl.models import MLP
from rl.tools import do_vpg_training, get_sample_trajectory
from rl.tools import do_episode, compute_rewards_to_go

policy = MLP([4,20,20,2], activation=torch.relu)
env = gym.make('CartPole-v0')
max_timesteps=200
batch_size = 16
num_batches = 40

# states, actions, rewards, log_probs = do_episode(policy, env, max_timesteps)

optimizer = torch.optim.Adam(policy.parameters(),lr=.01)

avg_returns = do_vpg_training(policy, env, max_timesteps, 
                            optimizer=optimizer,batch_size=batch_size, num_batches=num_batches, 
                            baseline='running_average_Q')
