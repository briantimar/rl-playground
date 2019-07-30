import numpy as np
import gym
import torch

from rl.models import MLP, ModelStepper, HyperParams
from rl.tools import do_pg_training, get_sample_trajectory
from rl.tools import do_episode, compute_rewards_to_go

policy = MLP([4,20,20,3], activation=torch.relu, output_critic=True)
env = gym.make('CartPole-v0')
max_timesteps=200
batch_size = 32
num_batches = 40

# states, actions, rewards, log_probs = do_episode(policy, env, max_timesteps)

policy_optimizer = torch.optim.Adam(policy.parameters(),lr=.01)
critic_optimizer = torch.optim.Adam(policy.parameters(),lr=.01)

# value_model_hp = HyperParams(lr=.1, layer_sizes=(4, 20, 20, 1))
# value_modelstepper = ModelStepper(MLP, lossfn=torch.nn.MSELoss(), 
#                                 hyperparams=value_model_hp, optimizer=torch.optim.Adam)

avg_returns = do_pg_training(policy, env, max_timesteps, 
                            policy_optimizer=policy_optimizer,batch_size=batch_size, num_batches=num_batches, 
                            critic_optimizer=critic_optimizer,
                            baseline='policy_value_model')
