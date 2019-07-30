from sacred import Experiment, Ingredient
import gym 
import torch
import sys
sys.path.append('../..')

from ingredients import model, environment
from ingredients import get_model, get_env
training = Experiment('do-cartpole-training', 
                        ingredients=(model, environment))

@training.config
def training_config():
    #baseline, if any
    baseline = None
    #number of episodes to train
    episodes = 500
    #max allowed timesteps
    max_episode_timesteps = 200

@training.config
def hyperparam_config():
    """Set hyperparams for model training"""
    #learning rate
    lr = .01
    # how many episodes to use per gradient update
    batch_size = 16
    #discount rate
    discount = 1.0

@training.automain
def train(baseline, episodes, max_episode_timesteps,
            lr, batch_size, discount, seed):
    """ Train a single model via policy gradient. """
    
    from torch.optim import Adam
    from rl.tools import do_pg_training

    torch.manual_seed(seed)    
    policy = get_model()
    policy_optimizer = Adam(policy.parameters(), lr=lr)


    env = get_env
    __ = do_pg_training(policy, env, max_episode_timesteps, )