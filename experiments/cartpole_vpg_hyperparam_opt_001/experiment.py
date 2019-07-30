from sacred import Experiment, Ingredient
from sacred.observers import MongoObserver
import gym 
import torch
import sys
sys.path.append('../..')

from ingredients import model, environment
from ingredients import get_model, get_env
name = "cartpole-vpg-hyperparam-001"
training = Experiment(name, 
                        ingredients=(model, environment))

#log training stats in Mongodb
training.observers.append(MongoObserver.create())

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
def train( max_episode_timesteps,
                episodes, batch_size,baseline,
            lr, discount, seed):
    """ Train a single model via policy gradient. """
    
    from torch.optim import Adam
    from rl.tools import do_pg_training

    torch.manual_seed(seed)    
    policy = get_model()
    policy_optimizer = Adam(policy.parameters(), lr=lr)

    env = get_env()
    def avg_return_logger(r):
        training.log_scalar(name + ".avg_return", r)
    def loss_logger(l):
        training.log_scalar(name + ".loss", l)

    avg_return = do_pg_training(policy, env, max_episode_timesteps,
                        policy_optimizer=policy_optimizer,
                        batch_size=batch_size, num_episodes=episodes,
                        baseline=baseline, 
                        avg_return_logger=avg_return_logger,
                        loss_logger=loss_logger)
