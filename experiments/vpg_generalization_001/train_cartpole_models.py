import sys
sys.path.append('../..')
from models import MLP
import os
from tools import do_vpg_training
import gym
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

# how long to train each model
T_train = 100
# how many policies to train per model
n_train = 10
#where to save the trained models
MODEL_SAVE_DIR = "models"

batch_size = 16
num_batches = 50

# constructors for a few different models.
linear_constructor = lambda: MLP([4,2], activation=None)
MLP_onelayer_constructor = lambda: MLP([4,10,2])
MLP_twolayer_constructor = lambda : MLP([4,10,10,2])

constructors = { 'linear': linear_constructor, 
                'mlp_onelayer': MLP_onelayer_constructor, 
                'mlp_twolayer': MLP_twolayer_constructor}

#cartpole environment
env = gym.make('CartPole-v0')

def make_filename(model_name, seed_index):
    return os.path.join(MODEL_SAVE_DIR, 
                        'cartpolev0_{0}_{1}'.format(model_name, seed_index))

def do_training(model_name, seed_index):
    """ Train specified model in cartpole environment and save."""
    policy = constructors[model_name]()
    optimizer = torch.optim.Adam(policy.parameters(), lr=.01)
    avg_return = do_vpg_training(policy, env, T_train,
                                    optimizer=optimizer,batch_size=batch_size, num_batches=num_batches )
    fname = make_filename(model_name, seed_index)
    model_name = fname + "_model"
    return_name = fname + "_returns"
    policy.save(model_name)
    np.save(return_name, avg_return)


if __name__ == "__main__":
    for model in constructors:
        for i in range(n_train):
            print("Training on {0} -- {1}".format(model, i))
            do_training(model, i)
