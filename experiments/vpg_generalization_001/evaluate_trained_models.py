import sys
sys.path.append('../..')
from models import MLP
import os
from tools import do_episode
import gym
import numpy as np
import torch
import json

torch.manual_seed(0)
np.random.seed(0)

# times at which to evaluate trained models
T_eval = [N * 100 for N in range(1, 11)]
# number of evaluations per model

SAVEDIR = "data"

n_train = 10
n_eval = 30

# constructors for a few different models.
linear_constructor = lambda: MLP([4,2], activation=None)
MLP_onelayer_constructor = lambda: MLP([4,10,2])
MLP_twolayer_constructor = lambda : MLP([4,10,10,2])

constructors = { 'linear': linear_constructor, 
                'mlp_onelayer': MLP_onelayer_constructor, 
                'mlp_twolayer': MLP_twolayer_constructor}

def make_filename(model_name, seed_index):
    return os.path.join("models", 
                        'cartpolev0_{0}_{1}'.format(model_name, seed_index))

env = gym.make('CartPole-v0')

returns = {}

for model in constructors:
    returns[model] = {}
    for n in range(n_train):
        returns[model][n] = {}
        policy = constructors[model]()
        model_name = make_filename(model, n) + "_model"
        policy.load(model_name)

        for T in T_eval:
            print("model {0}, {1} for {2} timesteps".format(model,n,T))
            ret = []
            for m in range(n_eval):
                __,__, rewards,__ = do_episode(policy, env, max_timesteps=T, stop_on_done=True)
                ret.append(sum(rewards).item())
            returns[model][n][T] = ret

with open(os.path.join(SAVEDIR, 'returns.json'),'w') as f:
    json.dump(returns, f)
            