from sacred import Experiment, Ingredient
import gym 
import torch
import sys
sys.path.append('../..')

model = Ingredient('model')
environment = Ingredient('environment')

@model.config
def model_config():
    """Define the size of the model, and how to train it"""

    #size of the state space
    state_dim = 4
    #model hidden layer sizes
    hidden_layer_sizes = [32, 32]
    #how many choices of action the model has
    output_dim = 2
    #how to baseline, if at all
    baseline = None
    #all layer sizes of the policy model
    layer_sizes = [state_dim] + hidden_layer_sizes + [output_dim]
    
@model.capture
def get_model(layer_sizes):
    """ Returns policy model."""
    from rl.models import MLP
    return MLP(layer_sizes, output_critic=False)

@environment.config
def environment_config():
    """Set environment, number of episodes."""
    #name of the openai environment
    env_name = 'CartPole-v0'
    

@environment.capture
def get_env(env_name):
    """Returns environment and episode limitations."""
    env = gym.make(env_name)
    return env

