from sacred import Experiment, Ingredient
import gym 
import torch

training = Ingredient('do-cartpole-training')

@training.config
def model_config():
    """Define the environment, the size of the model, and how to train it"""
    #name of the openai environment
    env_name = 'CartPole-v0'
    #size of the state space
    state_dim = 4
    #model hidden layer sizes
    hidden_layer_sizes = [32, 32]
    #how to baseline, if at all
    baseline = None

@training.config
def hyperparam_config():
    """Set hyperparams for model training"""
    lr = .01
    batch_size = 16
    num_batches = 50


