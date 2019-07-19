import torch
from torch.distributions.categorical import Categorical

class Policy(torch.nn.Module):
    """ Feed-forward network that represents a stochastic policy.
        Inputs are observations from environment.
        The output layer defines logits, which should be fed to a softmax to obtain action
        probabilities. """

    def __init__(self):
        super().__init__()
    
    def sample_action_with_log_prob(self, input):
        """Sample action and compute its log-probability under model and input. 
            input: tensor of input values
            action: (batch_size, k) tensor of actions, with k an integer in 0, ... , num_actions -1"""
        logits = self(input)
        probs = Categorical(logits=logits)
        action = probs.sample()
        logprobs = probs.log_prob(action)
        return action, logprobs

    def save(self, path):
        """Saves the model's state dict to the path provided"""
        torch.save(self.state_dict(), path)

    
    def load(self, path):
        """Load model from param_dict at the specified path."""
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)


class MLP(Policy):

    def __init__(self, layer_sizes, activation=torch.relu):
        super().__init__()
        self.layer_sizes = layer_sizes
        #number of layers including input and output
        self.num_layer = len(layer_sizes)
        self.activation = activation
        for i in range(len(layer_sizes)-1):
            l = torch.nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1])               
            self.add_module("layer_%d"%i, l)

    def forward(self, x):
        for i in range(self.num_layer-1):
            layer = getattr(self, "layer_%d"%i)
            x = layer(x)
            if i < self.num_layer - 2 and self.activation is not None:
                x = self.activation(x)
        return x
