import torch
from torch.distributions.categorical import Categorical
from dataclasses import dataclass

class Policy(torch.nn.Module):
    """ Feed-forward network that represents a stochastic policy.
        Inputs are observations from environment.
        The output layer defines logits, which should be fed to a softmax to obtain action
        probabilities. """

    def __init__(self, output_critic=False):
        """ `output_critic`: if True, network will also emit scalar 'critic' signal."""
        super().__init__()
        self.output_critic = output_critic
    
    def sample_action_with_log_prob(self, input, stochastic=True):
        """Sample action and compute its log-probability under model and input. 
            input: tensor of input values
            action: (batch_size, k) tensor of actions, with k an integer in 0, ... , num_actions -1
            If `output_critic` is True: returns action, logprobs, critc
                Otherwise, returns action, logprobs
            stochastic: bool. If false, the most likely action is returned deterministically.
                """

        outputs = self(input)
        # if model computes critic, don't include in logits
        if self.output_critic:
            # print(outputs.shape)
            logits = outputs[..., :-1]
            critic = outputs[..., -1]
        else:
            logits = outputs
            critic = None
        if stochastic:
            probs = Categorical(logits=logits)
            action = probs.sample()
            logprobs = probs.log_prob(action)
        else:
            action = logits.argmax(dim=-1)
            logprobs = torch.zeros_like(action, dtype=torch.float)
        if critic is None:
            return action, logprobs
        return action, logprobs, critic

    def save(self, path):
        """Saves the model's state dict to the path provided"""
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        """Load model from param_dict at the specified path."""
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)


class MLP(Policy):

    def __init__(self, layer_sizes, output_critic=False,activation=torch.relu):
        super().__init__(output_critic=output_critic)
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

    @classmethod
    def create_from_hyperparams(cls, hp):
        layer_sizes = hp.layer_sizes
        if hasattr(hp, 'activation'):
            activation = hp.activation
        else:
            activation = torch.relu
        return cls(layer_sizes, activation=activation)

class PolicyFamily:
    """ A collection of policies, which can vote on actions during rollout."""

    def __init__(self, policies):
        self.policies = policies
        self.output_critic = False

    def sample_action_with_log_prob(self, state, stochastic=True):
        """Returns a single action by averaging over model logits.
             Each policy is assigned an equal weight. """

        logits_all = [p.forward(state) for p in self.policies]
        mean_logits = torch.stack(logits_all).mean(dim=0)
        if stochastic:
            probs = Categorical(logits=mean_logits)
            action = probs.sample()
            logprobs = probs.log_prob(action)
        else:
            action = mean_logits.argmax(dim=-1)
            logprobs = torch.zeros_like(action, dtype=torch.float)
        return action, logprobs



@dataclass
class HyperParams:
    """Class holding hyperparameters for defining and training pytorch models"""
    lr: float = .01
    layer_sizes: tuple = ()
    batch_size: int = 0
    epochs: int = 0


class ModelStepper:
    """Class to package a trainable model. Holds the model's trainable params and hyperparams, 
    and updates and evaluates on demand."""

    def __init__(self, modelfactory, lossfn, hyperparams, optimizer):
        """ modelfactory: model constructor implementing a create_from_hyperparams method.
            lossfn: given model output and targets, computes loss.
            hyperparams: dataclass holding architecture and training specs
            optimizer: something from torch.optim. Currently only takes an lr argument!
            """
        self.model = modelfactory.create_from_hyperparams(hyperparams)
        self.hyperparams = hyperparams
        self.optimizer = optimizer(self.model.parameters(), lr=hyperparams.lr)
        self.lossfn = lossfn
        self.losses = []

    def step(self, inputs, targets):
        """ Take a gradient descent step"""
        self.model.train()
        outputs = self.model(inputs)
        loss = self.lossfn(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.detach().numpy())

    def eval(self, inputs):
        """ Evaluate model on the inputs provided """
        self.model.eval()
        return self.model(inputs)

    
    