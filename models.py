import torch

class MLP(torch.nn.Module):

    def __init__(self, layer_sizes):
        super().__init__()
        self.layer_sizes = layer_sizes
        for i in range(len(layer_sizes)-1):
            l = torch.nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1])
            self.add_module("layer_%d"%i, l)

    def forward(self, x):
        for layer in self.modules():
            x = layer(x)
        return x
    
class MLPDeterministic(MLP):
    """ MLP for making discrete decisions deterministically"""

    def __init__(self, input_size, hidden_layer_sizes, output_size):
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        super().__init__(layer_sizes)

    def decide(self, x):
        """ Make a decision based on input based on argmax of the output. 
            Returns: integer in [0, ..., output_size -1] """
        logits = self(x)
        return logits.argmax(dim=-1)

    

class MLPStochastic(MLP):
    """ Multi-layer perceptron for making decisions probabilistically 
    """
    pass