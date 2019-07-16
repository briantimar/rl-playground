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
