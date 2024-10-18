import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    Implements an MLP with parameterized layers and activation function.
'''

class BaseMLP(nn.Module):
    def __init__(self, input_size, layer_sizes, activ_func):
        super(BaseMLP, self).__init__()
        
        self.fcs = nn.ModuleList([nn.Linear(input_size, layer_sizes[0])] + 
                                 [nn.Linear(layer_sizes[i-1], layer_sizes[i]) 
                                  for i in range(1, len(layer_sizes))])
        self.final_layer = nn.Linear(layer_sizes[-1], 1)
        self.activ = activ_func
        
    def forward(self, x):
        for fc in self.fcs:
            x = self.activ(fc(x))
        return self.final_layer(x)
