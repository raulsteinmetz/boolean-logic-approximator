import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseMLP(nn.Module):
    def __init__(self, input_size, layer_sizes, activ_func):
        super(BaseMLP, self).__init__()
        self.fcs = [nn.Linear(input_size, layer_sizes[0])] + \
                   [nn.Linear(layer_sizes[i-1], layer_sizes[i]) \
                    for i in range(1, len(layer_sizes))]
        self.activ = activ_func
        
    def forward(self, x):
        for i in range(len(self.fcs) - 1):
            x = self.activ(self.fcs[i](x))
        return self.fcs[-1](x)
        

def main():
    model = BaseMLP(10, [10, 100, 10, 5, 2], F.relu)
    print(model(torch.randn(10)))


if __name__ == '__main__':
    main()