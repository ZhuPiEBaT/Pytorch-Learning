import torch
import torch.nn as nn


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()

    def forward(self, input):
        output = input + 1
        return output

module = Module()
x = torch.tensor(1.0)
output = module(x)
print(output)
