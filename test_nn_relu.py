import torch
import torch.nn as nn

input = torch.tensor([[1, -0.5],
                      [-1, 3]])


class ReLu(nn.Module):
    def __init__(self):
        super(ReLu, self).__init__()
        self.relu1 = nn.ReLU()

    def forward(self, input):
        output = self.relu1(input)
        return output


relu = ReLu()
output = relu(input)
print(output)
