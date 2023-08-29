import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.module1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=(5, 5), padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.module1(x)
        return x


net = Net()
# print(net)
# 检验模型是否正确
input = torch.ones(64, 3, 32, 32)
output = net(input)
# print(output.shape)

# 模型可视化
writer = SummaryWriter('Seq_logs')
writer.add_graph(net, input)
writer.close()
