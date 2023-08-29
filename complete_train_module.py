import torch
from torch import nn


# 搭建网络
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


# 调试
if __name__ == '__main__':
    net = Net()  # 创建一个网络模型
    input = torch.ones((64, 3, 32, 32))
    output = net(input)
    print(output.shape)
