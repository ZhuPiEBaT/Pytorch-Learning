import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

data_set = torchvision.datasets.CIFAR10('./CIFAR10', train=False, transform=torchvision.transforms.ToTensor())
data_loader = DataLoader(data_set, batch_size=64)
# print(len(data_set))  # 10000
# print(len(data_loader))   # 157  156*64+16=10000


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
loss = nn.CrossEntropyLoss()    # 损失函数
optim = torch.optim.SGD(net.parameters(), lr=0.01)

for epoch in range(20):
    running_loss = 0
    for data in data_loader:
        imgs, targets = data
        outputs = net(imgs)
        result_loss = loss(outputs, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        # print(outputs)
        # print(targets)
        # print(result_loss)
        running_loss = running_loss + result_loss
    print(epoch, running_loss)

