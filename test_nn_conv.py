import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, transform=torchvision.transforms.ToTensor())
data_loader = DataLoader(test_set, batch_size=64)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


cnn = CNN()
# print(cnn)

writer = SummaryWriter('CNN_logs')
step = 0
for data in data_loader:
    imgs, targets = data
    output = cnn(imgs)
    # print(imgs.shape)
    # print(output.shape)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images('Input', imgs, step)
    writer.add_images('Output', output, step)
    step += 1

writer.close()
