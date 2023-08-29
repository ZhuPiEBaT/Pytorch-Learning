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
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


cnn = CNN()

writer = SummaryWriter('CNN_MaxPool_logs')
step = 0

for data in data_loader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output = cnn(imgs)
    writer.add_images('output', output, step)
    step += 1

writer.close()
