import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备的测试集

test_data = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# img, target = test_data[0]
# print(img.shape, target)
writer = SummaryWriter('CIFAR10_Dataloader')
step = 0
for data in test_loader:
    imgs, targets = data
    writer.add_images("DataLoader", imgs, step)
    step += 1
writer.close()
