import time
import torch.optim
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from complete_train_module import Net

# 准备数据集
"""数据集分为训练数据集和测试数据集"""
data_train = torchvision.datasets.CIFAR10('./CIFAR10', train=True, transform=torchvision.transforms.ToTensor())
data_test = torchvision.datasets.CIFAR10('./CIFAR10', train=False, transform=torchvision.transforms.ToTensor())

# 定义训练的设备GPU/CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # 三元运算符
print(device)

# 查看数据集大小
data_train_size = len(data_train)
data_test_size = len(data_test)
# print(f"训练集的数据长度:{data_train_size}")
# print(f"测试集的数据长度:{data_test_size}")

# 加载数据集
dataloader_train = DataLoader(data_train, batch_size=64)
dataloader_test = DataLoader(data_test, batch_size=64)

# 创建网络模型
net = Net()
# if torch.cuda.is_available():
#     net = net.cuda()
net = net.to(device)

# 损失函数
loss_function = nn.CrossEntropyLoss()
# if torch.cuda.is_available():
#     loss_function = loss_function.cuda()
loss_function = loss_function.to(device)

# 优化器(选择SGD,随机梯度下降)
learning_rate = 1e-2  # 学习率
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# 设置训练网络的一些变量
total_train_step = 0    # 训练的次数
total_test_step = 0    # 测试的次数
epoch = 10   # 训练的轮数

# 添加tensorboard
writer = SummaryWriter('./complete_train_logs')

# 开始训练
net.train()  # 如果网络中无Dropout和BatchNorm层则此语句可有可无
start_time = time.time()
for i in range(epoch):
    print(f"-----第{i+1}轮训练开始----")
    for data in dataloader_train:
        imgs, targets = data
        # if torch.cuda.is_available():
        #     imgs = imgs.cuda()
        #     targets = targets.cuda()
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = net(imgs)
        loss = loss_function(outputs, targets)
        # 优化器调用
        optimizer.zero_grad()   # 梯度清零
        loss.backward()     # 反向传播
        optimizer.step()    # 开始优化
        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            # print(f"运行时间：{end_time - start_time}")
            print(f"训练次数：{total_train_step},Loss:{loss}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试训练效果
    net.eval()  # 如果网络中无Dropout和BatchNorm层则此语句可有可无
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():   # 测试不需要对梯度进行调整
        for data in dataloader_test:
            imgs, targets = data
            # if torch.cuda.is_available():
            #     imgs = imgs.cuda()
            #     targets = targets.cuda()
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = net(imgs)
            loss = loss_function(outputs, targets)
            total_test_loss = total_test_loss + loss.item()  # loss为tensor数据类型，加上item则为数字类型
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print(f"整体测试集上的Loss:{total_test_loss}")
    print(f"整体测试集上的准确率：{total_accuracy.item()/data_test_size}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy, total_test_step)
    total_test_step += 1

    # 保存模型
    torch.save(net, f"complete_train_module{i+1}_gpu.pth")
    print("模型已保存")
writer.close()
