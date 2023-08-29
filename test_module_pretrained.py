import torch
import torchvision

# data_set = torchvision.datasets.ImageNet('./data', split='train', transform=torchvision.transforms.ToTensor)

# vgg16_true = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
vgg16_false = torchvision.models.vgg16(weights=None)
# print(vgg16_false)

# 保存方式1,保存模型结构和参数
torch.save(vgg16_false, 'vgg16_method1.pth')

# 保存方式2，以字典形式保存模型参数（官方推荐）
torch.save(vgg16_false.state_dict(), 'vgg16_method2.pth')

# 打开方式1（打开保存方式1保存的模型）
module1 = torch.load('vgg16_method1.pth')
print(module1)

# 打开方式2（打开保存方式2保存的模型）
module2 = torch.load('vgg16_method2.pth')   # 加载模型的参数
print(module2)
# vgg16_false.load_state_dict(torch.load('vgg16_method2.pth'))  # 加载模型结构和参数
