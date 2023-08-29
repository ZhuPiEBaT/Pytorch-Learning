import torch
import torchvision
from PIL import Image
# from complete_train_module import Net

image_path = "./image/dog.png"
image = Image.open(image_path)
# print(image)
image = image.convert('RGB')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
image = torch.reshape(image, (1, 3, 32, 32))
# image = image.cuda()    # 1.将输入转为cuda类型
# print(image.shape)

# 加载模型
# module = torch.load('complete_train_module1.pth')  # 数据为cuda类型时语句
module = torch.load('complete_train_module10_gpu.pth', map_location=torch.device('cpu'))     # 将用GPU训练好的模型映射到CPU上
print(module)
module.eval()
with torch.no_grad():
    output = module(image)
print(output)
print(output.argmax(1))

