from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('transforms_logs')
img_path = 'Data_Set/train/ants/0013035.jpg'
img = Image.open(img_path)

# ToTensor
tensor_trans = transforms.ToTensor()
out_img_tensor = tensor_trans(img)
writer.add_image('ToTensor', out_img_tensor)

# Normalize
norm_trans = transforms.Normalize([1, 0.5, 3], [0.5, 2, 0.5])
out_img_norm = norm_trans(out_img_tensor)
writer.add_image('Normalize', out_img_norm)

# Resize
# print(img.size)
resize_trans = transforms.Resize((512, 512))
out_img_resize = resize_trans(img)
out_img_resize = tensor_trans(out_img_resize)
# print(out_img_resize)
writer.add_image('Resize', out_img_resize, 0)

# Compose+Resize
resize_trans_2 = transforms.Resize(512)
compose_trans = transforms.Compose([resize_trans_2, tensor_trans])
out_img_resize_2 = compose_trans(img)
writer.add_image('Resize', out_img_resize_2, 1)

writer.close()
