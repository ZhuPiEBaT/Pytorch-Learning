import numpy as np
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

writer = SummaryWriter('logs')

# add_image的应用
img_path = 'Data_Set/train/bees/16838648_415acd9e3f.jpg'
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)
writer.add_image('test', img_array, 2, dataformats='HWC')


# add_scalar的应用
for i in range(100):
    writer.add_scalar("y=2x ", 2*i, i)

writer.close()
# 要查看过程，需在终端里输入tensorboard --logdir=logs
