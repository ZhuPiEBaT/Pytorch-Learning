import os
from torch.utils.data import Dataset
from PIL import Image


class MyData(Dataset):

    """Dataset用于自己设置训练数据集，
       而datasets则是设置官方提供的数据集。
    """

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir  # 获取根路径
        self.label_dir = label_dir  # 获取图片标签路径
        self.path = os.path.join(self.root_dir, self.label_dir)  # 将路径拼接起来
        self.img_path = os.listdir(self.path)  # 将所有图片的名称打包为列表

    def __getitem__(self, idx):
        img_name = self.img_path[idx]  # 获取不同索引下的图片名称
        img_itm_path = os.path.join(self.root_dir, self.label_dir, img_name)  # 获取每个图片的路径
        img = Image.open(img_itm_path)  # 打开图片
        label = self.label_dir  # 图片标签
        return img, label

    def __len__(self):
        return len(self.img_path)


out_root_dir = 'Data_Set/train'
ants_label_dir = 'ants'
bees_label_dir = 'bees'
ants_dataset = MyData(out_root_dir, ants_label_dir)
bees_dataset = MyData(out_root_dir, bees_label_dir)
train_dataset = ants_dataset + bees_dataset
