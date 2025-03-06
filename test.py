import numpy as np

import torch

import torchvision.models as models

from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader

from compute_pretrained_embeddings import get_embeddings

from torchvision import transforms



# 自定义数据集，返回图像、路径和索引

class PathImageFolder(ImageFolder):

    def __getitem__(self, index):

        path, _ = self.samples[index]

        img = super().__getitem__(index)[0]  # 获取处理后的图像

        return img, path, index



# 自定义collate函数处理字符串路径

def collate_fn(batch):

    images = torch.stack([item[0] for item in batch])

    paths = [item[1] for item in batch]

    indices = torch.tensor([item[2] for item in batch])

    return images, paths, indices



# 初始化模型（以ResNet-50为例）

model = models.resnet50(pretrained=True)

model.fc = torch.nn.Identity()  # 获取特征向量

model.eval()



# 数据预处理（根据模型调整）

transform = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])



# 创建数据集和数据加载器

dataset = PathImageFolder(root='./data', transform=transform)

dataloader = DataLoader(

    dataset,

    batch_size=128,

    shuffle=False,

    num_workers=4,

    collate_fn=collate_fn

)



# 设置内存映射文件参数

dataset_size = len(dataset)

emb_size = 2048  # ResNet-50特征维度

path_str_type = 'U200'  # 假设路径最大200字符

emb_file = 'embeddings.mmap'

paths_file = 'paths.mmap'



# 创建内存映射文件

emb_mmap = np.memmap(emb_file, dtype='float32', mode='w+', shape=(dataset_size, emb_size))

paths_mmap = np.memmap(paths_file, dtype=path_str_type, mode='w+', shape=(dataset_size,))



# 生成嵌入

get_embeddings(model, dataloader, emb_mmap, paths_mmap)



# 确保数据写入磁盘

emb_mmap.flush()

paths_mmap.flush()

print("Embeddings generated successfully!")
