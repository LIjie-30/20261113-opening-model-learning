import numpy as np
from medmnist import PathMNIST
import torchvision.transforms as transforms
from PIL import Image
import torch

root = r"E:\Project\medical_image_learning\project1\data\medmnist"

# 不加载 transform，先看原始数据
ds = PathMNIST(split="train", download=False, root=root)

print("数据集长度:", len(ds))
print("类别数:", len(ds.info["label"]))
print("类别名:", ds.info["label"])
print("原始图像形状:", ds[0][0].size if hasattr(ds[0][0], 'size') else ds[0][0].shape)
print("原始标签形状:", np.array(ds[0][1]).shape)

# 看一批标签的分布
labels = [int(ds[i][1]) for i in range(1000)]
import collections
print("前1000个样本标签分布:", collections.Counter(labels))

img, label = ds[0]  # PIL Image, label

# Step 1: ToTensor
t1 = transforms.ToTensor()(img)
print("ToTensor 后 shape:", t1.shape, "min/max:", t1.min().item(), t1.max().item())

# Step 2: Lambda 扩展 3 通道
t2 = t1.mean(dim=0, keepdim=True).expand(3, -1, -1)
print("Lambda 后 shape:", t2.shape, "min/max:", t2.min().item(), t2.max().item())

# Step 3: Normalize
t3 = transforms.Normalize(mean=[0.5], std=[0.5])(t2)
print("Normalize 后 shape:", t3.shape, "min/max:", t3.min().item(), t3.max().item())

"""
Day 3 数据探索脚本
目的：打印 PathMNIST 的类别、图像 shape、标签 shape，
     并手动拆解 transform 三步（ToTensor / Lambda / Normalize）的数值范围。

关键发现：
1. PathMNIST 原生是 3 通道 RGB，不是灰度。
2. 原 transform 中的 Lambda 把 RGB 取均值变单通道，再复制成 3 份，丢失颜色信息。
3. Normalize(mean=[0.5], std=[0.5]) 只作用于 channel 0，其余通道未归一化。
"""