import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import metrics
import torchvision.models as models

# #goolgenet 14
# model=models.googlenet(pretrained=True)
# print(sum(p.numel() for p in model.parameters() if p.requires_grad)) #六百万
# #resnet 15
# model=models.resnet18(pretrained=True)
# print(sum(p.numel() for p in model.parameters() if p.requires_grad)) #一千一百万
# #squeeze 16
# model=models.squeezenet1_0(pretrained=True)
# print(sum(p.numel() for p in model.parameters() if p.requires_grad)) #一百万
# #densenet 17
# model=models.densenet121(pretrained=True)
# print(sum(p.numel() for p in model.parameters() if p.requires_grad)) #八百万
# #mobilenet 18
# model=models.mobilenet_v2(pretrained=True)
# print(sum(p.numel() for p in model.parameters() if p.requires_grad)) #三百万
# #efficientnet 19
# model=models.efficientnet_b0(pretrained=True)
# print(sum(p.numel() for p in model.parameters() if p.requires_grad)) #五百万
# #vit 20
# model=models.vit_b_16(pretrained=False)
# print(sum(p.numel() for p in model.parameters() if p.requires_grad)) #八千六百万
# #swin_transformer 21
# model=models.swin_b(pretrained=False)
# print(sum(p.numel() for p in model.parameters() if p.requires_grad)) #八千七百万
# #convnext 22
# model=models.convnext_base(pretrained=False)
# print(sum(p.numel() for p in model.parameters() if p.requires_grad)) #八千八百万

import torch
import torch.nn as nn
from torchvision import models

# 加载预训练的 GoogLeNet 模型
model = models.googlenet(pretrained=True)
print(model)
# 定义一个钩子函数，用于获取每一层的输出形状
def hook_fn(module, input, output):
    print(f"Layer: {module.__class__.__name__}")
    print(f"Input shape: {tuple(input[0].shape)}")
    print(f"Output shape: {tuple(output.shape)}\n")

# 注册钩子到每一层
hooks = []
for layer in model.children():
    hook = layer.register_forward_hook(hook_fn)
    hooks.append(hook)

# 创建一个假数据输入，形状为 (batch_size=1, channels=3, height=224, width=224)
dummy_input = torch.randn(1, 3, 224, 224)

# 通过模型进行一次前向传播，钩子函数会打印每一层的输入和输出形状
model(dummy_input)

# 清除钩子，避免对后续操作产生影响
for hook in hooks:
    hook.remove()


