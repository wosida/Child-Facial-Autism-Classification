import torch
import sys
import os
from Vim-main.vim.models_mamba import VisionMamba
import torch.nn as nn
from einops.layers.torch import Rearrange

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
# x=torch.rand(1,3,224,224)
# x=x.to(device)
# model = VisionMamba(
#         patch_size=16, stride=16, embed_dim=384, depth=1, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
#         final_pool_type='all', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False,
#         bimamba_type="v2", if_cls_token=False, if_devide_out=True, use_middle_cls_token=True).to(device)
# out = model(x,return_features=True)
#
# print(out.shape)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.InstanceNorm2d(out_channels)
        self.relu = nn.GELU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class seq2vol(nn.Module):
    def __init__(self,patch_size, H):
        super(seq2vol, self).__init__()
        self.proj=nn.Sequential(
            #nn.Linear(self.dmodel, patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=(H // patch_size[0]), p1=patch_size[0],
                      p2=patch_size[1]),
        )
    def forward(self, x):
        x = self.proj(x)
        return x
class PHMBA(nn.Module):
    def __init__(self,):
        super(PHMBA, self).__init__()
        self.conv = ConvBlock(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.mamba1 = VisionMamba(
        patch_size=16, stride=16, embed_dim=768, depth=1, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        final_pool_type='all', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False,
        bimamba_type="v2", if_cls_token=False, if_devide_out=True, use_middle_cls_token=True)
        self.seq2vol=seq2vol((16,16),224)
        self.conv1 = ConvBlock(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.mamba2 = VisionMamba(
        patch_size=16, stride=16, embed_dim=768, depth=1, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        final_pool_type='all', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False,
        bimamba_type="v2", if_cls_token=False, if_devide_out=True, use_middle_cls_token=True)
        self.seq2vol=seq2vol((16,16),224)
        self.conv2 = ConvBlock(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvBlock(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x1 = self.conv1(x)
        x2 = self.mamba1(x,return_features=True)
        x2=self.seq2vol(x2)
        x=x1+x2
        x3 = self.conv2(x)
        x4 = self.mamba2(x,return_features=True)
        x4=self.seq2vol(x4)
        x=x3+x4
        x=self.conv3(x)
        return x

# model=PHMBA().to(device)
# x=torch.rand(1,3,224,224)
# x=x.to(device)
# out=model(x)
# print(out.shape)

















