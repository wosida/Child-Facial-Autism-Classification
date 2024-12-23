#vit_block LSRA代替MHA
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
from thop import profile
import math
from torch.nn.init import trunc_normal_

np.set_printoptions(threshold=1000)


class Map_reshape(nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.p1, self.p2 = patch_size[0], patch_size[1]
        self.num_heads = num_heads

    def forward(self, map):
        map = rearrange(map, 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=self.p1, p2=self.p2)
        map = map.max(-1)[0]
        map_attn = map.unsqueeze(2).repeat(1, 1, map.shape[-1]).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        return map_attn


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, map_attn, add_Map, **kwargs):
        return self.fn(self.norm(x), map_attn, add_Map, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, map_attn, add_Map):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, H, W, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.H, self.W = H, W

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, map_attn, add_Map):
        B, N, C = x.shape
        H, W = self.H, self.W
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)

        if add_Map and (map_attn is not None):
            attn = map_attn * attn
        else:
            attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Transformer(nn.Module):
    def __init__(self, dim, H, W, depth, num_heads, mlp_dim=1024, dropout=0., attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, H=H, W=W, num_heads=num_heads, sr_ratio=sr_ratio, attn_drop=attn_drop,
                                       proj_drop=proj_drop, linear=linear)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, map_attn, add_Map):
        for attn, ff in self.layers:
            att_out = attn(x, map_attn, add_Map)
            x = att_out + x
            x = ff(x, None, None) + x
        return x


class Vision_Transformer(nn.Module):
    def __init__(self, dim, dmodel, input_resolution, num_heads, mlp_ratio=2, patch_size=[1, 1], dropout=0.1,
                 emb_dropout=0.1, attn_drop=0., proj_drop=0., in_depth=1, sr_ratio=1,
                 add_Map=True, linear=False):  # dim是输入通道数, dmodel是transformer的编码dim,sr_ratio如果等于1，就是普通VIT
        super().__init__()

        self.dim = dim
        self.dmodel = dmodel
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.mlp_dim = self.dmodel * mlp_ratio
        self.add_Map = add_Map
        self.sr_ratio = sr_ratio

        H, W = self.input_resolution
        assert H % patch_size[0] == 0 and W % patch_size[
            1] == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (H // patch_size[0]) * (W // patch_size[1])
        patch_dim = patch_size[0] * patch_size[1] * dim  # 这个dim是[b,c,h,w]中的c

        self.map_reshape = Map_reshape(patch_size, self.num_heads)

        self.patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size[0], p2=patch_size[1]),
            nn.Linear(patch_dim, self.dmodel),  # dmodel是transformer的编码dim
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(self.dmodel, H // self.patch_size[0],
                                       W // self.patch_size[1], in_depth, num_heads,
                                       self.mlp_dim, dropout, attn_drop, proj_drop, sr_ratio, linear)

        self.recover_patch_embedding = nn.Sequential(
            nn.Linear(self.dmodel, patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=(H // patch_size[0]), p1=patch_size[0],
                      p2=patch_size[1]),
        )
        self.recover_patch_embedding1 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size[0], p2=patch_size[1]), )

    def forward(self, x, A_map):
        h, w = self.input_resolution
        B, C, H, W = x.shape
        assert H == h and W == w, "input feature has wrong size"

        x = self.patch_embedding(x)
        x = x + self.pos_embedding
        x = self.dropout(x)

        if A_map is not None:
            map_attn = self.map_reshape(A_map)
        else:
            map_attn = None

        x = self.transformer(x, map_attn, self.add_Map)

        vit_out = self.recover_patch_embedding(x)

        return vit_out


if __name__ == "__main__":
    x = torch.randn(1, 3, 512, 512)
    A_map = torch.randn(1, 64, 64)
    model = Vision_Transformer(dim=3, dmodel=512, input_resolution=(512, 512), num_heads=8, patch_size=[8,8],
                               dropout=0.5, emb_dropout=0.5, in_depth=1, sr_ratio=2, add_Map=False,linear=True)
    out = model(x, A_map)
    flops, params = profile(model, inputs=(x, A_map))
    print(flops, params)
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

    print(out.shape)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
