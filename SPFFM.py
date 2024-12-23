from Transformer import Vision_Transformer
import torch
import torch.nn as nn
class DPN(nn.Module):  #dwconv+pwconv+norm
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, cnn_drop=0.):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True),
        )
        self.apply(self._iniweights)

    def _iniweights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        # nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.proj(x)
class Simam_module(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.act(y)

class SPFFM(
    nn.Module):
    def __init__(self, img_size, patch_size, in_channels, dim, num_heads, out_channels, cnn_drop=0., vit_drop=0.,
                 down_sampling=True, ):
        super().__init__()

        self.down_sampling = down_sampling
        self.dpn = DPN(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=3, stride=1, padding=1,
                       cnn_drop=cnn_drop)

        self.vit_block = Vision_Transformer(dim=in_channels, dmodel=dim, input_resolution=img_size, num_heads=num_heads,
                                            patch_size=patch_size, sr_ratio=2,
                                            dropout=vit_drop, emb_dropout=vit_drop, in_depth=1, add_Map=False)

        self.pwconv = nn.Conv2d(in_channels, out_channels // 2, 1, 1, 0, bias=False)
        self.norm = nn.BatchNorm2d(out_channels // 2)
        self.simam = Simam_module()
        #激活函数调整
        self.relu = nn.ReLU(inplace=True)

        self.dpn1 = DPN(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                        cnn_drop=cnn_drop)
        self.vit_block1 = Vision_Transformer(dim=out_channels, dmodel=dim, input_resolution=img_size,
                                             num_heads=num_heads,
                                             patch_size=patch_size, sr_ratio=2,
                                             dropout=vit_drop, emb_dropout=vit_drop, in_depth=1, add_Map=False)
        self.dwconv = nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)

        self.pwconv1 = nn.Conv2d(out_channels * 2, out_channels, 1, 1, 0, bias=False)
        self.simam1 = Simam_module()
        #self.pool = nn.Conv2d(out_channels, out_channels, 2, 2, 0)
        self.pool = nn.AvgPool2d(2, 2)
        self.apply(self._iniweights)

    def _iniweights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            #nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, A_map):
        x1 = self.dpn(x)
        x2 = self.vit_block(x, A_map)
        x2 = self.pwconv(x2)
        x2 = self.norm(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.simam(x)
        x = self.relu(x)
        x_copy = x
        x = self.dpn1(x)
        x_copy1 = x
        x = self.vit_block1(x, A_map)
        x = self.dwconv(x)
        x = self.norm1(x)
        x = torch.cat([x, x_copy1], dim=1)
        x = self.pwconv1(x)
        x = x + x_copy
        x = self.simam1(x)
        x = self.relu(x)
        x1 = x

        if self.down_sampling:
            #x = self.conv(x)  #下采样
            x = self.pool(x)

        return x, x1

if __name__=="__main__":
    model = SPFFM(img_size=(14,14), patch_size=(2, 2), in_channels=512, dim=128, out_channels=512, num_heads=4, cnn_drop=0.1, vit_drop=0.1, down_sampling=False)
    img = torch.randn(1, 512, 14, 14)
    A_map = None
    out,_ = model(img, A_map)
    print(out.shape)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

