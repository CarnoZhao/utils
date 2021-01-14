import torch
import torch.nn as nn
from ..acti.mish import Mish

import torchvision
torchvision.models.ResNet

class ConvBN(nn.Module):
    def __init__(self, in_c, out_c, size):
        super(ConvBN, self).__init__()
        pad = [(i - 1) // 2 for i in size]
        self.conv = nn.Conv2d(in_c, out_c, size, 1, pad, bias = False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, ch, nblocks = 1):
        super(ResBlock, self).__init__()
        self.module_list = nn.ModuleList()
        for _ in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(ConvBN(ch, ch, (1, 1)))
            resblock_one.append(ConvBN(ch, ch, (3, 3)))
            self.module_list.append(resblock_one)
 
    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_c = 128, blocks = 2):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            ConvBN(in_c, in_c, (1, 9)),
            ConvBN(in_c, in_c, (9, 1)),
            ConvBN(in_c, in_c, (1, 1))
        )
        self.resBlock = ResBlock(ch = in_c, nblocks = blocks)
        self.conv2 = nn.Sequential(
            ConvBN(in_c, in_c, (1, 7)),
            ConvBN(in_c, in_c, (7, 1)),
            ConvBN(in_c, in_c, (1, 1))
        )
 
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.resBlock(x1)
        x2 = self.conv2(x2)
        return x1 + x2

 
class CRBlock(nn.Module):
    def __init__(self, n_c = 64):
        super(CRBlock, self).__init__()
        self.conv = ConvBN(n_c, n_c * 2, (3, 3))
        self.dec1 = ConvBlock(n_c * 2, 2)
        self.dec2 = nn.Sequential(
            ConvBN(n_c * 2, n_c * 2, (1, 5)),
            ConvBN(n_c * 2, n_c * 2, (5, 1)),
            ConvBN(n_c * 2, n_c * 2, (1, 1)),
            ConvBN(n_c * 2, n_c * 2, (3, 3)),
        )
        self.merge = nn.Sequential(
            ConvBlock(n_c * 4, 2),
            ConvBN(n_c * 4, n_c, (1, 1))
        )
 
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.dec1(x1)
        x3 = self.dec2(x1)
        x4 = torch.cat([x2, x3], 1)
        x4 = self.merge(x4)
        return x4 + x
 
 
class CREncoder(nn.Module):
    B = 4
    def __init__(self, bits, dim1 = 24, dim2 = 16, n_c = 64, drop = 0):
        super(CREncoder, self).__init__()
        self.conv1 = ConvBN(2, n_c, (3, 3))
        self.enc1 = ConvBlock(n_c)
        self.enc2 = nn.Sequential(
            ConvBN(n_c, n_c, (1, 5)),
            ConvBN(n_c, n_c, (5, 1)),
            ConvBN(n_c, n_c, (3, 3))
        )
        self.merge = nn.Sequential(
            ConvBlock(n_c * 2),
            ConvBN(n_c * 2, 2, (1, 1))
        )
        self.fc = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(2 * dim1 * dim2, int(bits / self.B)),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        x = self.conv1(x)
        x1 = self.enc1(x)
        x2 = self.enc2(x)
        x3 = torch.cat([x1, x2], 1)
        x3 = self.merge(x3)
        x3 = x3.view(x3.shape[0], -1)
        x3 = self.fc(x3)
        return x3
 
 
class CRDecoder(nn.Module):
    B = 4
    def __init__(self, bits, dim1 = 24, dim2 = 16, n_c = 64, drop = 0):
        super(CRDecoder, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.fc = nn.Sequential(
            nn.Linear(int(bits / self.B), 2 * dim1 * dim2),
            nn.Dropout(drop)
        )
        self.dec = nn.Sequential(
            ConvBN(2, n_c, (3, 3)),
            CRBlock(),
            CRBlock()
        )
        self.conv = nn.Conv2d(n_c, 2, (3, 3), 1, 1, bias = False)
 
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 2, self.dim1, self.dim2)
        x = self.dec(x)
        x = self.conv(x)
        return x


if __name__ == "__main__":
    e = CREncoder(400)
    d = CRDecoder(400)
    a = torch.rand(10, 2, 24, 16)
    b = e(a)
    c = d(b)