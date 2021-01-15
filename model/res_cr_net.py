import torch
import torch.nn as nn
from ..acti.mish import Mish

class ConvBN(nn.Module):
    def __init__(self, in_c, out_c, size):
        super(ConvBN, self).__init__()
        pad = [(i - 1) // 2 for i in size]
        self.conv = nn.Conv2d(in_c, out_c, size, 1, pad)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, ch, nblocks = 1):
        super().__init__()
        self.module_list = nn.ModuleList()
        for _ in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(ConvBN(ch, ch, (1, 1)))
            resblock_one.append(Mish())
            resblock_one.append(ConvBN(ch, ch, (3, 3)))
            resblock_one.append(Mish())
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
        self.act = Mish()
 
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.resBlock(x1)
        x2 = self.conv2(x2)
        x = self.act(x1 + x2)
        return x

 
class CRBlock(nn.Module):
    def __init__(self, n_c = 64):
        super(CRBlock, self).__init__()
        self.conv = ConvBN(n_c, n_c * 2, (3, 3))
        self.dec1 = ConvBlock(n_c * 2, 4)
        self.dec2 = nn.Sequential(
            ConvBN(n_c * 2, n_c * 2, (1, 5)),
            ConvBN(n_c * 2, n_c * 2, (5, 1)),
            ConvBN(n_c * 2, n_c * 2, (1, 1)),
            ConvBN(n_c * 2, n_c * 2, (3, 3)),
        )
        self.merge = nn.Sequential(
            ConvBlock(n_c * 4, 4),
            ConvBN(n_c * 4, n_c, (1, 1))
        )
        self.act = Mish()
 
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.dec1(x1)
        x3 = self.dec2(x1)
        x1 = torch.cat([x2, x3], 1)
        x1 = self.act(x1)
        x1 = self.merge(x1)
        x = self.act(x1 + x)
        return x
 
 
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
            ConvBN(n_c * 2, 4, (1, 1))
        )
        self.fc = nn.Sequential(
            nn.Linear(4 * dim1 * dim2, int(bits / self.B)),
            nn.Dropout(drop),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        x = self.conv1(x)
        x1 = self.enc1(x)
        x2 = self.enc2(x)
        x = torch.cat([x1, x2], 1)
        x = self.merge(x)
        x = x.contiguous().view(x.shape[0], -1)
        x = self.fc(x)
        return x
 
 
class CRDecoder(nn.Module):
    B = 4
    def __init__(self, bits, dim1 = 24, dim2 = 16, n_c = 64, drop = 0):
        super(CRDecoder, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.fc = nn.Sequential(
            nn.Linear(int(bits / self.B), 4 * dim1 * dim2),
            nn.Dropout(drop)
        )
        self.dec = nn.Sequential(
            ConvBN(4, n_c, (3, 3)),
            CRBlock(),
            CRBlock()
        )
        self.conv = nn.Conv2d(n_c, 2, (3, 3), 1, 1)
 
    def forward(self, x):
        x = self.fc(x)
        x = x.contiguous().view(-1, 4, self.dim1, self.dim2)
        x = self.dec(x)
        x = self.conv(x)
        return x


if __name__ == "__main__":
    from torchviz import make_dot
    e = CREncoder(400)
    d = CRDecoder(400)
    a = torch.randn(10, 2, 24, 16)
    b = e(a)
    make_dot(b, dict(e.named_parameters())).view("encoder", cleanup = True)
    pass
    b = torch.tensor(b.detach().numpy())
    c = d(b)
    make_dot(c, dict(d.named_parameters())).view("decoder", cleanup = True)
    pass