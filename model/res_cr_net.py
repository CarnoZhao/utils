import torch
import torch.nn as nn
from ..acti.mish import Mish

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

class ResoBlock(nn.Module):
    def __init__(self, in_c, reso):
        super(ResoBlock, self).__init__()
        self.reso = nn.Sequential(
            ConvBN(in_c, in_c, (1, reso)),
            ConvBN(in_c, in_c, (reso, 1)),
            ConvBN(in_c, in_c, (3, 3)),
            ConvBN(in_c, in_c, (1, 1)),
        )

    def forward(self, x):
        return self.reso(x)

 
class CRBlock(nn.Module):
    def __init__(self, n_c = 64):
        super(CRBlock, self).__init__()
        self.conv = ConvBN(n_c, n_c, (3, 3))
        self.dec0 = ResoBlock(n_c, 3)
        self.dec1 = ResoBlock(n_c, 5)
        self.dec2 = ResoBlock(n_c, 7)
        self.dec3 = ResoBlock(n_c, 9)
        self.merge = ConvBN(n_c * 4, n_c, (1, 1))
 
    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.dec0(x0)
        x2 = self.dec1(x0)
        x3 = self.dec2(x0)
        x4 = self.dec3(x0)
        x5 = torch.cat([x1, x2, x3, x4], 1)
        x5 = self.merge(x5)
        return x5 + x
 
 
class CREncoder(nn.Module):
    B = 4
    def __init__(self, bits, dim1 = 24, dim2 = 16, n_c = 64, drop = 0):
        super(CREncoder, self).__init__()
        self.conv1 = ConvBN(2, n_c, (3, 3))
        self.enc0 = ResoBlock(n_c, 3)
        self.enc1 = ResoBlock(n_c, 5)
        self.enc2 = ResoBlock(n_c, 7)
        self.enc3 = ResoBlock(n_c, 9)
        self.merge = ConvBN(n_c * 5, 2, (1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(2 * dim1 * dim2, int(bits / self.B)),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        x = self.conv1(x)
        x0 = self.enc0(x)
        x1 = self.enc1(x)
        x2 = self.enc2(x)
        x3 = self.enc3(x)
        x4 = torch.cat([x, x0, x1, x2, x3], 1)
        x4 = self.merge(x4)
        x4 = x4.view(x4.shape[0], -1)
        x4 = self.fc(x4)
        return x4
 
 
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
    from torchviz import make_dot
    e = CREncoder(400)
    d = CRDecoder(400)
    a = torch.rand(10, 2, 24, 16)
    b = e(a)
    make_dot(b, dict(e.named_parameters())).view("encoder", cleanup = True)
    b = torch.tensor(b.detach().numpy())
    c = d(b)
    make_dot(c, dict(d.named_parameters())).view("decoder", cleanup = True)