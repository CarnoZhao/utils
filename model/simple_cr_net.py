from utils.acti.mish import Mish
import torch
import torch.nn as nn

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

class CREncoder(nn.Module):
    quan = 4
    def __init__(self, bits, dim1 = 24, dim2 = 16):
        super(CREncoder, self).__init__()
        self.enc1 = nn.Sequential(
            ConvBN(2, 2, (3, 3)),
            ConvBN(2, 2, (1, 9)),
            ConvBN(2, 2, (9, 1))
        )
        self.enc2 = nn.Sequential(
            ConvBN(2, 2, (3, 3))
        )
        self.merge = ConvBN(4, 2, (1, 1))
        self.fc = nn.Sequential(
            nn.Linear(2 * dim1 * dim2, int(bits / self.quan)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x)
        x = torch.cat([x1, x2], 1)
        x = self.merge(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

class CRBlock(nn.Module):
    def __init__(self):
        super(CRBlock, self).__init__()
        self.dec1 = nn.Sequential(
            ConvBN(2, 2, (3, 3)),
            ConvBN(2, 7, (1, 9)),
            ConvBN(7, 7, (9, 1))
        )
        self.dec2 = nn.Sequential(
            ConvBN(2, 2, (1, 5)),
            ConvBN(2, 7, (5, 1))
        )
        self.merge = ConvBN(14, 2, (1, 1))
    
    def forward(self, x):
        x1 = self.dec1(x)
        x2 = self.dec2(x)
        x3 = torch.cat([x1, x2], 1)
        x3 = self.merge(x3)
        x = x3 + x
        return x

class CRDecoder(nn.Module):
    quan = 4
    def __init__(self, bits, dim1 = 24, dim2 = 16):
        super(CRDecoder, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.fc = nn.Sequential(
            nn.Linear(int(bits / self.quan), 2 * dim1 * dim2),
            Mish()
        )
        self.conv1 = ConvBN(2, 2, (5, 5))
        self.crb = nn.Sequential(
            CRBlock(),
            CRBlock()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 2, self.dim1, self.dim2)
        x = self.conv1(x)
        x = self.crb(x)
        return x
        