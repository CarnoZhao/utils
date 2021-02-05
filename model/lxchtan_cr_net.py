import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import math

def conv3x3(in_planes, out_planes, size = 3, stride = 1, padding = 1, bias = False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size = size, stride = stride,
                     padding = padding, bias = bias)

def convbn(in_planes, out_planes, size = (3, 3)):
    pad = [(i - 1) // 2 for i in size]
    return nn.Sequential(
        conv3x3(in_planes, out_planes, size = size, padding = pad, bias = False), 
        nn.BatchNorm2d(out_planes), 
        nn.LeakyReLU(0.3)
    )

# create your own CREncoder
class CREncoder(nn.Module):
    B = 2

    def __init__(self, feedback_bits):
        super(CREncoder, self).__init__()
        self.fc = conv3x3(4, 2, stride = 2, bias = True)#nn.Linear(768 * 2, int(feedback_bits / self.B))
        self.sig = nn.Sigmoid()
        self.relu = nn.LeakyReLU(0.3)

        self.head = nn.Sequential(
            convbn(2, 64),
            convbn(64, 256)
        )

        self.enc1nums = 3
        self.enc2nums = 3
        self.mergenums = 2
        self.enc1 = nn.ModuleList()
        for _ in range(self.enc1nums):
            self.enc1.append(nn.Sequential(
                convbn(256, 256, (7, 1)),
                convbn(256, 256, (1, 7)),
                convbn(256, 256, (1, 1))
            ))
        self.enc2 = nn.ModuleList()
        for _ in range(self.enc2nums):
            self.enc2.append(nn.Sequential(
                convbn(256, 256, (5, 1)),
                convbn(256, 256, (1, 5)),
                convbn(256, 256, (1, 1))
            ))
        
        self.merge = nn.ModuleList()
        self.merge.append(nn.Sequential(
                convbn(512, 128),
                convbn(128, 32),
                convbn(32, 4)))
        for _ in range(self.mergenums):
            self.merge.append(nn.Sequential(
                convbn(4, 32),
                convbn(32, 32),
                convbn(32, 4)))

    def forward(self, x):
        out = self.head(x)
        out1 = out
        for _ in range(self.enc1nums):
            res = out1
            out1 = res + self.enc1[_](out1)
        out2 = out
        for _ in range(self.enc2nums):
            res = out2
            out2 = res + self.enc2[_](out2)
        out = torch.cat([out1, out2], 1)
        
        out = self.merge[0](out)
        for i in range(1, self.mergenums + 1):
            res = out
            out = res + self.merge[i](out)

        out = self.fc(out)
        out = self.sig(out)
        out = out.view(-1, 384 // self.B)
        return out


# create your own Decoder
class CRDecoder(nn.Module):
    B = 2

    def __init__(self, feedback_bits):
        super(CRDecoder, self).__init__()
        self.feedback_bits = feedback_bits
        self.relu = nn.LeakyReLU(0.3)

        self.conv2nums = 5
        self.conv5nums = 2

        self.multiConvs2 = nn.ModuleList()
        self.multiConvs5 = nn.ModuleList()

        self.fc = nn.ConvTranspose2d(2, 4, 4, 2, 1)#nn.Linear(int(feedback_bits / self.B), 768 * 2)
        self.out_cov = conv3x3(2, 2)
        self.sig = nn.LeakyReLU(0.3)

        self.multiConvs2.append(nn.Sequential(
                convbn(4, 64),
                convbn(64, 256)))
        self.multiConvs5.append(nn.Sequential(
                convbn(256, 128),
                convbn(128, 32),
                convbn(32, 2)))

        for _ in range(self.conv2nums):
            self.multiConvs2.append(nn.Sequential(
                convbn(256, 64),
                convbn(64, 64),
                convbn(64, 256)))
        for _ in range(self.conv5nums):
            self.multiConvs5.append(nn.Sequential(
                convbn(2, 32),
                convbn(32, 32),
                convbn(32, 2)))

    def forward(self, x):
        out = x.view(-1, 2, 12, 8)
        # out = x.view(-1, int(self.feedback_bits / self.B))
        # out = out.view(-1, 4, 24, 16)
        out = self.sig(self.fc(out))

        out = self.multiConvs2[0](out)
        for i in range(1, self.conv2nums + 1):
            residual = out
            out = self.multiConvs2[i](out)
            out = residual + out

        out = self.multiConvs5[0](out)
        for i in range(1, self.conv5nums + 1):
            residual = out
            out = self.multiConvs5[i](out)
            out = residual + out

        out = self.out_cov(out)
        return out 
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