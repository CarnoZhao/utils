import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import math

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
    # return nn.Conv2d(in_planes, out_planes, kernel_size = (1, 3), padding = (0, 1))

def convbn(in_planes, out_planes):
    return conv3x3(in_planes, out_planes), nn.BatchNorm2d(out_planes), nn.LeakyReLU(0.3)

mid_channel = 4
# create your own CREncoder
class CREncoder(nn.Module):
    B = 3

    def __init__(self, feedback_bits):
        super(CREncoder, self).__init__()
        self.sig = nn.Sigmoid()
        self.relu = nn.LeakyReLU(0.3)

        self.conv2nums = 5
        self.conv5nums = 2
        self.multiConvs2 = nn.ModuleList()
        self.multiConvs2.append(nn.Sequential(
                *convbn(2, 64),
                *convbn(64, 256)))
        self.multiConvs5 = nn.ModuleList()
        self.multiConvs5.append(nn.Sequential(
                *convbn(256, 128),
                *convbn(128, 32),
                *convbn(32, mid_channel)))
        for _ in range(self.conv2nums):
            self.multiConvs2.append(nn.Sequential(
                *convbn(256, 64),
                *convbn(64, 64),
                *convbn(64, 256)))
        for _ in range(self.conv5nums):
            self.multiConvs5.append(nn.Sequential(
                *convbn(mid_channel, 32),
                *convbn(32, 32),
                *convbn(32, mid_channel)))
        self.fc = nn.Linear(384 * mid_channel, int(feedback_bits / self.B))

    def forward(self, x):
        out = self.multiConvs2[0](x)
        for i in range(1, self.conv2nums + 1):
            residual = out
            out = self.multiConvs2[i](out)
            out = residual + out

        out = self.multiConvs5[0](out)
        for i in range(1, self.conv5nums + 1):
            residual = out
            out = self.multiConvs5[i](out)
            out = residual + out
        out = out.reshape(-1, 384 * mid_channel)
        out = self.fc(out)
        out = self.sig(out)
        return out


# create your own Decoder
class CRDecoder(nn.Module):
    B = 3

    def __init__(self, feedback_bits):
        super(CRDecoder, self).__init__()
        self.feedback_bits = feedback_bits
        self.relu = nn.LeakyReLU(0.3)

        self.conv2nums = 6
        self.conv5nums = 3

        self.multiConvs2 = nn.ModuleList()
        self.multiConvs5 = nn.ModuleList()

        self.fc = nn.Linear(int(feedback_bits / self.B), 384 * mid_channel)
        self.out_cov = conv3x3(128, 2)
        self.sig = nn.LeakyReLU(0.3)

        self.multiConvs2.append(nn.Sequential(
            *convbn(mid_channel, 64),
            *convbn(64, 384),
        ))
        self.multiConvs5.append(nn.Sequential(
            *convbn(384, 128),
            *convbn(128, 128),
        ))

        for _ in range(self.conv2nums):
            self.multiConvs2.append(nn.Sequential(
                *convbn(384, 384),
                *convbn(384, 384)
            ))
        for _ in range(self.conv5nums):
            self.multiConvs5.append(nn.Sequential(
                *convbn(128, 128),
                *convbn(128, 128)
            ))

    def forward(self, x):
        out = self.sig(self.fc(x))
        out = out.view(-1, mid_channel, 24, 16)

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