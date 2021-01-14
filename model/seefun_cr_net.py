import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import OrderedDict

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))


class CRBlock(nn.Module):
    def __init__(self):
        super(CRBlock, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(32, 32, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv1x9', ConvBN(32, 32, [1, 9])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv9x1', ConvBN(32, 32, [9, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(32, 32, [1, 5])),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1', ConvBN(32, 32, [5, 1])),
        ]))
        self.conv1x1 = ConvBN(32 * 2, 32, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        identity = self.identity(x)

        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)

        out = self.relu(out + identity)
        return out


class CREncoder(nn.Module):
    B = 4

    def __init__(self, feedback_bits):
        super(CREncoder, self).__init__()
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(2, 32, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x9_bn", ConvBN(32, 32, [1, 9])),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv9x1_bn", ConvBN(32, 32, [9, 1])),
        ]))
        self.encoder2 = ConvBN(2, 32, 3)
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn", ConvBN(32*2, 2, 1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))

        self.fc = nn.Linear(768, int(feedback_bits / self.B))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        encode1 = self.encoder1(x)
        encode2 = self.encoder2(x)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
        out = out.view(-1, 768)
        out = self.fc(out)
        out = self.sig(out)
        return out


class CRDecoder(nn.Module):
    B = 4

    def __init__(self, feedback_bits):
        super(CRDecoder, self).__init__()
        self.feedback_bits = feedback_bits
        self.fc = nn.Linear(int(feedback_bits / self.B), 768)
        decoder = OrderedDict([
            ("conv5x5_bn", ConvBN(2, 32, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock()),
            ("CRBlock2", CRBlock()),
        ])
        self.decoder_feature = nn.Sequential(decoder)
        self.out_cov = conv3x3(32, 2)    

    def forward(self, x):
        out = x.view(-1, int(self.feedback_bits / self.B))
        out = self.fc(out)
        out = out.view(-1, 2, 24, 16)
        out = self.decoder_feature(out)
        out = self.out_cov(out)
        return out

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