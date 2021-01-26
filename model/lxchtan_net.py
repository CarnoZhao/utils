import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

def convbn(in_planes, out_planes):
    return conv3x3(in_planes, out_planes), nn.BatchNorm2d(out_planes), nn.LeakyReLU()

# create your own CREncoder
class CREncoder(nn.Module):
    B = 1

    def __init__(self, feedback_bits):
        super(CREncoder, self).__init__()
        self.conv1 = conv3x3(2, 2)
        self.conv2 = conv3x3(2, 2)
        self.fc = nn.Linear(768, int(feedback_bits / self.B))
        self.sig = nn.Sigmoid()
        self.relu = nn.LeakyReLU()
        # self.quantize = QuantizationLayer(self.B)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = out.view(-1, 768)
        out = self.fc(out)
        out = self.sig(out)
        # self.out_ori = out
        # out = self.quantize(out)
        return out


# create your own Decoder
class CRDecoder(nn.Module):
    B = 1

    def __init__(self, feedback_bits):
        super(CRDecoder, self).__init__()
        self.feedback_bits = feedback_bits
        # self.dequantize = DequantizationLayer(self.B)

        self.conv2nums = 5
        self.conv3nums = 5
        self.conv4nums = 5
        self.conv5nums = 5

        self.multiConvs2 = nn.ModuleList()
        self.multiConvs3 = nn.ModuleList()
        self.multiConvs4 = nn.ModuleList()
        self.multiConvs5 = nn.ModuleList()

        self.fc = nn.Linear(int(feedback_bits / self.B), 768)
        self.out_cov = conv3x3(2, 2)
        self.sig = nn.Sigmoid()

        self.multiConvs2.append(nn.Sequential(
                *convbn(2, 64),
                *convbn(64, 64)))
        self.multiConvs3.append(nn.Sequential(
                *convbn(64, 128),
                *convbn(128, 128)))
        self.multiConvs4.append(nn.Sequential(
                *convbn(128, 256),
                *convbn(256, 256)))
        self.multiConvs5.append(nn.Sequential(
                *convbn(256, 128),
                *convbn(128, 32),
                *convbn(32, 2)))
                
        for _ in range(self.conv2nums):
            self.multiConvs2.append(nn.Sequential(
                *convbn(64, 64),
                *convbn(64, 64),
                *convbn(64, 64)))
        for _ in range(self.conv3nums):
            self.multiConvs3.append(nn.Sequential(
                *convbn(128, 128),
                *convbn(128, 128),
                *convbn(128, 128)))
        for _ in range(self.conv4nums):
            self.multiConvs4.append(nn.Sequential(
                *convbn(256, 256),
                *convbn(256, 256),
                *convbn(256, 256)))
        for _ in range(self.conv5nums):
            self.multiConvs5.append(nn.Sequential(
                *convbn(2, 32),
                *convbn(32, 32),
                *convbn(32, 2)))

    def forward(self, x):
        # out = self.dequantize(x)
        # self.out_de = out
        out = x.view(-1, int(self.feedback_bits / self.B))
        out = self.sig(self.fc(out))
        out = out.view(-1, 2, 24, 16)

        out = self.multiConvs2[0](out)
        for i in range(1, self.conv2nums + 1):
            residual = out
            out = self.multiConvs2[i](out)
            out = residual + out

        out = self.multiConvs3[0](out)
        for i in range(1, self.conv3nums + 1):
            residual = out
            out = self.multiConvs3[i](out)
            out = residual + out

        out = self.multiConvs4[0](out)
        for i in range(1, self.conv4nums + 1):
            residual = out
            out = self.multiConvs4[i](out)
            out = residual + out

        out = self.multiConvs5[0](out)
        for i in range(1, self.conv5nums + 1):
            residual = out
            out = self.multiConvs5[i](out)
            out = residual + out

        out = self.out_cov(out)
        # out = self.sig(out)
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