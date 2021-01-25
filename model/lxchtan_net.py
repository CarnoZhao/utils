import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import math

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

def conv1x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=True)

def conv2x2(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=2, stride=stride,
                     padding=1, bias=True)

# create your own CREncoder
class CREncoder(nn.Module):
    B = 1

    def __init__(self, feedback_bits):
        super(CREncoder, self).__init__()
        self.conv1 = conv3x3(2, 2)
        self.conv2 = conv3x3(2, 2)
        self.fc = nn.Linear(768, int(feedback_bits / self.B))
        self.sig = nn.Sigmoid()
        # self.quantize = QuantizationLayer(self.B)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
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

        self.conv2nums = 2
        self.conv3nums = 3
        self.conv4nums = 5
        self.conv5nums = 3

        self.multiConvs2 = nn.ModuleList()
        self.multiConvs3 = nn.ModuleList()
        self.multiConvs4 = nn.ModuleList()
        self.multiConvs5 = nn.ModuleList()

        self.fc = nn.Linear(int(feedback_bits / self.B), 768)
        self.out_cov = conv3x3(2, 2)
        self.sig = nn.Sigmoid()

        self.multiConvs2.append(nn.Sequential(
                conv3x3(2, 64),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                conv3x3(64, 256),
                nn.BatchNorm2d(256),
                nn.ReLU()))
        self.multiConvs3.append(nn.Sequential(
                conv3x3(256, 512),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                conv3x3(512, 512),
                nn.BatchNorm2d(512),
                nn.ReLU()))
        self.multiConvs4.append(nn.Sequential(
                conv3x3(512, 1024),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                conv3x3(1024, 1024),
                nn.BatchNorm2d(1024),
                nn.ReLU()))
        self.multiConvs5.append(nn.Sequential(
                conv3x3(1024, 128),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                conv3x3(128, 32),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                conv3x3(32, 2),
                nn.BatchNorm2d(2),
                nn.ReLU()))
                
        for _ in range(self.conv2nums):
            self.multiConvs2.append(nn.Sequential(
                conv3x3(256, 64),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                conv3x3(64, 64),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                conv3x3(64, 256),
                nn.BatchNorm2d(256),
                nn.ReLU()))
        for _ in range(self.conv3nums):
            self.multiConvs3.append(nn.Sequential(
                conv3x3(512, 128),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                conv3x3(128, 128),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                conv3x3(128, 512),
                nn.BatchNorm2d(512),
                nn.ReLU()))
        for _ in range(self.conv4nums):
            self.multiConvs4.append(nn.Sequential(
                conv3x3(1024, 256),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                conv3x3(256, 256),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                conv3x3(256, 1024),
                nn.BatchNorm2d(1024),
                nn.ReLU()))
        for _ in range(self.conv5nums):
            self.multiConvs5.append(nn.Sequential(
                conv3x3(2, 32),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                conv3x3(32, 32),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                conv3x3(32, 2),
                nn.BatchNorm2d(2),
                nn.ReLU()))

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