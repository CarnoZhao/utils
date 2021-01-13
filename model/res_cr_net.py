
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import OrderedDict
from .quant import QuantizationLayer, DequantizationLayer
 
channelNum = 64


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x

class ResBlock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """
 
    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(ConvBN(ch, ch, 1))
            resblock_one.append(Mish())
            resblock_one.append(ConvBN(ch, ch, 3))
            resblock_one.append(Mish())
            self.module_list.append(resblock_one)
 
    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x

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
            ('bn', nn.BatchNorm2d(out_planes)),
            ('Mish', Mish())
        ]))
 
 
 
class CRBlock64(nn.Module):
    def __init__(self):
        super(CRBlock64, self).__init__()
        self.convbncrb = ConvBN(channelNum, channelNum * 2, 3)
        self.path1 = Encoder_conv(channelNum * 2, 4)
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(channelNum * 2, channelNum * 2, [1, 5])),
            ('conv5x1', ConvBN(channelNum * 2, channelNum * 2, [5, 1])),
            ('conv5x1', ConvBN(channelNum * 2, channelNum * 2, 1)),
            ('conv5x1', ConvBN(channelNum * 2, channelNum * 2, 3)),
        ]))
        self.encoder_conv = Encoder_conv(channelNum * 4, 4)
        self.encoder_conv1 = ConvBN(channelNum * 4, channelNum, 1)
        self.identity = nn.Identity()
        self.relu = Mish()
 
    def forward(self, x):
        identity = self.identity(x)
        x = self.convbncrb(x)
        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.encoder_conv(out)
        out = self.encoder_conv1(out)
        out = self.relu(out + identity)
        return out
 
 
class CRBlock(nn.Module):
    def __init__(self):
        super(CRBlock, self).__init__()
        self.convban = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(channelNum, channelNum, 3)),
        ]))
        self.path1 = Encoder_conv(channelNum, 4)
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(channelNum, channelNum, [1, 5])),
            ('conv5x1', ConvBN(channelNum, channelNum, [5, 1])),
            ("conv9x1_bn", ConvBN(channelNum, channelNum, 1)),
        ]))
        self.encoder_conv = Encoder_conv(channelNum * 2)
        self.encoder_conv1 = ConvBN(channelNum * 2, channelNum, 1)
        self.identity = nn.Identity()
        self.relu = Mish()
 
    def forward(self, x):
        identity = self.identity(x)
        x = self.convban(x)
        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.encoder_conv(out)
        out = self.encoder_conv1(out)
        out = self.relu(out + identity)
        return out
 
 
class Encoder_conv(nn.Module):
    def __init__(self, in_planes=128, blocks=2):
        super().__init__()
        self.conv2 = ConvBN(in_planes, in_planes, [1, 9])
        self.conv3 = ConvBN(in_planes, in_planes, [9, 1])
        self.conv4 = ConvBN(in_planes, in_planes, 1)
        self.resBlock = ResBlock(ch=in_planes, nblocks=blocks)
        self.conv5 = ConvBN(in_planes, in_planes, [1, 7])
        self.conv6 = ConvBN(in_planes, in_planes, [7, 1])
        self.conv7 = ConvBN(in_planes, in_planes, 1)
        self.relu = Mish()
 
    def forward(self, input):
        x2 = self.conv2(input)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        r1 = self.resBlock(x4)
        x5 = self.conv5(r1)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x7 = self.relu(x7 + x4)
        return x7
 
 
class Encoder(nn.Module):
    B = 4
 
    def __init__(self, feedback_bits, quantization=True):
        super(Encoder, self).__init__()
        self.convban = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(2, channelNum, 3)),
        ]))
        self.encoder1 = Encoder_conv(channelNum)
        self.encoder2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(channelNum, channelNum, [1, 5])),
            ('conv5x1', ConvBN(channelNum, channelNum, [5, 1])),
            ("conv9x1_bn", ConvBN(channelNum, channelNum, 3)),
        ]))
        self.encoder_conv = Encoder_conv(channelNum * 2)
        self.encoder_conv1 = nn.Sequential(OrderedDict([
            ("conv1x1_bn", ConvBN(channelNum * 2, 2, 1)),
        ]))
        self.fc = nn.Linear(768, int(feedback_bits / self.B))
        self.sig = nn.Sigmoid()
        self.quantize = QuantizationLayer(self.B)
        self.quantization = quantization
 
    def forward(self, x):
        x = self.convban(x)
        encode1 = self.encoder1(x)
        encode2 = self.encoder2(x)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
        out = self.encoder_conv1(out)
        out = out.view(-1, 768)
        out = self.fc(out)
        out = self.sig(out)
        if self.quantization:
            out = self.quantize(out)
        else:
            out = out
        return out
 
 
class Decoder(nn.Module):
    B = 4
    def __init__(self, feedback_bits, quantization=True):
        super(Decoder, self).__init__()
        self.feedback_bits = feedback_bits
        self.dequantize = DequantizationLayer(self.B)
        self.fc = nn.Linear(int(feedback_bits / self.B), 768)
        decoder = OrderedDict([
            ("conv3x3_bn", ConvBN(2, channelNum, 3)),
            ("CRBlock1", CRBlock64()),
            ("CRBlock2", CRBlock()),
        ])
        self.decoder_feature = nn.Sequential(decoder)
        self.out_cov = conv3x3(channelNum, 2)
        self.sig = nn.Sigmoid()
        self.quantization = quantization
 
    def forward(self, x):
        if self.quantization:
            out = self.dequantize(x)
        else:
            out = x
        out = out.view(-1, int(self.feedback_bits / self.B))
        out = self.fc(out)
        out = out.view(-1, 2, 24, 16)
        out = self.decoder_feature(out)
        out = self.out_cov(out)
        return out