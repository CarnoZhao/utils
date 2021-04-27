from typing import Optional, Union, List
import torch.nn as nn

import timm
from timm.models.layers import ConvBnAct, get_act_layer, create_attn, create_conv2d
from timm.models.cspnet import _create_cspnet
import segmentation_models_pytorch as smp
# from .hardnet import HarDNet
from ..acti.acon import AconC, MetaAconC

class Conv3x3_ACON(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding='', dilation=1, groups=1,
                 bias=False, norm_layer=nn.BatchNorm2d):
        super(Conv3x3_ACON, self).__init__()
        self.conv = create_conv2d(
            in_channels, out_channels, kernel_size, stride=1,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = norm_layer(out_channels)
        self.acon = AconC(out_channels)

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.acon(x)
        return x


class DarkBlock_ACON(nn.Module):
    """ DarkNet Block
    """

    def __init__(self, in_chs, out_chs, dilation=1, bottle_ratio=0.5, groups=1,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_layer=None, aa_layer=None,
                 drop_block=None, drop_path=None):
        super(DarkBlock_ACON, self).__init__()
        mid_chs = int(round(out_chs * bottle_ratio))
        ckwargs = dict(act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer, drop_block=drop_block)
        self.conv1 = ConvBnAct(in_chs, mid_chs, kernel_size=1, **ckwargs)
        self.conv2 = Conv3x3_ACON(mid_chs, out_chs, kernel_size=3, dilation=dilation, groups=groups)
        self.attn = create_attn(attn_layer, channels=out_chs)
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv2.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.attn is not None:
            x = self.attn(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        x = x + shortcut
        return x


class Conv3x3_MetaACON(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding='', dilation=1, groups=1,
                 bias=False, norm_layer=nn.BatchNorm2d):
        super(Conv3x3_MetaACON, self).__init__()
        self.conv = create_conv2d(
            in_channels, out_channels, kernel_size, stride=1,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = norm_layer(out_channels)
        self.acon = MetaAconC(out_channels)

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.acon(x)
        return x


class DarkBlock_MetaACON(nn.Module):
    """ DarkNet Block
    """

    def __init__(self, in_chs, out_chs, dilation=1, bottle_ratio=0.5, groups=1,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_layer=None, aa_layer=None,
                 drop_block=None, drop_path=None):
        super(DarkBlock_MetaACON, self).__init__()
        mid_chs = int(round(out_chs * bottle_ratio))
        ckwargs = dict(act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer, drop_block=drop_block)
        self.conv1 = ConvBnAct(in_chs, mid_chs, kernel_size=1, **ckwargs)
        self.conv2 = Conv3x3_MetaACON(mid_chs, out_chs, kernel_size=3, dilation=dilation, groups=groups)
        self.attn = create_attn(attn_layer, channels=out_chs)
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv2.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.attn is not None:
            x = self.attn(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        x = x + shortcut
        return x


class CSPDarkNet53Encoder(nn.Module):
    def __init__(self, pretrain=True, act_layer='leaky_relu', acon_layer='none'):
        super(CSPDarkNet53Encoder, self).__init__()
        if acon_layer == 'acon':
            base = _create_cspnet('cspdarknet53', pretrained=pretrain, pretrained_strict=False, act_layer=get_act_layer(act_layer), block_fn=DarkBlock_ACON)
        elif acon_layer == 'meta_acon':
            base = _create_cspnet('cspdarknet53', pretrained=pretrain, pretrained_strict=False, act_layer=get_act_layer(act_layer), block_fn=DarkBlock_MetaACON)
        elif acon_layer == 'none':
            base = timm.create_model('cspdarknet53', pretrained=pretrain, act_layer=get_act_layer(act_layer))
        self.stem = base.stem
        self.layer0 = base.stages[0]
        self.layer1 = base.stages[1]
        self.layer2 = base.stages[2]
        self.layer3 = base.stages[3]
        self.layer4 = base.stages[4]
        del base
        
        self.depth = 5
        self.out_channels = (32, 64, 128, 256, 512, 1024)
        self.in_channels = 3
        
        
    def get_stages(self):
        return [
            self.stem,
            self.layer0,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ]
    
    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self.depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

class CSPDarkNet53FPN(smp.base.SegmentationModel):
    def __init__(
        self,
        pretrain=True,
        decoder_pyramid_channels: int = 256,
        decoder_segmentation_channels: int = 128,
        decoder_merge_policy: str = "add",
        decoder_dropout: float = 0.2,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()
        
        self.encoder = CSPDarkNet53Encoder(pretrain)
        
        self.decoder = smp.fpn.decoder.FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=self.encoder.depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy,
        )

        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = smp.base.ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "fpn-cspdarknet53"
        self.initialize()


class CSPDarkNet53Unet(smp.base.SegmentationModel):
    def __init__(
        self,
        pretrain=True,
        act_layer='leaky_relu',
        acon_layer='none',
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        aux_params: Optional[dict] = None,
        
    ):
        super().__init__()
        
        self.encoder = CSPDarkNet53Encoder(pretrain, act_layer, acon_layer)
        
        self.decoder = smp.unet.decoder.UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=self.encoder.depth,
            use_batchnorm=decoder_use_batchnorm,
            center=False,
            attention_type=decoder_attention_type
        )

        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = smp.base.ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "unet-cspdarknet53"
        self.initialize()


class ReXNetEncoder(nn.Module):
    def __init__(self, encoder, pretrain=True):
        super(ReXNetEncoder, self).__init__()
        self.base = timm.create_model(encoder, features_only=True, out_indices=(0, 1, 2, 3, 4), pretrained=pretrain)
        self.depth = 5
        self.in_channels = 3
        self.out_channels = (self.in_channels, ) + tuple(self.base.feature_info.channels())
        self.identity = nn.Identity()
    
    def forward(self, x):
        features = []
        features.append(self.identity(x))
        out = self.base(x)
        features.extend(out)

        return features


class ReXNetFPN(smp.base.SegmentationModel):
    def __init__(
        self,
        encoder,
        pretrain=True,
        decoder_pyramid_channels: int = 256,
        decoder_segmentation_channels: int = 128,
        decoder_merge_policy: str = "add",
        decoder_dropout: float = 0.2,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()
        
        self.encoder = ReXNetEncoder(encoder, pretrain)
        
        self.decoder = smp.fpn.decoder.FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=self.encoder.depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy,
        )

        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = smp.base.ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "fpn-rexnet"
        self.initialize()



# class HarDNetEncoder(nn.Module):
#     def __init__(self, arch=85, pretrain=True):
#         super(HarDNetEncoder, self).__init__()
#         base = HarDNet(depth_wise=False, arch=arch, pretrained=pretrain).base

#         self.id = nn.Identity()
#         self.layer0 = nn.Sequential(*base[:2])
#         self.layer1 = nn.Sequential(*base[2:5])
#         self.layer2 = nn.Sequential(*base[5:10])
#         self.layer3 = nn.Sequential(*base[10:15])
#         self.layer4 = nn.Sequential(*base[15:19])
#         del base
        
#         self.depth = 5
#         self.in_channels = 3
#         self.out_channels = (self.in_channels, ) + (96, 192, 320, 720, 1280)
        
        
#     def get_stages(self):
#         return [
#             self.id,
#             self.layer0,
#             self.layer1,
#             self.layer2,
#             self.layer3,
#             self.layer4,
#         ]
    
#     def forward(self, x):
#         stages = self.get_stages()

#         features = []
#         for i in range(self.depth + 1):
#             x = stages[i](x)
#             features.append(x)

#         return features

# class HarDNetFPN(smp.base.SegmentationModel):
#     def __init__(
#         self,
#         arch,
#         pretrain=True,
#         decoder_pyramid_channels: int = 256,
#         decoder_segmentation_channels: int = 128,
#         decoder_merge_policy: str = "add",
#         decoder_dropout: float = 0.2,
#         in_channels: int = 3,
#         classes: int = 1,
#         activation: Optional[str] = None,
#         upsampling: int = 4,
#         aux_params: Optional[dict] = None,
#     ):
#         super().__init__()
        
#         self.encoder = HarDNetEncoder(arch=arch, pretrain=pretrain)
        
#         self.decoder = smp.fpn.decoder.FPNDecoder(
#             encoder_channels=self.encoder.out_channels,
#             encoder_depth=self.encoder.depth,
#             pyramid_channels=decoder_pyramid_channels,
#             segmentation_channels=decoder_segmentation_channels,
#             dropout=decoder_dropout,
#             merge_policy=decoder_merge_policy,
#         )

#         self.segmentation_head = smp.base.SegmentationHead(
#             in_channels=self.decoder.out_channels,
#             out_channels=classes,
#             activation=activation,
#             kernel_size=1,
#             upsampling=upsampling,
#         )

#         if aux_params is not None:
#             self.classification_head = smp.base.ClassificationHead(
#                 in_channels=self.encoder.out_channels[-1], **aux_params
#             )
#         else:
#             self.classification_head = None

#         self.name = "fpn-hardnet"
#         self.initialize()


# class EffNetEncoder(nn.Module):
#     def __init__(self, encoder, pretrain=True):
#         super(EffNetEncoder, self).__init__()
#         self.base = timm.create_model(encoder, features_only=True, pretrained=pretrain)
#         self.depth = 5
#         self.in_channels = 3
#         self.out_channels = (self.in_channels, ) + tuple(self.base.feature_info.channels())
#         self.identity = nn.Identity()
    
#     def forward(self, x):
#         features = []
#         features.append(self.identity(x))
#         out = self.base(x)
#         features.extend(out)

#         return features

# class EffNetUnet(smp.base.SegmentationModel):
    # def __init__(
    #     self,
    #     encoder,
    #     pretrain=True,
    #     decoder_use_batchnorm: bool = True,
    #     decoder_channels: List[int] = (256, 128, 64, 32, 16),
    #     decoder_attention_type: Optional[str] = None,
    #     in_channels: int = 3,
    #     classes: int = 1,
    #     activation: Optional[str] = None,
    #     aux_params: Optional[dict] = None,
    # ):
    #     super().__init__()
        
    #     self.encoder = EffNetEncoder(encoder, pretrain)
        
    #     self.decoder = smp.unet.decoder.UnetDecoder(
    #         encoder_channels=self.encoder.out_channels,
    #         decoder_channels=decoder_channels,
    #         n_blocks=self.encoder.depth,
    #         use_batchnorm=decoder_use_batchnorm,
    #         center=False,
    #         attention_type=decoder_attention_type
    #     )

    #     self.segmentation_head = smp.base.SegmentationHead(
    #         in_channels=decoder_channels[-1],
    #         out_channels=classes,
    #         activation=activation,
    #         kernel_size=3,
    #     )

    #     if aux_params is not None:
    #         self.classification_head = smp.base.ClassificationHead(
    #             in_channels=self.encoder.out_channels[-1], **aux_params
    #         )
    #     else:
    #         self.classification_head = None

    #     self.name = "unet-effnet"
    #     self.initialize()