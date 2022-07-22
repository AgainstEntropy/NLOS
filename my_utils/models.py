# -*- coding: utf-8 -*-
# @Date    : 2021/12/18 19:37
# @Author  : WangYihao
# @File    : model.py
import math
from typing import Tuple, Optional, Callable, List, Type, Any, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F


# from timm.models.layers import trunc_normal_, DropPath


class NLOS_Conv(nn.Module):
    r"""
    A reproduced version from paper 'What You Can Learn by Staring at a Blank Wall'.
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 5
        depths (tuple(int)): Number of blocks at each stage. Default: (3, 1)
        dims (tuple(int)): Feature dimension at each stage. Default: (16, 16)
    """

    def __init__(self,
                 in_chans: int = 3,
                 num_classes: int = 20,
                 kernel_size: int = 7,
                 depths: Tuple = (4, 1),
                 dims: Tuple = (16, 32)):
        super().__init__()

        self.num_classes = num_classes
        assert len(depths) == len(dims)
        self.num_stages = len(dims)

        self.stages = nn.ModuleList()
        start_layer = self.conv_block(in_chans, dims[0], kernel_size)
        self.stages.append(start_layer)

        for i in range(self.num_stages - 1):
            if depths[i] - 1 > 0:
                self.stages.append(nn.Sequential(
                    *[self.conv_block(dims[i], dims[i], kernel_size) for _ in range(depths[i] - 1)]
                ))
            if i < self.num_stages - 1:
                self.stages.append(self.conv_block(dims[i], dims[i + 1], kernel_size))

        self.GAP = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.head = nn.Linear(dims[-1], num_classes)
        self.loss_func = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            # trunc_normal_(m.weight, std=.02)
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def conv_block(self, in_chans, out_chans, kernel_size=5):
        r"""
        Args:
            in_chans (int): Number of input image channels.
            out_chans (int): Number of output image channels.
            kernel_size (int): Kernel size of Conv layer. Default: 7
        """
        block = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=1e-1),
            # nn.MaxPool2d(kernel_size=2)
        )
        return block

    def forward(self, xx):
        x, labels = xx
        for stage in self.stages:
            x = stage(x)  # (N, C[i], H, W) -> (N, C[i+1], H, W)
        x = self.GAP(x)
        x = x.flatten(start_dim=1)
        scores = self.head(x)
        loss = self.loss_func(scores, labels)
        preds = scores.argmax(axis=1)

        return loss, preds


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class my_NLOS_r21d(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            num_classes: int = 20,
            kernel_size: int = 5,
            depths: Tuple = (2, 1),
            dims: Tuple = (16, 32)
    ) -> None:
        super(my_NLOS_r21d, self).__init__()

        assert len(depths) == len(dims)
        self.num_stages = len(dims)

        self.stages = nn.ModuleList()
        stem_layer = self.r21d_block(in_chans, dims[0], kernel_size)
        self.stages.append(stem_layer)

        for i in range(self.num_stages - 1):
            if depths[i] - 1 > 0:
                self.stages.append(nn.Sequential(
                    *[self.r21d_block(dims[i], dims[i], kernel_size) for _ in range(depths[i] - 1)]
                ))
            if i < self.num_stages - 1:
                self.stages.append(self.r21d_block(dims[i], dims[i + 1], kernel_size))

        self.GAP = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.head = nn.Linear(dims[-1], num_classes)
        self.loss_func = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            # trunc_normal_(m.weight, std=.02)
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    def r21d_block(self,
                   in_chans: int,
                   out_chans: int,
                   ks: int = 5):
        r"""
        Args:
            in_chans (int): Number of input image channels.
            out_chans (int): Number of output image channels.
            ks (int): Kernel size of Conv layer. Default: 7
        """
        mid_chans = in_chans if in_chans == out_chans else int(math.sqrt(in_chans * out_chans))
        stride = 1 if in_chans == out_chans else 2
        return nn.Sequential(
            nn.Conv3d(in_chans, mid_chans, bias=False,
                      kernel_size=(1, ks, ks),
                      stride=(1, 1, 1),
                      padding=(0, 2, 2)),
            nn.BatchNorm3d(mid_chans),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_chans, out_chans, bias=False,
                      kernel_size=(ks, 1, 1),
                      stride=(1, stride, stride),
                      padding=(2, 0, 0)),
            nn.BatchNorm3d(out_chans),
            nn.ReLU(inplace=True),
        )

    def forward(self, xx):
        x, labels = xx
        for stage in self.stages:
            x = stage(x)  # (N, C[i], H, W) -> (N, C[i+1], H, W)
        x = self.GAP(x)
        x = x.flatten(start_dim=1)
        scores = self.head(x)
        loss = self.loss_func(scores, labels)
        preds = scores.argmax(axis=1)

        return loss, preds


class Conv2Plus1D(nn.Sequential):

    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            midplanes: int,
            stride: int = 1,
            padding: int = 1
    ) -> None:
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1),
                      stride=(stride, 1, 1), padding=(padding, 0, 0),
                      bias=False))

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        return stride, stride, stride


class R2Plus1DBlock(nn.Module):
    def __init__(
            self,
            inplanes: int,
            planes: int,
            conv_builder: nn.Module,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
    ) -> None:
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(R2Plus1DBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """

    def __init__(self, stem_in: int = 8, stem_out: int = 16) -> None:
        super(R2Plus1dStem, self).__init__(
            nn.Conv3d(3, stem_in, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False),
            nn.BatchNorm3d(stem_in),
            nn.ReLU(inplace=True),
            nn.Conv3d(stem_in, stem_out, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False),
            nn.BatchNorm3d(stem_out),
            nn.ReLU(inplace=True))


class NLOS_r21d(nn.Module):

    def __init__(
            self,
            depths: List[int],
            dims: List[int],
            block: Type[R2Plus1DBlock] = R2Plus1DBlock,
            conv_maker: Type[Conv2Plus1D] = Conv2Plus1D,
            stem: Callable[..., nn.Module] = R2Plus1dStem,
            num_classes: int = 20
    ) -> None:
        """Generic resnet video generator.

        Args:
            block: resnet building block. Defaults to R2Plus1DBlock.
            conv_maker: generator function for each layer. Defaults to Conv2Plus1D.
            depths: number of blocks per layer. Defaults to [2,2,2,2].
            stem: module specifying the ResNet stem. Defaults to R2Plus1dStem.
            num_classes: Dimension of the final FC layer. Defaults to 20.
        """
        super(NLOS_r21d, self).__init__()
        assert len(depths) == len(dims)
        self.num_blocks = len(depths)
        self.inplanes = 16
        strides = [1, 2, 2, 2]

        self.stem = stem(stem_in=8, stem_out=dims[0])

        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            self.blocks.append(self._make_layer(block, conv_maker, dims[i], depths[i], stride=strides[i]))

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(dims[-1], num_classes)

        self.loss_func = nn.CrossEntropyLoss()

        # init weights
        self._initialize_weights()

    def forward(self, xx: Union[Tensor]):
        x, labels = xx
        x = self.stem(x)

        for block in self.blocks:
            x = block(x)

        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(start_dim=1)
        scores = self.fc(x)

        loss = self.loss_func(scores, labels)
        preds = scores.argmax(axis=1)

        return loss, preds

    def _make_layer(
            self,
            block: Type[R2Plus1DBlock],
            conv_builder: Type[Conv2Plus1D],
            planes: int,
            blocks: int,
            stride: int = 1
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes)
            )
        layers = [block(self.inplanes, planes, conv_builder, stride, downsample)]

        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
