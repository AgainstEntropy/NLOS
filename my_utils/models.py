# -*- coding: utf-8 -*-
# @Date    : 2021/12/18 19:37
# @Author  : WangYihao
# @File    : model.py

import torch
from torch import nn
import torch.nn.functional as F

# from timm.models.layers import trunc_normal_, DropPath


class NLOS_Conv(nn.Module):
    r"""
    A reproduced version from paper 'What You Can Learn by Staring at a Blank Wall'.
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 5
        depths (tuple(int)): Number of blocks at each stage. Default: (4, 1)
        dims (tuple(int)): Feature dimension at each stage. Default: (64, 64)
    """

    def __init__(self, in_chans=3, num_classes=5, kernel_size=5,
                 depths=(4, 1), dims=(64, 64)):
        super().__init__()

        assert len(depths) == len(dims)
        self.num_stages = len(dims)

        self.stages = nn.ModuleList()
        start_layer = self.conv_block(in_chans, dims[0])
        self.stages.append(start_layer)

        for i in range(self.num_stages - 1):
            stage = nn.Sequential(
                *[self.conv_block(dims[i], dims[i], kernel_size) for _ in range(depths[i] - 1)]
            )
            self.stages.append(stage)
            self.stages.append(self.conv_block(dims[i], dims[i + 1]))

        self.GAP = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.head = nn.Linear(dims[-1], num_classes)

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
            kernel_size (int): Kernel size of Conv layer. Default: 5
        """
        block = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=1e-1),
            nn.MaxPool2d(kernel_size=2)
        )
        return block

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)  # (N, C[i], H, W) -> (N, C[i+1], H, W)
        x = self.GAP(x).squeeze()  # global average pooling, (N, C, H, W) -> (N, C)
        scores = self.head(x)

        return scores


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
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
