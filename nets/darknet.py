#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
from torch import nn
from nets.swinnet import PatchEmbed, BasicLayer, PatchMerging


def merge_singe(tensor, H=0, W=0, model="CSP"):
    """
    tensor: 经过dark后的中间量x, 经过swin后的中间量y
    H: swin_y保存的高
    W: swin_y保存的宽
    """
    if model == "CSP":
        B, L, C = tensor.shape
        assert L == H * W, "input feature has wrong size"

        swin_y = tensor.transpose(1, 2).view(B, C, H, W)

        return swin_y
    elif model == "Swin":
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        csp_x = tensor.flatten(2).transpose(1, 2)
        return csp_x

    else:
        assert False, "模式不为<CSP> 或者 <Swin>"


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1, )
        return self.conv(x)


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act, )
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


# --------------------------------------------------#
#   残差结构的构建，小的残差结构
# --------------------------------------------------#
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act="silu", ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        # --------------------------------------------------#
        #   利用1x1卷积进行通道数的缩减。缩减率一般是50%
        # --------------------------------------------------#
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # --------------------------------------------------#
        #   利用3x3卷积进行通道数的拓张。并且完成特征提取
        # --------------------------------------------------#
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act="silu", ):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        # --------------------------------------------------#
        #   主干部分的初次卷积
        # --------------------------------------------------#
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # --------------------------------------------------#
        #   大的残差边部分的初次卷积
        # --------------------------------------------------#
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # -----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        # -----------------------------------------------#
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)

        # --------------------------------------------------#
        #   根据循环的次数构建上述Bottleneck残差结构
        # --------------------------------------------------#
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in
                       range(n)]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        # -------------------------------#
        #   x_1是主干部分
        # -------------------------------#
        x_1 = self.conv1(x)
        # -------------------------------#
        #   x_2是大的残差边部分
        # -------------------------------#
        x_2 = self.conv2(x)

        # -----------------------------------------------#
        #   主干部分利用残差结构堆叠继续进行特征提取
        # -----------------------------------------------#
        x_1 = self.m(x_1)
        # -----------------------------------------------#
        #   主干部分和大的残差边部分进行堆叠
        # -----------------------------------------------#
        x = torch.cat((x_1, x_2), dim=1)
        # -----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        # -----------------------------------------------#
        return self.conv3(x)


class CSPDarknet(nn.Module):
    def __init__(self, dep_mul, wid_mul, out_features=("dark2", "dark3", "dark4", "dark5"), depthwise=False, act="silu",
                 patch_size=4, in_chans=3, embed_dim=48,
                 # depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 depths=(6,), num_heads=(12,),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        # -----------------------------------------------#
        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # -----------------------------------------------#
        #   利用focus网络结构进行特征提取
        #   640, 640, 3 -> 320, 320, 12 -> 320, 320, 64
        # -----------------------------------------------#
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # -----------------------------------------------#
        #   完成卷积之后，320, 320, 64 -> 160, 160, 128
        #   完成CSPlayer之后，160, 160, 128 -> 160, 160, 128
        # -----------------------------------------------#
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act),
        )

        self.cat_conv = Conv(base_channels * 2 + embed_dim, base_channels * 2, 1, 1, act=act)
        self.cat_csp = CSPLayer((base_channels * 2 + embed_dim) * 2, base_channels * 4, n=base_depth,
                                depthwise=depthwise, act=act)

        # -----------------------------------------------#
        #   完成卷积之后，160, 160, 128 -> 80, 80, 256
        #   完成CSPlayer之后，80, 80, 256 -> 80, 80, 256
        # -----------------------------------------------#
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        # -----------------------------------------------#
        #   完成卷积之后，80, 80, 256 -> 40, 40, 512
        #   完成CSPlayer之后，40, 40, 512 -> 40, 40, 512
        # -----------------------------------------------#
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        # -----------------------------------------------#
        #   完成卷积之后，40, 40, 512 -> 20, 20, 1024
        #   完成SPP之后，20, 20, 1024 -> 20, 20, 1024
        #   完成CSPlayer之后，20, 20, 1024 -> 20, 20, 1024
        # -----------------------------------------------#
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise,
                     act=act),
        )

        # swin transformer部分
        # self.num_classes = num_classes
        # embed_dim = wid_mul * embed_dim  # "s" 模型

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        # self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # dim_list = [1, 1.5, 2, 2]
        # build layers
        # self.layers = nn.ModuleList()
        # for i_layer in range(self.num_layers):
        #     # 注意这里构建的stage和论文图中有些差异
        #     # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的
        #     layers = BasicLayer(dim=int(embed_dim * 3 ** i_layer),
        #                         depth=depths[i_layer],
        #                         num_heads=num_heads[i_layer],
        #                         window_size=window_size,
        #                         mlp_ratio=self.mlp_ratio,
        #                         qkv_bias=qkv_bias,
        #                         drop=drop_rate,
        #                         attn_drop=attn_drop_rate,
        #                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
        #                         norm_layer=norm_layer,
        #                         # downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
        #                         downsample=PatchMerging if (i_layer <= self.num_layers - 1) else None,
        #                         use_checkpoint=use_checkpoint)

        # self.layers = nn.ModuleList()
        # for i_layer in range(self.num_layers):
        # 注意这里构建的stage和论文图中有些差异
        # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的
        i_layer = 0
        self.layers = BasicLayer(dim=int(embed_dim * 3 ** i_layer),
                                 depth=depths[i_layer],
                                 num_heads=num_heads[i_layer],
                                 window_size=window_size,
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop_rate,
                                 attn_drop=attn_drop_rate,
                                 drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                 norm_layer=norm_layer,
                                 # downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                 downsample=PatchMerging if (i_layer <= self.num_layers - 1) else None,
                                 use_checkpoint=use_checkpoint)

        # self.csp_swin = CSPLayer(embed_dim * 2, int(embed_dim * 1.5), n=base_depth * 3, depthwise=depthwise, act=act)

    def forward(self, x):
        y, H, W = self.patch_embed(x)
        conv_y = merge_singe(y, H, W, model="CSP")  # 转换为conv的维度
        # -----------------------------------------------#
        #   Patch Partition的输出为batch, 160*160, embed_dim
        # -----------------------------------------------#

        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        # -----------------------------------------------#
        #   dark2的输出为160, 160, 64，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark2(x)
        outputs["dark2"] = x
        x = torch.cat([x, conv_y], dim=1)  # channel: 32+48
        x = self.cat_conv(x)
        # outputs["dark2"] = x

        # -----------------------------------------------#
        #   dark3的输出为80, 80, 128，是一个有效特征层
        # -----------------------------------------------#
        y, H, W = self.layers(y, H, W)
        conv_y = merge_singe(y, H, W, model="CSP")

        x = self.dark3(x)
        outputs["dark3"] = x
        x = torch.cat([x, conv_y], dim=1)
        x = self.cat_csp(x)
        # outputs["dark3"] = x

        # -----------------------------------------------#
        #   dark4的输出为40, 40, 256，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark4(x)
        outputs["dark4"] = x

        # -----------------------------------------------#
        #   dark5的输出为20, 20, 512，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


if __name__ == '__main__':
    print(CSPDarknet(1, 1))
