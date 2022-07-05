#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import torch.nn as nn

from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
from .attention import eca_block, cbam_block, eam_block, CA_Block
from .ercs import ercs_block

attention_block = [eca_block, cbam_block, eam_block, CA_Block, ercs_block]


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=1.0, in_channels=[128, 256, 512, 1024], act="silu", depthwise=False, ):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(in_channels=int(in_channels[i] * width), out_channels=int(256 * width), ksize=1, stride=1,
                         act=act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=num_classes, kernel_size=1, stride=1, padding=0)
            )

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=4, kernel_size=1, stride=1, padding=0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, inputs):
        # ---------------------------------------------------#
        #   inputs输入
        #   P2_out  160, 160, 64
        #   P3_out  80, 80, 128
        #   P4_out  40, 40, 256
        #   P5_out  20, 20, 512
        # ---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            # ---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            # ---------------------------------------------------#
            x = self.stems[k](x)
            # ---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            # ---------------------------------------------------#
            cls_feat = self.cls_convs[k](x)
            # ---------------------------------------------------#
            #   判断特征点所属的种类
            #   160, 160, num_classes
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            # ---------------------------------------------------#
            cls_output = self.cls_preds[k](cls_feat)

            # ---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            # ---------------------------------------------------#
            reg_feat = self.reg_convs[k](x)
            # ---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 160, 160, 4
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            # ---------------------------------------------------#
            reg_output = self.reg_preds[k](reg_feat)
            # ---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 160, 160, 1
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            # ---------------------------------------------------#
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs


class YOLOPAFPN(nn.Module):
    def __init__(self, depth=1.0, width=1.0, in_features=("dark2", "dark3", "dark4", "dark5"),
                 in_channels=[128, 256, 512, 1024],
                 depthwise=False, act="silu", att_phi=5):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.att_phi = att_phi

        if 1 <= self.att_phi <= 5:
            self.csp4Up_att = attention_block[self.att_phi - 1](int(in_channels[2] * width))  # 256
            self.csp3Up_att = attention_block[self.att_phi - 1](int(in_channels[1] * width))  # 128
            self.csp2Up_att = attention_block[self.att_phi - 1](int(in_channels[0] * width))  # 64

            self.csp4Down_att = attention_block[self.att_phi - 1](int(in_channels[2] * width))  # 256
            self.csp3Down_att = attention_block[self.att_phi - 1](int(in_channels[1] * width))  # 128
            # self.csp2Down_att = attention_block[self.att_phi - 1](int(in_channels[0] * width))  # 64

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # -------------------------------------------#
        #   20, 20, 512 -> 20, 20, 256
        # -------------------------------------------#
        self.lateral_conv0 = BaseConv(int(in_channels[3] * width), int(in_channels[2] * width), 1, 1, act=act)

        # -------------------------------------------#
        #   40, 40, 256 -> 40, 40, 256
        # -------------------------------------------#
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[2] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # -------------------------------------------#
        #   40, 40, 128 -> 40, 40, 64
        # -------------------------------------------#
        self.reduce_conv0 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)

        # -------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        # -------------------------------------------#
        self.C2_p2 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # -------------------------------------------#
        #   40, 40, 256 -> 40, 40, 128
        # -------------------------------------------#
        self.reduce_conv1 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        # -------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        # -------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # -------------------------------------------#
        #   160, 160, 64 -> 80, 80, 64
        # -------------------------------------------#
        self.bu_conv0 = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)

        # -------------------------------------------#
        #   80, 80, 128 -> 80, 80, 128
        # -------------------------------------------#
        self.C2_n2 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # -------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        # -------------------------------------------#
        self.bu_conv2 = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        # -------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        # -------------------------------------------#
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # -------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        # -------------------------------------------#
        self.bu_conv1 = Conv(int(in_channels[2] * width), int(in_channels[2] * width), 3, 2, act=act)
        # -------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        # -------------------------------------------#
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[2] * width),
            int(in_channels[3] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        out_features = self.backbone.forward(input)
        [feat0, feat1, feat2, feat3] = [out_features[f] for f in self.in_features]

        # -------------------------------------------#
        #   20, 20, 512 -> 20, 20, 256
        # -------------------------------------------#
        P5 = self.lateral_conv0(feat3)

        # -------------------------------------------#
        #  20, 20, 256 -> 40, 40, 256
        # -------------------------------------------#
        P5_upsample = self.upsample(P5)

        # -------------------------------------------#
        #  40, 40, 256 + 40, 40, 256 -> 40, 40, 512
        # -------------------------------------------#
        P5_upsample = torch.cat([P5_upsample, feat2], 1)

        # -------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        # -------------------------------------------#
        P5_upsample = self.C3_p4(P5_upsample)
        if 1 <= self.att_phi <= 5:
            P5_upsample = self.csp4Up_att(P5_upsample)

        # -------------------------------------------#
        #   40, 40, 256 -> 40, 40, 128
        # -------------------------------------------#
        P4 = self.reduce_conv1(P5_upsample)

        # -------------------------------------------#
        #   40, 40, 128 -> 80, 80, 128
        # -------------------------------------------#
        P4_upsample = self.upsample(P4)

        # -------------------------------------------#
        #   80, 80, 128 + 80, 80, 128 -> 80, 80, 256
        # -------------------------------------------#
        P4_upsample = torch.cat([P4_upsample, feat1], 1)

        # -------------------------------------------#
        #   80, 80, 256 -> 80, 80, 128
        # -------------------------------------------#
        P3_temp = self.C3_p3(P4_upsample)
        if 1 <= self.att_phi <= 5:
            P3_temp = self.csp3Up_att(P3_temp)

        # -------------------------------------------#
        #   80, 80, 128 -> 80, 80, 64
        # -------------------------------------------#
        P2 = self.reduce_conv0(P3_temp)

        # -------------------------------------------#
        #   80, 80, 64 -> 160, 160, 64
        # -------------------------------------------#
        P2_upsample = self.upsample(P2)

        # -------------------------------------------#
        #   160, 160, 64 + 160, 160, 64 -> 160, 160, 128
        # -------------------------------------------#
        P2_upsample = torch.cat([P2_upsample, feat0], 1)

        # -------------------------------------------#
        #   160, 160, 128 -> 160, 160, 64
        # -------------------------------------------#
        P2_out = self.C2_p2(P2_upsample)
        P2_att = P2_out.clone()
        if 1 <= self.att_phi <= 5:
            P2_att = self.csp2Up_att(P2_att)

        # -------------------------------------------#
        #   160, 160, 64 -> 80, 80, 64
        # -------------------------------------------#
        P2_downsample = self.bu_conv0(P2_att)

        # -------------------------------------------#
        #   80, 80, 64 + 80, 80, 64 -> 80, 80, 128
        # -------------------------------------------#
        P2_downsample = torch.cat([P2_downsample, P2], 1)

        # -------------------------------------------#
        #   80, 80, 128 -> 80, 80, 128
        # -------------------------------------------#
        P3_out = self.C2_n2(P2_downsample)
        P3_att = P3_out.clone()
        if 1 <= self.att_phi <= 5:
            P3_att = self.csp3Down_att(P3_att)

        # -------------------------------------------#
        #   80, 80, 128 -> 40, 40, 128
        # -------------------------------------------#
        P3_downsample = self.bu_conv2(P3_att)

        # -------------------------------------------#
        #   40, 40, 128 + 40, 40, 128 -> 40, 40, 256
        # -------------------------------------------#
        P3_downsample = torch.cat([P3_downsample, P4], 1)

        # -------------------------------------------#
        #   40, 40, 256 -> 40, 40, 256
        # -------------------------------------------#
        P4_out = self.C3_n3(P3_downsample)
        P4_att = P4_out.clone()
        if 1 <= self.att_phi <= 5:
            P4_att = self.csp4Down_att(P4_att)

        # -------------------------------------------#
        #   40, 40, 256 -> 20, 20, 256
        # -------------------------------------------#
        P4_downsample = self.bu_conv1(P4_att)

        # -------------------------------------------#
        #   20, 20, 256 + 20, 20, 256 -> 20, 20, 512
        # -------------------------------------------#
        P4_downsample = torch.cat([P4_downsample, P5], 1)

        # -------------------------------------------#
        #   20, 20, 512 -> 20, 20, 512
        # -------------------------------------------#
        P5_out = self.C3_n4(P4_downsample)

        return P2_out, P3_out, P4_out, P5_out


class YoloBody(nn.Module):
    def __init__(self, num_classes, phi):
        super().__init__()
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33, }
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        depth, width = depth_dict[phi], width_dict[phi]
        depthwise = True if phi == 'nano' else False

        self.backbone = YOLOPAFPN(depth, width, depthwise=depthwise)
        self.head = YOLOXHead(num_classes, width, depthwise=depthwise)

    def forward(self, x):
        fpn_outs = self.backbone.forward(x)
        outputs = self.head.forward(fpn_outs)
        return outputs
