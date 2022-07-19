import math
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ChannelAttention, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conva = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.convm = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

        self.gelu = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x.flatten(2)
        ya = self.avg_pool(x)
        ya = self.conva(ya.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)

        ym = self.max_pool(x)
        ym = self.convm(ym.squeeze(-1).transpose(-1, -2))

        y = torch.matmul(ya, ym)  # 矩阵相乘尺寸c*c

        out = torch.matmul(y, residual).view(B, C, H, W)

        out = self.gelu(out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(SpatialAttention, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        x = x * residual
        return self.gelu(x)


class ChannelBatchAtt(nn.Module):
    def __init__(self, channels):
        super(ChannelBatchAtt, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class ercs_block(nn.Module):
    def __init__(self, channel):
        super(ercs_block, self).__init__()
        self.channelattention = ChannelAttention(channel)
        self.spatialattention = SpatialAttention(channel)
        self.channelbatchatt = ChannelBatchAtt(channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        yc = self.channelattention(x)
        ys = self.spatialattention(x)
        yb = self.channelbatchatt(x)
        y = yc + ys + yb
        y = self.sigmoid(y)
        return x * y.expand_as(x)


if __name__ == '__main__':
    x = torch.randn([2, 16, 20, 20])
    # model = ChannelAttention(16)
    # model = SpatialAttention(16)
    # model = ChannelBatchAtt(16)
    model = ercs_block(16)
    out = model(x)

    print(out)
