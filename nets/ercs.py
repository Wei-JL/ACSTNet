import math
import torch
import torch.nn as nn


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
