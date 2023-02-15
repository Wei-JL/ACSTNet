import math
import torch
import torch.nn as nn


class eam_block(nn.Module):
    def __init__(self, channels, shape=None, out_channels=None, no_spatial=True):
        super(eam_block, self).__init__()
        self.Channel_Att = Channel_Att(channels)

    def forward(self, x):
        x_out1 = self.Channel_Att(x)

        return x_out1


if __name__ == '__main__':
    input = torch.rand([2, 3, 64, 64])
    # print(input)
    model = eam_block(3, [2, 3, 64, 64])
    x = model(input)
    print(x, x.shape)
