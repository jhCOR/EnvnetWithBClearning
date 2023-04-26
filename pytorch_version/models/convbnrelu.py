import torch
import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 initialW=nn.init.kaiming_uniform_, nobias=True):
        super(ConvBNReLU, self).__init__(
            conv = nn.Conv2d(in_channels, out_channels, ksize, stride, pad,
                                 initialW=initialW, nobias=nobias),
            bn = nn.BatchNorm2d(out_channels)
        )

    def __call__(self, x, train):
        h = self.conv(x)
        if train is True:
            print("ConvBNReLU__call__:", train)
            h = self.bn(h)

        return nn.ReLU(h)
