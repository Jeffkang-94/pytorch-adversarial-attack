from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F


def init_weight(*args):
   return nn.Parameter(nn.init.kaiming_normal_(torch.zeros(*args), mode='fan_out', nonlinearity='relu'))


class Block(nn.Module):
    """
    Pre-activated ResNet block.
    """
    def __init__(self, width):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(width, affine=False)
        self.register_parameter('conv0', init_weight(width, width, 3, 3))
        self.bn1 = nn.BatchNorm2d(width, affine=False)
        self.register_parameter('conv1', init_weight(width, width, 3, 3))

    def forward(self, x):
        h = F.conv2d(F.relu(self.bn0(x)), self.conv0, padding=1)
        h = F.conv2d(F.relu(self.bn1(h)), self.conv1, padding=1)
        return x + h


class DownsampleBlock(nn.Module):
    """
    Downsample block.
    Does F.avg_pool2d + torch.cat instead of strided conv.
    """

    def __init__(self, width):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(width // 2, affine=False)
        self.register_parameter('conv0', init_weight(width, width // 2, 3, 3))
        self.bn1 = nn.BatchNorm2d(width, affine=False)
        self.register_parameter('conv1', init_weight(width, width, 3, 3))

    def forward(self, x):
        h = F.conv2d(F.relu(self.bn0(x)), self.conv0, padding=1, stride=2)
        h = F.conv2d(F.relu(self.bn1(h)), self.conv1, padding=1)
        x_d = F.avg_pool2d(x, kernel_size=3, padding=1, stride=2)
        x_d = torch.cat([x_d, torch.zeros_like(x_d)], dim=1)
        return x_d + h


class WRN(nn.Module):
    """
    Implementation of modified Wide Residual Network.
    Differences with pre-activated ResNet and Wide ResNet:
       * BatchNorm has no affine weight and bias parameters
       * First layer has 16 * width channels
       * Last fc layer is removed in favor of 1x1 conv + F.avg_pool2d
       * Downsample is done by F.avg_pool2d + torch.cat instead of strided conv
    First and last convolutional layers are kept in float32.
    """

    def __init__(self, depth, width, num_classes):
        super().__init__()
        widths = [int(v * width) for v in (16, 32, 64)]
        n = (depth - 2) // 6
        self.register_parameter('conv0', init_weight(widths[0], 3, 3, 3))
        self.group0 = self._make_block(widths[0], n)
        self.group1 = self._make_block(widths[1], n, downsample=True)
        self.group2 = self._make_block(widths[2], n, downsample=True)
        self.bn = nn.BatchNorm2d(widths[2], affine=False)
        self.register_parameter('conv_last', init_weight(num_classes, widths[2], 1, 1))
        self.bn_last = nn.BatchNorm2d(num_classes)


    def _make_block(self, width, n, downsample=False):
        def select_block(j):
            if downsample and j == 0:
                return DownsampleBlock(width)
            return Block(width)

        return nn.Sequential(OrderedDict(('block%d' % i, select_block(i)) for i in range(n)))


    def forward(self, x):
        h = F.conv2d(x, self.conv0, padding=1)
        h = self.group0(h)
        h = self.group1(h)
        h = self.group2(h)
        h = F.relu(self.bn(h))
        h = F.conv2d(h, self.conv_last)
        h = self.bn_last(h)
        out = F.avg_pool2d(h, kernel_size=h.shape[-2:]).view(h.shape[0], -1)
        return out
   
    

    


