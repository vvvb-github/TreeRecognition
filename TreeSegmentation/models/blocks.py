import torch.nn as nn


BN_MOMENTUM = 0.01


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None) -> None:
        super(BasicBlock, self).__init__()
        self._stride = stride
        self._downsample = downsample
        self._conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self._bn1 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self._conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self._bn2 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self._relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self._conv1(x)
        out = self._bn1(out)
        out = self._relu(out)

        out = self._conv2(out)
        out = self._bn2(out)

        res = x if self._downsample is None else self._downsample(x)
        out += res
        out = self._relu(out)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None) -> None:
        super(BottleNeck, self).__init__()
        self._stride = stride
        self._downsample = downsample
        self._conv1 = nn.Conv2d(in_channels, out_channels,
                                kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self._conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self._bn2 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self._conv3 = nn.Conv2d(
            out_channels, out_channels*self.expansion, kernel_size=1, bias=False)
        self._bn3 = nn.BatchNorm2d(
            out_channels*self.expansion, momentum=BN_MOMENTUM)
        self._relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self._conv1(x)
        out = self._bn1(out)
        out = self._relu(out)

        out = self._conv2(out)
        out = self._bn2(out)
        out = self._relu(out)

        out = self._conv3(out)
        out = self._bn3(out)

        res = x if self._downsample is None else self._downsample(x)
        out += res
        out = self._relu(out)

        return out


block_table = {
    'BASIC': BasicBlock,
    'BOTTLENECK': BottleNeck
}
