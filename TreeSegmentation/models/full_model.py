from .HRNetV2 import HRNetV2
from .loss import Loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import log_error, log_info, log_success, log_warn


class FullModel(nn.Module):
    def __init__(self, config) -> None:
        super(FullModel, self).__init__()
        self.model = HRNetV2(config)
        self.loss = Loss()

    def forward(self, x, gt):
        x = self.model(x)

        # upsample for stem
        h, w = gt.size(1), gt.size(2)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        # squeeze for channel 1
        x = x.squeeze(1)

        return x, self.loss(x, gt)


class SynthModel(nn.Module):
    def __init__(self, config) -> None:
        super(SynthModel, self).__init__()
        self.model = HRNetV2(config)

    def forward(self, x,origin_size):
        x = self.model(x)

        x = F.interpolate(x, size=origin_size, mode='bilinear', align_corners=True)
        x = x.squeeze(1)

        return x
