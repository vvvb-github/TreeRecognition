import torch
import torch.nn as nn
from utils import log_error, log_info, log_success, log_warn


class DiceLoss(nn.Module):
    def __init__(self, loss_weight=1.0) -> None:
        super(DiceLoss, self).__init__()
        self._loss_weight = loss_weight

    def forward(self, input, target, reduce=True):
        batch_size = input.size(0)
        input = torch.sigmoid(input)

        input = input.contiguous().view(batch_size, -1)
        target = target.contiguous().view(batch_size, -1).float()

        a = torch.sum(input * target, dim=1)
        b = torch.sum(input * input, dim=1) + 0.001
        c = torch.sum(target * target, dim=1) + 0.001
        d = (2 * a) / (b + c)
        loss = 1 - d

        loss = self._loss_weight * loss

        if reduce:
            loss = torch.mean(loss)

        return loss


class Loss(nn.Module):
    def __init__(self) -> None:
        super(Loss, self).__init__()

        self._dice_loss = DiceLoss()
        self._l2loss = nn.MSELoss(reduce=False)

    def forward(self, input, target):
        dice_loss = self._dice_loss(input, target, reduce=False)
        l2loss = self._l2loss(torch.sigmoid(
            input), target.float()).mean([1, 2])
        dice_loss = torch.mean(dice_loss)
        l2loss = torch.mean(l2loss)
        loss = dice_loss + l2loss * 1

        return loss
