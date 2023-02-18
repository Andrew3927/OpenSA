import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class MeanLoss(nn.Module):
    def __init__(self):
        super(MeanLoss, self).__init__()

    def forward(self, output: torch.Tensor, target: Tensor) -> torch.Tensor:
        losses = [F.mse_loss(output, target), F.l1_loss(output, target), F.smooth_l1_loss(output, target)]
        loss = torch.mean(torch.stack(losses))
        return loss

    def __call__(self, output: torch.Tensor, target: Tensor) -> torch.Tensor:
        return self.forward(output, target)

class QuantileLoss(nn.Module):
    def __init__(self, quantile: int = ...):
        super(QuantileLoss, self).__init__()
        self.__quantile = quantile

    def forward(self, output: torch.Tensor, target: Tensor) -> torch.Tensor:
        quantile_loss = F.smooth_l1_loss(output, target, reduction='none')
        mask = (target < output)
        loss = ((1.0 - self.__quantile) * quantile_loss * mask + self.__quantile * quantile_loss * (~mask)).mean()
        return loss

    def __call__(self, output: torch.Tensor, target: Tensor) -> torch.Tensor:
        return self.forward(output, target)
