import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import MODULES
from lakonlab.ops.gmflow_ops.gmflow_ops import gm_logprob
from .utils import weighted_loss


_reduction_modes = ['none', 'mean', 'sum', 'batchmean', 'flatmean']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@MODULES.register_module()
class L1Loss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, avg_factor=None):
        return self.loss_weight * l1_loss(
            pred, target, weight=weight, reduction=self.reduction, avg_factor=avg_factor)


@MODULES.register_module()
class TemporalL1Loss(L1Loss):

    def forward(self, pred, target, weight=None, avg_factor=None):
        if pred.ndim == 5 and pred.size(2) > 1:
            t = pred.size(2)
            pred = pred.diff(dim=2)
            target = target.diff(dim=2)
            if weight is not None and weight.ndim >= 3 and weight.size(-3) == t:
                weight = weight.unfold(-3, 2, 1).mean(dim=-1)
        else:
            t = 1
        return super().forward(pred, target, weight=weight, avg_factor=avg_factor) * (t - 1) / t


@weighted_loss
def gaussian_nll_loss(pred, target, logstd, eps=1e-4):
    inverse_std = torch.exp(-logstd).clamp(max=1 / eps)
    diff_weighted = (pred - target) * inverse_std
    loss = 0.5 * (diff_weighted.square() + math.log(2 * math.pi)) + logstd
    return loss


@weighted_loss
def mse_loss(pred, target):
    """MSE loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target (Tensor): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated MSE loss.
    """
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def gaussian_mixture_nll_loss(
        pred_means, target, pred_logstds, pred_logweights):
    """
    Args:
        pred_means (torch.Tensor): Shape (bs, *, num_gaussians, c, h, w)
        target (torch.Tensor): Shape (bs, *, c, h, w)
        pred_logstds (torch.Tensor): Shape (bs, *, 1 or num_gaussians, 1 or c, 1 or h, 1 or w)
        pred_logweights (torch.Tensor): Shape (bs, *, num_gaussians, 1, h, w)

    Returns:
        torch.Tensor: Shape (bs, *, h, w)
    """
    num_channels = pred_means.size(-3)
    loss = -gm_logprob(
        dict(
            means=pred_means,
            logstds=pred_logstds,
            logweights=pred_logweights),
        target.unsqueeze(-4))[0]
    return loss.squeeze(-3) / num_channels
