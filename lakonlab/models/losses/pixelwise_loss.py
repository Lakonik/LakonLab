import math

import torch
import torch.nn.functional as F

from lakonlab.ops.gmflow_ops.gmflow_ops import gm_logprob
from .utils import weighted_loss


_reduction_modes = ['none', 'mean', 'sum', 'batchmean', 'flatmean']


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
