from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn

from mmcv.cnn import constant_init, xavier_init
from diffusers.utils import BaseOutput


@dataclass
class GMFlowModelOutput(BaseOutput):
    """
    The output of GMFlow models.

    Args:
        means (`torch.Tensor` of shape `(batch_size, num_gaussians, num_channels, height, width)` or
        `(batch_size, num_gaussians, num_channels, frame, height, width)`):
            Gaussian mixture means.
        logweights (`torch.Tensor` of shape `(batch_size, num_gaussians, 1, height, width)` or
        `(batch_size, num_gaussians, 1, frame, height, width)`):
            Gaussian mixture log-weights (logits).
        logstds (`torch.Tensor` of shape `(batch_size, 1, 1, 1, 1)` or `(batch_size, 1, 1, 1, 1, 1)`):
            Gaussian mixture log-standard-deviations (logstds are shared across all Gaussians and channels).
    """

    means: torch.Tensor
    logweights: torch.Tensor
    logstds: torch.Tensor


class GMOutput2D(nn.Module):

    def __init__(self,
                 num_gaussians,
                 out_channels,
                 embed_dim,
                 constant_logstd=None,
                 logstd_inner_dim=1024,
                 num_logstd_layers=2,
                 activation_fn='silu'):
        super(GMOutput2D, self).__init__()
        self.num_gaussians = num_gaussians
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.constant_logstd = constant_logstd

        if constant_logstd is None:
            if activation_fn == 'gelu-approximate':
                act = partial(nn.GELU, approximate='tanh')
            elif activation_fn == 'silu':
                act = nn.SiLU
            else:
                raise ValueError(f'Unsupported activation function: {activation_fn}')

            assert num_logstd_layers >= 1
            in_dim = self.embed_dim
            logstd_layers = []
            for _ in range(num_logstd_layers - 1):
                logstd_layers.extend([
                    act(),
                    nn.Linear(in_dim, logstd_inner_dim)])
                in_dim = logstd_inner_dim
            self.logstd_layers = nn.Sequential(
                *logstd_layers,
                act(),
                nn.Linear(in_dim, 1))

        self.init_weights()

    def init_weights(self):
        if self.constant_logstd is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    xavier_init(m, distribution='uniform')
            constant_init(self.logstd_layers[-1], val=0)

    def forward(self, x, emb):
        bs, c, h, w = x.size()
        means, logweights = x.split([self.num_gaussians * self.out_channels, self.num_gaussians], dim=1)
        means = means.view(bs, self.num_gaussians, self.out_channels, h, w)
        logweights = logweights.view(bs, self.num_gaussians, 1, h, w).log_softmax(dim=1)
        if self.constant_logstd is None:
            logstds = self.logstd_layers(emb).view(bs, 1, 1, 1, 1)
        else:
            logstds = torch.full(
                (bs, 1, 1, 1, 1), float(self.constant_logstd),
                dtype=x.dtype, device=x.device)
        return GMFlowModelOutput(
            means=means,
            logweights=logweights,
            logstds=logstds)
