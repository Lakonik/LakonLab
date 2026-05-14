# Copyright (c) 2026 Hansheng Chen

import math

import torch
import torch.nn as nn
from diffusers.models.embeddings import Timesteps

from ...builder import MODULES
from .gm_output import GMFlowModelOutput


def get_1d_sincos_pos_embed(
        embed_dim, pos, min_period=1e-3, max_period=10):
    if embed_dim % 2 != 0:
        raise ValueError('embed_dim must be divisible by 2')
    half_dim = embed_dim // 2
    period = torch.logspace(
        math.log(min_period), math.log(max_period), half_dim, base=math.e,
        dtype=pos.dtype, device=pos.device)
    out = pos.unsqueeze(-1) * (2 * math.pi / period)
    emb = torch.cat(
        [torch.sin(out), torch.cos(out)],
        dim=-1)
    return emb


def get_2d_sincos_pos_embed(
        embed_dim, crd, min_period=1e-3, max_period=10):
    """
    Args:
        embed_dim (int)
        crd (torch.Tensor): Shape (bs, 2)
    """
    if embed_dim % 2 != 0:
        raise ValueError('embed_dim must be divisible by 2')
    bs = crd.size(0)
    emb = get_1d_sincos_pos_embed(
        embed_dim // 2, crd.flatten(), min_period, max_period)  # (bs * 2, embed_dim // 2)
    return emb.reshape(bs, embed_dim)


class SinCos2DPosEmbed(nn.Module):
    def __init__(self, num_channels=256, min_period=1e-3, max_period=10):
        super().__init__()
        self.num_channels = num_channels
        self.min_period = min_period
        self.max_period = max_period

    def forward(self, hidden_states):
        """
        Args:
            hidden_states (torch.Tensor): Shape (B, 2)

        Returns:
            torch.Tensor: Shape (B, num_channels)
        """
        return get_2d_sincos_pos_embed(self.num_channels, hidden_states, self.min_period, self.max_period)


@MODULES.register_module()
class GMFlowMLP2DDenoiser(nn.Module):
    def __init__(
            self,
            num_gaussians=32,
            pos_min_period=5e-3,
            pos_max_period=50,
            embed_dim=256,
            hidden_dim=512,
            constant_logstd=None,
            num_layers=5):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.constant_logstd = constant_logstd
        self.time_proj = Timesteps(num_channels=embed_dim, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.pos_emb = SinCos2DPosEmbed(num_channels=embed_dim, min_period=pos_min_period, max_period=pos_max_period)

        in_dim = embed_dim * 2
        mlp = []
        for _ in range(num_layers):
            mlp.append(nn.Linear(in_dim, hidden_dim))
            mlp.append(nn.SiLU())
            in_dim = hidden_dim
        self.net = nn.Sequential(*mlp)
        self.out_means = nn.Linear(hidden_dim, num_gaussians * 2)
        self.out_logweights = nn.Linear(hidden_dim, num_gaussians)
        if constant_logstd is None:
            self.out_logstds = nn.Linear(hidden_dim, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.out_logweights.weight)
        if self.constant_logstd is None:
            nn.init.zeros_(self.out_logstds.weight)

    def forward(self, hidden_states, timestep):
        shape = hidden_states.shape
        bs = shape[0]
        extra_dims = shape[2:]

        t_emb = self.time_proj(timestep).to(hidden_states)
        pos_emb = self.pos_emb(hidden_states.reshape(bs, 2)).to(hidden_states)
        embeddings = torch.cat([t_emb, pos_emb], dim=-1)

        feat = self.net(embeddings)
        means = self.out_means(feat).reshape(bs, self.num_gaussians, 2, *extra_dims)
        logweights = self.out_logweights(feat).log_softmax(dim=-1).reshape(bs, self.num_gaussians, 1, *extra_dims)
        if self.constant_logstd is None:
            logstds = self.out_logstds(feat).reshape(bs, 1, 1, *extra_dims)
        else:
            logstds = torch.full(
                (bs, 1, 1, *extra_dims), float(self.constant_logstd),
                dtype=hidden_states.dtype, device=hidden_states.device)
        return GMFlowModelOutput(
            means=means,
            logweights=logweights,
            logstds=logstds)
