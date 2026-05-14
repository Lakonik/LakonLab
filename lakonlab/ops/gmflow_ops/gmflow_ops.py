# Copyright (c) 2026 Hansheng Chen

# A complete library for Gaussian mixture operations in PyTorch.
# Some of the functions are not used in GMFlow but are kept for future reference.

import math
import torch

from torch.autograd import Function
from torch.amp import custom_fwd

BACKEND = None


def get_backend():
    global BACKEND

    if BACKEND is None:
        try:
            import _gmflow_ops as _backend
        except ImportError:
            from .backend import _backend

        BACKEND = _backend

    return BACKEND


class _gm1d_inverse_cdf(Function):
    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, gm1d, scaled_cdfs, n_steps, eps, max_step_size, init_samples):
        means = gm1d['means']
        logstds = gm1d['logstds']
        logweights = gm1d['logweights']
        gm_weights = gm1d['gm_weights']

        batch_shapes = max(means.shape[:-3], scaled_cdfs.shape[:-3])
        num_gaussians, h, w = means.shape[-3:]
        batch_numel = batch_shapes.numel()
        n_samples = scaled_cdfs.size(-3)
        assert n_steps >= 0

        samples = init_samples.expand(*batch_shapes, n_samples, h, w).reshape(
            batch_numel, n_samples, h, w).contiguous()
        get_backend().gm1d_inverse_cdf(
            means.expand(*batch_shapes, num_gaussians, h, w).reshape(
                batch_numel, num_gaussians, h, w).contiguous(),
            logstds.expand(*batch_shapes, 1, 1, 1).reshape(
                batch_numel, 1, 1, 1).contiguous(),
            logweights.expand(*batch_shapes, num_gaussians, h, w).reshape(
                batch_numel, num_gaussians, h, w).contiguous(),
            gm_weights.expand(*batch_shapes, num_gaussians, h, w).reshape(
                batch_numel, num_gaussians, h, w).contiguous(),
            scaled_cdfs.expand(*batch_shapes, n_samples, h, w).reshape(
                batch_numel, n_samples, h, w).contiguous(),
            samples,
            n_steps, eps, max_step_size)

        return samples.reshape(*batch_shapes, n_samples, h, w)


def psd_inverse(x):
    return torch.cholesky_inverse(torch.linalg.cholesky(x))


def gm1d_pdf_cdf(gm1d, samples):
    """
    Args:
        gm1d (dict):
            means (torch.Tensor): (bs, *, num_gaussians, h, w)
            logstds (torch.Tensor): (bs, *, 1, 1, 1)
            logweights (torch.Tensor): (bs, *, num_gaussians, h, w)
            (Optional) gm_vars (torch.Tensor): (bs, *, 1, 1, 1)
            (Optional) gm_weights (torch.Tensor): (bs, *, num_gaussians, h, w)
        samples (torch.Tensor): (bs, *, n_samples, h, w)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (pdf, cdf) in shape (bs, *, n_samples, h, w)
            Note that the CDF is scaled to [-1, 1].
    """
    gm_logstds = gm1d['logstds'].unsqueeze(-4)  # (bs, *, 1, 1, 1, 1)
    gm_stds = gm_logstds.exp()  # (bs, *, 1, 1, 1, 1)
    gm_logweights = gm1d['logweights'].unsqueeze(-4)  # (bs, *, 1, num_gaussians, h, w)
    if 'gm_weights' in gm1d:
        gm_weights = gm1d['gm_weights'].unsqueeze(-4)  # (bs, *, 1, num_gaussians, h, w)
    else:
        gm_weights = gm_logweights.exp()  # (bs, *, 1, num_gaussians, h, w)
    # (bs, *, n_samples, num_gaussians, h, w)
    gm1d_norm_diffs = (samples.unsqueeze(-3) - gm1d['means'].unsqueeze(-4)) / gm_stds
    pdf = (-0.5 * gm1d_norm_diffs.square() - gm_logstds + gm_logweights).exp().sum(dim=-3) / math.sqrt(2 * math.pi)
    cdf = (gm_weights * torch.erf(gm1d_norm_diffs / math.sqrt(2))).sum(dim=-3)
    return pdf, cdf


@torch.jit.script
def gm1d_inverse_cdf_init_jit(gaussian_samples, var, mean):
    init_samples = gaussian_samples * var.sqrt() + mean
    return init_samples


def gm1d_inverse_cdf(
        gm1d, scaled_cdfs, n_steps=8, eps=1e-6, max_step_size=1.5, gaussian_samples=None, backward_steps=2):
    """
    Inverse CDF for 1D Gaussian mixture using Newton-Raphson method.

    Args:
        gm1d (dict):
            means (torch.Tensor): (bs, *, num_gaussians, h, w)
            logstds (torch.Tensor): (bs, *, 1, 1, 1)
            logweights (torch.Tensor): (bs, *, num_gaussians, h, w)
            (Optional) gm_vars (torch.Tensor): (bs, *, 1, 1, 1)
            (Optional) gm_weights (torch.Tensor): (bs, *, num_gaussians, h, w)
        scaled_cdfs (torch.Tensor): (bs, *, n_samples, h, w) scaled to range [-1, 1]
        gaussian_samples (torch.Tensor | None): (bs, *, n_samples, h, w) standard Gaussian samples with scaled_cdfs
        n_steps (int)
        eps (float)
        max_step_size (float)

    Returns:
        torch.Tensor: (bs, *, n_samples, h, w)
    """
    _gm1d = {k: v.unsqueeze(-3) for k, v in gm1d.items()}
    gaussian_proxy = gm_to_iso_gaussian(_gm1d)[0]
    gm1d['gm_weights'] = _gm1d['gm_weights'].squeeze(-3)
    gm1d['gm_vars'] = _gm1d['gm_vars'].squeeze(-3)

    with torch.no_grad():
        if gaussian_samples is None:
            gaussian_samples = (torch.erfinv(
                scaled_cdfs.float().clamp(min=-1 + eps, max=1 - eps)
            ) * math.sqrt(2)).to(dtype=scaled_cdfs.dtype)  # (bs, *, n_samples, h, w)
        # (bs, *, n_samples, h, w)
        init_samples = gm1d_inverse_cdf_init_jit(gaussian_samples, gaussian_proxy['var'], gaussian_proxy['mean'])
        samples = _gm1d_inverse_cdf.apply(
            gm1d, scaled_cdfs, n_steps - backward_steps, eps, max_step_size, init_samples)

    if backward_steps > 0:  # fallback to pytorch implementation, which supports autograd but is slower
        clamp_range = max_step_size * gm1d['logstds'].exp()
        for i in range(n_steps):
            cur_pdfs, cur_cdfs = gm1d_pdf_cdf(gm1d, samples)  # (bs, *, n_samples, h, w)
            delta = 0.5 * (cur_cdfs - scaled_cdfs) / cur_pdfs.clamp(min=eps)  # (bs, *, n_samples, h, w)
            delta = torch.maximum(torch.minimum(delta, clamp_range), -clamp_range)
            samples = samples - delta  # (bs, *, n_samples, h, w)
    return samples


@torch.jit.script
def gm_to_iso_gaussian_jit(gm_weights, gm_means, gm_vars):
    g_mean = (gm_weights * gm_means).sum(-4, keepdim=True)  # (bs, *, 1, out_channels, h, w)
    # (bs, *, num_gaussians, out_channels, h, w)
    gm_diffs = gm_means - g_mean
    g_var = (gm_weights * (gm_diffs * gm_diffs)).sum(-4, keepdim=True).mean(-3, keepdim=True) + gm_vars  # (bs, *, 1, 1, h, w)
    return g_mean, g_var, gm_diffs


def gm_to_iso_gaussian(gm):
    """
    Args:
        gm (dict):
            means (torch.Tensor): (bs, *, num_gaussians, out_channels, h, w)
            logstds (torch.Tensor): (bs, *, 1, 1, 1, 1)
            logweights (torch.Tensor): (bs, *, num_gaussians, 1, h, w)
            (Optional) gm_vars (torch.Tensor): (bs, *, 1, 1, 1, 1)
            (Optional) gm_weights (torch.Tensor): (bs, *, num_gaussians, 1, h, w)
            or
            means (torch.Tensor): (bs, *, num_gaussians, h, w, out_channels)
            covs (torch.Tensor): (bs, *, 1 or num_gaussians, h, w, out_channels, out_channels)
            logweights (torch.Tensor): (bs, *, num_gaussians, h, w)

    Returns:
        tuple[dict, torch.Tensor, torch.Tensor]:
            dict: Output gaussian
                mean (torch.Tensor): (bs, *, out_channels, h, w)
                var (torch.Tensor): (bs, *, 1, h, w)
            torch.Tensor: (bs, *, num_gaussians, out_channels, h, w)
            torch.Tensor: (bs, *, 1, 1, 1, 1) or (bs, *, 1, 1, h, w)
    """
    gm_means = gm['means']
    gm_logweights = gm['logweights']

    if 'covs' in gm:
        batch_shapes = gm_means.shape[:-4]
        num_gaussians, h, w, out_channels = gm_means.shape[-4:]
        batch_numel = batch_shapes.numel()

        gm_means = gm_means.reshape(
            batch_numel, num_gaussians, h, w, out_channels).permute(0, 1, 4, 2, 3).reshape(
            *batch_shapes, num_gaussians, out_channels, h, w)
        gm_weights = gm_logweights.exp().unsqueeze(-3)  # (bs, *, num_gaussians, 1, h, w)

        gm_vars = gm['covs'].diagonal(
            offset=0, dim1=-1, dim2=-2).mean(dim=-1).unsqueeze(-3)  # (bs, *, 1 or num_gaussians, 1, h, w)
        if gm_vars.size(-4) > 1:
            gm_vars = (gm_weights * gm_vars).sum(dim=-4, keepdim=True)  # (bs, *, 1, 1, h, w)

    else:
        if 'gm_weights' in gm:
            gm_weights = gm['gm_weights']
        else:
            gm_weights = gm_logweights.exp()  # (bs, *, num_gaussians, 1, h, w)
            gm['gm_weights'] = gm_weights

        gm_logstds = gm['logstds']  # (bs, *, 1, 1, 1, 1)
        if 'gm_vars' in gm:
            gm_vars = gm['gm_vars']
        else:
            gm_vars = (gm_logstds * 2).exp()  # (bs, *, 1, 1, 1, 1)
            gm['gm_vars'] = gm_vars

    g_mean, g_var, gm_diffs = gm_to_iso_gaussian_jit(gm_weights, gm_means, gm_vars)
    gaussian = dict(
        mean=g_mean.squeeze(-4),
        var=g_var.squeeze(-4))

    return gaussian, gm_diffs, gm_vars


@torch.jit.script
def gm_to_gaussian_jit(
        gm_means, gm_weights, gm_covs,
        batch_numel: int, num_gaussians: int, out_channels: int, h: int, w: int):
    g_mean = (gm_weights * gm_means).sum(-4, keepdim=True)  # (bs, *, 1, out_channels, h, w)
    # (batch_numel, num_gaussians, h, w, out_channels)
    gm_diffs = (gm_means - g_mean).reshape(batch_numel, num_gaussians, out_channels, h, w).permute(0, 1, 3, 4, 2)
    # (batch_numel, h, w, out_channels, out_channels) + (batch_numel, 1, 1, out_channels, out_channels)
    g_cov = (gm_weights.reshape(
        batch_numel, num_gaussians, h, w, 1, 1
    ) * gm_diffs.unsqueeze(-1) * gm_diffs.unsqueeze(-2)).sum(1) + gm_covs
    return g_mean, g_cov, gm_diffs


def gm_to_gaussian(gm, cov_scale=1.0):
    """
    Args:
        gm (dict):
            means (torch.Tensor): (bs, *, num_gaussians, out_channels, h, w)
            logstds (torch.Tensor): (bs, *, 1, 1, 1, 1)
            logweights (torch.Tensor): (bs, *, num_gaussians, 1, h, w)
            (Optional) gm_vars (torch.Tensor): (bs, *, 1, 1, 1, 1)
            (Optional) gm_weights (torch.Tensor): (bs, *, num_gaussians, 1, h, w)
            or
            means (torch.Tensor): (bs, *, num_gaussians, h, w, out_channels)
            covs (torch.Tensor): (bs, *, 1 or num_gaussians, h, w, out_channels, out_channels)
            logweights (torch.Tensor): (bs, *, num_gaussians, h, w)
        cov_scale (float): scale factor for the covariance matrix

    Returns:
        tuple[dict, torch.Tensor, torch.Tensor]:
            dict: Output gaussian
                mean (torch.Tensor): (bs, *, h, w, out_channels)
                cov (torch.Tensor): (bs, *, h, w, out_channels, out_channels)
            torch.Tensor: (bs, *, num_gaussians, h, w, out_channels)
            torch.Tensor: (bs, *, out_channels, out_channels)
    """
    gm_means = gm['means']
    gm_logweights = gm['logweights']

    if 'covs' in gm:
        batch_shapes = gm_means.shape[:-4]
        num_gaussians, h, w, out_channels = gm_means.shape[-4:]
        batch_numel = batch_shapes.numel()

        gm_means = gm_means.reshape(
            batch_numel, num_gaussians, h, w, out_channels).permute(0, 1, 4, 2, 3).reshape(
            *batch_shapes, num_gaussians, out_channels, h, w)
        gm_weights = gm_logweights.exp().unsqueeze(-3)  # (bs, *, num_gaussians, 1, h, w)

        gm_covs_num_gaussians = gm['covs'].size(-5)
        if gm_covs_num_gaussians == 1:
            gm_covs = gm['covs'].reshape(
                batch_numel, h, w, out_channels, out_channels)
        else:
            gm_covs = gm['covs'].reshape(
                batch_numel, gm['covs'].size(-5), h, w, out_channels, out_channels)
            gm_covs = (gm_weights.reshape(batch_numel, num_gaussians, h, w, 1, 1) * gm_covs).sum(dim=1)

    else:
        if 'gm_weights' in gm:
            gm_weights = gm['gm_weights']
        else:
            gm_weights = gm_logweights.exp()  # (bs, *, num_gaussians, 1, h, w)
            gm['gm_weights'] = gm_weights

        gm_logstds = gm['logstds']  # (bs, *, 1, 1, 1, 1)
        if 'gm_vars' in gm:
            gm_vars = gm['gm_vars']
        else:
            gm_vars = (gm_logstds * 2).exp()  # (bs, *, 1, 1, 1, 1)
            gm['gm_vars'] = gm_vars

        batch_shapes = gm_means.shape[:-4]
        num_gaussians, out_channels, h, w = gm_means.shape[-4:]
        batch_numel = batch_shapes.numel()
        dtype = gm_means.dtype
        device = gm_means.device

        gm_covs = (torch.eye(out_channels, dtype=dtype, device=device) * gm_vars.reshape(
            batch_numel, 1, 1, 1, 1))  # (batch_numel, 1, 1, out_channels, out_channels)

    g_mean, g_cov, gm_diffs = gm_to_gaussian_jit(
        gm_means, gm_weights, gm_covs, batch_numel, num_gaussians, out_channels, h, w)
    gaussian = dict(
        mean=g_mean.reshape(batch_numel, out_channels, h, w).permute(
            0, 2, 3, 1).reshape(*batch_shapes, h, w, out_channels),
        cov=g_cov.reshape(*batch_shapes, h, w, out_channels, out_channels) * cov_scale)

    return (gaussian,
            gm_diffs.reshape(*batch_shapes, num_gaussians, h, w, out_channels),
            gm_covs.reshape(*batch_shapes, 1, 1, 1, out_channels, out_channels))


def gm_mul_gaussian(gm, gaussian, gm_power, gaussian_power, gm_diffs=None, gm_covs=None):
    """
    Args:
        gm (dict):
            means (torch.Tensor): (bs, *, num_gaussians, out_channels, h, w)
            logstds (torch.Tensor): (bs, *, 1, 1, 1, 1)
            logweights (torch.Tensor): (bs, *, num_gaussians, 1, h, w)
            (Optional) gm_vars (torch.Tensor): (bs, *, 1, 1, 1, 1)
        gaussian (dict):
            mean (torch.Tensor): (bs, *, h, w, out_channels)
            cov (torch.Tensor): (bs, *, h, w, out_channels, out_channels)
        gm_power (float): power for the Gaussian mixture
        gaussian_power (float): power for the Gaussian
        gm_diffs (torch.Tensor | None): (bs, *, num_gaussians, h, w, out_channels)
        gm_covs (torch.Tensor | None): (bs, *, 1, 1, 1, out_channels, out_channels)

    Returns:
        tuple[dict, float]:
            dict: Output gausian mixture
                means (torch.Tensor): (bs, *, num_gaussians, h, w, out_channels)
                covs (torch.Tensor): (bs, *, num_gaussians, h, w, out_channels, out_channels)
                logweights (torch.Tensor): (bs, *, num_gaussians, h, w)
            float: Output gaussian mixture power
    """
    gm_means = gm['means']  # (bs, *, num_gaussians, out_channels, h, w)
    gm_logstds = gm['logstds']  # (bs, *, 1, 1, 1, 1)
    gm_logweights = gm['logweights']  # (bs, *, num_gaussians, 1, h, w)
    if 'gm_vars' in gm:
        gm_vars = gm['gm_vars']
    else:
        gm_vars = (gm_logstds * 2).exp()  # (bs, *, 1, 1, 1, 1)
        gm['gm_vars'] = gm_vars

    batch_shapes = gm_means.shape[:-4]
    num_gaussians, out_channels, h, w = gm_means.shape[-4:]
    batch_numel = batch_shapes.numel()
    dtype = gm_means.dtype
    device = gm_means.device
    eye = torch.eye(out_channels, dtype=dtype, device=device)

    gm_means = gm_means.reshape(batch_numel, num_gaussians, out_channels, h, w).permute(
        0, 1, 3, 4, 2)  # (batch_numel, num_gaussians, h, w, out_channels)
    gm_vars = gm_vars.reshape(batch_numel, 1, 1, 1, 1)
    g_mean = gaussian['mean'].reshape(batch_numel, h, w, out_channels)
    g_cov = gaussian['cov'].reshape(batch_numel, h, w, out_channels, out_channels)

    if gm_diffs is None:
        gm_diffs = (gm_means - g_mean.unsqueeze(1))  # (batch_numel, num_gaussians, h, w, out_channels)
    else:
        gm_diffs = gm_diffs.reshape(batch_numel, num_gaussians, h, w, out_channels)
    if gm_covs is None:
        gm_covs = eye * gm_vars.unsqueeze(-1)  # (batch_numel, 1, 1, 1, out_channels, out_channels)
    else:
        gm_covs = gm_covs.reshape(batch_numel, 1, 1, 1, out_channels, out_channels)

    gm_weight = eye / gm_vars  # (batch_numel, 1, 1, out_channels, out_channels)
    g_weight = (gaussian_power / gm_power) * psd_inverse(g_cov)  # (batch_numel, h, w, out_channels, out_channels)
    out_covs = psd_inverse(gm_weight + g_weight)  # (batch_numel, h, w, out_channels, out_channels)
    out_means = (out_covs.unsqueeze(1) @ (
        # (batch_numel, num_gaussians, h, w, out_channels, 1)
        (gm_weight[..., :1, :1] * gm_means).unsqueeze(-1)
        # (batch_numel, 1, h, w, out_channels, 1)
        + (g_weight @ g_mean.unsqueeze(-1)).unsqueeze(1)
    )).squeeze(-1)  # (batch_numel, num_gaussians, h, w, out_channels)
    # (batch_numel, num_gaussians, h, w, 1, out_channels) @ (batch_numel, 1, h, w, out_channels, out_channels)
    # @ (batch_numel, num_gaussians, h, w, out_channels, 1) -> (batch_numel, num_gaussians, h, w, 1)
    logweights_delta = (
        gm_diffs.unsqueeze(-2)
        @ psd_inverse(gm_covs * gaussian_power + g_cov.unsqueeze(1) * gm_power)
        @ gm_diffs.unsqueeze(-1)
    ).squeeze(-1) * (-0.5 * gaussian_power)

    # (bs, *, num_gaussians, h, w)
    out_logweights = gm_logweights.squeeze(-3) + logweights_delta.reshape(*batch_shapes, num_gaussians, h, w)
    out_logweights = out_logweights.log_softmax(dim=-3)
    out_means = out_means.reshape(*batch_shapes, num_gaussians, h, w, out_channels)
    out_covs = out_covs.reshape(*batch_shapes, 1, h, w, out_channels, out_channels)

    return dict(means=out_means, covs=out_covs, logweights=out_logweights), gm_power


@torch.jit.script
def gm_mul_iso_gaussian_jit(
        gm_means, gm_vars, gm_logstds, gm_logweights, g_mean, g_var, g_logstd,
        gm_power: float, gaussian_power: float, eps: float):
    gm_diffs = gm_means - g_mean  # (bs, *, num_gaussians, out_channels, h, w)

    power_ratio = gaussian_power / gm_power
    norm_factor = (g_var + power_ratio * gm_vars).clamp(min=eps)

    out_means = (g_var * gm_means + power_ratio * gm_vars * g_mean) / norm_factor
    # (bs, *, num_gaussians, 1, h, w)
    logweights_delta = gm_diffs.square().sum(
        dim=-3, keepdim=True) * (-0.5 * power_ratio / norm_factor)

    out_logweights = (gm_logweights + logweights_delta).log_softmax(dim=-4)
    out_logstds = gm_logstds + g_logstd - 0.5 * torch.log(norm_factor)

    return out_means, out_logstds, out_logweights


def gm_mul_iso_gaussian(gm, gaussian, gm_power, gaussian_power, eps=1e-6):
    """
    Args:
        gm (dict):
            means (torch.Tensor): (bs, *, num_gaussians, out_channels, h, w)
            logstds (torch.Tensor): (bs, *, 1, 1, 1, 1)
            logweights (torch.Tensor): (bs, *, num_gaussians, 1, h, w)
            (Optional) gm_vars (torch.Tensor): (bs, *, 1, 1, 1, 1)
        gaussian (dict):
            mean (torch.Tensor): (bs, *, out_channels, h, w)
            var (torch.Tensor): (bs, *, 1, 1, 1)
            (Optional) logstd (torch.Tensor): (bs, *, 1, 1, 1)
        gm_power (float): power for the Gaussian mixture
        gaussian_power (float): power for the Gaussian

    Returns:
        tuple[dict, float]:
            dict: Output gausian mixture
                means (torch.Tensor): (bs, *, num_gaussians, out_channels, h, w)
                logstds (torch.Tensor): (bs, *, 1, 1, 1, 1)
                logweights (torch.Tensor): (bs, *, num_gaussians, 1, h, w)
            float: Output gaussian mixture power
    """
    gm_means = gm['means']  # (bs, *, num_gaussians, out_channels, h, w)
    gm_logstds = gm['logstds']  # (bs, *, 1, 1, 1, 1)
    gm_logweights = gm['logweights']  # (bs, *, num_gaussians, 1, h, w)
    if 'gm_vars' in gm:
        gm_vars = gm['gm_vars']
    else:
        gm_vars = (gm_logstds * 2).exp()  # (bs, *, 1, 1, 1, 1)
        gm['gm_vars'] = gm_vars
    if 'logstd' not in gaussian:
        gaussian['logstd'] = torch.log(gaussian['var']) / 2
    g_mean = gaussian['mean'].unsqueeze(-4)
    g_var = gaussian['var'].unsqueeze(-4)
    g_logstd = gaussian['logstd'].unsqueeze(-4)

    out_means, out_logstds, out_logweights = gm_mul_iso_gaussian_jit(
        gm_means, gm_vars, gm_logstds, gm_logweights, g_mean, g_var, g_logstd, gm_power, gaussian_power, eps)
    return dict(means=out_means, logstds=out_logstds, logweights=out_logweights), gm_power


@torch.jit.script
def gm_mul_gm_jit(gm1_means, gm1_logstds, gm1_vars, gm1_logweights, gm2_means, gm2_logstds, gm2_vars, gm2_logweights):
    gm1_means = gm1_means.unsqueeze(-4)  # (bs, *, num_gaussians_1, 1, out_channels, h, w)
    gm1_vars = gm1_vars.unsqueeze(-4)  # (bs, *, 1, 1, 1, 1, 1)
    gm1_logweights = gm1_logweights.unsqueeze(-4)  # (bs, *, num_gaussians_1, 1, 1, h, w)

    gm2_means = gm2_means.unsqueeze(-5)  # (bs, *, 1, num_gaussians_2, out_channels, h, w)
    gm2_vars = gm2_vars.unsqueeze(-5)  # (bs, *, 1, 1, 1, 1, 1)
    gm2_logweights = gm2_logweights.unsqueeze(-5)  # (bs, *, 1, num_gaussians_2, 1, h, w)

    # (bs, *, num_gaussians_1, num_gaussians_2, out_channels, h, w)
    gm_diffs = gm1_means - gm2_means
    norm_factor = gm1_vars + gm2_vars  # (bs, *, 1, 1, 1, 1, 1)

    # (bs, *, num_gaussians_1, num_gaussians_2, out_channels, h, w)
    out_means = (gm2_vars * gm1_means + gm1_vars * gm2_means) / norm_factor
    out_means = out_means.flatten(-5, -4)  # (bs, *, num_gaussians_1 * num_gaussians_2, out_channels, h, w)

    # (bs, *, num_gaussians_1, num_gaussians_2, 1, h, w)
    logweights_delta = gm_diffs.square().sum(dim=-3, keepdim=True) * (-0.5 / norm_factor)
    # (bs, *, num_gaussians_1, num_gaussians_2, 1, h, w)
    out_logweights = (gm1_logweights + gm2_logweights + logweights_delta)
    # (bs, *, num_gaussians_1 * num_gaussians_2, 1, h, w)
    out_logweights = out_logweights.flatten(-5, -4).log_softmax(dim=-4)

    out_logstds = gm1_logstds + gm2_logstds - 0.5 * torch.logaddexp(gm1_logstds * 2, gm2_logstds * 2)

    return out_means, out_logstds, out_logweights


def gm_mul_gm(gm1, gm2):
    """
    Args:
        gm1 (dict):
            means (torch.Tensor): (bs, *, num_gaussians_1, out_channels, h, w)
            logstds (torch.Tensor): (bs, *, 1, 1, 1, 1)
            logweights (torch.Tensor): (bs, *, num_gaussians_1, 1, h, w)
            (Optional) gm_vars (torch.Tensor): (bs, *, 1, 1, 1, 1)
        gm2 (dict):
            means (torch.Tensor): (bs, *, num_gaussians_2, out_channels, h, w)
            logstds (torch.Tensor): (bs, *, 1, 1, 1, 1)
            logweights (torch.Tensor): (bs, *, num_gaussians_2, 1, h, w)
            (Optional) gm_vars (torch.Tensor): (bs, *, 1, 1, 1, 1)

    Returns:
        dict: Output gausian mixture
            means (torch.Tensor): (bs, *, num_gaussians_1 * num_gaussians_2, out_channels, h, w)
            logstds (torch.Tensor): (bs, *, 1, 1, 1, 1)
            logweights (torch.Tensor): (bs, *, num_gaussians_1 * num_gaussians_2, 1, h, w)
    """
    gm1_means = gm1['means']  # (bs, *, num_gaussians_1, out_channels, h, w)
    gm1_logstds = gm1['logstds']  # (bs, *, 1, 1, 1, 1)
    gm1_logweights = gm1['logweights']  # (bs, *, num_gaussians_1, 1, h, w)
    if 'gm_vars' in gm1:
        gm1_vars = gm1['gm_vars']
    else:
        gm1_vars = (gm1_logstds * 2).exp()  # (bs, *, 1, 1, 1, 1)
        gm1['gm_vars'] = gm1_vars

    gm2_means = gm2['means']  # (bs, *, num_gaussians_2, out_channels, h, w)
    gm2_logstds = gm2['logstds']  # (bs, *, 1, 1, 1, 1)
    gm2_logweights = gm2['logweights']  # (bs, *, num_gaussians_2, 1, h, w)
    if 'gm_vars' in gm2:
        gm2_vars = gm2['gm_vars']
    else:
        gm2_vars = (gm2_logstds * 2).exp()  # (bs, *, 1, 1, 1, 1)
        gm2['gm_vars'] = gm2_vars

    out_means, out_logstds, out_logweights = gm_mul_gm_jit(
        gm1_means, gm1_logstds, gm1_vars, gm1_logweights, gm2_means, gm2_logstds, gm2_vars, gm2_logweights)
    return dict(means=out_means, logstds=out_logstds, logweights=out_logweights)


@torch.jit.script
def gm_to_mean_jit(gm_means, gm_logweights, gm_power: float):
    mean = ((gm_logweights * gm_power).softmax(dim=-4) * gm_means).sum(dim=-4)
    return mean


def gm_to_mean(gm, gm_power=1):
    """
    Args:
        gm (dict):
            means (torch.Tensor): (bs, *, num_gaussians, out_channels, h, w)
            logstds (torch.Tensor): (bs, *, 1, 1, 1, 1)
            logweights (torch.Tensor): (bs, *, num_gaussians, 1, h, w)
            or
            means (torch.Tensor): (bs, *, num_gaussians, h, w, out_channels)
            covs (torch.Tensor): (bs, *, 1 or num_gaussians, h, w, out_channels, out_channels)
            logweights (torch.Tensor): (bs, *, num_gaussians, h, w)
        gm_power (float)

    Returns:
        torch.Tensor: (bs, *, out_channels, h, w)
    """
    gm_means = gm['means']
    gm_logweights = gm['logweights']
    if 'covs' in gm:
        batch_shapes = gm_means.shape[:-4]
        num_gaussians, h, w, out_channels = gm_means.shape[-4:]
        batch_numel = batch_shapes.numel()
        gm_means = gm_means.reshape(
            batch_numel, num_gaussians, h, w, out_channels
        ).permute(0, 1, 4, 2, 3).reshape(*batch_shapes, num_gaussians, out_channels, h, w)
        gm_logweights = gm_logweights.unsqueeze(-3)
    return gm_to_mean_jit(gm_means, gm_logweights, gm_power)


def gm_to_sample(
        gm,
        gm_power=1,
        n_samples=1,
        cov_sharpen=False):
    """
    Args:
        gm (dict):
            means (torch.Tensor): (bs, *, num_gaussians, out_channels, h, w)
            logstds (torch.Tensor): (bs, *, 1, 1, 1, 1) or (bs, *, num_gaussians, 1, h, w)
            logweights (torch.Tensor): (bs, *, num_gaussians, 1, h, w)
            or
            means (torch.Tensor): (bs, *, num_gaussians, h, w, out_channels)
            covs (torch.Tensor): (bs, *, 1 or num_gaussians, h, w, out_channels, out_channels)
            logweights (torch.Tensor): (bs, *, num_gaussians, h, w)
        gm_power (float): power for the Gaussian mixture, samples are approximated if power is not 1
        n_samples (int): number of samples
        cov_sharpen (bool): whether to sharpen the covariance matrix when power is greater than 1

    Returns:
        torch.Tensor: (bs, *, n_samples, out_channels, h, w)
    """
    gm_means = gm['means']
    gm_logweights = gm['logweights']

    if 'covs' in gm:
        gm_covs = gm['covs']

        batch_shapes = gm_means.shape[:-4]
        num_gaussians, h, w, out_channels = gm_means.shape[-4:]
        batch_numel = batch_shapes.numel()

        inds = torch.multinomial(
            (gm_logweights.reshape(batch_numel, num_gaussians, h, w).permute(0, 2, 3, 1).reshape(
                batch_numel * h * w, num_gaussians) * gm_power).softmax(dim=-1),
            n_samples, replacement=True
        ).reshape(batch_numel, h, w, n_samples).permute(0, 3, 1, 2).reshape(*batch_shapes, n_samples, 1, h, w)

        means = gm_means.gather(  # (bs, *, n_samples, h, w, out_channels)
            dim=-4,
            index=inds.reshape(*batch_shapes, n_samples, h, w, 1).expand(
                *batch_shapes, n_samples, h, w, out_channels))
        if gm_covs.size(-5) == 1:
            tril = torch.linalg.cholesky(gm_covs)  # (bs, *, 1, h, w, out_channels, out_channels)
        elif n_samples < num_gaussians:
            covs = gm_covs.gather(  # (bs, *, n_samples, h, w, out_channels, out_channels)
                dim=-5,
                index=inds.reshape(*batch_shapes, n_samples, h, w, 1, 1).expand(
                    *batch_shapes, n_samples, h, w, out_channels, out_channels))
            if cov_sharpen:
                covs = covs / gm_power
            tril = torch.linalg.cholesky(covs)
        else:
            tril = torch.linalg.cholesky(gm_covs)
            if cov_sharpen:
                tril = tril / math.sqrt(gm_power)
            tril = tril.gather(  # (bs, *, n_samples, h, w, out_channels, out_channels)
                dim=-5,
                index=inds.reshape(*batch_shapes, n_samples, h, w, 1, 1).expand(
                    *batch_shapes, n_samples, h, w, out_channels, out_channels))

        # (bs, *, n_samples, h, w, out_channels)
        samples = (tril @ torch.randn(
            (*batch_shapes, n_samples, h, w, out_channels, 1),
            dtype=means.dtype, device=means.device)).squeeze(-1) + means
        samples = samples.reshape(batch_numel, n_samples, h, w, out_channels).permute(0, 1, 4, 2, 3).reshape(
            *batch_shapes, n_samples, out_channels, h, w)

    else:
        batch_shapes = gm_means.shape[:-4]
        num_gaussians, out_channels, h, w = gm_means.shape[-4:]
        batch_numel = batch_shapes.numel()

        inds = torch.multinomial(
            (gm_logweights.reshape(batch_numel, num_gaussians, h, w).permute(0, 2, 3, 1).reshape(
                batch_numel * h * w, num_gaussians) * gm_power).softmax(dim=-1),
            n_samples, replacement=True
        ).reshape(batch_numel, h, w, n_samples).permute(0, 3, 1, 2).reshape(*batch_shapes, n_samples, 1, h, w)

        means = gm_means.gather(  # (bs, *, n_samples, out_channels, h, w)
            dim=-4,
            index=inds.expand(*batch_shapes, n_samples, out_channels, h, w))
        stds = gm['logstds'].exp()  # (bs, *, 1, 1, 1, 1) or (bs, *, num_gaussians, 1, h, w)
        if cov_sharpen:
            stds = stds / math.sqrt(gm_power)
        if stds.size(-4) == num_gaussians and num_gaussians > 1:
            stds = stds.gather(dim=-4, index=inds)  # (bs, *, n_samples, 1, h, w)

        # (bs, *, n_samples, out_channels, h, w)
        samples = stds * torch.randn(
            (*batch_shapes, n_samples, out_channels, h, w),
            dtype=means.dtype, device=means.device) + means

    return samples


def gaussian_mul_gaussian(gaussian_1, gaussian_2, gaussian_1_power, gaussian_2_power):
    """
    Args:
        gaussian_1 (dict):
            mean (torch.Tensor): (bs, *, h, w, out_channels)
            cov (torch.Tensor): (bs, *, h, w, out_channels, out_channels)
        gaussian_2 (dict):
            mean (torch.Tensor): (bs, *, h, w, out_channels)
            cov (torch.Tensor): (bs, *, h, w, out_channels, out_channels)

    Returns:
        dict:
            mean (torch.Tensor): (bs, *, h, w, out_channels)
            cov (torch.Tensor): (bs, *, h, w, out_channels, out_channels)
    """
    g1_mean = gaussian_1['mean']
    g1_cov = gaussian_1['cov']
    g2_mean = gaussian_2['mean']
    g2_cov = gaussian_2['cov']

    g1_cov_inv = gaussian_1_power * psd_inverse(g1_cov)
    g2_cov_inv = gaussian_2_power * psd_inverse(g2_cov)
    out_cov = psd_inverse(g1_cov_inv + g2_cov_inv)
    out_mean = out_cov @ (g1_cov_inv @ g1_mean.unsqueeze(-1) + g2_cov_inv @ g2_mean.unsqueeze(-1))
    out_mean = out_mean.squeeze(-1)
    return dict(mean=out_mean, cov=out_cov)


@torch.jit.script
def iso_gaussian_mul_iso_gaussian_jit(
        g1_mean, g1_var, g2_mean, g2_var,
        gaussian_1_power: float, gaussian_2_power: float, eps: float):
    norm_factor = (gaussian_1_power * g2_var + gaussian_2_power * g1_var).clamp(min=eps)
    out_var = g2_var * g1_var / norm_factor
    out_mean = (gaussian_1_power * g2_var * g1_mean + gaussian_2_power * g1_var * g2_mean) / norm_factor
    return out_mean, out_var


def iso_gaussian_mul_iso_gaussian(gaussian_1, gaussian_2, gaussian_1_power, gaussian_2_power, eps=1e-6):
    """
    Args:
        gaussian_1 (dict):
            mean (torch.Tensor): (bs, *, out_channels, h, w)
            var (torch.Tensor): (bs, *, 1, h, w)
        gaussian_2 (dict):
            mean (torch.Tensor): (bs, *, out_channels, h, w)
            var (torch.Tensor): (bs, *, 1, h, w)

    Returns:
        dict:
            mean (torch.Tensor): (bs, *, out_channels, h, w)
            var (torch.Tensor): (bs, *, 1, h, w)
    """
    out_mean, out_var = iso_gaussian_mul_iso_gaussian_jit(
        gaussian_1['mean'],
        gaussian_1['var'],
        gaussian_2['mean'],
        gaussian_2['var'],
        gaussian_1_power, gaussian_2_power, eps)
    return dict(mean=out_mean, var=out_var)


def iso_gaussian_logprob(iso_gaussian, samples):
    """
    Args:
        iso_gaussian (dict):
            mean (torch.Tensor): (bs, *, out_channels, h, w)
            var (torch.Tensor): (bs, *, 1, h, w)
        samples (torch.Tensor): (bs, *, n_samples, out_channels, h, w)

    Returns:
        torch.Tensor: (bs, *, n_samples, h, w)
    """
    mean = iso_gaussian['mean'].unsqueeze(-4)  # (bs, *, 1, out_channels, h, w)
    var = iso_gaussian['var']  # (bs, *, 1, h, w)
    out_channels = mean.size(-3)
    const = -0.5 * out_channels * math.log(2 * math.pi)
    logprob = -0.5 * (samples - mean).square().sum(dim=-3) / var - 0.5 * out_channels * var.log() + const
    return logprob


@torch.jit.script
def gm_iso_gaussian_logprobs_jit(
        means, logstds, samples, out_channels: int, const: float):
    logstds = logstds.unsqueeze(-5)
    inverse_std = torch.exp(-logstds)
    # (bs, *, n_samples, num_gaussians, out_channels, h, w)
    diff_weighted = (samples.unsqueeze(-4) - means.unsqueeze(-5)) * inverse_std
    # (bs, *, n_samples, num_gaussians, h, w)
    gaussian_logprobs = -0.5 * (diff_weighted.square().sum(dim=-3)) - out_channels * logstds.squeeze(-3) + const
    return gaussian_logprobs


def gm_logprob(gm, samples):
    """
    Args:
        gm (dict):
            means (torch.Tensor): (bs, *, num_gaussians, out_channels, h, w)
            logstds (torch.Tensor): (bs, *, 1, 1, 1, 1) or (bs, *, num_gaussians, 1, h, w)
            logweights (torch.Tensor): (bs, *, num_gaussians, 1, h, w)
            or
            means (torch.Tensor): (bs, *, num_gaussians, h, w, out_channels)
            covs (torch.Tensor): (bs, *, 1 or num_gaussians, h, w, out_channels, out_channels)
            logweights (torch.Tensor): (bs, *, num_gaussians, h, w)
            (Optional) invcov_trils (torch.Tensor): (bs, *, 1 or num_gaussians, h, w, out_channels, out_channels)
                lower triangular Cholesky decomposition of the inverse covariance matrix
            (Optional) logdets (torch.Tensor): (bs, *, 1 or num_gaussians, h, w)
                log-determinant of the covariance matrix
        samples (torch.Tensor): (bs, *, n_samples, out_channels, h, w)

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            torch.Tensor: (bs, *, n_samples, h, w)
            torch.Tensor: (bs, *, n_samples, num_gaussians, h, w)
    """
    n_samples = samples.size(-4)

    if 'covs' in gm:
        means = gm['means']
        batch_shapes = means.shape[:-4]
        num_gaussians, h, w, out_channels = means.shape[-4:]
        batch_numel = batch_shapes.numel()

        const = -0.5 * out_channels * math.log(2 * math.pi)

        covs = gm['covs']
        if 'invcov_trils' in gm:
            invcov_trils = gm['invcov_trils']
        else:
            invcov_trils = torch.linalg.cholesky(psd_inverse(covs))
            gm['invcov_trils'] = invcov_trils
        if 'logdets' in gm:
            logdets = gm['logdets']
        else:
            logdets = torch.logdet(covs)
            gm['logdets'] = logdets

        samples = samples.reshape(
            batch_numel, n_samples, out_channels, h, w
        ).permute(0, 1, 3, 4, 2).reshape(batch_numel, n_samples, 1, h, w, out_channels)
        # (batch_numel, n_samples, num_gaussians, h, w, out_channels)
        diffs = samples - means.reshape(batch_numel, 1, num_gaussians, h, w, out_channels)
        diff_weighted = (diffs.unsqueeze(-2) @ invcov_trils.reshape(
            batch_numel, 1, invcov_trils.size(-5), h, w, out_channels, out_channels)).squeeze(-2)
        # (batch_numel, n_samples, num_gaussians, h, w)
        gaussian_logprobs = (-0.5 * (diff_weighted.square().sum(dim=-1) + logdets.unsqueeze(-4)) + const).reshape(
            *batch_shapes, n_samples, num_gaussians, h, w)

    else:
        batch_shapes = gm['means'].shape[:-4]
        num_gaussians, out_channels, h, w = gm['means'].shape[-4:]
        const = -0.5 * out_channels * math.log(2 * math.pi)
        gaussian_logprobs = gm_iso_gaussian_logprobs_jit(
            gm['means'], gm['logstds'], samples, out_channels, const)

    # (bs, *, n_samples, h, w)
    logprob = (gm['logweights'].reshape(
        *batch_shapes, 1, num_gaussians, h, w) + gaussian_logprobs).logsumexp(dim=-3)

    return logprob, gaussian_logprobs


def gm_spectral_logprobs(
        gm, samples, power_spectrum=None, spectral_samples=None, n_axes=None, eps=1e-6, axis_aligned=True):
    """
    Args:
        gm (dict):
            means (torch.Tensor): (bs, *, num_gaussians, out_channels, h, w)
            logstds (torch.Tensor): (bs, *, 1, 1, 1, 1) or (bs, *, num_gaussians, 1, h, w)
            logweights (torch.Tensor): (bs, *, num_gaussians, 1, h, w)
            or
            means (torch.Tensor): (bs, *, num_gaussians, h, w, out_channels)
            covs (torch.Tensor): (bs, *, 1 or num_gaussians, h, w, out_channels, out_channels)
            logweights (torch.Tensor): (bs, *, num_gaussians, h, w)
            (Optional) invcov_trils (torch.Tensor): (bs, *, 1 or num_gaussians, h, w, out_channels, out_channels)
                lower triangular Cholesky decomposition of the inverse covariance matrix
            (Optional) logdets (torch.Tensor): (bs, *, 1 or num_gaussians, h, w)
                log-determinant of the covariance matrix
        samples (torch.Tensor): (bs, *, n_samples, out_channels, h, w)
        power_spectrum (torch.Tensor | None): (bs, *, out_channels, h, w)

    Returns:
        torch.Tensor: (bs, *, n_samples)
    """
    logprobs = gm_logprob(gm, samples)[0].sum(dim=(-2, -1))
    if power_spectrum is not None:
        if spectral_samples is None:
            z_kr = gm_samples_to_gaussian_samples(
                gm, samples, n_axes=n_axes, eps=eps, axis_aligned=axis_aligned)
            z_kr_fft = torch.fft.fft2(z_kr, norm='ortho')
            spectral_samples = z_kr_fft.real + z_kr_fft.imag
        out_channels = spectral_samples.size(-3)
        spectral_logprob_diff = -0.5 * spectral_samples.square().sum(
            dim=-3) * (torch.exp(-power_spectrum) - 1) - 0.5 * out_channels * power_spectrum
        logprobs = logprobs + spectral_logprob_diff.sum(dim=(-2, -1))
    return logprobs


def gm_kl_div(gm_p, gm_q, n_samples=32, use_kr=False, kr_backward_steps=1):
    """
    Args:
        gm_p (dict):
            means (torch.Tensor): (bs, *, num_gaussians, out_channels, h, w)
            logstds (torch.Tensor): (bs, *, 1, 1, 1, 1)
            logweights (torch.Tensor): (bs, *, num_gaussians, 1, h, w)
            (Optional) gm_weights (torch.Tensor): (bs, *, num_gaussians, 1, h, w)
        gm_q (dict):
            Same as gm_p
        n_samples (int): number of samples

    Returns:
        torch.Tensor: (bs, *, 1, h, w)
    """
    if use_kr:
        sample_size = list(gm_p['means'].shape)
        sample_size[-4] = n_samples
        gaussian_samples = torch.randn(sample_size, device=gm_p['means'].device, dtype=gm_p['means'].dtype)
        samples = gaussian_samples_to_gm_samples(
            gm_p, gaussian_samples, axis_aligned=True, backward_steps=kr_backward_steps)
    else:
        samples = gm_to_sample(gm_p, 1.0, n_samples=n_samples)
    return (gm_logprob(gm_p, samples)[0] - gm_logprob(gm_q, samples)[0]).mean(dim=-3, keepdim=True)


def gm_entropy(gm, n_samples=32):
    samples = gm_to_sample(gm, 1.0, n_samples=n_samples)
    return -gm_logprob(gm, samples)[0].mean(dim=-3, keepdim=True)


def gm_samples_to_gaussian_samples(gm, gm_samples, n_axes=None, eps=1e-6, axis_aligned=True):
    """
    Knothe-Rosenblatt transport for transforming samples from a Gaussian mixture to samples from a standard Gaussian.

    Args:
        gm (dict):
            means (torch.Tensor): (bs, *, num_gaussians, out_channels, h, w)
            logstds (torch.Tensor): (bs, *, 1, 1, 1, 1)
            logweights (torch.Tensor): (bs, *, num_gaussians, 1, h, w)
            (Optional) gm_weights (torch.Tensor): (bs, *, num_gaussians, 1, h, w)
        gm_samples (torch.Tensor): (bs, *, n_samples, out_channels, h, w)

    Returns:
        torch.Tensor: (bs, *, n_samples, out_channels, h, w)
    """
    assert 'covs' not in gm
    dtype = gm_samples.dtype
    gm_means = gm['means']
    batch_shapes = gm_means.shape[:-4]
    num_gaussians, out_channels, h, w = gm_means.shape[-4:]
    batch_numel = batch_shapes.numel()
    n_samples = gm_samples.size(-4)
    if n_axes is None:
        n_axes = out_channels

    with torch.no_grad():
        gaussians = gm_to_gaussian(gm)[0]
        covs = gaussians['cov']
        if axis_aligned:
            covs = covs.mean(dim=(-4, -3), keepdim=True)
        # (bs, *, h, w, out_channels, out_channels) or (bs, *, 1, 1, out_channels, out_channels)
        _, eigvecs = torch.linalg.eigh(covs.float())
        # eigenvalues in descending order
        eigvecs = eigvecs.flip(-1).to(dtype=dtype)

    eigvecs = eigvecs.reshape(
        (batch_numel, 1, 1, out_channels, out_channels) if axis_aligned
        else (batch_numel, h, w, out_channels, out_channels))
    gm_means = gm_means.reshape(
        batch_numel, num_gaussians, out_channels, h, w
    ).permute(0, 3, 4, 1, 2)  # (batch_numel, h, w, num_gaussians, out_channels)
    gm_logstds = gm['logstds'].reshape(batch_numel, 1, 1, 1, 1)
    gm_logweights = gm['logweights'].reshape(
        batch_numel, num_gaussians, 1, h, w
    ).permute(0, 3, 4, 1, 2)  # (batch_numel, h, w, num_gaussians, 1)
    gm_stds = gm_logstds.exp()  # (batch_numel, 1, 1, 1, 1)
    gm_samples = gm_samples.reshape(
        batch_numel, n_samples, out_channels, h, w
    ).permute(0, 3, 4, 1, 2)  # (batch_numel, h, w, n_samples, out_channels)

    eigvecs_ = eigvecs[..., :n_axes]
    gm_means_rot = gm_means @ eigvecs_  # (batch_numel, h, w, num_gaussians, n_axes)
    gm_samples_rot = gm_samples @ eigvecs_  # (batch_numel, h, w, n_samples, n_axes)

    # (batch_numel, h, w, n_samples, num_gaussians, n_axes)
    gm_norm_diffs = (gm_samples_rot.unsqueeze(-2)
                     - gm_means_rot.unsqueeze(-3)
                     ) / gm_stds.unsqueeze(-3)
    gm_norm_diffs_sq = gm_norm_diffs.square()

    # (batch_numel, h, w, n_samples, num_gaussians, n_axes - 1)
    gm_norm_diffs_sq_cumsumprev = torch.cumsum(gm_norm_diffs_sq[..., :-1], dim=-1)
    gm_slice_logweights = gm_logweights.unsqueeze(-3) - 0.5 * gm_norm_diffs_sq_cumsumprev
    gm_slice_weights = gm_slice_logweights.softmax(dim=-2)

    if 'gm_weights' in gm:
        gm_weights = gm['gm_weights'].reshape(
            batch_numel, num_gaussians, 1, h, w
        ).permute(0, 3, 4, 1, 2)  # (batch_numel, h, w, num_gaussians, 1)
    else:
        gm_weights = gm_logweights.exp()
    gm_slice_weights = torch.cat(
        [gm_weights.unsqueeze(-3).expand(batch_numel, h, w, n_samples, num_gaussians, 1),
         gm_slice_weights],
        dim=-1)  # (batch_numel, h, w, n_samples, num_gaussians, n_axes)

    sqrt_2 = math.sqrt(2)
    # (batch_numel, h, w, n_samples, n_axes)
    gaussian_samples_rot = (torch.erfinv(
        (gm_slice_weights * torch.erf(gm_norm_diffs / sqrt_2)).sum(dim=-2).float().clamp(min=-1 + eps, max=1 - eps)
    ) * sqrt_2).to(dtype=dtype)
    if n_axes < out_channels:
        gaussian_samples_rot = torch.cat(
            [gaussian_samples_rot,
             torch.randn(
                 (batch_numel, h, w, n_samples, out_channels - n_axes),
                 dtype=gm_samples.dtype, device=gm_samples.device)],
            dim=-1)  # (batch_numel, h, w, n_samples, out_channels)

    if axis_aligned:
        gaussian_samples = gaussian_samples_rot
    else:
        gaussian_samples = gaussian_samples_rot @ eigvecs.transpose(-1, -2)

    return gaussian_samples.permute(0, 3, 4, 1, 2).reshape(*batch_shapes, n_samples, out_channels, h, w)


@torch.jit.script
def gm_kr_slice_jit(last_samples, gm_means_1d, gm_logweights, gm_stds, gm_norm_diffs_sq_cumsumprev):
    gm_norm_diffs_prev = (last_samples  # (batch_numel, n_samples, 1, h, w)
                          - gm_means_1d  # (batch_numel, 1, num_gaussians, h, w)
                          ) / gm_stds  # (batch_numel, n_samples, num_gaussians, h, w)
    gm_norm_diffs_sq_cumsumprev = gm_norm_diffs_sq_cumsumprev + gm_norm_diffs_prev * gm_norm_diffs_prev
    # (batch_numel, n_samples, num_gaussians, h, w)
    gm_slice_logweights = (gm_logweights - 0.5 * gm_norm_diffs_sq_cumsumprev).log_softmax(dim=-3)
    return gm_slice_logweights, gm_norm_diffs_sq_cumsumprev


def gaussian_samples_to_gm_samples(
        gm, gaussian_samples, n_axes=None, n_steps=16, backward_steps=0, eps=1e-6,
        force_fp32=True, axis_aligned=True):
    """
    Knothe-Rosenblatt transport for transforming samples from a standard Gaussian to samples from a Gaussian mixture.

    Args:
        gm (dict):
            means (torch.Tensor): (bs, *, num_gaussians, out_channels, h, w)
            logstds (torch.Tensor): (bs, *, 1, 1, 1, 1)
            logweights (torch.Tensor): (bs, *, num_gaussians, 1, h, w)
            (Optional) gm_weights (torch.Tensor): (bs, *, num_gaussians, 1, h, w)
        gaussian_samples (torch.Tensor): (bs, *, n_samples, out_channels, h, w)

    Returns:
        torch.Tensor: (bs, *, n_samples, out_channels, h, w)
    """
    assert 'covs' not in gm
    ori_dtype = gaussian_samples.dtype
    if force_fp32:
        gm = {k: v.float() for k, v in gm.items()}
        gaussian_samples = gaussian_samples.float()

    dtype = gaussian_samples.dtype
    gm_means = gm['means']
    batch_shapes = gm_means.shape[:-4]
    num_gaussians, out_channels, h, w = gm_means.shape[-4:]
    batch_numel = batch_shapes.numel()
    n_samples = gaussian_samples.size(-4)
    if n_axes is None:
        n_axes = out_channels

    with torch.no_grad():
        gaussians = gm_to_gaussian(gm)[0]
        covs = gaussians['cov']
        if axis_aligned:
            covs = covs.mean(dim=(-4, -3), keepdim=True)
        # (bs, *, h, w, out_channels, out_channels) or (bs, *, 1, 1, out_channels, out_channels)
        _, eigvecs = torch.linalg.eigh(covs.float())
        # eigenvalues in descending order
        eigvecs = eigvecs.flip(-1).to(dtype=dtype)

    eigvecs = eigvecs.reshape(
        (batch_numel, 1, 1, out_channels, out_channels) if axis_aligned
        else (batch_numel, h, w, out_channels, out_channels))
    gm_means = gm_means.reshape(
        batch_numel, num_gaussians, out_channels, h, w
    ).permute(0, 3, 4, 1, 2)  # (batch_numel, h, w, num_gaussians, out_channels)
    gaussian_samples = gaussian_samples.reshape(
        batch_numel, n_samples, out_channels, h, w
    ).permute(0, 3, 4, 1, 2)  # (batch_numel, h, w, n_samples, out_channels)

    eigvecs_ = eigvecs[..., :n_axes]
    gm_means_rot = gm_means @ eigvecs  # (batch_numel, h, w, num_gaussians, out_channels)
    if axis_aligned:
        gaussian_samples_rot = gaussian_samples
    else:
        gaussian_samples_rot = gaussian_samples @ eigvecs_  # (batch_numel, h, w, n_samples, n_axes)

    gm_logstds = gm['logstds'].reshape(batch_numel, 1, 1, 1, 1)
    gm_logweights = gm['logweights'].reshape(batch_numel, 1, num_gaussians, h, w)
    gm_stds = gm_logstds.exp()  # (batch_numel, 1, 1, 1, 1)
    gm_means_rot = gm_means_rot.permute(0, 4, 3, 1, 2)  # (batch_numel, n_axes, num_gaussians, h, w)
    gaussian_samples_rot = gaussian_samples_rot.permute(0, 3, 4, 1, 2)  # (batch_numel, n_samples, n_axes, h, w)

    uniform_samples = torch.erf(gaussian_samples_rot / math.sqrt(2))  # in range [-1, 1]

    gm_samples_rot_list = []
    gm_norm_diffs_sq_cumsumprev = gaussian_samples.new_tensor([0.0])
    gm_means_1d = gm_means_rot[:, :1]  # (batch_numel, 1, num_gaussians, h, w)  for axis_id = 0
    for axis_id in range(n_axes):
        if axis_id == 0:
            gm1d = dict(
                means=gm_means_1d.squeeze(-4),  # (batch_numel, num_gaussians, h, w)
                logstds=gm_logstds.squeeze(-4),  # (batch_numel, 1, 1, 1)
                logweights=gm_logweights.squeeze(-4))  # (batch_numel, num_gaussians, h, w)
            if 'gm_weights' in gm:
                gm1d.update(gm_weights=gm['gm_weights'].reshape(batch_numel, num_gaussians, h, w))
            gm_samples_rot_list.append(
                gm1d_inverse_cdf(
                    gm1d,
                    uniform_samples[:, :, axis_id],  # (batch_numel, n_samples, h, w)
                    n_steps=n_steps,
                    eps=eps,
                    max_step_size=1.5,
                    gaussian_samples=gaussian_samples_rot[:, :, axis_id],  # (batch_numel, n_samples, h, w)
                    backward_steps=backward_steps
                ).unsqueeze(-3)  # (batch_numel, n_samples, 1, h, w)
            )
        else:
            gm_slice_logweights, gm_norm_diffs_sq_cumsumprev = gm_kr_slice_jit(
                gm_samples_rot_list[-1], gm_means_1d, gm_logweights, gm_stds, gm_norm_diffs_sq_cumsumprev)
            gm_means_1d = gm_means_rot[:, axis_id:axis_id + 1]  # (batch_numel, 1, num_gaussians, h, w)
            gm1d = dict(
                means=gm_means_1d,  # (batch_numel, 1, num_gaussians, h, w)
                logstds=gm_logstds,  # (batch_numel, 1, 1, 1, 1)
                logweights=gm_slice_logweights)  # (batch_numel, n_samples, num_gaussians, h, w)
            gm_samples_rot_list.append(
                gm1d_inverse_cdf(
                    gm1d,
                    uniform_samples[:, :, axis_id:axis_id + 1],  # (batch_numel, n_samples, 1, h, w)
                    n_steps=n_steps,
                    eps=eps,
                    max_step_size=1.5,
                    gaussian_samples=gaussian_samples_rot[:, :, axis_id:axis_id + 1],  # (batch_numel, n_samples, 1, h, w)
                    backward_steps=backward_steps
                )  # (batch_numel, n_samples, 1, h, w)
            )

    gm_samples_rot = torch.cat(gm_samples_rot_list, dim=-3)  # (batch_numel, n_samples, n_axes, h, w)

    if n_axes < out_channels:
        gm_slice_logweights, gm_norm_diffs_sq_cumsumprev = gm_kr_slice_jit(
            gm_samples_rot_list[-1], gm_means_1d, gm_logweights, gm_stds, gm_norm_diffs_sq_cumsumprev)
        gm_means = gm_means_rot[:, n_axes:]  # (batch_numel, num_channels - n_axes, num_gaussians, h, w)
        gm_slice = dict(
            means=gm_means.transpose(-3, -4).unsqueeze(-5).expand(
                batch_numel, n_samples, num_gaussians, out_channels - n_axes, h, w),
            logstds=gm_logstds.unsqueeze(-3),  # (batch_numel, 1, 1, 1, 1, 1)
            logweights=gm_slice_logweights.unsqueeze(-3))  # (batch_numel, n_samples, num_gaussians, 1, h, w)
        gm_samples_rot = torch.cat(
            [gm_samples_rot,
             gm_to_sample(gm_slice, 1).squeeze(-4)],
            dim=-3)    # (batch_numel, n_samples, out_channels, h, w)

    gm_samples = gm_samples_rot.permute(0, 3, 4, 1, 2) @ eigvecs.transpose(-1, -2)

    return gm_samples.permute(0, 3, 4, 1, 2).reshape(*batch_shapes, n_samples, out_channels, h, w).to(ori_dtype)


def gm_transpose_t_first(gm):
    # (bs, num_gaussians, out_channels, t, h, w) -> (bs, t, num_gaussians, out_channels, h, w)
    return dict(
        means=gm['means'].permute(0, 3, 1, 2, 4, 5),
        logweights=gm['logweights'].permute(0, 3, 1, 2, 4, 5),
        logstds=gm['logstds'].permute(0, 3, 1, 2, 4, 5)
    )


def gm_temperature(gm, temperature, gm_dim=-4, eps=1e-6):
    gm = gm.copy()
    temperature = max(temperature, eps)
    gm['logweights'] = (gm['logweights'] / temperature).log_softmax(dim=gm_dim)
    if 'logstds' in gm:
        gm['logstds'] = gm['logstds'] + (0.5 * math.log(temperature))
    if 'gm_vars' in gm:
        gm['gm_vars'] = gm['gm_vars'] * temperature
    return gm
