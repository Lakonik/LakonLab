# Copyright (c) 2026 Hansheng Chen

import sys
import inspect
from typing import Optional
from copy import deepcopy

import torch
import diffusers
import mmcv

from ..builder import MODULES, build_module
from . import GaussianFlow, schedulers
from lakonlab.models.architectures.utils import get_module_device
from lakonlab.ops.gmflow_ops.gmflow_ops import (
    gm_to_sample, gm_to_mean, gaussian_samples_to_gm_samples, gm_samples_to_gaussian_samples,
    iso_gaussian_mul_iso_gaussian, gm_mul_iso_gaussian, gm_to_iso_gaussian, gm_transpose_t_first)


@torch.jit.script
def probabilistic_guidance_jit(
        cond_mean, total_var, uncond_mean, guidance_scale,
        orthogonal: float = 1.0, parallel_dir: Optional[torch.Tensor] = None):
    dim = list(range(1, cond_mean.dim()))
    bias = cond_mean - uncond_mean
    if orthogonal > 0.0:
        if parallel_dir is None:
            parallel_dir = cond_mean
        bias = bias - ((bias * parallel_dir).mean(
            dim=dim, keepdim=True
        ) / (parallel_dir * parallel_dir).mean(
            dim=dim, keepdim=True
        ).clamp(min=1e-6) * parallel_dir).mul(orthogonal)
    bias_power = (bias * bias).mean(dim=dim, keepdim=True)
    avg_var = total_var.mean(dim=dim, keepdim=True)
    bias = bias * ((avg_var / bias_power.clamp(min=1e-6)).sqrt() * guidance_scale)
    gaussian_output = dict(
        mean=cond_mean + bias,
        var=total_var * (1 - (guidance_scale * guidance_scale)))
    return gaussian_output, bias, avg_var


@torch.jit.script
def gmflow_posterior_mean_jit(
        sigma_t_src, sigma_t, x_t_src, x_t,
        gm_means, gm_vars, gm_logweights,
        eps: float, gm_dim: int = -4, channel_dim: int = -3):
    sigma_t_src = sigma_t_src.clamp(min=eps)
    sigma_t = sigma_t.clamp(min=eps)

    alpha_t_src = 1 - sigma_t_src
    alpha_t = 1 - sigma_t

    alpha_over_sigma_t_src = alpha_t_src / sigma_t_src
    alpha_over_sigma_t = alpha_t / sigma_t

    zeta = alpha_over_sigma_t.square() - alpha_over_sigma_t_src.square()
    nu = alpha_over_sigma_t * x_t / sigma_t - alpha_over_sigma_t_src * x_t_src / sigma_t_src

    nu = nu.unsqueeze(gm_dim)  # (bs, *, 1, out_channels, h, w)
    zeta = zeta.unsqueeze(gm_dim)  # (bs, *, 1, 1, 1, 1)
    denom = (gm_vars * zeta + 1).clamp(min=eps)

    out_means = (gm_vars * nu + gm_means) / denom
    # (bs, *, num_gaussians, 1, h, w)
    logweights_delta = (gm_means * (nu - 0.5 * zeta * gm_means)).sum(
        dim=channel_dim, keepdim=True) / denom
    out_weights = (gm_logweights + logweights_delta).softmax(dim=gm_dim)

    out_mean = (out_means * out_weights).sum(dim=gm_dim)

    return out_mean


class GMFlowMixin:

    @property
    def time_scaling(self):
        if hasattr(self, 'scheduler'):  # for diffusers pipelines
            return self.scheduler.config.num_train_timesteps
        elif hasattr(self, 'num_timesteps'):
            return self.num_timesteps
        else:
            raise ValueError('num_timesteps or scheduler is not defined.')

    def u_to_x_0(self, denoising_output, x_t, t=None, sigma=None, eps=1e-6):
        if isinstance(denoising_output, dict) and 'logweights' in denoising_output:
            x_t = x_t.unsqueeze(-4)

        if sigma is None:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, device=x_t.device)
            t = t.reshape(*t.size(), *((x_t.dim() - t.dim()) * [1]))
            sigma = t / self.time_scaling
        else:
            assert sigma.dim() == x_t.dim() - 1
            sigma = sigma.unsqueeze(-4)

        if isinstance(denoising_output, dict):
            if 'logweights' in denoising_output:
                means_x_0 = x_t - sigma * denoising_output['means']
                logstds_x_0 = denoising_output['logstds'] + torch.log(sigma.clamp(min=eps))
                return dict(
                    means=means_x_0,
                    logstds=logstds_x_0,
                    logweights=denoising_output['logweights'])
            elif 'var' in denoising_output:
                mean = x_t - sigma * denoising_output['mean']
                var = denoising_output['var'] * sigma.square()
                return dict(mean=mean, var=var)
            else:
                raise ValueError('Invalid denoising_output.')

        else:  # sample mode
            x_0 = x_t - sigma * denoising_output
            return x_0

    def gmflow_posterior_mean(
            self, gm, x_t, x_t_src, t=None, t_src=None,
            sigma_t_src=None, sigma_t=None, eps=1e-6, prediction_type='x0',
            checkpointing=False):
        """
        Fuse gmflow_posterior and gm_to_mean to avoid redundant computation.
        """
        assert isinstance(gm, dict)

        if sigma_t_src is None:
            if not isinstance(t_src, torch.Tensor):
                t_src = torch.tensor(t_src, device=x_t_src.device)
            t_src = t_src.reshape(*t_src.size(), *((x_t_src.dim() - t_src.dim()) * [1]))
            sigma_t_src = t_src / self.time_scaling

        if sigma_t is None:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, device=x_t_src.device)
            t = t.reshape(*t.size(), *((x_t_src.dim() - t.dim()) * [1]))
            sigma_t = t / self.time_scaling

        if prediction_type == 'u':
            gm = self.u_to_x_0(gm, x_t_src, sigma=sigma_t_src)
        else:
            assert prediction_type == 'x0'

        gm_means = gm['means']  # (bs, *, num_gaussians, out_channels, h, w)
        gm_logweights = gm['logweights']  # (bs, *, num_gaussians, 1, h, w)
        if 'gm_vars' in gm:
            gm_vars = gm['gm_vars']
        else:
            gm_vars = (gm['logstds'] * 2).exp()  # (bs, *, 1, 1, 1, 1)
            gm['gm_vars'] = gm_vars

        if checkpointing and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                gmflow_posterior_mean_jit,
                sigma_t_src, sigma_t, x_t_src, x_t,
                gm_means, gm_vars, gm_logweights, eps,
                use_reentrant=True)  # use_reentrant=False does not work with jit
        else:
            return gmflow_posterior_mean_jit(
                sigma_t_src, sigma_t, x_t_src, x_t,
                gm_means, gm_vars, gm_logweights, eps)

    def reverse_transition(self, denoising_output, x_t_high, t_low, t_high, eps=1e-6, prediction_type='u'):
        if isinstance(denoising_output, dict):
            x_t_high = x_t_high.unsqueeze(-4)

        bs = x_t_high.size(0)
        if not isinstance(t_low, torch.Tensor):
            t_low = torch.tensor(t_low, device=x_t_high.device)
        if not isinstance(t_high, torch.Tensor):
            t_high = torch.tensor(t_high, device=x_t_high.device)
        if t_low.dim() == 0:
            t_low = t_low.expand(bs)
        if t_high.dim() == 0:
            t_high = t_high.expand(bs)
        t_low = t_low.reshape(*t_low.size(), *((x_t_high.dim() - t_low.dim()) * [1]))
        t_high = t_high.reshape(*t_high.size(), *((x_t_high.dim() - t_high.dim()) * [1]))

        sigma = t_high / self.time_scaling
        sigma_to = t_low / self.time_scaling
        alpha = 1 - sigma
        alpha_to = 1 - sigma_to

        sigma_to_over_sigma = sigma_to / sigma.clamp(min=eps)
        alpha_over_alpha_to = alpha / alpha_to.clamp(min=eps)
        beta_over_sigma_sq = 1 - (sigma_to_over_sigma * alpha_over_alpha_to) ** 2

        c1 = sigma_to_over_sigma ** 2 * alpha_over_alpha_to
        c2 = beta_over_sigma_sq * alpha_to

        if isinstance(denoising_output, dict):
            c3 = beta_over_sigma_sq * sigma_to ** 2
            if prediction_type == 'u':
                means_x_0 = x_t_high - sigma * denoising_output['means']
                logstds_x_t_low = torch.logaddexp(
                    (denoising_output['logstds'] + torch.log((sigma * c2).clamp(min=eps))) * 2,
                    torch.log(c3.clamp(min=eps))
                ) / 2
            elif prediction_type == 'x0':
                means_x_0 = denoising_output['means']
                logstds_x_t_low = torch.logaddexp(
                    (denoising_output['logstds'] + torch.log(c2.clamp(min=eps))) * 2,
                    torch.log(c3.clamp(min=eps))
                ) / 2
            else:
                raise ValueError('Invalid prediction_type.')
            means_x_t_low = c1 * x_t_high + c2 * means_x_0
            return dict(
                means=means_x_t_low,
                logstds=logstds_x_t_low,
                logweights=denoising_output['logweights'])

        else:  # sample mode
            c3_sqrt = beta_over_sigma_sq ** 0.5 * sigma_to
            noise = torch.randn_like(denoising_output)
            if prediction_type == 'u':
                x_0 = x_t_high - sigma * denoising_output
            elif prediction_type == 'x0':
                x_0 = denoising_output
            else:
                raise ValueError('Invalid prediction_type.')
            x_t_low = c1 * x_t_high + c2 * x_0 + c3_sqrt * noise
            return x_t_low

    @staticmethod
    def gm_sample(gm, power_spectrum=None, n_samples=1, generator=None):
        device = gm['means'].device
        if power_spectrum is not None:
            power_spectrum = power_spectrum.to(dtype=torch.float32).unsqueeze(-4)
            shape = list(gm['means'].size())
            shape[-4] = n_samples
            half_size = shape[-1] // 2 + 1
            spectral_samples = torch.randn(
                shape, dtype=torch.float32, device=device, generator=generator) * (power_spectrum / 2).exp()
            z_1 = spectral_samples.roll((-1, -1), dims=(-2, -1)).flip((-2, -1))[..., :half_size]
            z_0 = spectral_samples[..., :half_size]
            z_kr = torch.complex(z_0 + z_1, z_0 - z_1) / 2
            gaussian_samples = torch.fft.irfft2(z_kr, norm='ortho')
            samples = gaussian_samples_to_gm_samples(gm, gaussian_samples, axis_aligned=True)
        else:
            samples = gm_to_sample(gm, n_samples=n_samples)
            spectral_samples = None
        return samples, spectral_samples

    def gm_to_model_output(self, gm, output_mode, power_spectrum=None, generator=None):
        assert output_mode in ['mean', 'sample']
        if output_mode == 'mean':
            output = gm_to_mean(gm)
        else:  # sample
            output = self.gm_sample(gm, power_spectrum, generator=generator)[0].squeeze(-4)
        return output

    def gm_2nd_order(
            self, gm_output, gaussian_output, x_t, t, h,
            guidance_scale=0.0, gm_cond=None, gaussian_cond=None, avg_var=None, cfg_bias=None, ca=0.005, cb=1.0,
            gm2_correction_steps=0):
        if self.prev_gm is not None:
            dim = list(range(1, x_t.dim()))

            if cfg_bias is not None:
                gm_mean = gm_to_mean(gm_output)
                base_gaussian = gaussian_cond
                base_gm = gm_cond
            else:
                gm_mean = gaussian_output['mean']
                base_gaussian = gaussian_output
                base_gaussian['var'] = base_gaussian['var'].mean(dim=dim[:-3] + dim[-2:], keepdim=True)  # exclude channel dim
                avg_var = base_gaussian['var'].mean(dim=dim, keepdim=True)
                base_gm = gm_output

            mean_from_prev = self.gmflow_posterior_mean(
                self.prev_gm, x_t, self.prev_x_t, t, self.prev_t, prediction_type='x0')
            self.prev_gm = gm_output

            # Compute rescaled 2nd-order mean difference
            k = 0.5 * h / self.prev_h
            prev_h_norm = self.prev_h / self.time_scaling
            _guidance_scale = guidance_scale * cb
            err_power = avg_var * (_guidance_scale * _guidance_scale + ca)
            mean_diff = (gm_mean - mean_from_prev) * (
                (1 - err_power / (prev_h_norm * prev_h_norm)).clamp(min=0).sqrt() * k)

            bias = mean_diff
            # Here we fuse probabilistic guidance bias and 2nd-order mean difference to perform one single
            # update to the base GM, which avoids cumulative errors.
            if cfg_bias is not None:
                bias = mean_diff + cfg_bias
            bias_power = bias.square().mean(dim=dim, keepdim=True)
            bias = bias * (avg_var / bias_power.clamp(min=1e-6)).clamp(max=1).sqrt()

            gaussian_output = dict(
                mean=base_gaussian['mean'] + bias,
                var=base_gaussian['var'] * (1 - bias_power / avg_var.clamp(min=1e-6)).clamp(min=1e-6))
            gm_output = gm_mul_iso_gaussian(
                base_gm, iso_gaussian_mul_iso_gaussian(gaussian_output, base_gaussian, 1, -1),
                1, 1)[0]

            # Additional correction steps for strictly matching the 2nd order mean difference
            if gm2_correction_steps > 0:
                adjusted_bias = bias
                tgt_bias = mean_diff + gm_mean - base_gaussian['mean']
                for _ in range(gm2_correction_steps):
                    out_bias = gm_to_mean(gm_output) - base_gaussian['mean']
                    err = out_bias - tgt_bias
                    adjusted_bias = adjusted_bias - err * (
                        adjusted_bias.norm(dim=-3, keepdim=True) / out_bias.norm(dim=-3, keepdim=True).clamp(min=1e-6)
                    ).clamp(max=1)
                    adjusted_bias_power = adjusted_bias.square().mean(dim=dim, keepdim=True)
                    adjusted_bias = adjusted_bias * (avg_var / adjusted_bias_power.clamp(min=1e-6)).clamp(max=1).sqrt()
                    adjusted_gaussian_output = dict(
                        mean=base_gaussian['mean'] + adjusted_bias,
                        var=base_gaussian['var'] * (1 - adjusted_bias_power / avg_var.clamp(min=1e-6)).clamp(min=1e-6))
                    gm_output = gm_mul_iso_gaussian(
                        base_gm, iso_gaussian_mul_iso_gaussian(adjusted_gaussian_output, base_gaussian, 1, -1),
                        1, 1)[0]

        else:
            self.prev_gm = gm_output

        self.prev_x_t = x_t
        self.prev_t = t
        self.prev_h = h

        return gm_output, gaussian_output

    def init_gm_cache(self):
        self.prev_gm = None
        self.prev_x_t = None
        self.prev_t = None
        self.prev_h = None


@MODULES.register_module()
class GMFlow(GMFlowMixin, GaussianFlow):

    def __init__(
            self,
            *args,
            spectrum_net=None,
            spectral_loss_weight=1.0,
            **kwargs):
        super().__init__(*args, **kwargs)
        assert self.denoising_mean_mode.upper() == 'U'
        self.spectrum_net = build_module(spectrum_net) if spectrum_net is not None else None
        self.spectral_loss_weight = spectral_loss_weight
        self.intermediate_x_t = []
        self.intermediate_x_0 = []

    def loss(self, denoising_output, x_t_low, x_t_high, t_low, t_high):
        """
        GMFlow transition loss.
        """
        x_t_low = x_t_low.float()
        x_t_high = x_t_high.float()
        t_low = t_low.float()
        t_high = t_high.float()

        x_t_low_gm = self.reverse_transition(denoising_output, x_t_high, t_low, t_high)
        loss_kwargs = {k: v for k, v in x_t_low_gm.items()}

        loss_kwargs.update(x_t_low=x_t_low, timesteps=t_high)
        return self.flow_loss(loss_kwargs)

    def spectral_loss(self, denoising_output, x_0, x_t, t, eps=1e-6):
        x_0 = x_0.float()
        x_t = x_t.float()
        t = t.float()

        t = t.reshape(*t.size(), *((x_t.dim() - t.dim()) * [1]))
        inv_sigma = self.num_timesteps / t.clamp(min=eps)

        with torch.no_grad():
            output_g = self.u_to_x_0(gm_to_iso_gaussian(denoising_output)[0], x_t, t)
            u = (x_t - x_0) * inv_sigma
            z_kr = gm_samples_to_gaussian_samples(
                denoising_output, u.unsqueeze(-4), axis_aligned=True).squeeze(-4)
            z_kr_fft = torch.fft.fft2(z_kr, norm='ortho')
            z_kr_fft = z_kr_fft.real + z_kr_fft.imag

        log_var = self.spectrum_net(output_g)

        loss = z_kr_fft.square() * (torch.exp(-log_var) - 1) + log_var
        loss = loss.mean() * (0.5 * self.spectral_loss_weight)
        return loss

    def pred(self, x_t=None, t=None, test_cfg=dict(), **kwargs):
        ndim = x_t.dim()
        assert ndim in [4, 5], f'Invalid x_t shape: {x_t.shape}. Expected 4D or 5D tensor.'
        if ndim == 5:  # (bs, t, c, h, w)
            x_t = x_t.permute(0, 2, 1, 3, 4)  # (bs, c, t, h, w)
        output = super().pred(x_t=x_t, t=t, **kwargs)
        if ndim == 5:
            output = gm_transpose_t_first(output)  # (bs, t, c, h, w)
        return output

    def forward_train(self, x_0, **kwargs):
        device = get_module_device(self)

        num_batches = x_0.size(0)
        seq_len = x_0.shape[2:].numel()  # h * w or t * h * w
        ndim = x_0.dim()
        assert ndim in [4, 5], f'Invalid x_0 shape: {x_0.shape}. Expected 4D or 5D tensor.'
        if ndim == 5:  # (bs, c, t, h, w)
            x_0 = x_0.permute(0, 2, 1, 3, 4)  # (bs, t, c, h, w)

        trans_ratio = self.train_cfg.get('trans_ratio', 1.0)
        eps = self.train_cfg.get('eps', 1e-4)
        max_raw_t = self.train_cfg.get('max_raw_t', 1.0)
        min_raw_t = self.train_cfg.get('min_raw_t', 0.0)

        t_high = self.timestep_sampler(
            num_batches,
            seq_len=seq_len,
            device=device,
            raw_t_range=(min_raw_t, max_raw_t)
        ).clamp(min=eps, max=self.num_timesteps)
        t_low = t_high * (1 - trans_ratio)
        t_low = torch.minimum(t_low, t_high - eps).clamp(min=0)

        noise = torch.randn((num_batches * 2, *x_0.shape[1:]), device=device, dtype=x_0.dtype)
        noise_0, noise_1 = torch.chunk(noise, 2, dim=0)

        x_t_low, _, _ = self.sample_forward_diffusion(x_0, t_low, noise_0)
        x_t_high = self.sample_forward_transition(x_t_low, noise_1, t_src=t_low, t_tgt=t_high)

        denoising_output = self.pred(x_t_high, t_high, **kwargs)
        loss = self.loss(denoising_output, x_t_low, x_t_high, t_low, t_high)
        log_vars = self.flow_loss.log_vars
        log_vars.update(loss_transition=float(loss))

        if self.spectrum_net is not None:
            # Note: only support 2D power spectrum for now.
            loss_spectral = self.spectral_loss(denoising_output, x_0, x_t_high, t_high)
            log_vars.update(loss_spectral=float(loss_spectral))
            loss = loss + loss_spectral

        return loss, log_vars

    def forward_test(
            self, x_0=None, noise=None, guidance_scale=0.0,
            test_cfg_override=dict(), show_pbar=False, **kwargs):
        x_t = torch.randn_like(x_0) if noise is None else noise
        num_batches = x_t.size(0)
        seq_len = x_t.shape[2:].numel()  # h * w or t * h * w
        ori_dtype = x_t.dtype
        x_t = x_t.float()
        ndim = x_t.dim()
        assert ndim in [4, 5], f'Invalid x_t shape: {x_t.shape}. Expected 4D or 5D tensor.'
        if ndim == 5:  # (bs, c, t, h, w)
            x_t = x_t.permute(0, 2, 1, 3, 4)  # (bs, t, c, h, w)

        cfg = deepcopy(self.test_cfg)
        cfg.update(test_cfg_override)

        output_mode = cfg.get('output_mode', 'mean')
        assert output_mode in ['mean', 'sample']

        sampler = cfg['sampler']
        sampler_class = getattr(diffusers.schedulers, sampler + 'Scheduler', None)
        if sampler_class is None:
            sampler_class = getattr(schedulers, sampler + 'Scheduler', None)
        if sampler_class is None:
            raise AttributeError(f'Cannot find sampler [{sampler}].')

        sampler_kwargs = cfg.get('sampler_kwargs', {})
        signatures = inspect.signature(sampler_class).parameters.keys()
        for key in ['shift', 'use_dynamic_shifting', 'base_seq_len', 'max_seq_len', 'base_logshift', 'max_logshift']:
            if key in signatures and key not in sampler_kwargs:
                sampler_kwargs[key] = cfg.get(key, getattr(self.timestep_sampler, key))
        for key in ['max_raw_t', 'min_raw_t']:
            if key in signatures and key in cfg and key not in sampler_kwargs:
                sampler_kwargs[key] = cfg[key]
        sampler = sampler_class(self.num_timesteps, **sampler_kwargs)

        num_timesteps = cfg.get('num_timesteps', self.num_timesteps)
        num_substeps = cfg.get('num_substeps', 1)
        guidance_interval = cfg.get('guidance_interval', [0, self.num_timesteps])
        orthogonal_guidance = cfg.get('orthogonal_guidance', 1.0)
        save_intermediate = cfg.get('save_intermediate', False)
        order = cfg.get('order', 1)
        gm2_coefs = cfg.get('gm2_coefs', [0.005, 1.0])
        gm2_correction_steps = cfg.get('gm2_correction_steps', 0)

        set_timesteps_signatures = inspect.signature(sampler.set_timesteps).parameters.keys()
        if 'seq_len' in set_timesteps_signatures:
            sampler.set_timesteps(num_timesteps * num_substeps, seq_len=seq_len, device=x_t.device)
        else:
            sampler.set_timesteps(num_timesteps * num_substeps, device=x_t.device)

        timesteps = sampler.timesteps
        self.intermediate_x_t = []
        self.intermediate_x_0 = []
        self.intermediate_gm_x_0 = []
        self.intermediate_gm_trans = []
        self.intermediate_t = []
        assert order in [1, 2]
        use_guidance = 0.0 < guidance_scale < 1.0
        if use_guidance:
            guidance_scale = x_t.new_tensor(  # to tensor
                [guidance_scale]
            ).expand(num_batches).reshape([num_batches] + [1] * (x_t.dim() - 1))

        x_t = timesteps[0] / self.num_timesteps * x_t

        if show_pbar:
            pbar = mmcv.ProgressBar(num_timesteps)

        # ========== Main sampling loop ==========
        self.init_gm_cache()

        for timestep_id in range(num_timesteps):
            t = timesteps[timestep_id * num_substeps]

            if save_intermediate:
                self.intermediate_x_t.append(x_t)
                self.intermediate_t.append(t)

            x_t_input = x_t
            _kwargs = kwargs
            if use_guidance:
                x_t_input = torch.cat([x_t_input, x_t_input], dim=0)

            gm_output = self.pred(x_t_input, t, test_cfg=cfg, **_kwargs)
            assert isinstance(gm_output, dict)
            gm_output = self.u_to_x_0(gm_output, x_t_input, t)

            # ========== Probabilistic CFG ==========
            if use_guidance:
                _guidance_scale = guidance_scale
                guidance_active = guidance_interval[0] <= t <= guidance_interval[1]
                if not guidance_active:
                    _guidance_scale = torch.zeros_like(_guidance_scale)
                gm_cond = {k: v[num_batches:] for k, v in gm_output.items()}
                gm_uncond = {k: v[:num_batches] for k, v in gm_output.items()}
                uncond_mean = gm_to_mean(gm_uncond)
                gaussian_cond = gm_to_iso_gaussian(gm_cond)[0]
                if ndim == 5:
                    gaussian_cond['var'] = gaussian_cond['var'].mean(dim=(-4, -2, -1), keepdim=True)  # exclude channel dim
                else:
                    gaussian_cond['var'] = gaussian_cond['var'].mean(dim=(-2, -1), keepdim=True)  # exclude channel dim
                gaussian_output, cfg_bias, avg_var = probabilistic_guidance_jit(
                    gaussian_cond['mean'], gaussian_cond['var'], uncond_mean, _guidance_scale,
                    orthogonal=orthogonal_guidance)
                gm_output = gm_mul_iso_gaussian(
                    gm_cond, iso_gaussian_mul_iso_gaussian(gaussian_output, gaussian_cond, 1, -1),
                    1, 1)[0]
            else:
                gaussian_output = gm_to_iso_gaussian(gm_output)[0]
                gm_cond = gaussian_cond = avg_var = cfg_bias = None

            # ========== 2nd order GM ==========
            if order == 2:
                if timestep_id < num_timesteps - 1:
                    h = t - timesteps[(timestep_id + 1) * num_substeps]
                else:
                    h = t
                gm_output, gaussian_output = self.gm_2nd_order(
                    gm_output, gaussian_output, x_t, t, h,
                    guidance_scale if guidance_active else 0.0, gm_cond, gaussian_cond, avg_var, cfg_bias,
                    ca=gm2_coefs[0], cb=gm2_coefs[1], gm2_correction_steps=gm2_correction_steps)

            if save_intermediate:
                self.intermediate_gm_x_0.append(gm_output)
                if timestep_id < num_timesteps - 1:
                    t_next = timesteps[(timestep_id + 1) * num_substeps]
                else:
                    t_next = 0
                gm_trans = self.reverse_transition(gm_output, x_t, t_next, t, prediction_type='x0')
                self.intermediate_gm_trans.append(gm_trans)

            # ========== GM SDE step or GM ODE substeps ==========
            x_t_base = x_t
            t_base = t
            for substep_id in range(num_substeps):
                if substep_id == 0:
                    if self.spectrum_net is not None and output_mode == 'sample':
                        # Note: only support 2D power spectrum for now.
                        power_spectrum = self.spectrum_net(gaussian_output)
                    else:
                        power_spectrum = None
                    model_output = self.gm_to_model_output(gm_output, output_mode, power_spectrum=power_spectrum)
                else:
                    assert output_mode == 'mean'
                    t = timesteps[timestep_id * num_substeps + substep_id]
                    model_output = self.gmflow_posterior_mean(
                        gm_output, x_t, x_t_base, t, t_base, prediction_type='x0')
                x_t = sampler.step(model_output, t, x_t, return_dict=False, prediction_type='x0')[0]

            if save_intermediate:
                self.intermediate_x_0.append(model_output)

            if show_pbar:
                pbar.update()

        if show_pbar:
            sys.stdout.write('\n')

        if ndim == 5:  # (bs, t, c, h, w)
            x_t = x_t.permute(0, 2, 1, 3, 4)  # (bs, c, t, h, w)

        return x_t.to(ori_dtype)

    def forward_u(self, x_t, t, guidance_scale=0.0, test_cfg=dict(), **kwargs):
        ori_dtype = x_t.dtype
        x_t = x_t.float()
        num_batches = x_t.size(0)
        ndim = x_t.dim()
        assert ndim in [4, 5], f'Invalid x_t shape: {x_t.shape}. Expected 4D or 5D tensor.'
        if ndim == 5:  # (bs, c, t, h, w)
            x_t = x_t.permute(0, 2, 1, 3, 4)  # (bs, t, c, h, w)

        orthogonal_guidance = test_cfg.get('orthogonal_guidance', 1.0)
        guidance_interval = test_cfg.get('guidance_interval', [0, self.num_timesteps])

        use_guidance = 0.0 < guidance_scale < 1.0

        x_t_input = x_t
        t_input = t
        if use_guidance:
            guidance_scale = x_t.new_tensor(
                [guidance_scale]
            ).expand(num_batches).reshape([num_batches] + [1] * (x_t.dim() - 1))
            guidance_active = ((t >= guidance_interval[0]) & (t <= guidance_interval[1])).reshape_as(
                guidance_scale)
            guidance_scale = guidance_scale * guidance_active.float()

            x_t_input = torch.cat([x_t_input, x_t_input], dim=0)
            t_input = torch.cat([t_input, t_input], dim=0)

        gm_output = self.pred(x_t_input, t_input, **kwargs)
        assert isinstance(gm_output, dict)

        # ========== Probabilistic CFG ==========
        if use_guidance:
            gm_cond = {k: v[num_batches:] for k, v in gm_output.items()}
            gm_uncond = {k: v[:num_batches] for k, v in gm_output.items()}
            uncond_mean = gm_to_mean(gm_uncond)
            gaussian_cond = gm_to_iso_gaussian(gm_cond)[0]
            if ndim == 5:
                gaussian_cond['var'] = gaussian_cond['var'].mean(dim=(-4, -2, -1), keepdim=True)  # exclude channel dim
            else:
                gaussian_cond['var'] = gaussian_cond['var'].mean(dim=(-2, -1), keepdim=True)
            gaussian_output = probabilistic_guidance_jit(
                gaussian_cond['mean'], gaussian_cond['var'], uncond_mean, guidance_scale,
                orthogonal=orthogonal_guidance,
                parallel_dir=self.u_to_x_0(gaussian_cond['mean'], x_t, t))[0]
            gm_output = gm_mul_iso_gaussian(
                gm_cond, iso_gaussian_mul_iso_gaussian(gaussian_output, gaussian_cond, 1, -1),
                1, 1)[0]

        u = gm_to_mean(gm_output)

        if ndim == 5:  # (bs, t, c, h, w)
            u = u.permute(0, 2, 1, 3, 4)  # (bs, c, t, h, w)

        return u.to(ori_dtype)
