# Copyright (c) 2026 Hansheng Chen

import sys
import inspect
from typing import Optional
from copy import deepcopy

import torch
import torch.nn as nn
import diffusers

import mmcv
from mmcv.runner.fp16_utils import force_fp32

from ..builder import MODULES, build_module
from . import schedulers
from lakonlab.models.architectures.utils import get_module_device
from lakonlab.runner.timer import default_timers


@torch.jit.script
def guidance_jit(
        pos_mean, neg_mean, guidance_scale,
        orthogonal: float = 1.0, parallel_dir: Optional[torch.Tensor] = None):
    bias = (pos_mean - neg_mean) * (guidance_scale - 1)
    if orthogonal:
        dim = list(range(1, pos_mean.dim()))
        if parallel_dir is None:
            parallel_dir = pos_mean
        bias = bias - ((bias * parallel_dir).mean(
            dim=dim, keepdim=True
        ) / (parallel_dir * parallel_dir).mean(
            dim=dim, keepdim=True
        ).clamp(min=1e-6) * parallel_dir).mul(orthogonal)
    return bias


@MODULES.register_module()
class GaussianFlow(nn.Module):

    def __init__(self,
                 denoising=None,
                 flow_loss=None,
                 repa_loss=None,
                 num_timesteps=1000,
                 timestep_sampler=dict(type='ContinuousTimeStepSampler', shift=1.0),
                 flip_model_timesteps=False,
                 denoising_mean_mode='U',
                 sigma_min=1e-4,  # for training loss only, JiT uses 0.05
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        # build denoising module in this function
        self.num_timesteps = num_timesteps
        self.denoising = build_module(denoising) if isinstance(denoising, dict) else denoising
        self.repa_loss = build_module(repa_loss) if repa_loss is not None else None
        self.denoising_mean_mode = denoising_mean_mode
        self.sigma_min = sigma_min

        self.flip_model_timesteps = flip_model_timesteps
        self.train_cfg = deepcopy(train_cfg) if train_cfg is not None else dict()
        self.test_cfg = deepcopy(test_cfg) if test_cfg is not None else dict()

        # build sampler
        self.timestep_sampler = build_module(
            timestep_sampler,
            default_args=dict(num_timesteps=num_timesteps))
        self.flow_loss = build_module(flow_loss) if flow_loss is not None else None

        default_timers.add_timer('network time')

    def forward_transition(
            self, x_t_src, t_src=None, t_tgt=None, sigma_src=None, sigma_tgt=None, eps=1e-6):
        if sigma_src is None:
            if not isinstance(t_src, torch.Tensor):
                t_src = torch.tensor(t_src, device=x_t_src.device)
            t_src = t_src.reshape(*t_src.size(), *((x_t_src.dim() - t_src.dim()) * [1]))
            sigma_src = t_src / self.num_timesteps

        if sigma_tgt is None:
            if not isinstance(t_tgt, torch.Tensor):
                t_tgt = torch.tensor(t_tgt, device=x_t_src.device)
            t_tgt = t_tgt.reshape(*t_tgt.size(), *((x_t_src.dim() - t_tgt.dim()) * [1]))
            sigma_tgt = t_tgt / self.num_timesteps

        alpha_src = 1 - sigma_src
        alpha_tgt = 1 - sigma_tgt

        scale_trans = alpha_tgt / alpha_src.clamp(min=eps)
        var_trans = (sigma_tgt ** 2 - (scale_trans * sigma_src) ** 2).clamp(min=0)
        return dict(mean=x_t_src * scale_trans, var=var_trans), scale_trans

    def sample_forward_transition(self, x_t_src, noise, t_src=None, t_tgt=None, sigma_src=None, sigma_tgt=None):
        trans_g = self.forward_transition(
            x_t_src, t_src=t_src, t_tgt=t_tgt, sigma_src=sigma_src, sigma_tgt=sigma_tgt)[0]
        return trans_g['mean'] + noise * trans_g['var'].sqrt()

    def sample_forward_diffusion(self, x_0, t, noise):
        if t.dim() == 0:
            t = t.expand(x_0.size(0))
        std = t.reshape(*t.size(), *((x_0.dim() - t.dim()) * [1])) / self.num_timesteps
        mean = 1 - std
        return x_0 * mean + noise * std, mean, std

    def u_to_x_0(self, u, x_t, t=None, sigma=None):
        if sigma is None:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, device=x_t.device)
            t = t.reshape(*t.size(), *((x_t.dim() - t.dim()) * [1]))
            sigma = t / self.num_timesteps
        else:
            assert sigma.dim() == x_t.dim()

        x_0 = x_t - sigma * u
        return x_0

    def x_0_to_u(self, x_0, x_t, t=None, sigma=None, eps=1e-4):
        if sigma is None:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, device=x_t.device)
            t = t.reshape(*t.size(), *((x_t.dim() - t.dim()) * [1]))
            sigma = t / self.num_timesteps
        else:
            assert sigma.dim() == x_t.dim()

        denoising_output = (x_t - x_0) / sigma.clamp(min=eps)
        return denoising_output

    def get_clamp_coef(self, t=None, sigma=None, x_t=None):
        if sigma is None:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, device=x_t.device)
            t = t.reshape(*t.size(), *((x_t.dim() - t.dim()) * [1]))
            sigma = t / self.num_timesteps
        else:
            assert sigma.dim() == x_t.dim()
        sigma_clamped = sigma.clamp(min=self.sigma_min)
        clamp_coef = sigma / sigma_clamped
        return sigma, sigma_clamped, clamp_coef

    def pred(self, x_t=None, t=None, **kwargs):
        ori_dtype = x_t.dtype
        if hasattr(self.denoising, 'dtype'):
            denoising_dtype = self.denoising.dtype
        else:
            denoising_dtype = next(self.denoising.parameters()).dtype
        x_t = x_t.to(denoising_dtype)
        num_batches = x_t.size(0)
        if t.dim() == 0 or len(t) != num_batches:
            t = t.expand(num_batches)
        if self.flip_model_timesteps:
            t = self.num_timesteps - t
        output = self.denoising(x_t, t, **kwargs)
        if isinstance(output, dict):
            output = {k: v.to(ori_dtype) for k, v in output.items() if isinstance(v, torch.Tensor)}
        else:
            output = output.to(ori_dtype)
        return output

    @force_fp32()
    def loss(self, denoising_output, x_0, noise, x_t, t):
        _, sigma_clamped, clamp_coef = self.get_clamp_coef(t=t, x_t=x_t)
        if self.denoising_mean_mode.upper() == 'U':
            u_t_pred = denoising_output * clamp_coef
        elif self.denoising_mean_mode.upper() == 'X0':
            u_t_pred = (x_t - denoising_output) / sigma_clamped
        else:
            raise AttributeError('Unknown denoising mean output type '
                                 f'[{self.denoising_mean_mode}].')
        u_t = (noise - x_0) * clamp_coef
        loss_kwargs = dict(
            u_t_pred=u_t_pred,
            u_t=u_t,
            timesteps=t)
        return self.flow_loss(loss_kwargs)

    def forward_train(
            self,
            x_0,
            visual_encoder_features=None,
            **kwargs):
        device = get_module_device(self)

        num_batches = x_0.size(0)
        seq_len = x_0.shape[2:].numel()  # h * w or t * h * w

        eps = self.train_cfg.get('eps', 1e-4)
        max_raw_t = self.train_cfg.get('max_raw_t', 1.0)
        min_raw_t = self.train_cfg.get('min_raw_t', 0.0)

        t = self.timestep_sampler(
            num_batches,
            seq_len=seq_len,
            device=device,
            raw_t_range=(min_raw_t, max_raw_t)
        ).clamp(min=eps, max=self.num_timesteps)

        noise = torch.randn_like(x_0)
        x_t, _, _ = self.sample_forward_diffusion(x_0, t, noise)

        use_repa = self.repa_loss is not None and visual_encoder_features is not None
        if use_repa:
            kwargs = kwargs.copy()
            kwargs.update(cache_mode='save', cache_config=self.repa_loss.cache_config)

        denoising_output = self.pred(x_t, t, **kwargs)
        loss_diffusion = self.loss(denoising_output, x_0, noise, x_t, t)
        loss = loss_diffusion
        log_vars = self.flow_loss.log_vars
        log_vars.update(loss_diffusion=float(loss_diffusion.detach()))

        if use_repa:
            pred_features = self.denoising.get_cache_hidden_states()
            self.denoising.clear_cache()
            loss_align = self.repa_loss(pred_features, visual_encoder_features)
            loss = loss + loss_align
            log_vars.update(loss_align=float(loss_align.detach()))
        return loss, log_vars

    def forward_test(
            self, x_0=None, noise=None, guidance_scale=1.0,
            test_cfg_override=dict(), show_pbar=False, sample_callback=None, **kwargs):
        x_t = torch.randn_like(x_0) if noise is None else noise
        num_batches = x_t.size(0)
        ori_dtype = x_t.dtype
        x_t = x_t.float()

        cfg = deepcopy(self.test_cfg)
        cfg.update(test_cfg_override)
        sample_callback = cfg.pop('sample_callback', sample_callback)

        sampler = cfg.get('sampler', 'FlowEulerODE')
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
        if 'flow_shift' in signatures and 'use_flow_sigmas' in signatures:
            sampler_kwargs['prediction_type'] = 'flow_prediction'
            sampler_kwargs['use_flow_sigmas'] = True
            if 'flow_shift' not in sampler_kwargs:
                sampler_kwargs['flow_shift'] = cfg.get('shift', self.timestep_sampler.shift)
        for key in ['max_raw_t', 'min_raw_t']:
            if key in signatures and key in cfg and key not in sampler_kwargs:
                sampler_kwargs[key] = cfg[key]
        sampler = sampler_class(self.num_timesteps, **sampler_kwargs)

        num_timesteps = cfg.get('num_timesteps', None)
        sigmas = cfg.get('sigmas', None)
        guidance_interval = cfg.get('guidance_interval', [0, self.num_timesteps])
        orthogonal_guidance = cfg.get('orthogonal_guidance', 0.0)
        use_guidance = guidance_scale > 1.0
        if use_guidance:
            guidance_scale = x_t.new_tensor(  # to tensor
                [guidance_scale]
            ).expand(num_batches).reshape([num_batches] + [1] * (x_t.dim() - 1))

        set_timesteps_signatures = inspect.signature(sampler.set_timesteps).parameters.keys()
        seq_len = x_t.shape[2:].numel()  # h * w or t * h * w
        if 'seq_len' in set_timesteps_signatures:
            sampler.set_timesteps(num_timesteps, sigmas=sigmas, seq_len=seq_len, device=x_t.device)
        else:
            sampler.set_timesteps(num_timesteps, sigmas=sigmas, device=x_t.device)

        timesteps = sampler.timesteps

        x_t = timesteps[0] / self.num_timesteps * x_t

        if show_pbar:
            pbar = mmcv.ProgressBar(len(timesteps))

        for t in timesteps:
            x_t_input = x_t
            _kwargs = kwargs
            if use_guidance:
                x_t_input = torch.cat([x_t_input, x_t_input], dim=0)

            with default_timers['network time']:
                denoising_output = self.pred(x_t_input, t, **_kwargs)

            if self.denoising_mean_mode.upper() == 'X0':
                _, sigma_clamped, _ = self.get_clamp_coef(t=t, x_t=x_t_input)
                denoising_output = (x_t_input - denoising_output) / sigma_clamped

            if use_guidance:
                _guidance_scale = guidance_scale
                guidance_active = guidance_interval[0] <= t <= guidance_interval[1]
                if not guidance_active:
                    _guidance_scale = torch.ones_like(_guidance_scale)
                mean_neg, mean_pos = denoising_output.chunk(2, dim=0)
                bias = guidance_jit(
                    mean_pos, mean_neg, _guidance_scale,
                    orthogonal_guidance,
                    self.u_to_x_0(mean_pos, x_t, t)
                )
                denoising_output = mean_pos + bias

            if sample_callback is not None:
                callback_outputs = sample_callback(
                    self, dict(x_t=x_t, denoising_output=denoising_output, t=t, sampler=sampler))
                x_t = callback_outputs.get('x_t', x_t)
                denoising_output = callback_outputs.get('denoising_output', denoising_output)

            x_t = sampler.step(denoising_output, t, x_t, return_dict=False)[0]

            if show_pbar:
                pbar.update()

        if show_pbar:
            sys.stdout.write('\n')

        return x_t.to(ori_dtype)

    def forward_u(
            self, x_t=None, t=None, guidance_scale=1.0, test_cfg=dict(), return_cfg_bias=False, **kwargs):
        ori_dtype = x_t.dtype
        x_t = x_t.float()
        num_batches = x_t.size(0)

        orthogonal_guidance = test_cfg.get('orthogonal_guidance', 0.0)
        guidance_interval = test_cfg.get('guidance_interval', [0, self.num_timesteps])

        use_guidance = guidance_scale > 1.0

        x_t_input = x_t
        t_input = t
        if use_guidance:
            guidance_scale = x_t.new_tensor(
                [guidance_scale]
            ).expand(num_batches).reshape([num_batches] + [1] * (x_t.dim() - 1))
            guidance_active = ((t >= guidance_interval[0]) & (t <= guidance_interval[1])).reshape_as(
                guidance_scale)
            guidance_scale = torch.where(
                guidance_active,
                guidance_scale,
                torch.ones_like(guidance_scale)
            )
            x_t_input = torch.cat([x_t_input, x_t_input], dim=0)
            t_input = torch.cat([t_input, t_input], dim=0)

        denoising_output = self.pred(x_t_input, t_input, **kwargs)

        if self.denoising_mean_mode.upper() == 'X0':
            _, sigma_clamped, _ = self.get_clamp_coef(t=t_input, x_t=x_t_input)
            denoising_output = (x_t_input - denoising_output) / sigma_clamped

        if use_guidance:
            mean_neg, mean_pos = denoising_output.chunk(2, dim=0)
            bias = guidance_jit(
                mean_pos, mean_neg, guidance_scale,
                orthogonal_guidance, self.u_to_x_0(mean_pos, x_t, t))
            if return_cfg_bias:
                return mean_pos.to(ori_dtype), bias.to(ori_dtype)
            else:
                return (mean_pos + bias).to(ori_dtype)

        else:
            if return_cfg_bias:
                return denoising_output.to(ori_dtype), torch.zeros_like(denoising_output, dtype=ori_dtype)
            else:
                return denoising_output.to(ori_dtype)

    def forward_x_0(self, x_t=None, t=None, t_dst=None, **kwargs):
        raise NotImplementedError

    def forward(
            self,
            x_0=None,
            return_loss=False,
            return_u=False,
            return_denoising_output=False,
            return_x_0=False,
            **kwargs):
        if return_loss:
            return self.forward_train(x_0, **kwargs)
        elif return_u:
            return self.forward_u(**kwargs)
        elif return_denoising_output:
            return self.pred(**kwargs)
        elif return_x_0:
            return self.forward_x_0(**kwargs)
        else:
            return self.forward_test(x_0, **kwargs)
