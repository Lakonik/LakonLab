# Copyright (c) 2026 Hansheng Chen

from copy import deepcopy

import tqdm
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel, FSDPModule
from diffusers.utils.torch_utils import randn_tensor

from ..builder import MODULES, build_module
from .gaussian_flow import GaussianFlow, guidance_jit


@MODULES.register_module()
class MultiGaussianFlow(nn.Module):
    """A joint velocity denoiser with shared raw time and per-modality Gaussian flows.

    ``modalities`` maps names to GaussianFlow sampler/loss/sigma_min settings.
    Losses are summed; each flow_loss's rescale_cfg controls its relative weight.
    Modality overrides belong to train_cfg.modalities and test_cfg.modalities.
    Each flow receives its resolved training and test settings at construction.
    """

    flow_class = GaussianFlow

    def __init__(self,
                 denoising,
                 modalities,
                 num_timesteps=1000,
                 flip_model_timesteps=False,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.denoising = build_module(denoising) if isinstance(denoising, dict) else denoising
        self.num_timesteps = num_timesteps
        self.flip_model_timesteps = flip_model_timesteps
        self.train_cfg = deepcopy(train_cfg) if train_cfg is not None else dict()
        self.test_cfg = deepcopy(test_cfg) if test_cfg is not None else dict()

        train_cfg = self.train_cfg.copy()
        modality_train_cfg = train_cfg.pop('modalities', {})
        test_cfg = self.test_cfg.copy()
        modality_test_cfg = test_cfg.pop('modalities', {})
        self.flows = nn.ModuleDict({
            name: self.flow_class(
                num_timesteps=num_timesteps,
                flip_model_timesteps=flip_model_timesteps,
                train_cfg=dict(train_cfg, **modality_train_cfg.get(name, {})),
                test_cfg=dict(test_cfg, **modality_test_cfg.get(name, {})),
                **config)
            for name, config in modalities.items()})
        self.check_timestep_samplers()
        for module in self.flows.modules():
            module.lakonlab_no_tie = True

    def check_timestep_samplers(self):
        first_name, first_flow = next(iter(self.flows.items()))
        first_sampler = first_flow.timestep_sampler
        for name, flow in self.flows.items():
            sampler = flow.timestep_sampler
            if (first_sampler.logit_normal_enable != sampler.logit_normal_enable
                    or tuple(first_sampler.raw_t_bleed) != tuple(sampler.raw_t_bleed)
                    or (first_sampler.logit_normal_enable and (
                        first_sampler.logit_normal_mean != sampler.logit_normal_mean
                        or first_sampler.logit_normal_std != sampler.logit_normal_std))):
                raise ValueError(
                    f'Modalities {first_name!r} and {name!r} must use the same raw-time sampling distribution.')

    def pred(self, x_t, t, **kwargs):
        ori_dtype = {name: value.dtype for name, value in x_t.items()}
        denoising_fsdp = next((
            module for module in self.denoising.modules()
            if isinstance(module, (FullyShardedDataParallel, FSDPModule))), None)
        if isinstance(denoising_fsdp, FSDPModule):
            mp_policy = denoising_fsdp._get_fsdp_state()._mp_policy
        else:
            mp_policy = getattr(denoising_fsdp, 'mixed_precision', None)
        if mp_policy is not None and mp_policy.param_dtype is not None:
            denoising_dtype = mp_policy.param_dtype
        elif hasattr(self.denoising, 'dtype'):
            denoising_dtype = self.denoising.dtype
        else:
            denoising_dtype = next(self.denoising.parameters()).dtype
        x_t = {name: value.to(denoising_dtype) for name, value in x_t.items()}
        t = {name: t[name].expand(value.size(0)) for name, value in x_t.items()}
        if self.flip_model_timesteps:
            t = {name: self.num_timesteps - value for name, value in t.items()}

        output = self.denoising(x_t, t, **kwargs)

        extra = None
        if isinstance(output, tuple):
            output, extra = output
        output = {
            name: {k: v.to(dtype) for k, v in output[name].items() if isinstance(v, torch.Tensor)}
            if isinstance(output[name], dict) else output[name].to(dtype)
            for name, dtype in ori_dtype.items()}
        return output if extra is None else (output, extra)

    def forward_u(
            self, x_t, t, guidance_scale=1.0, test_cfg=dict(), sample_callback=None, **kwargs):
        ori_dtype = {name: value.dtype for name, value in x_t.items()}
        x_t = {name: value.float() for name, value in x_t.items()}
        t = {name: t[name].expand(value.size(0)) for name, value in x_t.items()}

        test_cfg = deepcopy(test_cfg)
        modality_cfg = test_cfg.pop('modalities', {})
        use_guidance = guidance_scale > 1.0

        x_t_input = x_t
        t_input = t
        if use_guidance:
            x_t_input = {name: torch.cat([value, value], dim=0) for name, value in x_t.items()}
            t_input = {name: torch.cat([value, value], dim=0) for name, value in t.items()}

        denoising_output = self.pred(x_t_input, t_input, **kwargs)

        for name, flow in self.flows.items():
            if flow.denoising_mean_mode.upper() == 'X0':
                denoising_output[name] = flow.x_0_to_u(
                    denoising_output[name], x_t_input[name], t=t_input[name], eps=flow.sigma_min)

            if use_guidance:
                num_batches = x_t[name].size(0)
                flow_cfg = dict(test_cfg, **modality_cfg.get(name, {}))
                orthogonal_guidance = flow_cfg.get('orthogonal_guidance', 0.0)
                guidance_interval = flow_cfg.get('guidance_interval', [0, self.num_timesteps])

                _guidance_scale = x_t[name].new_tensor(
                    [guidance_scale]
                ).expand(num_batches).reshape([num_batches] + [1] * (x_t[name].dim() - 1))
                guidance_active = ((t[name] >= guidance_interval[0]) & (
                    t[name] <= guidance_interval[1])).reshape_as(_guidance_scale)
                _guidance_scale = torch.where(
                    guidance_active,
                    _guidance_scale,
                    torch.ones_like(_guidance_scale)
                )

                mean_neg, mean_pos = denoising_output[name].chunk(2, dim=0)
                bias = guidance_jit(
                    mean_pos, mean_neg, _guidance_scale,
                    orthogonal_guidance,
                    flow.u_to_x_0(mean_pos, x_t[name], t[name]) if orthogonal_guidance != 0.0 else None
                )
                denoising_output[name] = mean_pos + bias

        if sample_callback is not None:
            callback_outputs = sample_callback(
                self, dict(x_t=x_t, u=denoising_output, t=t))
            denoising_output = callback_outputs.get('u', denoising_output)

        return {name: denoising_output[name].to(dtype) for name, dtype in ori_dtype.items()}

    def forward_train(self, x_0, **kwargs):
        max_raw_t = self.train_cfg.get('max_raw_t', 1.0)
        min_raw_t = self.train_cfg.get('min_raw_t', 0.0)

        name, value = next(iter(x_0.items()))
        num_batches = value.size(0)
        raw_t = self.flows[name].timestep_sampler(
            num_batches, warp_t=False, scale_t=False, device=value.device,
            raw_t_range=(min_raw_t, max_raw_t))

        t, noise, x_t = {}, {}, {}
        for name, value in x_0.items():
            seq_len = value.shape[2:].numel()
            flow = self.flows[name]
            t[name] = (flow.timestep_sampler.warp_t(
                raw_t, seq_len=seq_len) * self.num_timesteps
            ).clamp(min=flow.train_cfg.get('eps', 1e-4), max=self.num_timesteps)
            noise[name] = torch.randn_like(value)
            x_t[name] = flow.sample_forward_diffusion(value, t[name], noise[name])[0]

        denoising_output = self.pred(x_t, t, **kwargs)

        loss, log_vars = 0, {}
        for name, flow in self.flows.items():
            modality_loss = flow.loss(denoising_output[name], x_0[name], noise[name], x_t[name], t[name])
            loss = loss + modality_loss
            log_vars.update({f'{key}_{name}': value for key, value in flow.flow_loss.log_vars.items()})
            log_vars[f'loss_diffusion_{name}'] = modality_loss.detach()
        log_vars['loss_diffusion'] = loss.detach()

        return loss, log_vars

    def forward_test(
            self, x_0=None, noise=None, guidance_scale=1.0, test_cfg_override=dict(),
            show_pbar=False, sample_callback=None, stop_at_step=None, generator=None, tqdm_cls=tqdm.tqdm, **kwargs):
        x_t = {
            name: randn_tensor(value.shape, dtype=value.dtype, device=value.device, generator=generator)
            for name, value in x_0.items()} if noise is None else noise
        ori_dtype = {name: value.dtype for name, value in x_t.items()}
        x_t = {name: value.float() for name, value in x_t.items()}

        test_cfg_override = deepcopy(test_cfg_override)
        modality_cfg = test_cfg_override.pop('modalities', {})
        cfg = {
            name: {**flow.test_cfg, **test_cfg_override, **modality_cfg.get(name, {})}
            for name, flow in self.flows.items()}

        use_guidance = guidance_scale > 1.0

        samplers = {
            name: self.flows[name]._build_sampler(value, cfg[name])
            for name, value in x_t.items()}
        num_steps = len(next(iter(samplers.values())).timesteps)
        if any(len(sampler.timesteps) != num_steps for sampler in samplers.values()):
            raise ValueError('Joint sampling requires the same number of steps for every modality.')
        timesteps = [dict(zip(samplers, t)) for t in zip(*(sampler.timesteps for sampler in samplers.values()))]

        pbar = tqdm_cls(None, total=len(timesteps)) if show_pbar else tqdm.tqdm(disable=True)

        step_id = -1
        for step_id, t in enumerate(timesteps[:stop_at_step]):
            x_t_input = x_t
            if use_guidance:
                x_t_input = {name: torch.cat([value, value], dim=0) for name, value in x_t.items()}

            denoising_output = self.pred(x_t_input, t, **kwargs)

            for name, flow in self.flows.items():
                if flow.denoising_mean_mode.upper() == 'X0':
                    denoising_output[name] = flow.x_0_to_u(
                        denoising_output[name], x_t_input[name], t=t[name], eps=flow.sigma_min)

                if use_guidance:
                    num_batches = x_t[name].size(0)
                    flow_cfg = cfg[name]
                    orthogonal_guidance = flow_cfg.get('orthogonal_guidance', 0.0)
                    guidance_interval = flow_cfg.get('guidance_interval', [0, self.num_timesteps])

                    _guidance_scale = x_t[name].new_tensor(
                        [guidance_scale]
                    ).expand(num_batches).reshape([num_batches] + [1] * (x_t[name].dim() - 1))
                    guidance_active = guidance_interval[0] <= t[name] <= guidance_interval[1]
                    if not guidance_active:
                        _guidance_scale = torch.ones_like(_guidance_scale)
                    mean_neg, mean_pos = denoising_output[name].chunk(2, dim=0)
                    bias = guidance_jit(
                        mean_pos, mean_neg, _guidance_scale,
                        orthogonal_guidance,
                        flow.u_to_x_0(mean_pos, x_t[name], t[name]) if orthogonal_guidance != 0.0 else None
                    )
                    denoising_output[name] = mean_pos + bias

            if sample_callback is not None:
                callback_outputs = sample_callback(self, dict(x_t=x_t, u=denoising_output, t=t))
                x_t = callback_outputs.get('x_t', x_t)
                denoising_output = callback_outputs.get('u', denoising_output)

            x_t = {
                name: sampler.step(denoising_output[name], t[name], x_t[name], return_dict=False)[0]
                for name, sampler in samplers.items()}

            pbar.update()

        if stop_at_step is not None:
            return x_t, timesteps[step_id + 1]

        return {name: x_t[name].to(dtype) for name, dtype in ori_dtype.items()}

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
        if return_u:
            return self.forward_u(**kwargs)
        if return_denoising_output:
            return self.pred(**kwargs)
        if return_x_0:
            return self.forward_x_0(**kwargs)
        else:
            return self.forward_test(x_0, **kwargs)
