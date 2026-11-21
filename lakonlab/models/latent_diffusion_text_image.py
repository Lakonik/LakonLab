# Copyright (c) 2026 Hansheng Chen

import inspect
from copy import deepcopy

import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor

from .builder import MODELS, build_module
from .base_diffusion import BaseDiffusion
from .base_diffusion_sd import BaseDiffusionSD
from lakonlab.utils import rgetattr, first_tensor_device, is_warmup


def pad_prompt_embed_kwargs(prompt_embed_kwargs, seq_len):
    # MMH3 needs leading padding to preserve text/media rotary distances and varlen boundaries.
    # Check positional and attention-mask conventions before reusing this for another model.
    prompt_embed_kwargs = prompt_embed_kwargs.copy()
    for k in ('encoder_hidden_states', 'encoder_hidden_states_mask'):
        v = prompt_embed_kwargs[k]
        if v.size(1) != seq_len:
            prompt_embed_kwargs[k] = F.pad(v, (0, 0) * (v.dim() - 2) + (seq_len - v.size(1), 0))
    return prompt_embed_kwargs


def cat_prompt_embed_kwargs(prompt_embed_kwargs_list):
    if 'encoder_hidden_states_mask' in prompt_embed_kwargs_list[0]:
        seq_len = max(pekw['encoder_hidden_states'].size(1) for pekw in prompt_embed_kwargs_list)
        prompt_embed_kwargs_list = [pad_prompt_embed_kwargs(pekw, seq_len) for pekw in prompt_embed_kwargs_list]
    cat_kwargs = dict()
    for k in prompt_embed_kwargs_list[0].keys():
        if isinstance(prompt_embed_kwargs_list[0][k], torch.Tensor):
            cat_kwargs[k] = torch.cat(
                [pekw[k] for pekw in prompt_embed_kwargs_list], dim=0)
        elif isinstance(prompt_embed_kwargs_list[0][k], (list, tuple)):
            cat_kwargs[k] = []
            for pekw in prompt_embed_kwargs_list:
                cat_kwargs[k].extend(pekw[k])
        else:
            raise TypeError(
                f'Unsupported type {type(prompt_embed_kwargs_list[0][k])} '
                f'for prompt_embed_kwargs[{k}] concatenation.')
    return cat_kwargs


def select_prompt_embed_kwargs(keep, prompt_embed_kwargs, negative_prompt_embed_kwargs):
    """Per-sample choice between prompt and negative prompt embed kwargs, for randomly
    unconditioning samples during CFG training. Keys absent from the negative kwargs (e.g.
    `condition_latents`) are passed through unchanged."""
    if ('encoder_hidden_states_mask' in prompt_embed_kwargs
            and 'encoder_hidden_states_mask' in negative_prompt_embed_kwargs):
        seq_len = max(
            prompt_embed_kwargs['encoder_hidden_states'].size(1),
            negative_prompt_embed_kwargs['encoder_hidden_states'].size(1))
        prompt_embed_kwargs = pad_prompt_embed_kwargs(prompt_embed_kwargs, seq_len)
        negative_prompt_embed_kwargs = pad_prompt_embed_kwargs(negative_prompt_embed_kwargs, seq_len)
    select_kwargs = dict()
    for k, v in prompt_embed_kwargs.items():
        if k not in negative_prompt_embed_kwargs:
            select_kwargs[k] = v
        elif isinstance(v, torch.Tensor):
            negative_v = negative_prompt_embed_kwargs[k].to(dtype=v.dtype, device=v.device)
            select_kwargs[k] = torch.where(
                keep.reshape(-1, *((v.dim() - 1) * [1])), v, negative_v)
        elif isinstance(v, (list, tuple)):
            negative_v = negative_prompt_embed_kwargs[k]
            select_kwargs[k] = [
                v[i] if keep[i] else negative_v[i % len(negative_v)] for i in range(len(v))]
        else:
            raise TypeError(
                f'Unsupported type {type(v)} for prompt_embed_kwargs[{k}] selection.')
    return select_kwargs


class LatentDiffusionTextImageMixin:

    def train_minibatch(self, data, loss_scaler=None, running_status=None):
        denoising = rgetattr(self.diffusion, 'denoising')
        if (is_warmup() and self.text_encoder is not None
                and rgetattr(self.text_encoder, 'pad_to_multiple_of', None) is not None
                and rgetattr(denoising, '_compiled_forward', None) is not None):
            max_length = rgetattr(self.text_encoder, 'max_sequence_length')
            multiple = rgetattr(self.text_encoder, 'pad_to_multiple_of')
            lengths = range(multiple, max_length + 1, multiple)
            if not rgetattr(self.text_encoder, 'trim_output', False):
                lengths = [max_length]
            # Use normal forward/backward calls so distributed wrappers and compiled
            # backward are warmed too. The runner supplies no optimizers during warmup.
            for length in lengths:
                warmup_data = data.copy()
                cond_kwargs = self._get_prompt_embed_kwargs(data)
                pad = length - cond_kwargs['encoder_hidden_states'].size(1)
                cond_kwargs['encoder_hidden_states'] = F.pad(
                    cond_kwargs['encoder_hidden_states'],
                    (0, 0) * (cond_kwargs['encoder_hidden_states'].dim() - 2) + (pad, 0))
                cond_kwargs['encoder_hidden_states_mask'] = F.pad(
                    cond_kwargs['encoder_hidden_states_mask'], (pad, 0))
                warmup_data['prompt_embed_kwargs'] = cond_kwargs
                outputs = super().train_minibatch(warmup_data, loss_scaler, running_status)
            return outputs
        return super().train_minibatch(data, loss_scaler, running_status)

    def _get_prompt_embed_kwargs(self, data):
        if 'prompt_embed_kwargs' in data:
            prompt_embed_kwargs = data['prompt_embed_kwargs'].copy()
        elif 'prompt_kwargs' in data:
            assert self.text_encoder is not None, 'Text encoder must be provided for encoding text to embeddings.'
            prompt_embed_kwargs = self.text_encoder(**data['prompt_kwargs'])
        else:
            raise ValueError('Either `prompt_embed_kwargs` or `prompt_kwargs` should be provided in the input data.')

        if self.use_condition_latents and ('condition_latents' in data or 'condition_images' in data):
            if 'condition_latents' in data:
                condition_latents = data['condition_latents']
            else:
                assert self.vae is not None, 'VAE must be provided for encoding images to latents.'
                if hasattr(self.vae, 'dtype'):
                    vae_dtype = self.vae.dtype
                else:
                    vae_dtype = next(self.vae.parameters()).dtype
                kwargs = dict()
                if 'sample_mode' in inspect.signature(rgetattr(self.vae, 'encode')).parameters:
                    kwargs.update(sample_mode='argmax')
                condition_latents = self.vae.encode(
                    (data['condition_images'] * 2 - 1).to(vae_dtype), **kwargs).float()
            prompt_embed_kwargs['condition_latents'] = self.patchify(condition_latents)

        return prompt_embed_kwargs

    def _get_negative_prompt_embed_kwargs(self, data):
        if 'negative_prompt_embed_kwargs' in data:
            return data['negative_prompt_embed_kwargs']
        elif 'negative_prompt_kwargs' in data:
            return self.text_encoder(**data['negative_prompt_kwargs'])
        else:
            raise ValueError(
                'Either `negative_prompt_embed_kwargs` or `negative_prompt_kwargs` should be provided in the '
                'input data for classifier-free guidance.')

    def _prepare_train_minibatch_base_args(self, data):
        if getattr(self, 'train_cached_latents_as_latents_2', False) and 'latents' in data:
            data = data.copy()
            data['latents_2'] = data.pop('latents')

        cond_kwargs = self._get_prompt_embed_kwargs(data)

        if getattr(self, 'train_images_as_latents', False):
            latents = data['images'] * 2 - 1
        elif 'latents' in data:
            latents = data['latents']
        elif 'images' in data:
            assert self.vae is not None, 'VAE must be provided for encoding images to latents.'
            if hasattr(self.vae, 'dtype'):
                vae_dtype = self.vae.dtype
            else:
                vae_dtype = next(self.vae.parameters()).dtype
            latents = self.vae.encode((data['images'] * 2 - 1).to(vae_dtype)).float()
        else:
            raise ValueError('Either `latents` or `images` should be provided in the input data.')

        if 'latents_2' in data:
            latents_2 = data['latents_2']
        elif 'images' in data and getattr(self, 'vae_2', None) is not None:
            if hasattr(self.vae_2, 'dtype'):
                vae_2_dtype = self.vae_2.dtype
            else:
                vae_2_dtype = next(self.vae_2.parameters()).dtype
            latents_2 = self.vae_2.encode((data['images'] * 2 - 1).to(vae_2_dtype)).float()
        else:
            latents_2 = None

        v = next(iter(cond_kwargs.values()))
        bs = len(v)
        device = first_tensor_device(cond_kwargs)

        prob_prompt = self.train_cfg.get('prob_prompt', 1.0)
        if prob_prompt < 1.0:
            cond_kwargs = select_prompt_embed_kwargs(
                torch.rand(bs, device=device) < prob_prompt,
                cond_kwargs,
                self._get_negative_prompt_embed_kwargs(data))

        args = (self.patchify(latents), ) if latents_2 is None else (self.patchify(latents), self.patchify(latents_2))

        return args, cond_kwargs, bs, device

    @staticmethod
    def _prepare_train_minibatch_extra_kwargs(bs, device, train_cfg):
        extra_kwargs = dict()
        distilled_guidance_scale = train_cfg.get('distilled_guidance_scale', None)
        if distilled_guidance_scale is not None:
            distilled_guidance_scale = torch.full(
                (bs,), distilled_guidance_scale, dtype=torch.float32, device=device)
            extra_kwargs.update(guidance=distilled_guidance_scale)
        return extra_kwargs

    def _prepare_train_minibatch_test_kwargs(self, data, prompt_embed_kwargs, bs, device, test_cfg):
        guidance_scale = test_cfg.get('guidance_scale', None)
        use_guidance = (guidance_scale is not None
                        and guidance_scale != 0.0 and guidance_scale != 1.0)

        if use_guidance:
            negative_prompt_embed_kwargs = self._get_negative_prompt_embed_kwargs(data).copy()
            if 'condition_latents' in prompt_embed_kwargs:
                negative_prompt_embed_kwargs['condition_latents'] = prompt_embed_kwargs['condition_latents']
                if test_cfg.get('condition_guidance', False):  # keep the batch uniform but mask the negative condition
                    prompt_embed_kwargs = prompt_embed_kwargs.copy()
                    prompt_embed_kwargs['condition_mask'] = torch.ones(bs, dtype=torch.bool, device=device)
                    negative_prompt_embed_kwargs['condition_mask'] = ~prompt_embed_kwargs['condition_mask']
            test_kwargs = cat_prompt_embed_kwargs([
                negative_prompt_embed_kwargs, prompt_embed_kwargs])
            test_kwargs.update(guidance_scale=guidance_scale)
        else:
            test_kwargs = prompt_embed_kwargs.copy()

        distilled_guidance_scale = test_cfg.get('distilled_guidance_scale', None)
        if distilled_guidance_scale is not None:
            distilled_guidance_scale = torch.full(
                (bs * 2,) if use_guidance else (bs,),
                distilled_guidance_scale, dtype=torch.float32, device=device)
            test_kwargs.update(guidance=distilled_guidance_scale)

        if test_cfg.get('clamp_denoised', False):
            test_kwargs.update(sample_callback=self._get_clamp_denoised_callback())

        test_kwargs.update(test_cfg=test_cfg)

        return test_kwargs

    def _prepare_val_minibatch_args(self, data, cfg, **kwargs):
        kwargs = deepcopy(kwargs)
        diffusion = self.diffusion_ema if self.diffusion_use_ema else self.diffusion

        prompt_embed_kwargs = self._get_prompt_embed_kwargs(data)
        v = next(iter(prompt_embed_kwargs.values()))
        bs = len(v)
        device = first_tensor_device(prompt_embed_kwargs)

        guidance_scale = cfg.get('guidance_scale', 1.0)
        kwargs.update(guidance_scale=guidance_scale)

        use_guidance = guidance_scale != 0.0 and guidance_scale != 1.0
        if use_guidance:
            negative_prompt_embed_kwargs = self._get_negative_prompt_embed_kwargs(data).copy()
            if self.use_condition_latents:
                negative_prompt_embed_kwargs['condition_latents'] = prompt_embed_kwargs['condition_latents']
                if cfg.get('condition_guidance', False):  # keep the batch uniform but mask the negative condition
                    prompt_embed_kwargs = prompt_embed_kwargs.copy()
                    prompt_embed_kwargs['condition_mask'] = torch.ones(bs, dtype=torch.bool, device=device)
                    negative_prompt_embed_kwargs['condition_mask'] = ~prompt_embed_kwargs['condition_mask']
            prompt_embed_kwargs = cat_prompt_embed_kwargs(
                [negative_prompt_embed_kwargs, prompt_embed_kwargs])
        kwargs.update(prompt_embed_kwargs)

        distilled_guidance_scale = cfg.get('distilled_guidance_scale', None)
        if distilled_guidance_scale is not None:
            distilled_guidance_scale = torch.full(
                (bs * 2,) if use_guidance else (bs,),
                distilled_guidance_scale, dtype=torch.float32, device=device)
            kwargs.update(guidance=distilled_guidance_scale)

        if cfg.get('clamp_denoised', False):
            kwargs['sample_callback'] = self._get_clamp_denoised_callback()

        if cfg.get('seed_trajectory', False):
            kwargs['generator'] = [
                torch.Generator(device=device).manual_seed(int(idx) + 1000000)
                for idx in data['ids']]

        parameters = inspect.signature(rgetattr(diffusion, 'forward_test')).parameters
        if 'vae' in parameters and self.vae is not None:
            assert isinstance(self.patch_size, int) and self.patch_size == 1
            kwargs.update(vae=self.vae)
        if 'vae_2' in parameters and self.vae_2 is not None:
            kwargs.update(vae_2=self.vae_2)

        if 'noise' in data:
            noise = data['noise']
        else:
            latent_size = cfg['latent_size']
            noise = randn_tensor((bs, *latent_size), device=device, generator=kwargs.get('generator', None))
        noise = self.patchify(noise)

        return noise, kwargs, bs, device

    @torch.no_grad()
    def val_step(self, data, test_cfg_override=dict(), **kwargs):
        cfg = deepcopy(self.test_cfg)
        cfg.update(test_cfg_override)
        noise, diffusion_kwargs, bs, device = self._prepare_val_minibatch_args(data, cfg, **kwargs)

        diffusion = self.diffusion_ema if self.diffusion_use_ema else self.diffusion
        diffusion_out = diffusion(noise=noise, test_cfg_override=test_cfg_override, **diffusion_kwargs)

        if 'vae' in diffusion_kwargs:
            images = diffusion_out
        else:
            latents_out = self.unpatchify(diffusion_out)
            if hasattr(self.vae, 'dtype'):
                vae_dtype = self.vae.dtype
            else:
                vae_dtype = next(self.vae.parameters()).dtype
            images = self.vae.decode(latents_out.to(vae_dtype))
        out_images = images.float() / 2
        out_images.add_(0.5).clamp_(min=0, max=1)

        return dict(num_samples=bs, pred_imgs=out_images)


@MODELS.register_module()
class LatentDiffusionTextImage(LatentDiffusionTextImageMixin, BaseDiffusion):

    def __init__(self,
                 *args,
                 vae=None,
                 vae_2=None,
                 text_encoder=None,
                 use_condition_latents=False,
                 train_cached_latents_as_latents_2=False,  # True for AsymFlowVR training
                 train_images_as_latents=False,  # True for DWD training (raw pixels as x_0)
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.vae = build_module(vae) if vae is not None else None
        self.vae_2 = build_module(vae_2) if vae_2 is not None else None
        self.text_encoder = build_module(text_encoder) if text_encoder is not None else None
        self.use_condition_latents = use_condition_latents
        self.train_cached_latents_as_latents_2 = train_cached_latents_as_latents_2
        self.train_images_as_latents = train_images_as_latents

    def _prepare_train_minibatch_args(self, data, running_status=None):
        diffusion_args, cond_kwargs, bs, device = self._prepare_train_minibatch_base_args(data)
        diffusion_kwargs = cond_kwargs.copy()
        diffusion_kwargs.update(
            self._prepare_train_minibatch_extra_kwargs(bs, device, self.train_cfg))

        parameters = inspect.signature(rgetattr(self.diffusion, 'forward_train')).parameters
        if 'running_status' in parameters:
            diffusion_kwargs['running_status'] = running_status

        if 'teacher' in parameters and self.teacher is not None:
            diffusion_kwargs.update(teacher=self.teacher)

            if 'teacher_kwargs' in parameters:
                teacher_kwargs = self._prepare_train_minibatch_test_kwargs(
                    data, cond_kwargs, bs, device, self.train_cfg.get('teacher_test_cfg', dict()))
                diffusion_kwargs.update(teacher_kwargs=teacher_kwargs)

            if 'teacher_kwargs_2' in parameters:
                teacher_kwargs_2 = self._prepare_train_minibatch_test_kwargs(
                    data, cond_kwargs, bs, device, self.train_cfg.get('teacher_test_cfg_2', dict()))
                diffusion_kwargs.update(teacher_kwargs_2=teacher_kwargs_2)

        if 'teacher_2' in parameters and self.teacher_2 is not None:
            diffusion_kwargs.update(teacher_2=self.teacher_2)

        if 'vae' in parameters and self.vae is not None:
            diffusion_kwargs.update(vae=self.vae)
        if 'vae_2' in parameters and self.vae_2 is not None:
            diffusion_kwargs.update(vae_2=self.vae_2)

        return bs, diffusion_args, diffusion_kwargs


@MODELS.register_module()
class LatentDiffusionTextImageSD(LatentDiffusionTextImageMixin, BaseDiffusionSD):

    def __init__(self,
                 *args,
                 vae=None,
                 text_encoder=None,
                 use_condition_latents=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.vae = build_module(vae) if vae is not None else None
        self.text_encoder = build_module(text_encoder) if text_encoder is not None else None
        self.use_condition_latents = use_condition_latents

    def _prepare_train_minibatch_args(self, data, running_status=None):
        diffusion_args, cond_kwargs, bs, device = self._prepare_train_minibatch_base_args(data)
        diffusion_kwargs = cond_kwargs.copy()
        diffusion_kwargs.update(
            self._prepare_train_minibatch_extra_kwargs(bs, device, self.train_cfg))

        parameters = inspect.signature(rgetattr(self.diffusion, 'forward_train')).parameters
        if 'running_status' in parameters:
            diffusion_kwargs['running_status'] = running_status

        assert 'teacher' in parameters and self.teacher is not None
        diffusion_kwargs.update(teacher=self.teacher)

        if 'teacher_kwargs' in parameters:
            teacher_kwargs = self._prepare_train_minibatch_test_kwargs(
                data, cond_kwargs, bs, device, self.train_cfg.get('teacher_test_cfg', dict()))
            diffusion_kwargs.update(teacher_kwargs=teacher_kwargs)

        if 'teacher_kwargs_2' in parameters:
            teacher_kwargs_2 = self._prepare_train_minibatch_test_kwargs(
                data, cond_kwargs, bs, device, self.train_cfg.get('teacher_test_cfg_2', dict()))
            diffusion_kwargs.update(teacher_kwargs_2=teacher_kwargs_2)

        assert 'critic' in parameters and 'critic_kwargs' in parameters and self.critic is not None
        critic_kwargs = self._prepare_train_minibatch_test_kwargs(
            data, cond_kwargs, bs, device, self.train_cfg.get('critic_test_cfg', dict()))
        diffusion_kwargs.update(
            critic=self.critic,
            critic_kwargs=critic_kwargs)

        if 'vae' in parameters and self.vae is not None:
            diffusion_kwargs.update(vae=self.vae)
        if 'vae_2' in parameters and self.vae_2 is not None:
            diffusion_kwargs.update(vae_2=self.vae_2)

        return bs, diffusion_args, diffusion_kwargs

    def _prepare_train_minibatch_args_for_critic(self, data, running_status=None):
        critic_args, cond_kwargs, bs, device = \
            self._prepare_train_minibatch_base_args(data)

        diffusion_kwargs = cond_kwargs.copy()
        diffusion_kwargs.update(
            self._prepare_train_minibatch_extra_kwargs(bs, device, self.train_cfg))

        critic_kwargs = cond_kwargs.copy()
        critic_kwargs.update(
            self._prepare_train_minibatch_extra_kwargs(bs, device, self.train_cfg.get('critic_train_cfg', dict())))

        parameters = inspect.signature(rgetattr(self.critic, 'forward_train')).parameters
        if 'running_status' in parameters:
            critic_kwargs['running_status'] = running_status

        critic_kwargs.update(
            generator=self.diffusion,
            generator_kwargs=diffusion_kwargs)

        return bs, critic_args, critic_kwargs
