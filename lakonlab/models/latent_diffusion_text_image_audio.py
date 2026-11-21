# Copyright (c) 2026 Hansheng Chen

import inspect
from copy import deepcopy

import torch
from diffusers.utils.torch_utils import randn_tensor

from .builder import MODELS, build_module
from .latent_diffusion_text_image import LatentDiffusionTextImage, LatentDiffusionTextImageSD
from lakonlab.utils import rgetattr


class LatentDiffusionTextImageAudioMixin:

    def __init__(self,
                 *args,
                 audio_vae=None,
                 audio_vae_2=None,
                 train_audio_as_latents=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_vae = build_module(audio_vae) if audio_vae is not None else None
        self.audio_vae_2 = build_module(audio_vae_2) if audio_vae_2 is not None else None
        self.train_audio_as_latents = train_audio_as_latents

    def _prepare_train_minibatch_base_args(self, data):
        args, cond_kwargs, bs, device = super()._prepare_train_minibatch_base_args(data)

        if getattr(self, 'train_cached_latents_as_latents_2', False) and 'audio_latents' in data:
            data = data.copy()
            data['audio_latents_2'] = data.pop('audio_latents')

        if self.train_audio_as_latents:
            audio = data['audio']
        elif 'audio_latents' in data:
            audio = data['audio_latents']
        elif 'audio' in data:
            assert self.audio_vae is not None, 'Audio VAE must be provided for encoding audio to latents.'
            if hasattr(self.audio_vae, 'dtype'):
                dtype = self.audio_vae.dtype
            else:
                dtype = next(self.audio_vae.parameters()).dtype
            audio = self.audio_vae.encode(data['audio'].to(dtype)).float()
        else:
            raise ValueError('Either `audio_latents` or `audio` should be provided in the input data.')

        if 'audio_latents_2' in data:
            audio_2 = data['audio_latents_2']
        elif 'audio' in data and self.audio_vae_2 is not None:
            if hasattr(self.audio_vae_2, 'dtype'):
                dtype = self.audio_vae_2.dtype
            else:
                dtype = next(self.audio_vae_2.parameters()).dtype
            audio_2 = self.audio_vae_2.encode(data['audio'].to(dtype)).float()
        else:
            audio_2 = None

        latents = dict(image=args[0], audio=audio)
        if len(args) == 2 or audio_2 is not None:
            latents_2 = dict(image=args[-1], audio=audio if audio_2 is None else audio_2)
            args = (latents, latents_2)
        else:
            args = (latents,)

        return args, cond_kwargs, bs, device

    def _prepare_train_minibatch_args(self, data, running_status=None):
        bs, diffusion_args, diffusion_kwargs = super()._prepare_train_minibatch_args(data, running_status)
        parameters = inspect.signature(rgetattr(self.diffusion, 'forward_train')).parameters
        if 'vae' in parameters:
            diffusion_kwargs.update(vae=dict(image=self.vae, audio=self.audio_vae))
        if 'vae_2' in parameters:
            diffusion_kwargs.update(vae_2=dict(image=self.vae_2, audio=self.audio_vae_2))
        return bs, diffusion_args, diffusion_kwargs

    def _prepare_val_minibatch_args(self, data, cfg, **kwargs):
        noise, diffusion_kwargs, bs, device = super()._prepare_val_minibatch_args(data, cfg, **kwargs)

        diffusion = self.diffusion_ema if self.diffusion_use_ema else self.diffusion
        parameters = inspect.signature(rgetattr(diffusion, 'forward_test')).parameters
        if 'vae' in parameters:
            diffusion_kwargs.update(vae=dict(image=self.vae, audio=self.audio_vae))
        if 'vae_2' in parameters:
            diffusion_kwargs.update(vae_2=dict(image=self.vae_2, audio=self.audio_vae_2))

        if 'audio_noise' in data:
            audio_noise = data['audio_noise']
        else:
            audio_latent_size = cfg['audio_latent_size']
            audio_noise = randn_tensor(
                (bs, *audio_latent_size), device=device, generator=diffusion_kwargs.get('generator', None))
        noise = dict(image=noise, audio=audio_noise)

        return noise, diffusion_kwargs, bs, device

    @torch.no_grad()
    def val_step(self, data, test_cfg_override=dict(), **kwargs):
        cfg = deepcopy(self.test_cfg)
        cfg.update(test_cfg_override)
        noise, diffusion_kwargs, bs, device = self._prepare_val_minibatch_args(data, cfg, **kwargs)

        diffusion = self.diffusion_ema if self.diffusion_use_ema else self.diffusion
        diffusion_out = diffusion(noise=noise, test_cfg_override=test_cfg_override, **diffusion_kwargs)

        if 'vae' in diffusion_kwargs:
            images = diffusion_out['image']
        else:
            latents_out = self.unpatchify(diffusion_out['image'])
            if hasattr(self.vae, 'dtype'):
                vae_dtype = self.vae.dtype
            else:
                vae_dtype = next(self.vae.parameters()).dtype
            images = self.vae.decode(latents_out.to(vae_dtype))
        out_images = images.float() / 2
        out_images.add_(0.5).clamp_(min=0, max=1)

        if 'vae' in diffusion_kwargs:
            audio = diffusion_out['audio']
        else:
            latents_out = diffusion_out['audio']
            if hasattr(self.audio_vae, 'dtype'):
                vae_dtype = self.audio_vae.dtype
            else:
                vae_dtype = next(self.audio_vae.parameters()).dtype
            audio = self.audio_vae.decode(latents_out.to(vae_dtype))
        out_audio = audio.float()

        return dict(num_samples=bs, pred_imgs=out_images, pred_audio=out_audio)


@MODELS.register_module()
class LatentDiffusionTextImageAudio(LatentDiffusionTextImageAudioMixin, LatentDiffusionTextImage):
    pass


@MODELS.register_module()
class LatentDiffusionTextImageAudioSD(LatentDiffusionTextImageAudioMixin, LatentDiffusionTextImageSD):
    pass
