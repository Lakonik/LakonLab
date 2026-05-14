# Copyright (c) 2026 Hansheng Chen

import inspect
from copy import deepcopy

import torch

from .builder import MODELS, build_module
from .base_diffusion import BaseDiffusion
from lakonlab.utils import rgetattr


class LatentDiffusionClassImageMixin:

    def _prepare_train_minibatch_base_args(self, data):
        if 'latents' in data:
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

        args = (self.patchify(latents), ) if latents_2 is None else (self.patchify(latents), self.patchify(latents_2))

        labels = data['labels']
        prob_class = self.train_cfg.get('prob_class', 1.0)
        if prob_class < 1.0:
            labels = torch.where(
                torch.rand_like(labels, dtype=torch.float32) < prob_class,
                labels, data['negative_labels'])
        cond_kwargs = dict(class_labels=labels)

        bs = labels.size(0)

        return args, cond_kwargs, bs

    @staticmethod
    def _prepare_train_minibatch_test_kwargs(data, test_cfg):
        guidance_scale = test_cfg.get('guidance_scale', None)
        use_guidance = (guidance_scale is not None
                        and guidance_scale != 0.0 and guidance_scale != 1.0)

        labels = data['labels']
        if use_guidance:
            test_kwargs = dict(class_labels=torch.cat([data['negative_labels'], labels], dim=0))
            test_kwargs.update(guidance_scale=guidance_scale)
        else:
            test_kwargs = dict(class_labels=labels)

        test_kwargs.update(test_cfg=test_cfg)

        return test_kwargs

    def val_step(self, data, test_cfg_override=dict(), **kwargs):
        bs = len(data['labels'])
        cfg = deepcopy(self.test_cfg)
        cfg.update(test_cfg_override)
        guidance_scale = cfg.get('guidance_scale', 1.0)
        diffusion = self.diffusion_ema if self.diffusion_use_ema else self.diffusion

        with torch.no_grad():
            class_labels = data['labels']

            if guidance_scale != 0.0 and guidance_scale != 1.0:
                class_labels = torch.cat([data['negative_labels'], class_labels], dim=0)

            kwargs = dict(class_labels=class_labels)

            if cfg.get('clamp_denoised', False):
                kwargs['sample_callback'] = self._get_clamp_denoised_callback()

            if 'noise' in data:
                noise = data['noise']
            else:
                latent_size = cfg['latent_size']
                noise = torch.randn((bs, *latent_size), device=data['labels'].device)
            noise = self.patchify(noise)
            latents_out = diffusion(
                noise=noise,
                guidance_scale=guidance_scale,
                test_cfg_override=test_cfg_override,
                **kwargs)
            latents_out = self.unpatchify(latents_out)

            if hasattr(self.vae, 'dtype'):
                vae_dtype = self.vae.dtype
            else:
                vae_dtype = next(self.vae.parameters()).dtype
            latents_out = latents_out.to(vae_dtype)

            out_images = (self.vae.decode(latents_out).float() / 2 + 0.5).clamp(min=0, max=1)

            return dict(num_samples=bs, pred_imgs=out_images)


@MODELS.register_module()
class LatentDiffusionClassImage(LatentDiffusionClassImageMixin, BaseDiffusion):

    def __init__(self,
                 *args,
                 vae=None,
                 vae_2=None,
                 visual_encoder=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.vae = build_module(vae) if vae is not None else None
        self.vae_2 = build_module(vae_2) if vae_2 is not None else None
        self.visual_encoder = build_module(visual_encoder) if visual_encoder is not None else None

    def _prepare_train_minibatch_args(self, data, running_status=None):
        diffusion_args, diffusion_kwargs, bs = self._prepare_train_minibatch_base_args(data)

        parameters = inspect.signature(rgetattr(self.diffusion, 'forward_train')).parameters
        if 'running_status' in parameters:
            diffusion_kwargs['running_status'] = running_status

        if 'teacher' in parameters and self.teacher is not None:
            diffusion_kwargs.update(teacher=self.teacher)

            if 'teacher_kwargs' in parameters:
                teacher_kwargs = self._prepare_train_minibatch_test_kwargs(
                    data, self.train_cfg.get('teacher_test_cfg', dict()))
                diffusion_kwargs.update(teacher_kwargs=teacher_kwargs)

            if 'teacher_kwargs_2' in parameters:
                teacher_kwargs_2 = self._prepare_train_minibatch_test_kwargs(
                    data, self.train_cfg.get('teacher_test_cfg_2', dict()))
                diffusion_kwargs.update(teacher_kwargs_2=teacher_kwargs_2)

        if 'vae' in parameters and self.vae is not None:
            diffusion_kwargs.update(vae=self.vae)
        if 'vae_2' in parameters and self.vae_2 is not None:
            diffusion_kwargs.update(vae_2=self.vae_2)

        if 'visual_encoder_features' in parameters:
            if 'visual_encoder_features' in data:
                diffusion_kwargs.update(visual_encoder_features=data['visual_encoder_features'])
            elif 'images' in data and self.visual_encoder is not None:
                # visual encoder uses images in [0, 1] range, unlike VAE
                diffusion_kwargs.update(visual_encoder_features=self.visual_encoder(data['images']))

        return bs, diffusion_args, diffusion_kwargs
