# Copyright (c) 2026 Hansheng Chen

import inspect
from copy import deepcopy

import torch

from .builder import MODELS, build_module
from .base_diffusion import BaseDiffusion
from lakonlab.utils import rgetattr, first_tensor_device


def cat_prompt_embed_kwargs(prompt_embed_kwargs_list):
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


class LatentDiffusionTextImageMixin:

    def _prepare_train_minibatch_base_args(self, data):
        if getattr(self, 'train_cached_latents_as_latents_2', False) and 'latents' in data:
            data['latents_2'] = data.pop('latents')

        if 'prompt_embed_kwargs' in data:
            cond_kwargs = data['prompt_embed_kwargs']
        elif 'prompt_kwargs' in data:
            assert self.text_encoder is not None, 'Text encoder must be provided for encoding text to embeddings.'
            cond_kwargs = self.text_encoder(**data['prompt_kwargs'])
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
            cond_kwargs['condition_latents'] = self.patchify(condition_latents)

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

        v = next(iter(cond_kwargs.values()))
        bs = len(v)
        device = first_tensor_device(cond_kwargs)

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
            if 'negative_prompt_embed_kwargs' in data:
                negative_prompt_embed_kwargs = data['negative_prompt_embed_kwargs']
            elif 'negative_prompt_kwargs' in data:
                negative_prompt_embed_kwargs = self.text_encoder(**data['negative_prompt_kwargs'])
            else:
                raise ValueError(
                    'Either `negative_prompt_embed_kwargs` or `negative_prompt_kwargs` should be provided in the '
                    'input data for classifier-free guidance.')
            if 'condition_latents' in prompt_embed_kwargs:
                negative_prompt_embed_kwargs['condition_latents'] = prompt_embed_kwargs['condition_latents']
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

        test_kwargs.update(test_cfg=test_cfg)

        return test_kwargs

    def val_step(self, data, test_cfg_override=dict(), **kwargs):
        if 'prompt_embed_kwargs' in data:
            prompt_embed_kwargs = data['prompt_embed_kwargs']
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

        v = next(iter(prompt_embed_kwargs.values()))
        bs = len(v)
        device = first_tensor_device(prompt_embed_kwargs)

        cfg = deepcopy(self.test_cfg)
        cfg.update(test_cfg_override)
        guidance_scale = cfg.get('guidance_scale', 1.0)
        diffusion = self.diffusion_ema if self.diffusion_use_ema else self.diffusion

        with torch.no_grad():
            use_guidance = guidance_scale != 0.0 and guidance_scale != 1.0
            if use_guidance:
                if 'negative_prompt_embed_kwargs' in data:
                    negative_prompt_embed_kwargs = data['negative_prompt_embed_kwargs']
                elif 'negative_prompt_kwargs' in data:
                    negative_prompt_embed_kwargs = self.text_encoder(**data['negative_prompt_kwargs'])
                else:
                    raise ValueError(
                        'Either `negative_prompt_embed_kwargs` or `negative_prompt_kwargs` should be provided in the '
                        'input data for classifier-free guidance.')
                if self.use_condition_latents:
                    negative_prompt_embed_kwargs['condition_latents'] = prompt_embed_kwargs['condition_latents']
                kwargs = cat_prompt_embed_kwargs(
                    [negative_prompt_embed_kwargs, prompt_embed_kwargs])
            else:
                kwargs = prompt_embed_kwargs.copy()
            distilled_guidance_scale = cfg.get('distilled_guidance_scale', None)
            if distilled_guidance_scale is not None:
                distilled_guidance_scale = torch.full(
                    (bs * 2,) if use_guidance else (bs,),
                    distilled_guidance_scale, dtype=torch.float32, device=device)
                kwargs.update(guidance=distilled_guidance_scale)

            if cfg.get('clamp_denoised', False):
                kwargs['sample_callback'] = self._get_clamp_denoised_callback()

            if 'noise' in data:
                noise = data['noise']
            else:
                latent_size = cfg['latent_size']
                noise = torch.randn((bs, *latent_size), device=device)
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
class LatentDiffusionTextImage(LatentDiffusionTextImageMixin, BaseDiffusion):

    def __init__(self,
                 *args,
                 vae=None,
                 vae_2=None,
                 text_encoder=None,
                 use_condition_latents=False,
                 train_cached_latents_as_latents_2=False,  # True for AsymFlowVR training
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.vae = build_module(vae) if vae is not None else None
        self.vae_2 = build_module(vae_2) if vae_2 is not None else None
        self.text_encoder = build_module(text_encoder) if text_encoder is not None else None
        self.use_condition_latents = use_condition_latents
        self.train_cached_latents_as_latents_2 = train_cached_latents_as_latents_2

    def _prepare_train_minibatch_args(self, data, running_status=None):
        diffusion_args, cond_kwargs, bs, device = \
            self._prepare_train_minibatch_base_args(data)
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

        if 'vae' in parameters and self.vae is not None:
            diffusion_kwargs.update(vae=self.vae)
        if 'vae_2' in parameters and self.vae_2 is not None:
            diffusion_kwargs.update(vae_2=self.vae_2)

        return bs, diffusion_args, diffusion_kwargs
