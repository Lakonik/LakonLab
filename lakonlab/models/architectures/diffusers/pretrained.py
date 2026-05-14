import logging
from contextlib import contextmanager
from threading import Lock
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models import (
    AutoencoderKL, AutoencoderKLQwenImage, AutoencoderKLFlux2, AutoencoderRAE
)
from diffusers.pipelines import (
    FluxPipeline, QwenImagePipeline, StableDiffusion3Pipeline, Flux2Pipeline,
    ZImagePipeline, Flux2KleinPipeline
)

from ...builder import MODULES
from lakonlab.utils.io_utils import hf_model_loader

# Suppress truncation warnings from transformers and diffusers
for name in (
        'transformers.tokenization_utils_base',
        'transformers.tokenization_utils',
        'transformers.tokenization_utils_fast'):
    logging.getLogger(name).setLevel(logging.ERROR)

for name, logger in logging.root.manager.loggerDict.items():
    if isinstance(logger, logging.Logger) and (name.startswith('diffusers.pipelines.')):
        logger.setLevel(logging.ERROR)


_DINOV2_WITH_REGISTERS_INIT_PATCH_LOCK = Lock()


@MODULES.register_module()
class PretrainedVAE(nn.Module):
    def __init__(self,
                 model_name_or_path=None,
                 del_encoder=False,
                 del_decoder=False,
                 use_slicing=False,
                 max_bs=64,
                 freeze=True,
                 eval_mode=True,
                 torch_dtype='float32',
                 **kwargs):
        super().__init__()
        if torch_dtype is not None:
            kwargs.update(torch_dtype=getattr(torch, torch_dtype))
        self.vae = hf_model_loader(AutoencoderKL, model_name_or_path, **kwargs)
        if del_encoder:
            del self.vae.encoder
        if del_decoder:
            del self.vae.decoder
        if use_slicing:
            self.vae.enable_slicing()
        self.max_bs = max_bs
        self.freeze = freeze
        self.eval_mode = eval_mode
        if self.freeze:
            self.requires_grad_(False)
        if self.eval_mode:
            self.eval()
        self.vae.set_use_memory_efficient_attention_xformers(
            not hasattr(torch.nn.functional, 'scaled_dot_product_attention'))

    def train(self, mode=True):
        mode = mode and (not self.eval_mode)
        return super().train(mode)

    def forward(self, *args, **kwargs):
        return self.vae(*args, return_dict=False, **kwargs)[0]

    def _encode(self, img):
        bs = img.size(0)
        if bs <= self.max_bs:
            return self.vae.encode(img).latent_dist.sample()
        else:
            latents = []
            for i in range(0, img.size(0), self.max_bs):
                img_chunk = img[i:min(i + self.max_bs, bs)]
                latents_chunk = self.vae.encode(img_chunk).latent_dist.sample()
                latents.append(latents_chunk)
            return torch.cat(latents, dim=0)

    def _decode(self, code):
        bs = code.size(0)
        if bs <= self.max_bs:
            return self.vae.decode(code, return_dict=False)[0]
        else:
            imgs = []
            for i in range(0, code.size(0), self.max_bs):
                code_chunk = code[i:min(i + self.max_bs, bs)]
                img_chunk = self.vae.decode(code_chunk, return_dict=False)[0]
                imgs.append(img_chunk)
            return torch.cat(imgs, dim=0)

    def encode(self, img):
        if self.vae.config.latents_mean is not None and self.vae.config.latents_std is not None:
            device = img.device
            dtype = img.dtype
            latents_mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=dtype)[:, None, None]
            latents_std = torch.tensor(self.vae.config.latents_std, device=device, dtype=dtype)[:, None, None]
            return (self._encode(img) - latents_mean) / latents_std
        else:
            scaling_factor = self.vae.config.scaling_factor
            shift_factor = self.vae.config.shift_factor
            if scaling_factor is None:
                scaling_factor = 1.0
            if shift_factor is None:
                shift_factor = 0.0
            return (self._encode(img) - shift_factor) * scaling_factor

    def decode(self, code):
        if self.vae.config.latents_mean is not None and self.vae.config.latents_std is not None:
            device = code.device
            dtype = code.dtype
            latents_mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=dtype)[:, None, None]
            latents_std = torch.tensor(self.vae.config.latents_std, device=device, dtype=dtype)[:, None, None]
            return self._decode(code * latents_std + latents_mean)
        else:
            scaling_factor = self.vae.config.scaling_factor
            shift_factor = self.vae.config.shift_factor
            if scaling_factor is None:
                scaling_factor = 1.0
            if shift_factor is None:
                shift_factor = 0.0
            return self._decode(code / scaling_factor + shift_factor)


@MODULES.register_module()
class PretrainedVAEDecoder(PretrainedVAE):
    def __init__(self, **kwargs):
        super().__init__(
            del_encoder=True,
            del_decoder=False,
            **kwargs)

    def forward(self, code):
        return super().decode(code)


@MODULES.register_module()
class PretrainedVAEEncoder(PretrainedVAE):
    def __init__(self, **kwargs):
        super().__init__(
            del_encoder=False,
            del_decoder=True,
            **kwargs)

    def forward(self, img):
        return super().encode(img)


@MODULES.register_module()
class PretrainedVAEQwenImage(nn.Module):
    def __init__(self,
                 model_name_or_path=None,
                 use_slicing=False,
                 freeze=True,
                 eval_mode=True,
                 torch_dtype='float32',
                 **kwargs):
        super().__init__()
        if torch_dtype is not None:
            kwargs.update(torch_dtype=getattr(torch, torch_dtype))
        self.vae = hf_model_loader(AutoencoderKLQwenImage, model_name_or_path, **kwargs)
        if use_slicing:
            self.vae.enable_slicing()
        self.freeze = freeze
        self.eval_mode = eval_mode
        if self.freeze:
            self.requires_grad_(False)
        if self.eval_mode:
            self.eval()

    def train(self, mode=True):
        mode = mode and (not self.eval_mode)
        return super().train(mode)

    def forward(self, *args, **kwargs):
        return self.vae(*args, return_dict=False, **kwargs)[0]

    def encode(self, img, sample_mode='sample'):
        device = img.device
        dtype = img.dtype
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std, device=device, dtype=dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1)
        latent_dist = self.vae.encode(img.unsqueeze(-3)).latent_dist
        if sample_mode == 'sample':
            latents = latent_dist.sample()
        elif sample_mode == 'argmax':
            latents = latent_dist.mode()
        else:
            raise ValueError(f'Invalid sample_mode: {sample_mode}')
        return ((latents - latents_mean) / latents_std).squeeze(-3)

    def decode(self, code):
        device = code.device
        dtype = code.dtype
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std, device=device, dtype=dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1)
        return self.vae.decode(code.unsqueeze(-3) * latents_std + latents_mean, return_dict=False)[0].squeeze(-3)


@MODULES.register_module()
class PretrainedVAEFlux2(nn.Module):
    def __init__(self,
                 model_name_or_path=None,
                 use_slicing=False,
                 freeze=True,
                 eval_mode=True,
                 torch_dtype='float32',
                 **kwargs):
        super().__init__()
        if torch_dtype is not None:
            kwargs.update(torch_dtype=getattr(torch, torch_dtype))
        self.vae = hf_model_loader(AutoencoderKLFlux2, model_name_or_path, **kwargs)
        if use_slicing:
            self.vae.enable_slicing()
        self.freeze = freeze
        self.eval_mode = eval_mode
        if self.freeze:
            self.requires_grad_(False)
        if self.eval_mode:
            self.eval()

    def train(self, mode=True):
        mode = mode and (not self.eval_mode)
        return super().train(mode)

    def forward(self, *args, **kwargs):
        return self.vae(*args, return_dict=False, **kwargs)[0]

    @staticmethod
    def _patchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
        return latents

    @staticmethod
    def _unpatchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
        return latents

    def encode(self, img, sample_mode='sample'):
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(
            device=img.device, dtype=img.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            device=img.device, dtype=img.dtype)
        latent_dist = self.vae.encode(img).latent_dist
        if sample_mode == 'sample':
            latents = latent_dist.sample()
        elif sample_mode == 'argmax':
            latents = latent_dist.mode()
        else:
            raise ValueError(f'Invalid sample_mode: {sample_mode}')
        latents = (self._patchify_latents(latents) - latents_bn_mean) / latents_bn_std
        return self._unpatchify_latents(latents)

    def decode(self, code):
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(
            device=code.device, dtype=code.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            device=code.device, dtype=code.dtype)
        latents = self._patchify_latents(code) * latents_bn_std + latents_bn_mean
        return self.vae.decode(self._unpatchify_latents(latents), return_dict=False)[0]


@MODULES.register_module()
class PretrainedFluxTextEncoder(nn.Module):
    def __init__(self,
                 model_name_or_path='black-forest-labs/FLUX.1-dev',
                 freeze=True,
                 eval_mode=True,
                 torch_dtype='bfloat16',
                 max_sequence_length=512,
                 **kwargs):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.pipeline = hf_model_loader(
            FluxPipeline,
            model_name_or_path,
            scheduler=None,
            vae=None,
            transformer=None,
            image_encoder=None,
            feature_extractor=None,
            torch_dtype=getattr(torch, torch_dtype),
            **kwargs)
        self.text_encoder = self.pipeline.text_encoder
        self.text_encoder_2 = self.pipeline.text_encoder_2
        self.freeze = freeze
        self.eval_mode = eval_mode
        if self.freeze:
            self.requires_grad_(False)
        if self.eval_mode:
            self.eval()

    def train(self, mode=True):
        mode = mode and (not self.eval_mode)
        return super().train(mode)

    def forward(self, prompt, prompt_2=None):
        prompt_embeds, pooled_prompt_embeds, text_ids = self.pipeline.encode_prompt(
            prompt, prompt_2=prompt_2, max_sequence_length=self.max_sequence_length)
        return dict(
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds)


@MODULES.register_module()
class PretrainedQwenImageTextEncoder(nn.Module):
    def __init__(self,
                 model_name_or_path='Qwen/Qwen-Image',
                 freeze=True,
                 eval_mode=True,
                 torch_dtype='bfloat16',
                 max_sequence_length=512,
                 pad_seq_len=512,
                 **kwargs):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        if pad_seq_len is not None:
            assert pad_seq_len >= max_sequence_length
        self.pad_seq_len = pad_seq_len
        self.pipeline = hf_model_loader(
            QwenImagePipeline,
            model_name_or_path,
            scheduler=None,
            vae=None,
            transformer=None,
            torch_dtype=getattr(torch, torch_dtype),
            **kwargs)
        self.text_encoder = self.pipeline.text_encoder
        self.freeze = freeze
        self.eval_mode = eval_mode
        if self.freeze:
            self.requires_grad_(False)
        if self.eval_mode:
            self.eval()

    def train(self, mode=True):
        mode = mode and (not self.eval_mode)
        return super().train(mode)

    def forward(self, prompt):
        prompt_embeds, prompt_embeds_mask = self.pipeline.encode_prompt(
            prompt, max_sequence_length=self.max_sequence_length)
        if self.pad_seq_len is not None:
            pad_len = self.pad_seq_len - prompt_embeds.size(1)
            if prompt_embeds_mask is None:
                prompt_embeds_mask = torch.ones(
                    prompt_embeds.size(0), prompt_embeds.size(1),
                    device=prompt_embeds.device,
                    dtype=torch.long)
            prompt_embeds = F.pad(
                prompt_embeds, (0, 0, 0, pad_len), value=0.0)
            prompt_embeds_mask = F.pad(
                prompt_embeds_mask, (0, pad_len), value=0.0)
        return dict(
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask)


@MODULES.register_module()
class PretrainedStableDiffusion3TextEncoder(nn.Module):
    def __init__(self,
                 model_name_or_path='stabilityai/stable-diffusion-3.5-large',
                 freeze=True,
                 eval_mode=True,
                 torch_dtype='float32',
                 max_sequence_length=256,
                 **kwargs):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.pipeline = hf_model_loader(
            StableDiffusion3Pipeline,
            model_name_or_path,
            scheduler=None,
            vae=None,
            transformer=None,
            image_encoder=None,
            feature_extractor=None,
            torch_dtype=getattr(torch, torch_dtype),
            **kwargs)
        self.text_encoder = self.pipeline.text_encoder
        self.text_encoder_2 = self.pipeline.text_encoder_2
        self.text_encoder_3 = self.pipeline.text_encoder_3
        self.freeze = freeze
        self.eval_mode = eval_mode
        if self.freeze:
            self.requires_grad_(False)
        if self.eval_mode:
            self.eval()

    def train(self, mode=True):
        mode = mode and (not self.eval_mode)
        return super().train(mode)

    def forward(self, prompt, prompt_2=None, prompt_3=None):
        prompt_embeds, _, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt, prompt_2=prompt_2, prompt_3=prompt_3, do_classifier_free_guidance=False,
            max_sequence_length=self.max_sequence_length)
        return dict(
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds)


@MODULES.register_module()
class PretrainedFlux2TextEncoder(nn.Module):
    def __init__(self,
                 model_name_or_path='black-forest-labs/FLUX.2-dev',
                 freeze=True,
                 eval_mode=True,
                 torch_dtype='bfloat16',
                 max_sequence_length=512,
                 **kwargs):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.pipeline = hf_model_loader(
            Flux2Pipeline,
            model_name_or_path,
            scheduler=None,
            vae=None,
            transformer=None,
            torch_dtype=getattr(torch, torch_dtype),
            **kwargs)
        self.text_encoder = self.pipeline.text_encoder
        self.freeze = freeze
        self.eval_mode = eval_mode
        if self.freeze:
            self.requires_grad_(False)
        if self.eval_mode:
            self.eval()

    def train(self, mode=True):
        mode = mode and (not self.eval_mode)
        return super().train(mode)

    def forward(self, prompt):
        prompt_embeds, text_ids = self.pipeline.encode_prompt(
            prompt, max_sequence_length=self.max_sequence_length)
        return dict(
            encoder_hidden_states=prompt_embeds)


@MODULES.register_module()
class PretrainedZImageTextEncoder(nn.Module):
    def __init__(self,
                 from_pretrained='Tongyi-MAI/Z-Image-Turbo',
                 freeze=True,
                 eval_mode=True,
                 torch_dtype='bfloat16',
                 max_sequence_length=512,
                 **kwargs):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.pipeline = hf_model_loader(
            ZImagePipeline,
            from_pretrained,
            scheduler=None,
            vae=None,
            transformer=None,
            torch_dtype=getattr(torch, torch_dtype),
            **kwargs)
        self.text_encoder = self.pipeline.text_encoder
        self.freeze = freeze
        self.eval_mode = eval_mode
        if self.freeze:
            self.requires_grad_(False)
        if self.eval_mode:
            self.eval()

    def train(self, mode=True):
        mode = mode and (not self.eval_mode)
        return super().train(mode)

    def forward(self, prompt):
        prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt, do_classifier_free_guidance=False, max_sequence_length=self.max_sequence_length)
        return dict(
            cap_feats=prompt_embeds)


@MODULES.register_module()
class PretrainedFlux2KleinTextEncoder(nn.Module):
    def __init__(self,
                 model_name_or_path='black-forest-labs/FLUX.2-klein-base-9B',
                 freeze=True,
                 eval_mode=True,
                 torch_dtype='bfloat16',
                 max_sequence_length=512,
                 **kwargs):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.pipeline = hf_model_loader(
            Flux2KleinPipeline,
            model_name_or_path,
            scheduler=None,
            vae=None,
            transformer=None,
            torch_dtype=getattr(torch, torch_dtype),
            **kwargs)
        self.text_encoder = self.pipeline.text_encoder
        self.freeze = freeze
        self.eval_mode = eval_mode
        if self.freeze:
            self.requires_grad_(False)
        if self.eval_mode:
            self.eval()

    def train(self, mode=True):
        mode = mode and (not self.eval_mode)
        return super().train(mode)

    def forward(self, prompt):
        prompt_embeds, text_ids = self.pipeline.encode_prompt(
            prompt, max_sequence_length=self.max_sequence_length)
        return dict(
            encoder_hidden_states=prompt_embeds)


@contextmanager
def _patch_dinov2_with_registers_init_weights():
    try:
        from transformers.models.dinov2_with_registers.modeling_dinov2_with_registers import (
            Dinov2WithRegistersPreTrainedModel,
        )
    except ImportError:
        yield
        return

    old_init_weights = Dinov2WithRegistersPreTrainedModel._init_weights

    def patched_init_weights(self, module):
        # AutoencoderRAE immediately loads pretrained encoder weights, so touching
        # meta tensors during constructor-time init only causes compatibility issues
        # with older transformers.
        return None

    with _DINOV2_WITH_REGISTERS_INIT_PATCH_LOCK:
        Dinov2WithRegistersPreTrainedModel._init_weights = patched_init_weights
        try:
            yield
        finally:
            Dinov2WithRegistersPreTrainedModel._init_weights = old_init_weights


@MODULES.register_module()
class PretrainedRAE(nn.Module):
    def __init__(self,
                 model_name_or_path=None,
                 use_slicing=False,
                 freeze=True,
                 eval_mode=True,
                 torch_dtype='float32',
                 **kwargs):
        super().__init__()
        if torch_dtype is not None:
            kwargs.update(torch_dtype=getattr(torch, torch_dtype))
        with _patch_dinov2_with_registers_init_weights():
            self.vae = hf_model_loader(
                AutoencoderRAE,
                model_name_or_path,
                **kwargs)
        if use_slicing:
            self.vae.enable_slicing()
        self.freeze = freeze
        self.eval_mode = eval_mode
        if self.freeze:
            self.requires_grad_(False)
        if self.eval_mode:
            self.eval()

    def train(self, mode=True):
        mode = mode and (not self.eval_mode)
        return super().train(mode)

    def forward(self, img, *args, **kwargs):
        img = ((img + 1) / 2).clamp(0, 1)
        decoded = self.vae(img, *args, return_dict=False, **kwargs)[0].clamp(0, 1)
        return decoded * 2 - 1

    def encode(self, img):
        img = ((img + 1) / 2).clamp(0, 1)
        return self.vae.encode(img, return_dict=False)[0]

    def decode(self, code):
        decoded = self.vae.decode(code, return_dict=False)[0].clamp(0, 1)
        return decoded * 2 - 1
