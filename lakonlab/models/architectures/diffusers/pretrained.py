import logging
from contextlib import contextmanager
from threading import Lock
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models import (
    AutoencoderKL, AutoencoderKLWan, AutoencoderKLQwenImage, AutoencoderKLFlux2, AutoencoderRAE,
    AutoencoderKLMiniMaxH3 as _AutoencoderKLMiniMaxH3,
    AutoencoderKLMiniMaxH3Audio as _AutoencoderKLMiniMaxH3Audio
)
from diffusers.pipelines import (
    FluxPipeline, QwenImagePipeline, StableDiffusion3Pipeline, WanPipeline, Flux2Pipeline,
    ZImagePipeline, Flux2KleinPipeline, Krea2Pipeline
)
from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3TextEncoderStep
from transformers.utils.output_capturing import maybe_install_capturing_hooks

from ...builder import MODULES
from lakonlab.utils import is_warmup, onload
from lakonlab.utils.io_utils import hf_model_loader, hf_modular_model_loader

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
class PretrainedVAEWan(nn.Module):
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
        self.vae = hf_model_loader(AutoencoderKLWan, model_name_or_path, **kwargs)
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

    def encode(self, img):
        device = img.device
        dtype = img.dtype
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std, device=device, dtype=dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1)
        return (self.vae.encode(img).latent_dist.sample() - latents_mean) / latents_std

    def decode(self, code):
        device = code.device
        dtype = code.dtype
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std, device=device, dtype=dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1)
        return self.vae.decode(code * latents_std + latents_mean, return_dict=False)[0]


class AutoencoderKLMiniMaxH3(_AutoencoderKLMiniMaxH3):

    _keep_in_fp32_modules = None
    decode_tile_batch_size = 1

    def _decode_clip(self, z):
        if self.use_tiling and self.decode_tile_batch_size > 1:
            ratio = self.spatial_compression_ratio
            y_indices, y_lengths, y_overlaps = self._split_tiles(
                z.size(-2) * ratio, self.tile_sample_min_height, self.tile_sample_min_overlap_height)
            x_indices, x_lengths, x_overlaps = self._split_tiles(
                z.size(-1) * ratio, self.tile_sample_min_width, self.tile_sample_min_overlap_width)
            # Native tiling uses equal-sized tiles, including when either axis fits in one tile.
            tiles = [
                z[..., y // ratio:(y + h) // ratio, x // ratio:(x + w) // ratio]
                for y, h in zip(y_indices, y_lengths) for x, w in zip(x_indices, x_lengths)]
            decoded_tiles = []
            for i in range(0, len(tiles), self.decode_tile_batch_size):
                tile_batch = torch.cat(tiles[i:i + self.decode_tile_batch_size], dim=0)
                decoded_tiles.extend(self.decoder(self.post_quant_conv(tile_batch)).split(z.size(0), dim=0))
            rows = [decoded_tiles[i:i + len(x_indices)] for i in range(0, len(tiles), len(x_indices))]
            output = self._stitch_tiles(rows, y_overlaps, x_overlaps)
        else:
            output = super()._decode_clip(z)
        return output


@MODULES.register_module()
class PretrainedVAEMiniMaxH3(nn.Module):
    def __init__(self,
                 model_name_or_path=None,
                 use_slicing=False,
                 decode_tile_batch_size=1,
                 freeze=True,
                 eval_mode=True,
                 torch_dtype='float32',
                 **kwargs):
        super().__init__()
        if torch_dtype is not None:
            kwargs.update(torch_dtype=getattr(torch, torch_dtype))
        self.vae = hf_model_loader(AutoencoderKLMiniMaxH3, model_name_or_path, **kwargs)
        self.vae.decode_tile_batch_size = decode_tile_batch_size
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

    @onload
    def encode(self, img, sample_mode='sample'):
        pixel_mean = torch.tensor((-0.030, -0.088, -0.188), device=img.device, dtype=img.dtype).view(1, 3, 1, 1, 1)
        pixel_std = torch.tensor((0.458, 0.448, 0.450), device=img.device, dtype=img.dtype).view(1, 3, 1, 1, 1)
        img = (img - pixel_mean) / pixel_std
        latent_dist = self.vae.encode(img).latent_dist
        if sample_mode == 'sample':
            latents = latent_dist.sample()
        elif sample_mode == 'argmax':
            latents = latent_dist.mode()
        else:
            raise ValueError(f'Invalid sample_mode: {sample_mode}')
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=img.device, dtype=img.dtype).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std, device=img.device, dtype=img.dtype).view(1, -1, 1, 1, 1)
        return (latents - latents_mean) / latents_std

    @onload
    def decode(self, code):
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=code.device, dtype=code.dtype).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std, device=code.device, dtype=code.dtype).view(1, -1, 1, 1, 1)
        latents = code * latents_std + latents_mean
        img = self.vae.decode(latents, return_dict=False)[0]
        pixel_mean = torch.tensor((-0.030, -0.088, -0.188), device=code.device, dtype=code.dtype).view(1, 3, 1, 1, 1)
        pixel_std = torch.tensor((0.458, 0.448, 0.450), device=code.device, dtype=code.dtype).view(1, 3, 1, 1, 1)
        return img * pixel_std + pixel_mean


class AutoencoderKLMiniMaxH3Audio(_AutoencoderKLMiniMaxH3Audio):

    _keep_in_fp32_modules = None


@MODULES.register_module()
class PretrainedVAEMiniMaxH3Audio(nn.Module):
    def __init__(self,
                 model_name_or_path=None,
                 freeze=True,
                 eval_mode=True,
                 torch_dtype='float32',
                 compile_forward=False,
                 compile_kwargs=dict(
                     mode='default',
                     fullgraph=True,
                     dynamic=False),
                 **kwargs):
        super().__init__()
        if torch_dtype is not None:
            kwargs.update(torch_dtype=getattr(torch, torch_dtype))
        self.vae = hf_model_loader(AutoencoderKLMiniMaxH3Audio, model_name_or_path, **kwargs)
        self.freeze = freeze
        self.eval_mode = eval_mode
        if self.freeze:
            self.requires_grad_(False)
        if self.eval_mode:
            self.eval()
        self._compiled_encode = None
        self._compiled_decode = None
        if compile_forward:
            self._compiled_encode = torch.compile(self._encode, **compile_kwargs)
            self._compiled_decode = torch.compile(self._decode, **compile_kwargs)

    def train(self, mode=True):
        mode = mode and (not self.eval_mode)
        return super().train(mode)

    def _encode(self, audio, sample_mode='sample'):
        bs, channels, samples = audio.shape
        latent_dist = self.vae.encode(audio.reshape(bs * channels, 1, samples)).latent_dist
        if sample_mode == 'sample':
            latents = latent_dist.sample()
        elif sample_mode == 'argmax':
            latents = latent_dist.mode()
        else:
            raise ValueError(f'Invalid sample_mode: {sample_mode}')
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=audio.device, dtype=audio.dtype).view(1, -1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std, device=audio.device, dtype=audio.dtype).view(1, -1, 1)
        latents = (latents - latents_mean) / latents_std
        latents = latents[..., :round(samples / self.vae.hop_length)]
        return latents.reshape(bs, channels, *latents.shape[1:]).permute(0, 2, 1, 3)

    def _decode(self, code):
        bs, latent_channels, channels, frames = code.shape
        latents = code.permute(0, 2, 1, 3).reshape(bs * channels, latent_channels, frames)
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=code.device, dtype=code.dtype).view(1, -1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std, device=code.device, dtype=code.dtype).view(1, -1, 1)
        audio = self.vae.decode(latents * latents_std + latents_mean, return_dict=False)[0]
        return audio.reshape(bs, channels, -1)

    @onload
    def encode(self, audio, sample_mode='sample'):
        if self._compiled_encode is not None:
            return self._compiled_encode(audio, sample_mode=sample_mode)
        return self._encode(audio, sample_mode=sample_mode)

    @onload
    def decode(self, code):
        if self._compiled_decode is not None:
            return self._compiled_decode(code)
        return self._decode(code)


@MODULES.register_module()
class PretrainedVAEQwenImage(nn.Module):
    def __init__(self,
                 model_name_or_path=None,
                 use_slicing=False,
                 freeze=True,
                 eval_mode=True,
                 torch_dtype='float32',
                 compile_forward=False,
                 compile_kwargs=dict(
                     mode='default',
                     fullgraph=True,
                     dynamic=False),
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
        self._compiled_encode = None
        self._compiled_decode = None
        if compile_forward:
            self._compiled_encode = torch.compile(self._encode, **compile_kwargs)
            self._compiled_decode = torch.compile(self._decode, **compile_kwargs)

    def train(self, mode=True):
        mode = mode and (not self.eval_mode)
        return super().train(mode)

    def _encode(self, img, sample_mode='sample'):
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

    def _decode(self, code):
        device = code.device
        dtype = code.dtype
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std, device=device, dtype=dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1)
        return self.vae.decode(code.unsqueeze(-3) * latents_std + latents_mean, return_dict=False)[0].squeeze(-3)

    def encode(self, img, sample_mode='sample'):
        if self._compiled_encode is not None:
            return self._compiled_encode(img, sample_mode=sample_mode)
        return self._encode(img, sample_mode=sample_mode)

    def decode(self, code):
        if self._compiled_decode is not None:
            return self._compiled_decode(code)
        return self._decode(code)


@MODULES.register_module()
class PretrainedVAEFlux2(nn.Module):
    def __init__(self,
                 model_name_or_path=None,
                 use_slicing=False,
                 freeze=True,
                 eval_mode=True,
                 torch_dtype='float32',
                 compile_forward=False,
                 compile_kwargs=dict(
                     mode='default',
                     fullgraph=True,
                     dynamic=False),
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
        self._compiled_encode = None
        self._compiled_decode = None
        if compile_forward:
            self._compiled_encode = torch.compile(self._encode, **compile_kwargs)
            self._compiled_decode = torch.compile(self._decode, **compile_kwargs)

    def train(self, mode=True):
        mode = mode and (not self.eval_mode)
        return super().train(mode)

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

    def _encode(self, img, sample_mode='sample'):
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

    def _decode(self, code):
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(
            device=code.device, dtype=code.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            device=code.device, dtype=code.dtype)
        latents = self._patchify_latents(code) * latents_bn_std + latents_bn_mean
        return self.vae.decode(self._unpatchify_latents(latents), return_dict=False)[0]

    def encode(self, img, sample_mode='sample'):
        if self._compiled_encode is not None:
            return self._compiled_encode(img, sample_mode=sample_mode).clone()
        return self._encode(img, sample_mode=sample_mode)

    def decode(self, code):
        if self._compiled_decode is not None:
            return self._compiled_decode(code).clone()
        return self._decode(code)


@MODULES.register_module()
class PretrainedMiniMaxH3TextEncoder(nn.Module):

    def __init__(self,
                 model_name_or_path='MiniMaxAI/MiniMax-H3',
                 freeze=True,
                 eval_mode=True,
                 torch_dtype='bfloat16',
                 max_sequence_length=1024,
                 pad_to_multiple_of=256,
                 trim_output=True,
                 compile_forward=False,
                 compile_kwargs=dict(
                     mode='default',
                     fullgraph=True,
                     dynamic=False),
                 **kwargs):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.trim_output = trim_output
        text_encoder_step = hf_modular_model_loader(
            MiniMaxH3TextEncoderStep(),
            model_name_or_path,
            torch_dtype=getattr(torch, torch_dtype),
            **kwargs)
        self.text_encoder = text_encoder_step.text_encoder
        self.tokenizer = text_encoder_step.tokenizer
        self.text_encoder_layer = text_encoder_step.text_encoder_layer
        self.freeze = freeze
        self.eval_mode = eval_mode
        if self.freeze:
            self.requires_grad_(False)
        if self.eval_mode:
            self.eval()
        self._compiled_forward = None
        self._warmed_up = set()
        if compile_forward:
            # Transformers' lazy hidden-state hook installation cannot run inside fullgraph compile.
            maybe_install_capturing_hooks(self.text_encoder.model.language_model)
            self._compiled_forward = torch.compile(self._forward_impl, **compile_kwargs)

    def train(self, mode=True):
        return super().train(mode and not self.eval_mode)

    @property
    def _grad_enabled(self):
        return torch.is_grad_enabled() and not self.freeze

    def _forward_impl(self, input_ids, attention_mask):
        # Explicit text positions avoid Qwen3-VL's data-dependent multimodal layout.
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        outputs = self.text_encoder.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True)
        return outputs.hidden_states[self.text_encoder_layer]

    def _warmup_shapes(self, input_ids):
        for length in range(self.pad_to_multiple_of, self.max_sequence_length + 1, self.pad_to_multiple_of):
            padded = input_ids.new_full((input_ids.size(0), length), self.tokenizer.pad_token_id)
            self._compiled_forward(padded, torch.ones_like(padded))
        self._warmed_up.add(input_ids.size(0))

    def forward(self, prompt):
        tokens = self.tokenizer(
            prompt,
            add_special_tokens=False,
            padding=True,
            padding_side='left',
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors='pt')
        device = next(self.text_encoder.parameters()).device
        input_ids, attention_mask = tokens.input_ids.to(device), tokens.attention_mask.to(device)
        if input_ids.size(1) == 0:
            input_ids = input_ids.new_full((input_ids.size(0), self.pad_to_multiple_of), self.tokenizer.pad_token_id)
            attention_mask = torch.zeros_like(input_ids)
        with torch.set_grad_enabled(self._grad_enabled):
            if self._compiled_forward is not None:
                if is_warmup() and input_ids.size(0) not in self._warmed_up:
                    self._warmup_shapes(input_ids)
                prompt_embeds = self._compiled_forward(input_ids, attention_mask)
            else:
                prompt_embeds = self._forward_impl(input_ids, attention_mask)
        prompt_embeds = prompt_embeds * attention_mask[..., None].to(prompt_embeds.dtype)
        if not self.trim_output:
            pad = self.max_sequence_length - prompt_embeds.size(1)
            prompt_embeds = F.pad(prompt_embeds, (0, 0, pad, 0))
            attention_mask = F.pad(attention_mask, (pad, 0))
        return dict(
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=attention_mask)


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
class PretrainedWanTextEncoder(nn.Module):
    def __init__(self,
                 model_name_or_path='Wan-AI/Wan2.1-T2V-14B-Diffusers',
                 freeze=True,
                 eval_mode=True,
                 torch_dtype='bfloat16',
                 max_sequence_length=512,
                 **kwargs):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.pipeline = hf_model_loader(
            WanPipeline,
            model_name_or_path,
            scheduler=None,
            vae=None,
            transformer=None,
            transformer_2=None,
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
            prompt, do_classifier_free_guidance=False, max_sequence_length=512)
        return dict(encoder_hidden_states=prompt_embeds)


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
class PretrainedKrea2TextEncoder(nn.Module):
    def __init__(self,
                 model_name_or_path='krea/Krea-2-Raw',
                 freeze=True,
                 eval_mode=True,
                 torch_dtype='bfloat16',
                 max_sequence_length=512,
                 pad_to_multiple_of=256,
                 trim_output=True,
                 compile_forward=False,
                 compile_kwargs=dict(
                     mode='default',
                     fullgraph=True,
                     dynamic=False),
                 **kwargs):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.trim_output = trim_output
        pipeline = hf_model_loader(
            Krea2Pipeline,
            model_name_or_path,
            scheduler=None,
            vae=None,
            transformer=None,
            torch_dtype=getattr(torch, torch_dtype),
            **kwargs)
        self.text_encoder = pipeline.text_encoder
        self.tokenizer = pipeline.tokenizer
        self.text_encoder_select_layers = pipeline.text_encoder_select_layers
        self.prompt_template_encode_prefix = pipeline.prompt_template_encode_prefix
        self.prompt_template_encode_suffix = pipeline.prompt_template_encode_suffix
        self.prompt_template_encode_start_idx = pipeline.prompt_template_encode_start_idx
        self.prompt_template_encode_num_suffix_tokens = pipeline.prompt_template_encode_num_suffix_tokens
        self.freeze = freeze
        self.eval_mode = eval_mode
        if self.freeze:
            self.requires_grad_(False)
        if self.eval_mode:
            self.eval()
        self._compiled_forward = None
        self._warmed_up = set()
        if compile_forward:
            # Transformers' lazy hidden-state hook installation cannot run inside fullgraph compile.
            maybe_install_capturing_hooks(self.text_encoder.language_model)
            self._compiled_forward = torch.compile(self._forward_impl, **compile_kwargs)

    def train(self, mode=True):
        mode = mode and (not self.eval_mode)
        return super().train(mode)

    @property
    def _grad_enabled(self):
        return torch.is_grad_enabled() and not self.freeze

    def _forward_impl(self, input_ids, attention_mask):
        position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True)
        hidden_states = torch.stack([outputs.hidden_states[i] for i in self.text_encoder_select_layers], dim=2)
        return hidden_states[:, self.prompt_template_encode_start_idx:]

    def _warmup_shapes(self, input_ids):
        for length in range(self.pad_to_multiple_of, self.max_sequence_length + 1, self.pad_to_multiple_of):
            padded = input_ids.new_full(
                (input_ids.size(0), length + self.prompt_template_encode_start_idx), self.tokenizer.pad_token_id)
            self._compiled_forward(padded, torch.ones_like(padded, dtype=torch.bool))
        self._warmed_up.add(input_ids.size(0))

    def forward(self, prompt):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prefix_idx = self.prompt_template_encode_start_idx
        suffix_len = self.prompt_template_encode_num_suffix_tokens
        text = [self.prompt_template_encode_prefix + e for e in prompt]
        text_tokens = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            padding_side='right',
            max_length=self.max_sequence_length + prefix_idx - suffix_len,
            return_tensors='pt')
        suffix_tokens = self.tokenizer([self.prompt_template_encode_suffix] * len(text), return_tensors='pt')
        seq_len = text_tokens.input_ids.size(1) - prefix_idx + suffix_len
        seq_len = min(
            (seq_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of,
            self.max_sequence_length)
        pad = seq_len + prefix_idx - suffix_len - text_tokens.input_ids.size(1)
        input_ids = torch.cat([
            F.pad(text_tokens.input_ids, (0, pad), value=self.tokenizer.pad_token_id), suffix_tokens.input_ids], dim=1)
        attention_mask = torch.cat([
            F.pad(text_tokens.attention_mask, (0, pad)), suffix_tokens.attention_mask], dim=1).bool()
        device = next(self.text_encoder.parameters()).device
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        with torch.set_grad_enabled(self._grad_enabled):
            if self._compiled_forward is not None:
                if is_warmup() and input_ids.size(0) not in self._warmed_up:
                    self._warmup_shapes(input_ids)
                prompt_embeds = self._compiled_forward(input_ids, attention_mask)
            else:
                prompt_embeds = self._forward_impl(input_ids, attention_mask)
        attention_mask = attention_mask[:, prefix_idx:]
        if not self.trim_output:
            pad = self.max_sequence_length - prompt_embeds.size(1)
            prompt_embeds = F.pad(prompt_embeds, (0, 0, 0, 0, pad, 0))
            attention_mask = F.pad(attention_mask, (pad, 0))
        return dict(
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=attention_mask)


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
                 compile_forward=False,
                 compile_kwargs=dict(
                     mode='default',
                     fullgraph=True,
                     dynamic=False),
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
        if compile_forward:
            # Transformers' lazy hidden-state hook installation cannot run inside fullgraph compile.
            maybe_install_capturing_hooks(self.text_encoder.model)
            self.text_encoder.compile(**compile_kwargs)

    def train(self, mode=True):
        mode = mode and (not self.eval_mode)
        return super().train(mode)

    @property
    def _grad_enabled(self):
        return torch.is_grad_enabled() and not self.freeze

    def forward(self, prompt):
        with torch.set_grad_enabled(self._grad_enabled):
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
                 compile_forward=False,
                 compile_kwargs=dict(
                     mode='reduce-overhead',
                     fullgraph=True,
                     dynamic=False),
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
        self._compiled_encode = None
        self._compiled_decode = None
        if compile_forward:
            # Transformers' lazy hidden-state hook installation cannot run inside fullgraph compile.
            maybe_install_capturing_hooks(self.vae.encoder)
            self._compiled_encode = torch.compile(self._encode, **compile_kwargs)
            self._compiled_decode = torch.compile(self._decode, **compile_kwargs)

    def train(self, mode=True):
        mode = mode and (not self.eval_mode)
        return super().train(mode)

    def _encode(self, img):
        img = ((img + 1) / 2).clamp(0, 1)
        return self.vae.encode(img, return_dict=False)[0]

    def _decode(self, code):
        decoded = self.vae.decode(code, return_dict=False)[0].clamp(0, 1)
        return decoded * 2 - 1

    def encode(self, img):
        if self._compiled_encode is not None:
            return self._compiled_encode(img).clone()
        return self._encode(img)

    def decode(self, code):
        if self._compiled_decode is not None:
            return self._compiled_decode(code).clone()
        return self._decode(code)
