"""ComfyUI nodes for Lakonik's AsymFLUX.2-klein adapter.

Workflow:

    [UNETLoader (flux2-klein-base-9b)] -> MODEL
              \\
               -> [AsymFlux2LoadAdapter (path=...)] -> MODEL  (patched)
                              |
    [CLIP encode +/-] -> COND
    [AsymFlux2EmptyLatent (w,h)] -> LATENT  (3-channel pixel-space noise)
    [AsymFlux2Sigmas (steps,w,h)] -> SIGMAS
    [KSamplerSelect (uni_pc)] -> SAMPLER
              \\
               -> [SamplerCustom] -> LATENT  (denoised, 3-channel)
                              |
                              -> [OklabDecode] -> IMAGE -> [SaveImage]

We deliberately avoid replacing ComfyUI's KSampler -- standard SamplerCustom +
UniPC works once the sigmas are right and the model accepts (B, 3, H, W) input.
"""

import os
import torch

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None

import folder_paths  # ComfyUI module; only available when running inside ComfyUI

from .adapter_loader import apply_adapter
from .scheduler import compute_sigmas
from .oklab import (
    encode_image_to_oklab_latent,
    decode_oklab_latent_to_image,
)
from .key_map import (
    ASYMFLUX_PATCH_SIZE,
    ASYMFLUX_IN_CHANNELS,
    ASYMFLUX_OUT_CHANNELS,
)


# ---------------------------------------------------------------------------
# Adapter loader
# ---------------------------------------------------------------------------

class AsymFlux2LoadAdapter:
    """Mutates a stock FLUX.2-klein-9B model into AsymFLUX.2-klein.

    The base MODEL must already be a FLUX.2 klein checkpoint loaded via the
    standard UNETLoader / CheckpointLoaderSimple. This node performs in-place
    surgery: replaces img_in / final_layer.linear with new shapes, overwrites
    final_layer.adaLN_modulation, registers proj_buffer/scale_buffer, and
    fuses the rank-256 LoRA deltas into 58 weight tensors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # The adapter safetensors should sit in models/asymflux2/.
        return {
            'required': {
                'model': ('MODEL',),
                'adapter_name': (folder_paths.get_filename_list('asymflux2'),),
            },
            'optional': {
                'strict': ('BOOLEAN', {'default': True}),
            },
        }

    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'load'
    CATEGORY = 'AsymFlux2'

    def load(self, model, adapter_name: str, strict: bool = True):
        if load_safetensors is None:
            raise RuntimeError(
                'safetensors is not installed; run `pip install safetensors` '
                'inside the ComfyUI environment.')
        adapter_path = folder_paths.get_full_path('asymflux2', adapter_name)
        state = load_safetensors(adapter_path)

        # Clone the model patcher so we don't clobber the user's base model.
        patched = model.clone()

        # ComfyUI loads weights lazily into VRAM; force-load and then unpatch
        # so we can touch the underlying nn.Module without confusing the
        # patcher's bookkeeping.
        try:
            patched.unpatch_model(device_to=torch.device('cpu'), unpatch_weights=True)
        except Exception:
            pass

        flux = patched.model.diffusion_model

        # Pick compute dtype that matches the existing img_in (i.e. the base model).
        try:
            compute_dtype = flux.img_in.weight.dtype
        except AttributeError:
            compute_dtype = torch.bfloat16

        manifest = apply_adapter(
            flux, state, compute_dtype=compute_dtype, strict=strict)

        # Re-declare patch grid / channels so Flux.process_img patches correctly.
        flux.patch_size = ASYMFLUX_PATCH_SIZE
        flux.in_channels = ASYMFLUX_IN_CHANNELS * ASYMFLUX_PATCH_SIZE ** 2
        flux.out_channels = ASYMFLUX_OUT_CHANNELS * ASYMFLUX_PATCH_SIZE ** 2

        print(f'[AsymFlux2LoadAdapter] applied adapter `{adapter_name}`:'
              f' replaced={len(manifest["replaced_modules"])},'
              f' overwrote={len(manifest["overwrote_weights"])},'
              f' buffers={len(manifest["registered_buffers"])},'
              f' fused_loras={len(manifest["fused_loras"])},'
              f' skipped={len(manifest["skipped"])}')

        return (patched,)


# ---------------------------------------------------------------------------
# Empty pixel-space latent
# ---------------------------------------------------------------------------

class AsymFlux2EmptyLatent:
    """3-channel pixel-resolution latent of zeros, ready to be seeded with
    noise by SamplerCustom."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'width':  ('INT', {'default': 1024, 'min': 64, 'max': 4096, 'step': 16}),
                'height': ('INT', {'default': 1024, 'min': 64, 'max': 4096, 'step': 16}),
                'batch_size': ('INT', {'default': 1, 'min': 1, 'max': 16}),
            },
        }

    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'build'
    CATEGORY = 'AsymFlux2'

    def build(self, width: int, height: int, batch_size: int):
        # Width/height must be multiples of patch_size (16).
        if width % ASYMFLUX_PATCH_SIZE or height % ASYMFLUX_PATCH_SIZE:
            raise ValueError(
                f'width and height must be multiples of {ASYMFLUX_PATCH_SIZE}; '
                f'got {width}x{height}')
        samples = torch.zeros(
            (batch_size, ASYMFLUX_IN_CHANNELS, height, width), dtype=torch.float32)
        return ({'samples': samples},)


# ---------------------------------------------------------------------------
# Sigma schedule
# ---------------------------------------------------------------------------

class AsymFlux2Sigmas:
    """Produces the sqrt-shift sigmas LakonLab's FlowAdapterScheduler would
    have produced for the given image size and step count."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'steps':  ('INT', {'default': 38, 'min': 1, 'max': 500}),
                'width':  ('INT', {'default': 1024, 'min': 64, 'max': 4096}),
                'height': ('INT', {'default': 1024, 'min': 64, 'max': 4096}),
            },
            'optional': {
                'base_shift': ('FLOAT', {'default': 17.0, 'min': 0.1, 'max': 100.0}),
                'max_shift':  ('FLOAT', {'default': 34.0, 'min': 0.1, 'max': 100.0}),
            },
        }

    RETURN_TYPES = ('SIGMAS',)
    FUNCTION = 'build'
    CATEGORY = 'AsymFlux2'

    def build(self, steps: int, width: int, height: int,
              base_shift: float = 17.0, max_shift: float = 34.0):
        sigmas = compute_sigmas(
            num_inference_steps=steps,
            image_pixel_count=width * height,
            base_shift=base_shift,
            max_shift=max_shift,
        )
        return (sigmas,)


# ---------------------------------------------------------------------------
# Oklab encode / decode (replaces VAE)
# ---------------------------------------------------------------------------

class OklabEncode:
    """ComfyUI IMAGE -> 3-channel Oklab latent (after the affine norm)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',)}}

    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'encode'
    CATEGORY = 'AsymFlux2'

    def encode(self, image: torch.Tensor):
        latent = encode_image_to_oklab_latent(image)
        return ({'samples': latent},)


class OklabDecode:
    """3-channel Oklab latent -> ComfyUI IMAGE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'samples': ('LATENT',)}}

    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'decode'
    CATEGORY = 'AsymFlux2'

    def decode(self, samples: dict):
        latent = samples['samples']
        img = decode_oklab_latent_to_image(latent)
        return (img,)


NODE_CLASS_MAPPINGS = {
    'AsymFlux2LoadAdapter': AsymFlux2LoadAdapter,
    'AsymFlux2EmptyLatent': AsymFlux2EmptyLatent,
    'AsymFlux2Sigmas':      AsymFlux2Sigmas,
    'OklabEncode':          OklabEncode,
    'OklabDecode':          OklabDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'AsymFlux2LoadAdapter': 'AsymFLUX.2 Load Adapter',
    'AsymFlux2EmptyLatent': 'AsymFLUX.2 Empty Latent (pixel)',
    'AsymFlux2Sigmas':      'AsymFLUX.2 Sigmas (sqrt-shift)',
    'OklabEncode':          'Oklab Encode (image -> latent)',
    'OklabDecode':          'Oklab Decode (latent -> image)',
}
