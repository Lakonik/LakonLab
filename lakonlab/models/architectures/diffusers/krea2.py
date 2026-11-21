import torch

from accelerate import init_empty_weights
from diffusers.models import Krea2Transformer2DModel as _Krea2Transformer2DModel
from peft import LoraConfig

from ...builder import MODULES
from ..utils import flex_freeze
from lakonlab.utils import get_root_logger
from lakonlab.runner.checkpoint import load_checkpoint, _load_cached_checkpoint


@MODULES.register_module()
class Krea2Transformer2DModel(_Krea2Transformer2DModel):

    # Keep a uniform parameter dtype for FSDP1 flat-parameter sharding, as in Wan.
    _keep_in_fp32_modules = None

    def __init__(
            self,
            *args,
            patch_size=2,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            pretrained_lora=None,
            pretrained_lora_scale=1.0,
            torch_dtype='float32',
            freeze_exclude_fp32=True,
            freeze_exclude_autocast_dtype=None,
            checkpointing=True,
            compile_forward=False,
            compile_kwargs=dict(
                mode='default',
                fullgraph=True,
                dynamic=False),
            use_lora=False,
            lora_target_modules=None,
            lora_rank=16,
            lora_dropout=0.0,
            **kwargs):
        with init_empty_weights():
            super().__init__(*args, **kwargs)
        self.patch_size = patch_size

        self.init_weights(pretrained, pretrained_lora, pretrained_lora_scale)

        self.use_lora = use_lora
        self.lora_target_modules = lora_target_modules
        self.lora_rank = lora_rank
        if self.use_lora:
            transformer_lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights='gaussian',
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
            )
            self.add_adapter(transformer_lora_config)

        if torch_dtype is not None:
            self.to(getattr(torch, torch_dtype))

        self.freeze = freeze
        if self.freeze:
            flex_freeze(
                self,
                exclude_keys=freeze_exclude,
                exclude_fp32=freeze_exclude_fp32,
                exclude_autocast_dtype=freeze_exclude_autocast_dtype)

        if checkpointing:
            self.enable_gradient_checkpointing()

        self._compiled_forward = None
        if compile_forward:
            self._compiled_forward = torch.compile(self._forward_impl, **compile_kwargs)

    def init_weights(self, pretrained=None, pretrained_lora=None, pretrained_lora_scale=1.0):
        if pretrained is not None:
            logger = get_root_logger()
            load_checkpoint(
                self, pretrained,
                map_location='cpu', strict=False, logger=logger, assign=True, use_cache=True)
            if pretrained_lora is not None:
                if not isinstance(pretrained_lora, (list, tuple)):
                    assert isinstance(pretrained_lora, str)
                    pretrained_lora = [pretrained_lora]
                if not isinstance(pretrained_lora_scale, (list, tuple)):
                    assert isinstance(pretrained_lora_scale, (int, float))
                    pretrained_lora_scale = [pretrained_lora_scale]
                for pretrained_lora_single, pretrained_lora_scale_single in zip(pretrained_lora, pretrained_lora_scale):
                    lora_state_dict = _load_cached_checkpoint(
                        pretrained_lora_single, map_location='cpu', logger=logger)
                    self.load_lora_adapter(lora_state_dict)
                    self.fuse_lora(lora_scale=pretrained_lora_scale_single, safe_fusing=True)
                    self.unload_lora()
                    self.requires_grad_(True)  # PEFT freezes the weights after fusing

    def patchify(self, latents):
        if self.patch_size > 1:
            bs, c, h, w = latents.size()
            latents = latents.reshape(
                bs, c, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size
            ).permute(
                0, 1, 3, 5, 2, 4
            ).reshape(
                bs, c * self.patch_size * self.patch_size, h // self.patch_size, w // self.patch_size)
        return latents

    def unpatchify(self, latents):
        if self.patch_size > 1:
            bs, c, h, w = latents.size()
            latents = latents.reshape(
                bs, c // (self.patch_size * self.patch_size), self.patch_size, self.patch_size, h, w
            ).permute(
                0, 1, 4, 2, 5, 3
            ).reshape(
                bs, c // (self.patch_size * self.patch_size), h * self.patch_size, w * self.patch_size)
        return latents

    def _forward_impl(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            encoder_hidden_states_mask: torch.Tensor = None,
            **kwargs):
        hidden_states = self.patchify(hidden_states)
        bs, c, h, w = hidden_states.size()
        hidden_states = hidden_states.reshape(bs, c, h * w).permute(0, 2, 1).contiguous()
        text_ids = torch.zeros(encoder_hidden_states.size(1), 3, device=hidden_states.device)
        image_ids = torch.zeros(h, w, 3, device=hidden_states.device)
        image_ids[..., 1] = torch.arange(h, device=hidden_states.device)[:, None]
        image_ids[..., 2] = torch.arange(w, device=hidden_states.device)[None, :]
        position_ids = torch.cat([text_ids, image_ids.reshape(h * w, 3)], dim=0)

        output = super().forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_hidden_states_mask,
            timestep=timestep,
            position_ids=position_ids,
            return_dict=False,
            **kwargs)[0]

        output = output.permute(0, 2, 1).reshape(bs, self.out_channels, h, w)
        return self.unpatchify(output)

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            encoder_hidden_states_mask: torch.Tensor = None,
            **kwargs):
        dtype = hidden_states.dtype
        forward_fn = self._forward_impl if self._compiled_forward is None else self._compiled_forward
        return forward_fn(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states.to(dtype),
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            **kwargs)
