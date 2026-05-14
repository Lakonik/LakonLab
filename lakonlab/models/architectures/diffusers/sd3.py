import torch

from accelerate import init_empty_weights
from diffusers.models import SD3Transformer2DModel as _SD3Transformer2DModel
from peft import LoraConfig

from ...builder import MODULES
from ..utils import flex_freeze
from lakonlab.utils import get_root_logger
from lakonlab.runner.checkpoint import load_checkpoint


@MODULES.register_module()
class SD3Transformer2DModel(_SD3Transformer2DModel):

    def __init__(
            self,
            *args,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            torch_dtype='float32',
            freeze_exclude_fp32=True,
            freeze_exclude_autocast_dtype='float32',
            checkpointing=True,
            use_lora=False,
            lora_target_modules=None,
            lora_rank=16,
            **kwargs):
        with init_empty_weights():
            super().__init__(*args, **kwargs)
        self.init_weights(pretrained)

        self.use_lora = use_lora
        self.lora_target_modules = lora_target_modules
        self.lora_rank = lora_rank
        if self.use_lora:
            transformer_lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights='gaussian',
                target_modules=lora_target_modules,
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

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = get_root_logger()
            load_checkpoint(
                self, pretrained,
                map_location='cpu', strict=False, logger=logger, assign=True, use_cache=True)

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            pooled_projections: torch.Tensor = None,
            **kwargs):
        dtype = hidden_states.dtype

        return super().forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states.to(dtype),
            pooled_projections=pooled_projections.to(dtype),
            timestep=timestep,
            return_dict=False,
            **kwargs)[0]
