from typing import List, Union

import torch

from accelerate import init_empty_weights
from diffusers.models import ZImageTransformer2DModel as _ZImageTransformer2DModel
from peft import LoraConfig

from ...builder import MODULES
from ..utils import flex_freeze
from lakonlab.utils import get_root_logger
from lakonlab.runner.checkpoint import load_checkpoint, _load_cached_checkpoint


@MODULES.register_module()
class ZImageTransformer2DModel(_ZImageTransformer2DModel):

    def __init__(
            self,
            *args,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            pretrained_lora=None,
            pretrained_lora_scale=1.0,
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
                    self.fuse_lora(lora_scale=pretrained_lora_scale_single)
                    self.unload_lora()

    def forward(
            self,
            x: Union[torch.Tensor, List[torch.Tensor]],
            t: torch.Tensor,
            cap_feats: List[torch.Tensor] = None,
            **kwargs):
        if isinstance(x, torch.Tensor):
            _x = list(x.unsqueeze(-3).unbind(dim=0))  # (b, c, h, w) -> (bs, (c, 1, h, w))
        else:
            _x = [xi.unsqueeze(-3) for xi in x]  # (bs, (c, h, w)) -> (bs, (c, 1, h, w))

        dtype = _x[0].dtype
        cap_feats = [cf.to(dtype) for cf in cap_feats] if cap_feats is not None else None

        output = super().forward(
            x=_x, t=t, cap_feats=cap_feats, return_dict=False, **kwargs)[0]

        if isinstance(x, torch.Tensor):
            output = -torch.stack(output, dim=0).squeeze(-3)  # (bs, (c, 1, h, w)) -> (b, c, h, w)
        else:
            output = [-out.squeeze(-3) for out in output]  # (bs, (c, 1, h, w)) -> (bs, (c, h, w))
        return output
