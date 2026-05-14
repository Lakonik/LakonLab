import torch
import torch.nn as nn
from diffusers.models.embeddings import PatchEmbed

from ...builder import MODULES
from ..diffusers.dit import LabelEmbeddingMod
from ..utils import flex_freeze
from .modules import (
    VisionRotaryEmbeddingFast,
    LightningDiTBlock,
    FinalLayer,
    TimestepEmbedder,
)
from lakonlab.utils import get_root_logger
from lakonlab.runner.checkpoint import _load_cached_checkpoint, load_full_state_dict


@MODULES.register_module()
class LightningDiT(nn.Module):
    def __init__(
            self,
            input_size=16,
            patch_size=1,
            in_channels=64,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            num_classes=1000,
            use_qknorm=False,
            use_swiglu=True,
            use_rope=True,
            use_rmsnorm=True,
            wo_shift=False,
            use_pos_embed=True,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            torch_dtype='float32',
            compile_forward=False,
            compile_kwargs=dict(
                mode='reduce-overhead',
                fullgraph=True,
                dynamic=False),
            freeze_exclude_fp32=True,
            freeze_exclude_autocast_dtype='float32',
            checkpointing=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.depth = depth
        self.use_pos_embed = use_pos_embed

        self.x_embedder = PatchEmbed(
            height=input_size,
            width=input_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.hidden_size,
            bias=True,
            pos_embed_type='sincos' if use_pos_embed else None,
            # forcing persistent pos_embed, since pretrained models may incorrectly use learnable pos_embed
            pos_embed_max_size=input_size // self.patch_size,
        )

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbeddingMod(
            num_classes,
            self.hidden_size,
            dropout_prob=0.0,
            use_cfg_embedding=True,
        )

        if self.use_rope:
            half_head_dim = self.hidden_size // self.num_heads // 2
            hw_seq_len = input_size // self.patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=hw_seq_len)
        else:
            self.feat_rope = None

        self.blocks = nn.ModuleList([
            LightningDiTBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                use_qknorm=use_qknorm,
                use_swiglu=use_swiglu,
                use_rmsnorm=use_rmsnorm,
                wo_shift=wo_shift,
            )
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(
            self.hidden_size, self.patch_size, self.out_channels, use_rmsnorm=use_rmsnorm)

        self.init_weights(pretrained)

        if torch_dtype is not None:
            self.to(getattr(torch, torch_dtype))

        self.freeze = freeze
        if self.freeze:
            flex_freeze(
                self,
                exclude_keys=freeze_exclude,
                exclude_fp32=freeze_exclude_fp32,
                exclude_autocast_dtype=freeze_exclude_autocast_dtype,
            )

        self.gradient_checkpointing = checkpointing
        self._compiled_forward = None
        if compile_forward:
            self._compiled_forward = torch.compile(self._forward_impl, **compile_kwargs)

    def init_weights(self, pretrained):
        if pretrained is not None:
            logger = get_root_logger()
            state_dict = _load_cached_checkpoint(pretrained, map_location='cpu', logger=logger)
            # Convert original state dict to match diffusers' PatchEmbed
            if 'pos_embed' in state_dict and 'x_embedder.pos_embed' not in state_dict:
                state_dict['x_embedder.pos_embed'] = state_dict.pop('pos_embed')
            load_full_state_dict(self, state_dict)

    def _apply_blocks(self, x, cond):
        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x, cond, self.feat_rope,
                    use_reentrant=False,
                )
            else:
                x = block(x, cond, self.feat_rope)
        return x

    def _forward_impl(self, x, timestep=None, class_labels=None):
        bs, _, h, w = x.size()
        height, width = h // self.patch_size, w // self.patch_size

        x = self.x_embedder(x)
        t = self.t_embedder(timestep)
        y = self.y_embedder(class_labels)
        cond = t + y

        x = self._apply_blocks(x, cond)
        x = self.final_layer(x, cond)

        x = x.reshape(
            bs, height, width, self.patch_size, self.patch_size, self.out_channels
        ).permute(0, 5, 1, 3, 2, 4).reshape(
            bs, self.out_channels, height * self.patch_size, width * self.patch_size)
        return x

    def forward(self, x, timestep=None, class_labels=None):
        if self._compiled_forward is not None:
            return self._compiled_forward(x, timestep=timestep, class_labels=class_labels)
        return self._forward_impl(x, timestep=timestep, class_labels=class_labels)
