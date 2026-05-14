from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models import DiTTransformer2DModel, ModelMixin  # noqa: F401
from diffusers.models.embeddings import PatchEmbed
from diffusers.models.normalization import AdaLayerNormZero
from diffusers.configuration_utils import register_to_config
from mmcv.runner import load_checkpoint
from mmcv.cnn import constant_init, xavier_init

from ...builder import MODULES
from ..diffusers.dit import CombinedTimestepLabelEmbeddingsMod, BasicTransformerBlockMod
from ..utils import flex_freeze
from .gm_output import GMOutput2D
from lakonlab.utils import get_root_logger


class _GMDiTTransformer2DModel(DiTTransformer2DModel):

    @register_to_config
    def __init__(
            self,
            num_gaussians=16,
            constant_logstd=None,
            logstd_inner_dim=1024,
            gm_num_logstd_layers=2,
            class_dropout_prob=0.0,
            num_attention_heads: int = 16,
            attention_head_dim: int = 72,
            in_channels: int = 4,
            out_channels: Optional[int] = None,
            num_layers: int = 28,
            dropout: float = 0.0,
            norm_num_groups: int = 32,
            attention_bias: bool = True,
            sample_size: int = 32,
            patch_size: int = 2,
            activation_fn: str = 'gelu-approximate',
            num_embeds_ada_norm: Optional[int] = 1000,
            upcast_attention: bool = False,
            norm_type: str = 'ada_norm_zero',
            norm_elementwise_affine: bool = False,
            norm_eps: float = 1e-5):

        super(DiTTransformer2DModel, self).__init__()

        # Validate inputs.
        if norm_type != "ada_norm_zero":
            raise NotImplementedError(
                f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'."
            )
        elif norm_type == "ada_norm_zero" and num_embeds_ada_norm is None:
            raise ValueError(
                f"When using a `patch_size` and this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None."
            )

        # Set some common variables used across the board.
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.out_channels = in_channels if out_channels is None else out_channels
        self.gm_channels = num_gaussians * (self.out_channels + 1)
        self.gradient_checkpointing = False

        # 2. Initialize the position embedding and transformer blocks.
        self.height = self.config.sample_size
        self.width = self.config.sample_size

        self.patch_size = self.config.patch_size
        self.pos_embed = PatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim)
        self.emb = CombinedTimestepLabelEmbeddingsMod(
            num_embeds_ada_norm, self.inner_dim, class_dropout_prob=0.0)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlockMod(
                self.inner_dim,
                self.config.num_attention_heads,
                self.config.attention_head_dim,
                dropout=self.config.dropout,
                activation_fn=self.config.activation_fn,
                num_embeds_ada_norm=None,
                attention_bias=self.config.attention_bias,
                upcast_attention=self.config.upcast_attention,
                norm_type=norm_type,
                norm_elementwise_affine=self.config.norm_elementwise_affine,
                norm_eps=self.config.norm_eps)
            for _ in range(self.config.num_layers)])

        # 3. Output blocks.
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(
            self.inner_dim, self.config.patch_size * self.config.patch_size * self.gm_channels)

        self.gm_out = GMOutput2D(
            num_gaussians,
            self.out_channels,
            self.inner_dim,
            constant_logstd=constant_logstd,
            logstd_inner_dim=logstd_inner_dim,
            num_logstd_layers=gm_num_logstd_layers)

    # https://github.com/facebookresearch/DiT/blob/main/models.py
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.pos_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.pos_embed.proj.bias, 0)

        # Zero-out adaLN modulation layers in DiT blocks
        for m in self.modules():
            if isinstance(m, AdaLayerNormZero):
                constant_init(m.linear, val=0)

        # Zero-out output layers
        constant_init(self.proj_out_1, val=0)

        self.gm_out.init_weights()

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: Optional[torch.LongTensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None):
        # 1. Input
        bs, _, h, w = hidden_states.size()
        height, width = h // self.patch_size, w // self.patch_size
        hidden_states = self.pos_embed(hidden_states)

        cond_emb = self.emb(
            timestep, class_labels, hidden_dtype=hidden_states.dtype)
        dropout_enabled = self.config.class_dropout_prob > 0 and self.training
        if dropout_enabled:
            uncond_emb = self.emb(timestep, torch.full_like(
                class_labels, self.config.num_embeds_ada_norm), hidden_dtype=hidden_states.dtype)

        # 2. Blocks
        for block in self.transformer_blocks:
            if dropout_enabled:
                dropout_mask = torch.rand((bs, 1), device=hidden_states.device) < self.config.class_dropout_prob
                emb = torch.where(dropout_mask, uncond_emb, cond_emb)
            else:
                emb = cond_emb

            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    None,
                    None,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    emb,
                    use_reentrant=False)

            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    emb=emb)

        # 3. Output
        if dropout_enabled:
            dropout_mask = torch.rand((bs, 1), device=hidden_states.device) < self.config.class_dropout_prob
            emb = torch.where(dropout_mask, uncond_emb, cond_emb)
        else:
            emb = cond_emb
        shift, scale = self.proj_out_1(F.silu(emb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        hidden_states = self.proj_out_2(hidden_states).reshape(
                bs, height, width, self.patch_size, self.patch_size, self.gm_channels
            ).permute(0, 5, 1, 3, 2, 4).reshape(
                bs, self.gm_channels, height * self.patch_size, width * self.patch_size)

        return self.gm_out(hidden_states, cond_emb.detach())


@MODULES.register_module()
class GMDiTTransformer2DModel(_GMDiTTransformer2DModel):

    def __init__(
            self,
            *args,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            torch_dtype='float32',
            autocast_dtype=None,
            freeze_exclude_fp32=True,
            freeze_exclude_autocast_dtype='float32',
            checkpointing=True,
            **kwargs):
        super().__init__(*args, **kwargs)

        self.init_weights(pretrained)

        if autocast_dtype is not None:
            assert torch_dtype == 'float32'
        self.autocast_dtype = autocast_dtype

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
        super().init_weights()
        if pretrained is not None:
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: Optional[torch.LongTensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            **kwargs):
        if self.autocast_dtype is not None:
            dtype = getattr(torch, self.autocast_dtype)
        else:
            dtype = hidden_states.dtype
        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=dtype if self.autocast_dtype is not None else None):
            return super().forward(
                hidden_states.to(dtype),
                timestep=timestep,
                class_labels=class_labels,
                **kwargs)
