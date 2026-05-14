from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models import DiTTransformer2DModel, ModelMixin  # noqa: F401
from diffusers.models.embeddings import PatchEmbed
from diffusers.models.normalization import AdaLayerNormZero
from diffusers.configuration_utils import register_to_config
from mmcv.runner import load_checkpoint, _load_checkpoint, load_state_dict
from mmcv.cnn import constant_init, xavier_init

from ...builder import MODULES
from ..diffusers.dit import CombinedTimestepLabelEmbeddingsMod, BasicTransformerBlockMod
from ..utils import flex_freeze
from .gm_output import GMFlowModelOutput
from lakonlab.utils import get_root_logger


class _GMDiTTransformer2DModelV2(DiTTransformer2DModel):

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
        self.proj_out_means = nn.Linear(
            self.inner_dim,
            self.config.patch_size * self.config.patch_size * self.config.num_gaussians * self.out_channels)
        self.proj_out_logweights = nn.Linear(
            self.inner_dim,
            self.config.patch_size * self.config.patch_size * self.config.num_gaussians)
        self.constant_logstd = constant_logstd

        if self.constant_logstd is None:
            assert gm_num_logstd_layers >= 1
            in_dim = self.inner_dim
            logstd_layers = []
            for _ in range(gm_num_logstd_layers - 1):
                logstd_layers.extend([
                    nn.SiLU(),
                    nn.Linear(in_dim, logstd_inner_dim)])
                in_dim = logstd_inner_dim
            self.proj_out_logstds = nn.Sequential(
                *logstd_layers,
                nn.SiLU(),
                nn.Linear(in_dim, 1))

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

        # Output layers
        constant_init(self.proj_out_1, val=0)
        constant_init(self.proj_out_means, val=0)
        rand_noise = torch.randn((self.config.num_gaussians * self.out_channels)) * 0.1
        self.proj_out_means.bias.data.copy_(rand_noise[None, :].expand(
            self.config.patch_size * self.config.patch_size, -1).flatten())
        constant_init(self.proj_out_logweights, val=0)
        if self.constant_logstd is None:
            constant_init(self.proj_out_logstds[-1], val=0)

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
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    None,
                    None,
                    None,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    emb)
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

        out_means = self.proj_out_means(hidden_states).reshape(
                bs, height, width, self.patch_size, self.patch_size, self.config.num_gaussians * self.out_channels
            ).permute(0, 5, 1, 3, 2, 4).reshape(
                bs, self.config.num_gaussians, self.out_channels, height * self.patch_size, width * self.patch_size)
        out_logweights = self.proj_out_logweights(hidden_states).reshape(
                bs, height, width, self.patch_size, self.patch_size, self.config.num_gaussians
            ).permute(0, 5, 1, 3, 2, 4).reshape(
                bs, self.config.num_gaussians, 1, height * self.patch_size, width * self.patch_size
            ).log_softmax(dim=1)
        if self.constant_logstd is None:
            out_logstds = self.proj_out_logstds(cond_emb.detach()).reshape(bs, 1, 1, 1, 1)
        else:
            out_logstds = hidden_states.new_full((bs, 1, 1, 1, 1), float(self.constant_logstd))

        return GMFlowModelOutput(
            means=out_means,
            logweights=out_logweights,
            logstds=out_logstds)


@MODULES.register_module()
class GMDiTTransformer2DModelV2(_GMDiTTransformer2DModelV2):

    def __init__(
            self,
            *args,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            torch_dtype='float32',
            compile_forward=False,
            compile_kwargs=dict(
                mode='reduce-overhead',
                fullgraph=True,
                dynamic=False),
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

        self._compiled_forward = None
        if compile_forward:
            self._compiled_forward = torch.compile(self._forward_impl, **compile_kwargs)

    def init_weights(self, pretrained=None):
        super().init_weights()
        if pretrained is not None:
            logger = get_root_logger()
            # load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
            checkpoint = _load_checkpoint(pretrained, map_location='cpu', logger=logger)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # expand the output channels
            p2 = self.config.patch_size * self.config.patch_size
            ori_out_channels = p2 * self.out_channels
            if 'proj_out_2.weight' in state_dict:
                # if this is GMDiT V1 model with 1 Gaussian
                if state_dict['proj_out_2.weight'].size(0) == p2 * (self.out_channels + 1):
                    state_dict['proj_out_2.weight'] = state_dict['proj_out_2.weight'].reshape(
                        p2, self.out_channels + 1, -1
                    )[:, :-1].reshape(ori_out_channels, -1)
                if state_dict['proj_out_2.weight'].size(0) == ori_out_channels:
                    state_dict['proj_out_means.weight'] = state_dict['proj_out_2.weight'].reshape(
                        p2, 1, self.out_channels, -1
                    ).expand(-1, self.config.num_gaussians, -1, -1).reshape(
                        self.config.num_gaussians * ori_out_channels, -1)
                    del state_dict['proj_out_2.weight']
            if 'proj_out_2.bias' in state_dict:
                # if this is GMDiT V1 model with 1 Gaussian
                if state_dict['proj_out_2.bias'].size(0) == p2 * (self.out_channels + 1):
                    state_dict['proj_out_2.bias'] = state_dict['proj_out_2.bias'].reshape(
                        p2, self.out_channels + 1
                    )[:, :-1].reshape(ori_out_channels)
                if state_dict['proj_out_2.bias'].size(0) == ori_out_channels:
                    state_dict['proj_out_means.bias'] = state_dict['proj_out_2.bias'].reshape(
                        p2, 1, self.out_channels
                    ).expand(-1, self.config.num_gaussians, -1).reshape(
                        self.config.num_gaussians * ori_out_channels)
                    rand_noise = torch.randn(
                        (self.config.num_gaussians * self.out_channels),
                        dtype=state_dict['proj_out_means.bias'].dtype,
                        device=state_dict['proj_out_means.bias'].device) * 0.05
                    state_dict['proj_out_means.bias'] += rand_noise[None, :].expand(p2, -1).flatten()
                    del state_dict['proj_out_2.bias']
            if (self.constant_logstd is None
                    and 'proj_out_means.weight' in state_dict
                    and 'proj_out_means.bias' in state_dict):
                self.proj_out_logstds[-1].bias.data[:1] = np.log(0.05)  # reduce the initial logstd

            load_state_dict(self, state_dict, logger=logger)

    def _forward_impl(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

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
            if self._compiled_forward is not None:
                return self._compiled_forward(
                    hidden_states.to(dtype),
                    timestep=timestep,
                    class_labels=class_labels,
                    **kwargs)
            return self._forward_impl(
                hidden_states.to(dtype),
                timestep=timestep,
                class_labels=class_labels,
                **kwargs)
