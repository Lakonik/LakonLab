from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models import DiTTransformer2DModel, ModelMixin  # noqa: F401
from diffusers.models.attention import BasicTransformerBlock, _chunked_feed_forward, Attention, FeedForward
from diffusers.models.embeddings import (
    PatchEmbed, Timesteps, CombinedTimestepLabelEmbeddings, TimestepEmbedding, LabelEmbedding)
from diffusers.models.normalization import AdaLayerNormZero
from diffusers.configuration_utils import register_to_config
from mmcv.runner import load_checkpoint, _load_checkpoint, load_state_dict
from mmcv.cnn import constant_init, xavier_init

from ...builder import MODULES
from ..utils import flex_freeze
from lakonlab.utils import get_root_logger


class LabelEmbeddingMod(LabelEmbedding):
    def __init__(self, num_classes, hidden_size, dropout_prob=0.0, use_cfg_embedding=True):
        super(LabelEmbedding, self).__init__()
        if dropout_prob > 0:
            assert use_cfg_embedding
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob


class CombinedTimestepLabelEmbeddingsMod(CombinedTimestepLabelEmbeddings):
    """
    Modified CombinedTimestepLabelEmbeddings for reproducing the original DiT (downscale_freq_shift=0).
    """
    def __init__(
            self, num_classes, embedding_dim, class_dropout_prob=0.1, downscale_freq_shift=0, use_cfg_embedding=True):
        super(CombinedTimestepLabelEmbeddings, self).__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=downscale_freq_shift)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.class_embedder = LabelEmbeddingMod(num_classes, embedding_dim, class_dropout_prob, use_cfg_embedding)


class BasicTransformerBlockMod(BasicTransformerBlock):
    """
    Modified BasicTransformerBlock for reproducing the original DiT with shared time and class
    embeddings across all layers.
    """
    def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            dropout=0.0,
            cross_attention_dim: Optional[int] = None,
            activation_fn: str = 'geglu',
            num_embeds_ada_norm: Optional[int] = None,
            attention_bias: bool = False,
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_elementwise_affine: bool = True,
            norm_type: str = 'layer_norm',
            norm_eps: float = 1e-5,
            final_dropout: bool = False,
            attention_type: str = 'default',
            ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
            ada_norm_bias: Optional[int] = None,
            ff_inner_dim: Optional[int] = None,
            ff_bias: bool = True,
            attention_out_bias: bool = True):
        super(BasicTransformerBlock, self).__init__()
        self.only_cross_attention = only_cross_attention
        self.norm_type = norm_type
        self.num_embeds_ada_norm = num_embeds_ada_norm

        assert self.norm_type == 'ada_norm_zero'
        self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        self.norm2 = None
        self.attn2 = None

        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            emb: Optional[torch.Tensor] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype, emb=emb)

        if cross_attention_kwargs is None:
            cross_attention_kwargs = dict()
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs)
        attn_output = gate_msa.unsqueeze(1) * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        norm_hidden_states = self.norm3(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class _DiTTransformer2DModelMod(DiTTransformer2DModel):

    _cache_storage = dict()

    def get_cache_hidden_states(self):
        return self._cache_storage['hidden_states']

    def clear_cache(self):
        self._cache_storage.clear()

    @register_to_config
    def __init__(
            self,
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
        self.proj_out_2 = nn.Linear(
            self.inner_dim, self.config.patch_size * self.config.patch_size * self.out_channels)

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
        constant_init(self.proj_out_2, val=0)

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: Optional[torch.LongTensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            cache_mode: Optional[str] = None,
            cache_config: Optional[dict] = None,
            unpatchify_output: bool = True) -> torch.Tensor:
        # 1. Input
        bs, _, h, w = hidden_states.size()
        height, width = h // self.patch_size, w // self.patch_size
        dropout_enabled = self.config.class_dropout_prob > 0 and self.training

        if cache_mode is not None:
            cache_after_block = cache_config['cache_after_block']

        if cache_mode == 'load':
            hidden_states = self._cache_storage.pop('hidden_states')[-bs:]
            cond_emb = self._cache_storage.pop('cond_emb')[-bs:]
            uncond_emb = self._cache_storage.pop('uncond_emb')[-bs:] if dropout_enabled else None

        else:
            hidden_states = self.pos_embed(hidden_states)
            cond_emb = self.emb(
                timestep, class_labels, hidden_dtype=hidden_states.dtype)
            uncond_emb = self.emb(
                timestep,
                torch.full_like(class_labels, self.config.num_embeds_ada_norm),
                hidden_dtype=hidden_states.dtype
            ) if dropout_enabled else None

        # 2. Blocks
        start_block = 0
        if cache_mode == 'load':
            start_block = cache_after_block + 1

        for index_block in range(start_block, len(self.transformer_blocks)):
            block = self.transformer_blocks[index_block]

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

            if cache_mode == 'save' and index_block == cache_after_block:
                self._cache_storage['hidden_states'] = hidden_states
                self._cache_storage['cond_emb'] = cond_emb
                if dropout_enabled:
                    self._cache_storage['uncond_emb'] = uncond_emb

        # 3. Output
        if dropout_enabled:
            dropout_mask = torch.rand((bs, 1), device=hidden_states.device) < self.config.class_dropout_prob
            emb = torch.where(dropout_mask, uncond_emb, cond_emb)
        else:
            emb = cond_emb
        shift, scale = self.proj_out_1(F.silu(emb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        output = self.proj_out_2(hidden_states)

        if unpatchify_output:
            output = output.reshape(
                bs, height, width, self.patch_size, self.patch_size, self.out_channels
            ).permute(0, 5, 1, 3, 2, 4).reshape(
                bs, self.out_channels, height * self.patch_size, width * self.patch_size)

        return output


@MODULES.register_module()
class DiTTransformer2DModelMod(_DiTTransformer2DModelMod):

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
            # load from GMDiT V1 model with 1 Gaussian
            p2 = self.config.patch_size * self.config.patch_size
            ori_out_channels = p2 * self.out_channels
            if 'proj_out_2.weight' in state_dict:
                # if this is GMDiT V1 model with 1 Gaussian
                if state_dict['proj_out_2.weight'].size(0) == p2 * (self.out_channels + 1):
                    state_dict['proj_out_2.weight'] = state_dict['proj_out_2.weight'].reshape(
                        p2, self.out_channels + 1, -1
                    )[:, :-1].reshape(ori_out_channels, -1)
                # if this is original DiT with variance prediction
                if state_dict['proj_out_2.weight'].size(0) == 2 * ori_out_channels:
                    state_dict['proj_out_2.weight'] = state_dict['proj_out_2.weight'].reshape(
                        p2, 2 * self.out_channels, -1
                    )[:, :self.out_channels].reshape(ori_out_channels, -1)
            if 'proj_out_2.bias' in state_dict:
                # if this is GMDiT V1 model with 1 Gaussian
                if state_dict['proj_out_2.bias'].size(0) == p2 * (self.out_channels + 1):
                    state_dict['proj_out_2.bias'] = state_dict['proj_out_2.bias'].reshape(
                        p2, self.out_channels + 1
                    )[:, :-1].reshape(ori_out_channels)
                # if this is original DiT with variance prediction
                if state_dict['proj_out_2.bias'].size(0) == 2 * ori_out_channels:
                    state_dict['proj_out_2.bias'] = state_dict['proj_out_2.bias'].reshape(
                        p2, 2 * self.out_channels
                    )[:, :self.out_channels].reshape(ori_out_channels)
            if 'emb.class_embedder.embedding_table.weight' not in state_dict \
                    and 'transformer_blocks.0.norm1.emb.class_embedder.embedding_table.weight' in state_dict:
                # convert original diffusers DiT model to our modified DiT model with shared embeddings
                keys_to_remove = []
                state_update = {}
                for k, v in state_dict.items():
                    if k.startswith('transformer_blocks.0.norm1.emb.'):
                        new_k = k.replace('transformer_blocks.0.norm1.', '')
                        state_update[new_k] = v
                    if k.startswith('transformer_blocks.') and '.norm1.emb.' in k:
                        keys_to_remove.append(k)
                state_dict.update(state_update)
                for k in keys_to_remove:
                    del state_dict[k]
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
