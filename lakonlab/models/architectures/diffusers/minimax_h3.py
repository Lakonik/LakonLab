# Copyright (c) 2026 Hansheng Chen
# Timestep table and folded AdaLN adapted from code by the MiniMax and Hugging Face teams.
# Source: https://huggingface.co/multimodalart/MiniMax-H3-Pruned/blob/main/transformer/modeling_minimax_h3_pruned.py
# Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from diffusers.configuration_utils import register_to_config
from diffusers.models import MiniMaxH3Transformer3DModel as _MiniMaxH3Transformer3DModel
from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry, dispatch_attention_fn
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.models.transformers.transformer_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3AdaLayerNormModulation, MiniMaxH3AdaLayerNormOut, MiniMaxH3RotaryPosEmbed,
    MiniMaxH3Attention, MiniMaxH3AttnProcessor, MiniMaxH3TokenRefiner, MiniMaxH3TokenRefinerBlock,
    MiniMaxH3TransformerBlock,
    _apply_rotary_emb)
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep
from diffusers.utils import apply_lora_scale
from mmcv.cnn import constant_init
from peft import LoraConfig

from ...builder import MODULES
from ..dwd_head import DWDHeadMixin
from ..utils import flex_freeze
from .activations import SwiGLUFeedForward
from .attention import set_attention_backends
from lakonlab.ops.attention import sol_attention
from lakonlab.ops import quantize_frozen_linears
from lakonlab.runner.checkpoint import _load_cached_checkpoint, load_full_state_dict
from lakonlab.utils import get_root_logger

# The t a keyframe anchor is noised to and held at (native MiniMaxH3ModularPipeline.keyframe_noise_aug)
MINIMAX_H3_KEYFRAME_NOISE_AUG = 0.999


class MiniMaxH3AdaLayerNormModulationMod(MiniMaxH3AdaLayerNormModulation):

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        temb = self.linear(F.silu(temb))
        temb = temb.view(-1, 6 * self.hidden_size)
        return temb.chunk(6, dim=-1)


class MiniMaxH3AdaLayerNormOutMod(MiniMaxH3AdaLayerNormOut):

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        timestep_indices: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = self.linear(F.silu(temb)).chunk(2, dim=-1)
        return self.norm(hidden_states) * (1 + scale[timestep_indices]) + shift[timestep_indices]


class MiniMaxH3AttnProcessorMod(MiniMaxH3AttnProcessor):

    def __call__(
        self,
        attn: "MiniMaxH3AttentionMod",
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        sol_prefix=None,  # unused in base attention processor
        **kwargs,
    ) -> torch.Tensor:
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            attention_kwargs=kwargs,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class MiniMaxH3SolAttnProcessor(MiniMaxH3AttnProcessor):
    """Forward-only Sol sparse attention on the native packed inference layout. The prefix rows (text,
    reference/keyframe and audio) form an exact KV sink and get dense query attention from this processor's
    attention backend (native, flash 3 or sage), set independently of the other blocks'."""

    tau = 1.0
    thresh_type = 'exact'
    kv_splits = 0  # 0 selects four splits on SM90 at 65,536 tokens or more, and one otherwise

    def __call__(
        self,
        attn: "MiniMaxH3AttentionMod",
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        sol_prefix: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if attention_mask is not None or kwargs or self._parallel_config is not None:
            raise ValueError('Sol attention requires unmasked attention without context parallelism.')
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        prefix_backend = {
            AttentionBackendName.NATIVE: 'sdpa',
            AttentionBackendName._FLASH_3: 'flash3',
            AttentionBackendName.SAGE: 'sage',
        }[AttentionBackendName(self._attention_backend or _AttentionBackendRegistry.get_active_backend()[0])]
        hidden_states = sol_attention(
            query, key, value, sol_prefix, self.tau, self.thresh_type, self.kv_splits, prefix_backend)
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class MiniMaxH3AttentionMod(MiniMaxH3Attention):
    _default_processor_cls = MiniMaxH3AttnProcessorMod
    _available_processors = [MiniMaxH3AttnProcessorMod, MiniMaxH3SolAttnProcessor]

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, rotary_emb, attention_mask, **kwargs)


class MiniMaxH3TokenRefinerBlockMod(MiniMaxH3TokenRefinerBlock):

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
    ):
        super(MiniMaxH3TokenRefinerBlock, self).__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3AttentionMod(
            hidden_size=hidden_size,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qk_norm_eps=qk_norm_eps,
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.ff = SwiGLUFeedForward(hidden_size, inner_dim=ffn_dim, activation_fn="swiglu", bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_kwargs: dict | None = None,
    ) -> torch.Tensor:
        attention_kwargs = attention_kwargs or dict()
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), **attention_kwargs)
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states


class MiniMaxH3TokenRefinerMod(MiniMaxH3TokenRefiner):

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        num_layers: int,
        norm_eps: float,
        qk_norm_eps: float,
        final_norm_eps: float,
    ):
        super(MiniMaxH3TokenRefiner, self).__init__()
        self.refiner_blocks = nn.ModuleList(
            [
                MiniMaxH3TokenRefinerBlockMod(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    ffn_dim=ffn_dim,
                    norm_eps=norm_eps,
                    qk_norm_eps=qk_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(hidden_size, eps=final_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_kwargs: dict,
    ) -> torch.Tensor:
        for block in self.refiner_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(block, hidden_states, attention_kwargs)
            else:
                hidden_states = block(hidden_states, attention_kwargs=attention_kwargs)
        return self.final_norm(hidden_states)


class MiniMaxH3TransformerBlockMod(MiniMaxH3TransformerBlock):

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        time_embed_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
        adaln_rank: int | None = None,
    ):
        super(MiniMaxH3TransformerBlock, self).__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3AttentionMod(
            hidden_size=hidden_size,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qk_norm_eps=qk_norm_eps,
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.ff = SwiGLUFeedForward(hidden_size, inner_dim=ffn_dim, activation_fn="swiglu", bias=False)
        if adaln_rank is None:
            self.adaln_proj = MiniMaxH3AdaLayerNormModulationMod(time_embed_dim=time_embed_dim, hidden_size=hidden_size)
        else:
            self.adaln_proj = MiniMaxH3PrunedAdaLayerNormModulation(adaln_rank, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_kwargs: dict | None = None,
    ) -> torch.Tensor:
        attention_kwargs = attention_kwargs or dict()
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)

        # Unlike upstream's (S, ) indices, adaln_indices is (B, S) for independent
        # sample timesteps, producing (B, S, D) modulation
        residual = hidden_states
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa[adaln_indices]) + shift_msa[adaln_indices]
        attn_output = self.attn(norm_hidden_states, rotary_emb, **attention_kwargs)
        hidden_states = residual + gate_msa[adaln_indices] * attn_output

        residual = hidden_states
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[adaln_indices]) + shift_mlp[adaln_indices]
        ff_output = self.ff(norm_hidden_states)
        hidden_states = residual + gate_mlp[adaln_indices] * ff_output

        return hidden_states


class MiniMaxH3PrunedTimeEmbedder(nn.Module):

    def __init__(self, table_size: int, adaln_rank: int) -> None:
        super().__init__()
        self.register_buffer('table', torch.zeros(table_size, adaln_rank))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        steps = self.table.shape[0] - 1
        position = timestep.float().flatten().clamp(0, 1) * steps
        lower = position.floor().clamp(max=steps - 1).long()
        return torch.lerp(
            self.table[lower].float(), self.table[lower + 1].float(), (position - lower).unsqueeze(-1)
        )


class MiniMaxH3PrunedAdaLayerNormModulation(nn.Module):

    def __init__(self, adaln_rank: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(adaln_rank, 18 * hidden_size)

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        temb = self.linear(temb)
        return temb.reshape(-1, 6 * self.hidden_size).chunk(6, dim=-1)


class MiniMaxH3PrunedAdaLayerNormOut(nn.Module):

    def __init__(self, adaln_rank: int, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(adaln_rank, 2 * hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        timestep_indices: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = self.linear(temb).chunk(2, dim=-1)
        return self.norm(hidden_states) * (1 + scale[timestep_indices]) + shift[timestep_indices]


class MiniMaxH3DWDAdaLayerNormOut(AdaLayerNormContinuous):

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        timestep_indices: torch.Tensor,
    ) -> torch.Tensor:
        scale, shift = self.linear(self.silu(temb)).chunk(2, dim=-1)
        return self.norm(hidden_states) * (1 + scale[timestep_indices]) + shift[timestep_indices]


class _MiniMaxH3Transformer3DModelMod(DWDHeadMixin, _MiniMaxH3Transformer3DModel):

    _keep_in_fp32_modules = None
    _no_split_modules = [
        'MiniMaxH3TransformerBlockMod', 'MiniMaxH3TokenRefinerBlockMod',
        'MiniMaxH3AdaLayerNormOutMod', 'MiniMaxH3PrunedAdaLayerNormOut', 'MiniMaxH3DWDAdaLayerNormOut']
    _repeated_blocks = ['MiniMaxH3TransformerBlockMod', 'MiniMaxH3TokenRefinerBlockMod']
    _cache_storage = dict()

    def get_cache_hidden_states(self):
        return self._cache_storage['hidden_states']

    def clear_cache(self):
        self._cache_storage.clear()

    def _fix_state_dict_keys_on_load(self, state_dict):
        # The pruned checkpoint stores the AdaLN biases as `folded_bias` buffers.
        for k in [k for k in state_dict if k.endswith('.folded_bias')]:
            state_dict[k.removesuffix('folded_bias') + 'linear.bias'] = state_dict.pop(k)
        super()._fix_state_dict_keys_on_load(state_dict)

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        hidden_size: int = 5376,
        num_layers: int = 50,
        num_refiner_layers: int = 2,
        ffn_dim: int = 14336,
        in_channels: int = 24,
        audio_in_channels: int = 32,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_dim: int = 5120,
        freq_dim: int = 256,
        time_embed_hidden_dim: int = 5376,
        time_embed_dim: int = 2688,
        rope_freq_dim: int = 16,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
        adaln_rank: int | None = None,
        time_table_size: int = 1025,
        dwd_head_config: dict | None = None,
    ) -> None:
        super(_MiniMaxH3Transformer3DModel, self).__init__()

        video_patch_dim = in_channels * patch_size[0] * patch_size[1] * patch_size[2]

        # 1. Per-modality input projections
        self.proj_in = nn.Linear(video_patch_dim, hidden_size, bias=True)
        self.audio_proj_in = nn.Linear(audio_in_channels, hidden_size, bias=True)
        self.context_embedder = nn.Linear(text_dim, hidden_size, bias=True)

        # 2. Timestep embedding, shared by every AdaLN projection
        if adaln_rank is None:
            self.time_proj = Timesteps(num_channels=freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.time_embedder = TimestepEmbedding(
                in_channels=freq_dim, time_embed_dim=time_embed_hidden_dim, out_dim=time_embed_dim
            )
        else:
            self.time_embedder = MiniMaxH3PrunedTimeEmbedder(time_table_size, adaln_rank)
            self.register_buffer('adaln_basis', torch.zeros(adaln_rank, time_embed_dim))
            self.register_buffer('adaln_mean', torch.zeros(time_embed_dim))

        # 3. Rotary embedding over the packed (t, h, w) grid
        self.rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=rope_freq_dim, rope_theta=rope_theta)

        # 4. Text stream refiner
        self.token_refiner = MiniMaxH3TokenRefinerMod(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            ffn_dim=ffn_dim,
            num_layers=num_refiner_layers,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            final_norm_eps=final_norm_eps,
        )

        # 5. The block stack
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3TransformerBlockMod(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    ffn_dim=ffn_dim,
                    time_embed_dim=time_embed_dim,
                    norm_eps=norm_eps,
                    qk_norm_eps=qk_norm_eps,
                    adaln_rank=adaln_rank,
                )
                for _ in range(num_layers)
            ]
        )

        # 6. Shared output norm and the two per-modality output heads. Both heads run over every row of the packed
        # sequence; the rows of each modality are selected afterwards.
        if adaln_rank is None:
            self.norm_out = MiniMaxH3AdaLayerNormOutMod(
                hidden_size=hidden_size, time_embed_dim=time_embed_dim, eps=final_norm_eps
            )
        else:
            self.norm_out = MiniMaxH3PrunedAdaLayerNormOut(adaln_rank, hidden_size, final_norm_eps)
        self.proj_out = nn.Linear(hidden_size, video_patch_dim, bias=True)
        self.audio_proj_out = nn.Linear(hidden_size, audio_in_channels, bias=True)

        # 7. Optional DWD heads
        if dwd_head_config is not None:
            self.dwd_norm_out = MiniMaxH3DWDAdaLayerNormOut(
                hidden_size, time_embed_dim if adaln_rank is None else adaln_rank,
                elementwise_affine=False, eps=final_norm_eps)
            self.init_dwd_head(hidden_size, **dwd_head_config)

        self.gradient_checkpointing = False

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        text_attention_kwargs: dict,
        attention_kwargs: dict,
        cache_mode: str | None = None,
        cache_config: dict | None = None,
        return_features: bool = False,
    ):
        if cache_mode is not None:
            cache_after_block = cache_config['cache_after_block']

        if cache_mode == 'load':
            batch_size = hidden_states.size(0)
            hidden_states = self._cache_storage.pop('hidden_states')[-batch_size:]
            rotary_emb = self._cache_storage.pop('rotary_emb')
            temb = self._cache_storage.pop('temb')[-timestep.size(0):]

            start_block = cache_after_block + 1

        else:
            sequence_length = position_ids.shape[0]
            rotary_emb = self.rope(position_ids)

            # 1. Project each modality and scatter the rows into the packed sequence buffer.
            video_embeds = self.proj_in(hidden_states)
            audio_embeds = self.audio_proj_in(audio_hidden_states)
            text_embeds = self.context_embedder(encoder_hidden_states)
            text_embeds = self.token_refiner(text_embeds, attention_kwargs=text_attention_kwargs)

            hidden_states = text_embeds.new_zeros((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
            hidden_states.index_copy_(1, text_indices, text_embeds)
            hidden_states.index_copy_(1, video_indices, video_embeds)
            hidden_states.index_copy_(1, audio_indices, audio_embeds)
            del text_embeds, video_embeds, audio_embeds

            # 2. One timestep embedding per distinct noise level.
            if self.config.adaln_rank is None:
                temb = self.time_embedder(self.time_proj(timestep).type_as(hidden_states))
            else:
                temb = self.time_embedder(timestep).type_as(hidden_states)

            start_block = 0

        # 3. Row -> AdaLN table row.
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags

        for index_block in range(start_block, len(self.transformer_blocks)):
            block = self.transformer_blocks[index_block]
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, temb, adaln_indices, rotary_emb, attention_kwargs
                )
            else:
                hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb, attention_kwargs)

            if cache_mode == 'save' and index_block == cache_after_block:
                self._cache_storage['hidden_states'] = hidden_states
                self._cache_storage['rotary_emb'] = rotary_emb
                self._cache_storage['temb'] = temb

        # 5. Project each modality's normalized rows.
        norm_hidden_states = self.norm_out(hidden_states, temb, timestep_indices)
        video_output = self.proj_out(norm_hidden_states.index_select(1, video_indices))
        audio_output = self.audio_proj_out(norm_hidden_states.index_select(1, audio_indices))

        output = (video_output, audio_output)

        if return_features:
            if hasattr(self, 'dwd_norm_out'):
                del norm_hidden_states
                features = self.dwd_norm_out(hidden_states, temb, timestep_indices)
            else:
                features = norm_hidden_states
            features = dict(
                image=features.index_select(1, video_indices),
                audio=features.index_select(1, audio_indices))
            return output, features

        else:
            return output


@MODULES.register_module()
class MiniMaxH3Transformer3DModel(_MiniMaxH3Transformer3DModelMod):

    def __init__(
            self,
            *args,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            pretrained_dwd_head=None,
            pretrained_adapter=None,
            pretrained_adapter_scale=1.0,
            torch_dtype='float32',
            freeze_exclude_fp32=True,
            freeze_exclude_autocast_dtype=None,
            freeze_quant_config=None,
            attention_backend_config=None,
            checkpointing=True,
            use_lora=False,
            lora_target_modules=None,
            lora_rank=16,
            lora_dropout=0.0,
            compile_forward=False,
            compile_kwargs=dict(
                mode='default',
                fullgraph=True,
                dynamic=False),
            **kwargs):
        with init_empty_weights():
            super().__init__(*args, **kwargs)

        self.init_weights(pretrained, pretrained_dwd_head, pretrained_adapter, pretrained_adapter_scale)

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

        if freeze_quant_config is not None:
            assert torch_dtype in ('bfloat16', 'float16'), \
                'freeze_quant_config requires torch_dtype bfloat16 or float16'
            freeze_quant_config['exclude_keys'] = (
                ['self.proj_out', 'self.audio_proj_out', 'self.dwd_linear_head',
                 'self.first_frame_dwd_linear_head']
                + list(freeze_quant_config.get('exclude_keys', [])))
            quantize_frozen_linears(self, **freeze_quant_config)

        if attention_backend_config is not None:
            set_attention_backends(self, **attention_backend_config)
            for blocks in (self.token_refiner.refiner_blocks, self.transformer_blocks):
                if len({block.attn.processor._attention_backend for block in blocks}) > 1:
                    raise ValueError('H3 requires a uniform attention backend within each refiner or transformer stack.')

        if checkpointing:
            self.enable_gradient_checkpointing()

        self._compiled_forward = None
        if compile_forward:
            self._compiled_forward = torch.compile(self._forward_impl, **compile_kwargs)

    def init_weights(
            self,
            pretrained=None,
            pretrained_dwd_head=None,
            pretrained_adapter=None,
            pretrained_adapter_scale=1.0):
        super().init_weights()

        if hasattr(self, 'dwd_norm_out'):
            constant_init(self.dwd_norm_out.to_empty(device='cpu').linear, val=0)

        if pretrained is not None:
            logger = get_root_logger()
            checkpoint = _load_cached_checkpoint(pretrained, map_location='cpu', logger=logger)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            state_dict = {
                re.sub(r'\.folded_bias$', '.linear.bias', k.removeprefix('module.')): v
                for k, v in state_dict.items()}

            if pretrained_dwd_head is not None:
                state_dict.update(_load_cached_checkpoint(
                    pretrained_dwd_head, map_location='cpu', logger=logger))

            lora_state_dict = dict()
            if pretrained_adapter is not None:
                adapter_state_dict = _load_cached_checkpoint(
                    pretrained_adapter, map_location='cpu', logger=logger)
                for k, v in adapter_state_dict.items():
                    if 'lora' in k:
                        lora_state_dict[k.replace('.default.', '.')] = v
                    else:
                        state_dict[k] = v

            load_full_state_dict(self, state_dict, logger=logger, assign=True)

            if len(lora_state_dict) > 0:
                self.load_lora_adapter(lora_state_dict, prefix=None)
                self.fuse_lora(lora_scale=pretrained_adapter_scale, safe_fusing=True)
                self.unload_lora()
                self.requires_grad_(True)  # PEFT freezes the weights after fusing

    @staticmethod
    def patchify(latents, patch_size):
        bs, c, t, h, w = latents.size()
        p_t, p_h, p_w = patch_size
        return latents.reshape(
            bs, c, t // p_t, p_t, h // p_h, p_h, w // p_w, p_w
        ).permute(
            0, 1, 3, 5, 7, 2, 4, 6
        ).reshape(
            bs, c * p_t * p_h * p_w, t // p_t, h // p_h, w // p_w)

    @staticmethod
    def unpatchify(latents, patch_size):
        p_t, p_h, p_w = patch_size
        p = p_t * p_h * p_w
        bs, c, t, h, w = latents.size()
        return latents.reshape(
            bs, c // p, p_t, p_h, p_w, t, h, w
        ).permute(
            0, 1, 5, 2, 6, 3, 7, 4
        ).reshape(
            bs, c // p, t * p_t, h * p_h, w * p_w)

    @staticmethod
    def pack(latents):
        bs, c, t, h, w = latents.shape
        latents = latents.reshape(bs, c, t * h * w).permute(0, 2, 1)
        return latents

    @staticmethod
    def unpack(latents, t, h, w):
        bs, _, c = latents.shape
        latents = latents.permute(0, 2, 1).reshape(bs, c, t, h, w)
        return latents

    def prepare_layout(self, image, audio, text_length, keyframe_anchors=()):
        if self.config.patch_size[0] != 1:
            raise ValueError('H3 rotary positions require temporal patch size 1.')
        position_ids, token_tags, video_indices, audio_indices, text_indices, num_condition_rows, _ = (
            MiniMaxH3PrepareLayoutStep.build_packed_sequence(
                text_token_tags=torch.ones(text_length, dtype=torch.long),
                num_latent_frames=image.size(2),
                latent_height=image.size(3),
                latent_width=image.size(4),
                num_audio_latents=audio.size(3),
                patch_size=self.config.patch_size,
                audio_channels=audio.size(2),
                audio_tag=2,
                video_tag=0,
                keyframe_anchors=keyframe_anchors))
        # Move the keyframe anchors from after the text to the end of the sequence (rotary positions are explicit, so
        # this does not change the result), so that a sample's valid rows stay contiguous when its anchors are masked.
        order = torch.cat((
            torch.arange(text_length), torch.arange(text_length + num_condition_rows, token_tags.size(0)),
            torch.arange(text_length, text_length + num_condition_rows)))
        rank = order.argsort()
        position_ids, token_tags = position_ids[order], token_tags[order]
        video_indices, audio_indices, text_indices = rank[video_indices], rank[audio_indices], rank[text_indices]
        return dict(
            position_ids=position_ids.to(image.device),
            token_tags=token_tags.to(image.device),
            video_indices=video_indices.to(image.device),
            audio_indices=audio_indices.to(image.device),
            text_indices=text_indices.to(image.device))

    def _forward_impl(
            self, hidden_states, audio_hidden_states, condition_latents=None,
            return_pixels=False, return_features=False, **kwargs):
        hidden_states = self.patchify(hidden_states, self.config.patch_size)
        _, _, t, h, w = hidden_states.shape
        bs, d, c, audio_t = audio_hidden_states.shape
        hidden_states = self.pack(hidden_states)
        # Keyframe anchors are the leading video rows; only the generated rows are returned.
        num_condition_rows = 0
        if condition_latents is not None:
            condition_latents = self.pack(self.patchify(condition_latents, self.config.patch_size))
            num_condition_rows = condition_latents.size(1)
            hidden_states = torch.cat((condition_latents, hidden_states), dim=1)
        output = super().forward(
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states.permute(0, 2, 3, 1).reshape(bs, c * audio_t, d),
            return_features=return_pixels or return_features,
            **kwargs)
        auxiliary_output = {}
        if return_pixels or return_features:
            output, auxiliary_output = output
            auxiliary_output['image'] = auxiliary_output['image'][:, num_condition_rows:]
        output, audio_output = output
        output = self.unpatchify(self.unpack(output[:, num_condition_rows:], t, h, w), self.config.patch_size)
        audio_output = audio_output.reshape(bs, c, audio_t, d).permute(0, 3, 1, 2)
        output = dict(image=-output, audio=-audio_output)
        if return_pixels:
            auxiliary_output = dict(image=self.forward_dwd_head(auxiliary_output['image'], (t, h, w)))
        return output, auxiliary_output

    def prepare_attention_kwargs(self, mask, backend, attention_kwargs=None):
        if backend is None:
            backend = _AttentionBackendRegistry.get_active_backend()[0]
        kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
        if backend == AttentionBackendName.NATIVE:
            kwargs['attention_mask'] = mask[:, None, None, :].bool()
        elif backend == AttentionBackendName._FLASH_VARLEN_3:
            batch_size, sequence_length = mask.shape
            starts = torch.arange(batch_size + 1, device=mask.device, dtype=torch.int32) * sequence_length
            valid_starts = starts[:-1] + (mask.cumsum(-1) == 0).sum(-1, dtype=torch.int32)
            valid_ends = valid_starts + mask.sum(-1, dtype=torch.int32)
            # Each sample is three isolated documents: leading padding, its contiguous valid rows, and trailing masked
            # rows (masked keyframe anchors), any of which may be empty.
            offsets = torch.cat((torch.stack((starts[:-1], valid_starts, valid_ends), dim=1).flatten(), starts[-1:]))
            kwargs.update(
                cu_seqlens_q=offsets, cu_seqlens_k=offsets,
                max_seqlen_q=sequence_length, max_seqlen_k=sequence_length)
        else:
            raise ValueError(f'H3 padding is not supported by attention backend {backend!r}')
        return kwargs

    def forward(
        self,
        x: dict[str, torch.Tensor],
        t: dict[str, torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        attention_kwargs: dict | None = None,
        return_pixels: bool = False,
        return_features: bool = False,
        condition_latents: torch.Tensor | None = None,
        condition_mask: torch.Tensor | None = None,
        **kwargs
    ):
        x_video, x_audio = x['image'], x['audio']
        dtype = x_video.dtype
        bs = x_video.size(0)
        encoder_hidden_states = encoder_hidden_states[-bs:]
        encoder_hidden_states_mask = encoder_hidden_states_mask[-bs:]

        text_length = encoder_hidden_states.size(1)
        layout = self.prepare_layout(
            x_video, x_audio, text_length, keyframe_anchors=() if condition_latents is None else ('first', ))

        text_attention_kwargs = self.prepare_attention_kwargs(
            encoder_hidden_states_mask,
            self.token_refiner.refiner_blocks[0].attn.processor._attention_backend,
            attention_kwargs)
        joint_mask = torch.ones((bs, layout['position_ids'].size(0)), dtype=torch.bool, device=x_video.device)
        joint_mask = joint_mask.index_copy(1, layout['text_indices'], encoder_hidden_states_mask.bool())
        num_condition_rows = 0
        if condition_latents is not None:
            num_condition_rows = condition_latents.shape[2:].numel() // math.prod(self.config.patch_size)
            if condition_mask is not None:  # samples with a False condition_mask attend as if without the anchor
                joint_mask[:, layout['video_indices'][:num_condition_rows]] = condition_mask[-bs:, None].bool()
        block_attention_kwargs = self.prepare_attention_kwargs(
            joint_mask,
            self.transformer_blocks[0].attn.processor._attention_backend,
            attention_kwargs)

        # Each sample has two timestep rows: image (also used by text), then audio. A first-frame keyframe
        # anchor adds a third row: as in native H3, it is noised to and held at t = 0.999 or above.
        timestep_rows = [t['image'], t['audio']]
        row_timestep_ids = (layout['token_tags'] == 2).long()
        if condition_latents is not None:
            timestep_rows.append(t['image'].clamp(min=MINIMAX_H3_KEYFRAME_NOISE_AUG))
            condition_latents = condition_latents[-bs:]
            row_timestep_ids[layout['video_indices'][:num_condition_rows]] = 2
            condition_latents = (MINIMAX_H3_KEYFRAME_NOISE_AUG * condition_latents + (
                1 - MINIMAX_H3_KEYFRAME_NOISE_AUG) * torch.randn_like(condition_latents)).to(dtype)
        timestep = torch.stack(timestep_rows, dim=1).flatten()
        timestep_indices = torch.arange(bs, device=x_video.device)[:, None] * len(timestep_rows)
        timestep_indices = timestep_indices + row_timestep_ids[None, :]

        forward_fn = self._forward_impl if self._compiled_forward is None else self._compiled_forward
        output, auxiliary_output = forward_fn(
            hidden_states=x_video,
            audio_hidden_states=x_audio,
            condition_latents=condition_latents,
            encoder_hidden_states=encoder_hidden_states.to(dtype),
            text_attention_kwargs=text_attention_kwargs,
            timestep=timestep,
            timestep_indices=timestep_indices,
            attention_kwargs=block_attention_kwargs,
            return_pixels=return_pixels,
            return_features=return_features,
            **layout,
            **kwargs)
        if return_pixels or return_features:
            return output, auxiliary_output
        else:
            return output
