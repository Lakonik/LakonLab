import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

try:
    from torch.nn.attention import flex_attention as torch_flex_attention
    from torch.nn.attention.flex_attention import create_block_mask
except:
    torch_flex_attention = create_block_mask = None

from accelerate import init_empty_weights
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.utils import apply_lora_scale
from diffusers.models import ModelMixin, WanTransformer3DModel as _WanTransformer3DModel  # noqa: F401
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.models.transformers.transformer_wan import (
    WanTransformerBlock, WanRotaryPosEmbed, WanTimeTextImageEmbedding, FP32LayerNorm,
    WanAttnProcessor, _get_qkv_projections, _get_added_kv_projections)

from ...builder import MODULES
from ..dwd_head import DWDHeadMixin
from lakonlab.ops import quantize_frozen_linears
from ..utils import autocast_patch, flex_freeze
from .attention import set_attention_backends
from lakonlab.utils import get_root_logger
from lakonlab.runner.checkpoint import _load_cached_checkpoint, load_full_state_dict


autocast_patch(FP32LayerNorm, dtype=torch.float32, enabled=False)


_COMPILED_WAN_FLEX_ATTENTION = None


class WanTimeTextImageEmbeddingFix(WanTimeTextImageEmbedding):

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        timestep_seq_len: Optional[int] = None,
    ):
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        temb = self.time_embedder(timestep.type_as(encoder_hidden_states))
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanTransformerBlockFix(WanTransformerBlock):

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        self_attention_mask: Optional[Any] = None,
    ) -> torch.Tensor:
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).unbind(dim=2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states, None, self_attention_mask, rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


class FlexWanAttnProcessor(WanAttnProcessor):

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                # Reference: https://github.com/huggingface/diffusers/pull/12909
                parallel_config=None,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = self._compiled_flex_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attention_mask,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    @staticmethod
    def _compiled_flex_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[Any],
    ) -> torch.Tensor:
        global _COMPILED_WAN_FLEX_ATTENTION

        if torch_flex_attention is None:
            raise RuntimeError("Compiled Flex attention requires torch.nn.attention.flex_attention.")

        if _COMPILED_WAN_FLEX_ATTENTION is None:
            _COMPILED_WAN_FLEX_ATTENTION = torch.compile(
                torch_flex_attention.flex_attention,
                dynamic=False,
            )

        query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
        hidden_states = _COMPILED_WAN_FLEX_ATTENTION(
            query=query,
            key=key,
            value=value,
            score_mod=None,
            block_mask=attn_mask,
            scale=None,
            enable_gqa=False,
            return_lse=False,
        )
        return hidden_states.permute(0, 2, 1, 3)


@MODULES.register_module()
class _WanTransformer3DModelMod(DWDHeadMixin, _WanTransformer3DModel):

    # override the native list (time_embedder, scale_shift_table, norms): keeping learnable
    # parameters in fp32 breaks FSDP1 flat-parameter sharding
    _keep_in_fp32_modules = None
    _no_split_modules = ['WanTransformerBlockFix']
    _repeated_blocks = ['WanTransformerBlockFix']
    _cache_storage = dict()

    def get_cache_hidden_states(self):
        return self._cache_storage['hidden_states']

    def clear_cache(self):
        self._cache_storage.clear()

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: str = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: int | None = None,
        added_kv_proj_dim: int | None = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: int | None = None,
        causal_attention: bool = False,
        causal_num_frames_per_block: int = 1,
        secondary_temb_start_layer: int | None = None,
        dwd_head_config: dict | None = None,
    ) -> None:
        super(_WanTransformer3DModel, self).__init__()

        self._validate_causal_config(patch_size, causal_attention, causal_num_frames_per_block)

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbeddingFix(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        self.secondary_temb_start_layer = secondary_temb_start_layer
        if secondary_temb_start_layer is not None:
            assert 0 <= secondary_temb_start_layer < num_layers
            self.secondary_time_embedder = TimestepEmbedding(
                in_channels=freq_dim, time_embed_dim=inner_dim)
            self.secondary_time_proj = nn.Linear(inner_dim, inner_dim * 6)

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlockFix(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        # 5. Optional DWD heads
        if dwd_head_config is not None:
            self.init_dwd_head(inner_dim, **dwd_head_config)

        self.gradient_checkpointing = False

        self._causal_attention_mask_cache: dict[tuple[Any, ...], Any] = {}
        if causal_attention:
            self._enable_causal_attention_backend()

    @staticmethod
    def _validate_causal_config(
        patch_size: Tuple[int, ...],
        causal_attention: bool,
        causal_num_frames_per_block: int,
    ) -> None:
        if not causal_attention:
            return

        if create_block_mask is None:
            raise ValueError(
                "Causal attention requires `create_block_mask` from `torch.nn.attention.flex_attention`, "
                "which is not available in this version of PyTorch.")

        if causal_num_frames_per_block < 1:
            raise ValueError("`causal_num_frames_per_block` must be >= 1 when causal attention is enabled.")
        patch_frames = patch_size[0]
        if causal_num_frames_per_block % patch_frames != 0:
            raise ValueError(
                "`causal_num_frames_per_block` must be divisible by patch_size[0] when causal attention is enabled."
            )

    def _get_causal_attention_mask(
        self,
        post_patch_num_frames: int,
        post_patch_height: int,
        post_patch_width: int,
        device: torch.device,
    ) -> Optional[Any]:
        if not self.config.causal_attention:
            return None

        block_patch_frames = self.config.causal_num_frames_per_block // self.config.patch_size[0]
        tokens_per_frame = post_patch_height * post_patch_width

        cache_key = (
            device.type,
            device.index,
            post_patch_num_frames,
            tokens_per_frame,
            block_patch_frames,
        )
        cached_mask = self._causal_attention_mask_cache.get(cache_key)
        if cached_mask is not None:
            return cached_mask

        seq_len = post_patch_num_frames * tokens_per_frame

        def causal_mask(batch_idx, head_idx, q_idx, kv_idx):
            q_block = (q_idx // tokens_per_frame) // block_patch_frames
            kv_block = (kv_idx // tokens_per_frame) // block_patch_frames
            return q_block >= kv_block

        block_mask = create_block_mask(causal_mask, None, None, seq_len, seq_len, device=device)
        self._causal_attention_mask_cache[cache_key] = block_mask
        return block_mask

    def _enable_causal_attention_backend(self) -> None:
        for block in self.blocks:
            block.attn1.set_processor(FlexWanAttnProcessor())

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        cache_mode: Optional[str] = None,
        cache_config: Optional[dict] = None,
        return_features: bool = False,
    ) -> torch.Tensor | tuple:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        if cache_mode is not None:
            cache_after_block = cache_config['cache_after_block']
            if self.secondary_temb_start_layer is not None:
                assert cache_after_block < self.secondary_temb_start_layer

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        if cache_mode == 'load':
            hidden_states = self._cache_storage.pop('hidden_states')[-batch_size:]
            encoder_hidden_states = self._cache_storage.pop('encoder_hidden_states')[-batch_size:]
            rotary_emb = self._cache_storage.pop('rotary_emb')
            temb = self._cache_storage.pop('temb')[-batch_size:]
            timestep_proj = self._cache_storage.pop('timestep_proj')[-batch_size:]
            self_attention_mask = self._cache_storage.pop('self_attention_mask')

            start_block = cache_after_block + 1

        else:
            self_attention_mask = self._get_causal_attention_mask(
                post_patch_num_frames=post_patch_num_frames,
                post_patch_height=post_patch_height,
                post_patch_width=post_patch_width,
                device=hidden_states.device,
            )
            rotary_emb = self.rope(hidden_states)

            hidden_states = self.patch_embedding(hidden_states)
            hidden_states = hidden_states.flatten(2).transpose(1, 2)

            temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
                timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
            )
            if ts_seq_len is not None:
                # batch_size, seq_len, 6, inner_dim
                timestep_proj = timestep_proj.unflatten(2, (6, -1))
            else:
                # batch_size, 6, inner_dim
                timestep_proj = timestep_proj.unflatten(1, (6, -1))

            if encoder_hidden_states_image is not None:
                encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

            start_block = 0

        # 4. Transformer blocks
        for index_block in range(start_block, len(self.blocks)):
            # switch to the secondary time embedders when we reach the specified block
            if index_block == self.secondary_temb_start_layer:
                timestep_freq = self.condition_embedder.timesteps_proj(timestep)
                if ts_seq_len is not None:
                    timestep_freq = timestep_freq.unflatten(0, (-1, ts_seq_len))
                temb = self.secondary_time_embedder(timestep_freq.type_as(hidden_states))
                timestep_proj = self.secondary_time_proj(self.condition_embedder.act_fn(temb)).unflatten(
                    2 if ts_seq_len is not None else 1, (6, -1))

            block = self.blocks[index_block]
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, self_attention_mask)
            else:
                hidden_states = block(
                    hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, self_attention_mask)

            # Save cache after the specified block
            if cache_mode == 'save' and index_block == cache_after_block:
                self._cache_storage['hidden_states'] = hidden_states
                self._cache_storage['encoder_hidden_states'] = encoder_hidden_states
                self._cache_storage['rotary_emb'] = rotary_emb
                self._cache_storage['temb'] = temb
                self._cache_storage['timestep_proj'] = timestep_proj
                self._cache_storage['self_attention_mask'] = self_attention_mask

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).unbind(dim=2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        output = self.proj_out(hidden_states)

        output = output.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        output = output.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = output.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return (output, hidden_states) if return_features else output


@MODULES.register_module()
class WanTransformer3DModel(_WanTransformer3DModelMod):

    def __init__(
            self,
            *args,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            pretrained_dwd_head=None,
            torch_dtype='float32',
            autocast_dtype=None,
            freeze_exclude_fp32=True,
            freeze_exclude_autocast_dtype=None,
            freeze_quant_config=None,
            attention_backend_config=None,
            causal_attention=False,
            checkpointing=True,
            compile_forward=False,
            compile_kwargs=dict(
                mode='default',
                fullgraph=True,
                dynamic=False),
            **kwargs):
        with init_empty_weights():
            super().__init__(*args, causal_attention=causal_attention, **kwargs)
        self.init_weights(pretrained, pretrained_dwd_head)

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

        if freeze_quant_config is not None:
            assert torch_dtype in ('bfloat16', 'float16'), \
                'freeze_quant_config requires torch_dtype bfloat16 or float16'
            freeze_quant_config['exclude_keys'] = (
                ['self.proj_out', 'self.dwd_linear_head', 'self.first_frame_dwd_linear_head']
                + list(freeze_quant_config.get('exclude_keys', [])))
            quantize_frozen_linears(self, **freeze_quant_config)

        if attention_backend_config is not None:
            if causal_attention:
                raise ValueError('attention_backend_config is not supported with causal_attention')
            set_attention_backends(self, **attention_backend_config)

        if checkpointing:
            self.enable_gradient_checkpointing()

        self._compiled_forward = None
        if compile_forward:
            self._compiled_forward = torch.compile(self._forward_impl, **compile_kwargs)

    def init_weights(
            self,
            pretrained=None,
            pretrained_dwd_head=None):
        super().init_weights()

        if pretrained is not None:
            logger = get_root_logger()
            checkpoint = _load_cached_checkpoint(pretrained, map_location='cpu', logger=logger)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            if self.secondary_temb_start_layer is not None:
                # initialize the secondary time embedders from the pretrained primary weights
                secondary_temb_state_dict = dict()
                for k, v in state_dict.items():
                    if k.startswith('condition_embedder.time'):
                        secondary_k = 'secondary_' + k.removeprefix('condition_embedder.')
                        if secondary_k not in state_dict:
                            secondary_temb_state_dict[secondary_k] = v.clone()
                state_dict.update(secondary_temb_state_dict)

            if pretrained_dwd_head is not None:
                state_dict.update(_load_cached_checkpoint(
                    pretrained_dwd_head, map_location='cpu', logger=logger))

            load_full_state_dict(self, state_dict, logger=logger, assign=True)

    @staticmethod
    def unpatchify(latents, patch_size):
        p_t, p_h, p_w = patch_size
        p = p_t * p_h * p_w
        bs, c, t, h, w = latents.size()
        return latents.reshape(
            bs, p_t, p_h, p_w, c // p, t, h, w
        ).permute(
            0, 4, 5, 1, 6, 2, 7, 3
        ).reshape(
            bs, c // p, t * p_t, h * p_h, w * p_w)

    @staticmethod
    def unpack(latents, t, h, w):
        bs, _, c = latents.shape
        latents = latents.permute(0, 2, 1).reshape(bs, c, t, h, w)
        return latents

    def _forward_impl(self, hidden_states, return_pixels=False, return_features=False, **kwargs):
        output = super().forward(
            hidden_states=hidden_states, return_features=return_pixels or return_features, **kwargs)
        auxiliary_output = None
        if return_pixels or return_features:
            output, auxiliary_output = output
        if return_pixels:
            p_t, p_h, p_w = self.config.patch_size
            t, h, w = hidden_states.shape[2:]
            auxiliary_output = self.forward_dwd_head(
                auxiliary_output, (t // p_t, h // p_h, w // p_w))
        return output, auxiliary_output

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            return_pixels: bool = False,
            return_features: bool = False,
            **kwargs):
        encoder_hidden_states = encoder_hidden_states[-hidden_states.size(0):]
        if self.autocast_dtype is not None:
            dtype = getattr(torch, self.autocast_dtype)
        else:
            dtype = hidden_states.dtype
        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=dtype if self.autocast_dtype is not None else None):
            forward_fn = self._forward_impl if self._compiled_forward is None else self._compiled_forward
            output, auxiliary_output = forward_fn(
                hidden_states=hidden_states.to(dtype),
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states.to(dtype),
                encoder_hidden_states_image=None,  # CLIP image conditioning is not implemented
                return_pixels=return_pixels,
                return_features=return_features,
                **kwargs)
        if return_pixels or return_features:
            return output, auxiliary_output
        else:
            return output
