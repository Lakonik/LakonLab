# Copyright (c) 2026 Hansheng Chen

import torch
import torch.nn as nn

from diffusers.models import DiTTransformer2DModel, ModelMixin  # noqa: F401
from diffusers.models.embeddings import PatchEmbed
from diffusers.configuration_utils import register_to_config

from ...builder import MODULES
from ..diffusers.dit import CombinedTimestepLabelEmbeddingsMod, BasicTransformerBlockMod, _DiTTransformer2DModelMod
from ..utils import flex_freeze
from lakonlab.utils import get_root_logger
from lakonlab.runner.checkpoint import _load_cached_checkpoint, load_full_state_dict
from .common import AsymFlowMixin


class _AsymDiTTransformer2DModel(AsymFlowMixin, _DiTTransformer2DModelMod):

    @register_to_config
    def __init__(
            self,
            patch_size: int = 16,
            in_channels: int = 3,
            base_rank: int = 32,
            num_timesteps=1,
            class_dropout_prob=0.0,
            num_attention_heads: int = 16,
            attention_head_dim: int = 72,
            out_channels: int | None = None,
            num_layers: int = 28,
            dropout: float = 0.0,
            norm_num_groups: int = 32,
            attention_bias: bool = True,
            sample_size: int = 32,
            activation_fn: str = 'gelu-approximate',
            num_embeds_ada_norm: int = 1000,
            upcast_attention: bool = False,
            norm_type: str = 'ada_norm_zero',
            norm_elementwise_affine: bool = False,
            norm_eps: float = 1e-5,
            sigma_min: float = 1e-4):

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

        # 4. AsymFlow attributes and buffers.
        self.num_timesteps = num_timesteps
        self.sigma_min = sigma_min
        self.init_asymflow_buffers(in_channels * (patch_size ** 2), base_rank)

    @staticmethod
    def patchify(latents, patch_size, pack_channels=True):
        bs, c, h, w = latents.shape
        latents = latents.reshape(
            bs, c, h // patch_size, patch_size, w // patch_size, patch_size)
        if pack_channels:  # patch dim before channel dim (ImageNet DiT convention)
            latents = latents.permute(
                0, 3, 5, 1, 2, 4
            ).reshape(bs, patch_size * patch_size * c, h // patch_size, w // patch_size)
        else:  # channel dim before patch dim (for consistency with other models)
            latents = latents.permute(
                0, 1, 3, 5, 2, 4
            ).reshape(bs, c, patch_size * patch_size, h // patch_size, w // patch_size)
        return latents

    @staticmethod
    def unpatchify(latents, patch_size, packed_channels=True):
        if packed_channels:
            bs, c, h, w = latents.size()
            latents = latents.reshape(
                bs, patch_size, patch_size, c // (patch_size * patch_size), h, w
            ).permute(
                0, 3, 4, 1, 5, 2
            ).reshape(
                bs, c // (patch_size * patch_size), h * patch_size, w * patch_size)
        else:
            bs, c, _, h, w = latents.size()
            latents = latents.reshape(
                bs, c, patch_size, patch_size, h, w
            ).permute(
                0, 1, 4, 2, 5, 3
            ).reshape(
                bs, c, h * patch_size, w * patch_size)
        return latents

    @staticmethod
    def pack(latents):
        bs, c, h, w = latents.shape
        latents = latents.reshape(bs, c, h * w).permute(0, 2, 1)
        return latents

    @staticmethod
    def unpack(latents, h, w):
        bs, _, c = latents.shape
        latents = latents.permute(0, 2, 1).reshape(bs, c, h, w)
        return latents

    def forward(
            self,
            x_t: torch.Tensor,
            timestep: torch.Tensor,
            **kwargs):
        bs, _, h, w = x_t.size()
        ndim = x_t.dim()
        x_t_packed = self.pack(self.patchify(x_t, self.patch_size))
        packed_ndim = x_t_packed.dim()

        # scale and timestep calibration
        calibration = self.asymflow_calibration(timestep, bs, packed_ndim)
        hidden_states = x_t * calibration.k.reshape(bs, *((ndim - 1) * [1])).to(x_t.dtype)

        u_a_packed = super().forward(
            hidden_states, calibration.timestep, unpatchify_output=False, **kwargs)

        output_packed = self.asymflow_velocity(u_a_packed, x_t_packed, calibration)

        output = self.unpatchify(self.unpack(
            output_packed.to(x_t.dtype),
            h // self.patch_size, w // self.patch_size
        ), self.patch_size)

        return output


@MODULES.register_module()
class AsymDiTTransformer2DModel(_AsymDiTTransformer2DModel):

    def __init__(
            self,
            *args,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            pretrained_linear_proj=None,
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

        self.init_weights(pretrained, pretrained_linear_proj)

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

    @staticmethod
    def _project_patch_embed_weight(patch_embed_weight, in_proj_weight, patch_size, channels):
        # Rewrite PatchEmbed as explicit patchify(p, q, c) + linear, apply the
        # pretrained projector in that basis, then convert the linear weight back
        # to conv storage (c, p, q).
        linear_weight = patch_embed_weight.permute(0, 2, 3, 1).reshape(patch_embed_weight.size(0), -1)
        expected_shape = (linear_weight.size(1), channels * patch_size * patch_size)
        if tuple(in_proj_weight.shape) != expected_shape:
            raise RuntimeError(
                f'in_proj shape mismatch: expected {expected_shape}, got {tuple(in_proj_weight.shape)}. '
                'The pretrained linear projector likely does not match the latent/pixel patch sizes for this model.'
            )
        return (linear_weight @ in_proj_weight).reshape(
            patch_embed_weight.size(0), patch_size, patch_size, channels
        ).permute(
            0, 3, 1, 2
        )

    def init_weights(self, pretrained=None, pretrained_linear_proj=None):
        super().init_weights()
        if pretrained is None:
            return

        logger = get_root_logger()
        checkpoint = _load_cached_checkpoint(pretrained, map_location='cpu', logger=logger)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # load the pretrained linear projection weights
        if pretrained_linear_proj is not None:
            linear_proj_state_dict = _load_cached_checkpoint(
                pretrained_linear_proj, map_location='cpu', logger=logger)
            dtype = state_dict['pos_embed.proj.weight'].dtype
            p = self.patch_size
            proj_mat = linear_proj_state_dict[f'proj_mat_p{p}']  # (in_channels * p2, base_rank)
            if 'pos_embed.proj.weight' in state_dict:
                in_proj_weight = proj_mat.T.to(dtype)  # (base_rank, in_channels * p2)
                state_dict['pos_embed.proj.weight'] = self._project_patch_embed_weight(
                    state_dict['pos_embed.proj.weight'],
                    in_proj_weight,
                    self.patch_size,
                    self.config.in_channels)
            if 'proj_out_2.weight' in state_dict and 'proj_out_2.bias' in state_dict:
                out_proj_weight = proj_mat.to(dtype)  # (in_channels * p2, base_rank)
                state_dict['proj_out_2.bias'] = (
                    out_proj_weight @ state_dict['proj_out_2.bias'].unsqueeze(-1)
                ).squeeze(-1)
                state_dict['proj_out_2.weight'] = (
                    out_proj_weight @ state_dict['proj_out_2.weight']
                )
            state_dict['proj_buffer'] = proj_mat
            if f'scale_p{p}' in linear_proj_state_dict:
                state_dict['scale_buffer'] = linear_proj_state_dict[f'scale_p{p}']

        load_full_state_dict(self, state_dict, logger=logger)

    def _forward_impl(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            class_labels: torch.LongTensor | None = None,
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
