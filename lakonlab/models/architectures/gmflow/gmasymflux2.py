from typing import Any, Optional, List
import math

import torch
import torch.nn as nn

from accelerate import init_empty_weights
from diffusers.models import ModelMixin  # noqa: F401
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Transformer2DModel, Flux2PosEmbed, Flux2TransformerBlock, Flux2SingleTransformerBlock,
    Flux2TimestepGuidanceEmbeddings, Flux2Modulation)
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.configuration_utils import register_to_config
from diffusers.utils import apply_lora_scale
from peft import LoraConfig
from mmcv.cnn import constant_init, xavier_init

from ...builder import MODULES
from ..utils import flex_freeze
from ..asymflow.asymflux2 import _AsymFlux2Transformer2DModel
from .gm_output import GMFlowModelOutput
from lakonlab.utils import get_root_logger
from lakonlab.runner.checkpoint import _load_cached_checkpoint, load_full_state_dict


class _GMAsymFlux2Transformer2DModel(_AsymFlux2Transformer2DModel):

    @register_to_config
    def __init__(
            self,
            num_gaussians=8,
            constant_logstd=None,
            logstd_inner_dim=1024,
            gm_num_logstd_layers=2,
            patch_size=16,
            in_channels: int = 3,
            basis_rank: int = 128,
            num_layers: int = 8,
            num_single_layers: int = 48,
            attention_head_dim: int = 128,
            num_attention_heads: int = 48,
            joint_attention_dim: int = 15360,
            timestep_guidance_channels: int = 256,
            mlp_ratio: float = 3.0,
            axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32),
            rope_theta: int = 2000,
            eps: float = 1e-6,
            sigma_min: float = 1e-4,
            num_timesteps=1,
            use_calibration: bool = True,
            guidance_embeds: bool = True):
        super(Flux2Transformer2DModel, self).__init__()

        self.num_gaussians = num_gaussians

        self.patch_size = patch_size

        self.in_channels = in_channels
        self.out_channels = in_channels

        self.inner_dim = num_attention_heads * attention_head_dim

        # 1. Sinusoidal positional embedding for RoPE on image and text tokens
        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=axes_dims_rope)

        # 2. Combined timestep + guidance embedding
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels,
            embedding_dim=self.inner_dim,
            bias=False,
            guidance_embeds=guidance_embeds,
        )

        # 3. Modulation (double stream and single stream blocks share modulation parameters, resp.)
        # Two sets of shift/scale/gate modulation parameters for the double stream attn and FF sub-blocks
        self.double_stream_modulation_img = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.double_stream_modulation_txt = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        # Only one set of modulation parameters as the attn and FF sub-blocks are run in parallel for single stream
        self.single_stream_modulation = Flux2Modulation(self.inner_dim, mod_param_sets=1, bias=False)

        # 4. Input projections
        self.x_embedder = nn.Linear(in_channels * (patch_size ** 2), self.inner_dim, bias=False)
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim, bias=False)

        # 5. Double Stream Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                Flux2TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_layers)
            ]
        )

        # 6. Single Stream Transformer Blocks
        self.single_transformer_blocks = nn.ModuleList(
            [
                Flux2SingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_single_layers)
            ]
        )

        # 7. Output layers
        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=eps, bias=False
        )
        self.proj_out_means = nn.Linear(self.inner_dim, self.num_gaussians * self.out_channels * (patch_size ** 2))
        self.proj_out_logweights = nn.Linear(self.inner_dim, self.num_gaussians * (patch_size ** 2))
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

        # 8. AsymFlow attributes and buffers
        self.basis_rank = basis_rank
        self.sigma_min = sigma_min
        self.num_timesteps = num_timesteps
        self.init_asymflow_buffers(
            self.in_channels * (patch_size ** 2), self.basis_rank, use_calibration=use_calibration)

        self.gradient_checkpointing = False

    def init_weights(self):
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         xavier_init(m.to_empty(device='cpu'), distribution='uniform')
        #
        # # Zero-out adaLN modulation layers in DiT blocks
        # for m in self.modules():
        #     if isinstance(m, (AdaLayerNormZero, AdaLayerNormZeroSingle, AdaLayerNormContinuous)):
        #         constant_init(m.linear, val=0)

        # Output layers
        constant_init(self.proj_out_means.to_empty(device='cpu'), val=0)
        rand_noise = torch.randn((self.num_gaussians * self.out_channels)) * 0.05
        self.proj_out_means.bias.data.copy_(rand_noise[:, None].expand(-1, self.patch_size ** 2).flatten())
        constant_init(self.proj_out_logweights.to_empty(device='cpu'), val=0)
        if self.constant_logstd is None:
            # logstd layers
            for m in self.proj_out_logstds:
                if isinstance(m, nn.Linear):
                    xavier_init(m.to_empty(device='cpu'), distribution='uniform')
            constant_init(self.proj_out_logstds[-1], val=0)

    @apply_lora_scale("joint_attention_kwargs")
    def forward(
            self,
            x_t: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            condition_latents: List[torch.Tensor] | None = None,
            txt_ids: torch.Tensor = None,
            guidance: torch.Tensor = None,
            joint_attention_kwargs: dict[str, Any] | None = None,
            cache_mode: Optional[str] = None,
            cache_config: Optional[dict] = None):
        x_t = self.patchify(x_t, self.patch_size)
        img_ids = self._prepare_latent_ids(x_t)

        bs, _, h, w = x_t.size()
        x_t_packed = self.pack(x_t)
        num_x_tokens = x_t_packed.size(1)
        packed_ndim = x_t_packed.dim()
        num_txt_tokens = encoder_hidden_states.shape[1]

        # scale and timestep calibration
        calibration = self.asymflow_calibration(timestep, bs, packed_ndim)

        cache_after_block = None
        cache_after_single_block = None
        if cache_mode is not None:
            cache_after_block = cache_config.get('cache_after_block', None)
            cache_after_single_block = cache_config.get('cache_after_single_block', None)
            assert (cache_after_block is None) != (cache_after_single_block is None)
        load_dual_block_cache = cache_mode == 'load' and cache_after_block is not None
        load_single_block_cache = cache_mode == 'load' and cache_after_single_block is not None

        if cache_mode == 'load':
            hidden_states = self._cache_storage.pop('hidden_states')[-bs:]
            temb = self._cache_storage.pop('temb')[-bs:]
            single_stream_mod = self._cache_storage.pop('single_stream_mod')[-bs:]
            concat_rotary_emb = self._cache_storage.pop('concat_rotary_emb')
            if load_dual_block_cache:
                encoder_hidden_states = self._cache_storage.pop('encoder_hidden_states')[-bs:]
                double_stream_mod_img = self._cache_storage.pop('double_stream_mod_img')[-bs:]
                double_stream_mod_txt = self._cache_storage.pop('double_stream_mod_txt')[-bs:]

        else:
            if self.use_calibration:
                hidden_states = x_t_packed * calibration.k.to(x_t_packed.dtype)
            else:
                hidden_states = x_t_packed

            input_img_ids = img_ids
            if condition_latents is not None:
                condition_hidden_states = [self.patchify(z, self.patch_size) for z in condition_latents]
                condition_latent_ids = self._prepare_condition_latent_ids(condition_hidden_states)
                if self.use_calibration:
                    condition_hidden_states = [self.pack(z) / calibration.s for z in condition_hidden_states]
                else:
                    condition_hidden_states = [self.pack(z) for z in condition_hidden_states]
                hidden_states = torch.cat([hidden_states] + condition_hidden_states, dim=1)
                input_img_ids = torch.cat([img_ids, condition_latent_ids], dim=1)

            hidden_states = self.x_embedder(hidden_states)

            if guidance is not None:
                guidance = guidance.to(hidden_states.dtype) * 1000
            temb = self.time_guidance_embed(calibration.timestep.to(hidden_states.dtype) * 1000, guidance)

            double_stream_mod_img = self.double_stream_modulation_img(temb)
            double_stream_mod_txt = self.double_stream_modulation_txt(temb)
            single_stream_mod = self.single_stream_modulation(temb)

            encoder_hidden_states = self.context_embedder(encoder_hidden_states)
            concat_rotary_emb = self._get_rotary_emb(input_img_ids, txt_ids)

        start_block = 0
        if load_dual_block_cache:
            start_block = cache_after_block + 1

        if not load_single_block_cache:
            for index_block in range(start_block, len(self.transformer_blocks)):
                block = self.transformer_blocks[index_block]
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        double_stream_mod_img,
                        double_stream_mod_txt,
                        concat_rotary_emb,
                        joint_attention_kwargs,
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb_mod_img=double_stream_mod_img,
                        temb_mod_txt=double_stream_mod_txt,
                        image_rotary_emb=concat_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                if cache_mode == 'save' and index_block == cache_after_block:
                    self._cache_storage['hidden_states'] = hidden_states
                    self._cache_storage['temb'] = temb
                    self._cache_storage['single_stream_mod'] = single_stream_mod
                    self._cache_storage['concat_rotary_emb'] = concat_rotary_emb
                    self._cache_storage['encoder_hidden_states'] = encoder_hidden_states
                    self._cache_storage['double_stream_mod_img'] = double_stream_mod_img
                    self._cache_storage['double_stream_mod_txt'] = double_stream_mod_txt

            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        start_block = 0
        if load_single_block_cache:
            start_block = cache_after_single_block + 1

        for index_block in range(start_block, len(self.single_transformer_blocks)):
            block = self.single_transformer_blocks[index_block]
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    None,
                    single_stream_mod,
                    concat_rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=None,
                    temb_mod=single_stream_mod,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # Save cache after the specified block
            if cache_mode == 'save' and index_block == cache_after_single_block:
                self._cache_storage['hidden_states'] = hidden_states
                self._cache_storage['temb'] = temb
                self._cache_storage['single_stream_mod'] = single_stream_mod
                self._cache_storage['concat_rotary_emb'] = concat_rotary_emb

        hidden_states = hidden_states[:, num_txt_tokens:num_txt_tokens + num_x_tokens]
        hidden_states = self.norm_out(hidden_states, temb)

        bs, seq_len, _ = hidden_states.size()
        p = self.patch_size
        p2 = p ** 2

        u_a_packed = self.proj_out_means(hidden_states)

        out_means_packed = self.asymflow_velocity(
            u_a_packed.reshape(bs, seq_len * self.num_gaussians, self.out_channels * p2),
            x_t_packed.reshape(bs, seq_len, 1, self.out_channels * p2).expand(
                -1, -1, self.num_gaussians, -1).reshape(bs, seq_len * self.num_gaussians, self.out_channels * p2),
            calibration)

        out_means = out_means_packed.reshape(
            bs, h, w, self.num_gaussians, self.out_channels, p, p
        ).permute(0, 3, 4, 1, 5, 2, 6).reshape(
            bs, self.num_gaussians, self.out_channels, h * p, w * p
        )
        out_logweights = self.proj_out_logweights(hidden_states).reshape(
            bs, h, w, self.num_gaussians, p, p
        ).permute(0, 3, 1, 4, 2, 5).reshape(
            bs, self.num_gaussians, 1, h * p, w * p
        ).log_softmax(dim=1)
        if self.constant_logstd is None:
            out_logstds = self.proj_out_logstds(
                temb.detach()).reshape(bs, 1, 1, 1, 1)
        else:
            out_logstds = hidden_states.new_full((bs, 1, 1, 1, 1), float(self.constant_logstd))

        return GMFlowModelOutput(
            means=out_means,
            logweights=out_logweights,
            logstds=out_logstds)


@MODULES.register_module()
class GMAsymFlux2Transformer2DModel(_GMAsymFlux2Transformer2DModel):

    def __init__(
            self,
            *args,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            pretrained_adapter=None,
            pretrained_adapter_scale=1.0,
            torch_dtype='float32',
            autocast_dtype=None,
            freeze_exclude_fp32=True,
            freeze_exclude_autocast_dtype=None,
            checkpointing=True,
            use_lora=False,
            lora_target_modules=None,
            lora_rank=16,
            lora_dropout=0.0,
            **kwargs):
        with init_empty_weights():
            super().__init__(*args, **kwargs)

        self.init_weights(pretrained, pretrained_adapter, pretrained_adapter_scale)

        if autocast_dtype is not None:
            assert torch_dtype == 'float32'
        self.autocast_dtype = autocast_dtype

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

    def init_weights(self, pretrained=None, pretrained_adapter=None, pretrained_adapter_scale=1.0):
        super().init_weights()
        if pretrained is not None:
            logger = get_root_logger()
            checkpoint = _load_cached_checkpoint(pretrained, map_location='cpu', logger=logger)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            c_p2 = self.out_channels * self.patch_size ** 2

            if pretrained_adapter is not None:
                adapter_state_dict = _load_cached_checkpoint(
                    pretrained_adapter, map_location='cpu', logger=logger)
                lora_state_dict = dict()
                for k, v in adapter_state_dict.items():
                    if 'lora' in k:
                        lora_state_dict[k] = v
                    else:
                        state_dict[k] = v

                # expand the output channels
                if 'proj_out.weight' in state_dict and state_dict['proj_out.weight'].size(0) == c_p2:
                    state_dict['proj_out_means.weight'] = state_dict['proj_out.weight'][None].expand(
                        self.num_gaussians, -1, -1).reshape(self.num_gaussians * c_p2, -1)
                    del state_dict['proj_out.weight']
                if self.constant_logstd is None and 'proj_out_means.weight' in state_dict:
                    self.proj_out_logstds[-1].bias.data[:1] = math.log(0.05)  # reduce the initial logstd

                load_full_state_dict(self, state_dict, logger=logger, assign=True)
                if len(lora_state_dict) > 0:
                    self.load_lora_adapter(lora_state_dict, prefix=None)
                    self.fuse_lora(lora_scale=pretrained_adapter_scale, safe_fusing=True)
                    self.unload_lora()
                    self.requires_grad_(True)  # PEFT freezes the weights after fusing
            else:
                # expand the output channels
                if 'proj_out.weight' in state_dict and state_dict['proj_out.weight'].size(0) == c_p2:
                    state_dict['proj_out_means.weight'] = state_dict['proj_out.weight'][None].expand(
                        self.num_gaussians, -1, -1).reshape(self.num_gaussians * c_p2, -1)
                    del state_dict['proj_out.weight']
                if self.constant_logstd is None and 'proj_out_means.weight' in state_dict:
                    self.proj_out_logstds[-1].bias.data[:1] = math.log(0.05)  # reduce the initial logstd

                load_full_state_dict(self, state_dict, logger=logger, assign=True)

    @staticmethod
    def _prepare_text_ids(
            x: torch.Tensor,  # (B, L, D) or (L, D)
            t_coord: Optional[torch.Tensor] = None):
        """
        Copied from Diffusers
        """
        B, L, _ = x.shape
        out_ids = []

        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)

            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)

        return torch.stack(out_ids).to(device=x.device)

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            condition_latents: Optional[torch.Tensor] = None,
            **kwargs):
        if self.autocast_dtype is not None:
            dtype = getattr(torch, self.autocast_dtype)
        else:
            dtype = hidden_states.dtype
        txt_ids = self._prepare_text_ids(encoder_hidden_states)

        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=dtype if self.autocast_dtype is not None else None):
            output = super().forward(
                x_t=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states.to(dtype),
                condition_latents=[condition_latents.to(dtype)] if condition_latents is not None else None,
                txt_ids=txt_ids,
                **kwargs)

        return output
