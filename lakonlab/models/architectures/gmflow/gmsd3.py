from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn

from accelerate import init_empty_weights
from diffusers.models import ModelMixin  # noqa: F401
from diffusers.models.transformers.transformer_sd3 import (
    SD3Transformer2DModel, JointTransformerBlock)
from diffusers.models.embeddings import PatchEmbed, CombinedTimestepTextProjEmbeddings
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, SD35AdaLayerNormZeroX
from diffusers.configuration_utils import register_to_config
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from peft import LoraConfig
from mmcv.cnn import constant_init, xavier_init

from ...builder import MODULES
from ..utils import flex_freeze
from .gm_output import GMFlowModelOutput
from lakonlab.utils import get_root_logger
from lakonlab.runner.checkpoint import _load_cached_checkpoint, load_full_state_dict


class _GMSD3Transformer2DModel(SD3Transformer2DModel):

    @register_to_config
    def __init__(
            self,
            num_gaussians=16,
            constant_logstd=None,
            logstd_inner_dim=1024,
            gm_num_logstd_layers=2,
            sample_size: int = 128,
            patch_size: int = 2,
            in_channels: int = 16,
            num_layers: int = 18,
            attention_head_dim: int = 64,
            num_attention_heads: int = 18,
            joint_attention_dim: int = 4096,
            caption_projection_dim: int = 1152,
            pooled_projection_dim: int = 2048,
            out_channels: int = 16,
            pos_embed_max_size: int = 96,
            dual_attention_layers: Tuple[
                int, ...
            ] = (),  # () for sd3.0; (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) for sd3.5
            qk_norm: Optional[str] = None):
        super(SD3Transformer2DModel, self).__init__()

        self.num_gaussians = num_gaussians

        self.out_channels = out_channels if out_channels is not None else in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,  # hard-code for now.
        )
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )
        self.context_embedder = nn.Linear(joint_attention_dim, caption_projection_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                JointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    context_pre_only=i == num_layers - 1,
                    qk_norm=qk_norm,
                    use_dual_attention=True if i in dual_attention_layers else False,
                )
                for i in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_means = nn.Linear(
            self.inner_dim,
            self.config.patch_size * self.config.patch_size * self.num_gaussians * self.out_channels)
        self.proj_out_logweights = nn.Linear(
            self.inner_dim,
            self.config.patch_size * self.config.patch_size * self.num_gaussians)
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

        self.gradient_checkpointing = False

    def init_weights(self):
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         xavier_init(m, distribution='uniform')

        # # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.pos_embed.proj.weight.data
        # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # nn.init.constant_(self.pos_embed.proj.bias, 0)

        # # Zero-out adaLN modulation layers in DiT blocks
        # for m in self.modules():
        #     if isinstance(m, (AdaLayerNormZero, SD35AdaLayerNormZeroX, AdaLayerNormContinuous)):
        #         constant_init(m.linear, val=0)

        # Output layers
        constant_init(self.proj_out_means.to_empty(device='cpu'), val=0)
        rand_noise = torch.randn((self.config.num_gaussians * self.out_channels)) * 0.1
        self.proj_out_means.bias.data.copy_(rand_noise[None, :].expand(
            self.config.patch_size * self.config.patch_size, -1).flatten())
        constant_init(self.proj_out_logweights.to_empty(device='cpu'), val=0)
        if self.constant_logstd is None:
            # logstd layers
            for m in self.proj_out_logstds:
                if isinstance(m, nn.Linear):
                    xavier_init(m.to_empty(device='cpu'), distribution='uniform')
            constant_init(self.proj_out_logstds[-1], val=0)

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            pooled_projections: torch.Tensor = None,
            timestep: torch.LongTensor = None,
            block_controlnet_hidden_states: List = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            skip_layers: Optional[List[int]] = None):
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            assert joint_attention_kwargs is None or joint_attention_kwargs.get('scale', None) is None

        bs, _, h, w = hidden_states.size()
        height, width = h // self.patch_size, w // self.patch_size

        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states, ip_temb = self.image_proj(ip_adapter_image_embeds, timestep)

            joint_attention_kwargs.update(ip_hidden_states=ip_hidden_states, temb=ip_temb)

        for index_block, block in enumerate(self.transformer_blocks):
            # Skip specified layers
            is_skip = True if skip_layers is not None and index_block in skip_layers else False

            if torch.is_grad_enabled() and self.gradient_checkpointing and not is_skip:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    joint_attention_kwargs,
                )
            elif not is_skip:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.transformer_blocks) / len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[int(index_block / interval_control)]

        hidden_states = self.norm_out(hidden_states, temb)

        num_gaussians = self.config.num_gaussians
        patch_size = self.config.patch_size
        out_means = self.proj_out_means(hidden_states).reshape(
            bs, height, width, patch_size, patch_size, num_gaussians * self.out_channels
        ).permute(0, 5, 1, 3, 2, 4).reshape(
            bs, num_gaussians, self.out_channels, height * patch_size, width * patch_size)
        out_logweights = self.proj_out_logweights(hidden_states).reshape(
            bs, height, width, patch_size, patch_size, num_gaussians
        ).permute(0, 5, 1, 3, 2, 4).reshape(
            bs, num_gaussians, 1, height * patch_size, width * patch_size
        ).log_softmax(dim=1)
        if self.constant_logstd is None:
            out_logstds = self.proj_out_logstds(temb.detach()).reshape(bs, 1, 1, 1, 1)
        else:
            out_logstds = hidden_states.new_full((bs, 1, 1, 1, 1), float(self.constant_logstd))

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        return GMFlowModelOutput(
            means=out_means,
            logweights=out_logweights,
            logstds=out_logstds)


@MODULES.register_module()
class GMSD3Transformer2DModel(_GMSD3Transformer2DModel):

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
            use_lora=False,
            lora_target_modules=None,
            lora_rank=16,
            **kwargs):
        with init_empty_weights():
            super().__init__(*args, **kwargs)
        self.init_weights(pretrained)

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
        super().init_weights()
        if pretrained is not None:
            logger = get_root_logger()
            # load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
            checkpoint = _load_cached_checkpoint(pretrained, map_location='cpu', logger=logger)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            # expand the output channels
            p2 = self.config.patch_size * self.config.patch_size
            ori_out_channels = p2 * self.out_channels
            if 'proj_out.weight' in state_dict:
                if state_dict['proj_out.weight'].size(0) == ori_out_channels:
                    state_dict['proj_out_means.weight'] = state_dict['proj_out.weight'].reshape(
                        p2, 1, self.out_channels, -1
                    ).expand(-1, self.config.num_gaussians, -1, -1).reshape(
                        self.config.num_gaussians * ori_out_channels, -1)
                    del state_dict['proj_out.weight']
            if 'proj_out.bias' in state_dict:
                if state_dict['proj_out.bias'].size(0) == ori_out_channels:
                    state_dict['proj_out_means.bias'] = state_dict['proj_out.bias'].reshape(
                        p2, 1, self.out_channels
                    ).expand(-1, self.config.num_gaussians, -1).reshape(
                        self.config.num_gaussians * ori_out_channels)
                    rand_noise = torch.randn(
                        (self.config.num_gaussians * self.out_channels),
                        dtype=state_dict['proj_out_means.bias'].dtype,
                        device=state_dict['proj_out_means.bias'].device) * 0.05
                    state_dict['proj_out_means.bias'] += rand_noise[None, :].expand(p2, -1).flatten()
                    del state_dict['proj_out.bias']
            if (self.constant_logstd is None
                    and 'proj_out_means.weight' in state_dict
                    and 'proj_out_means.bias' in state_dict):
                self.proj_out_logstds[-1].bias.data = torch.full_like(
                    self.proj_out_logstds[-1].bias.data, np.log(0.05))  # reduce the initial logstd
            load_full_state_dict(self, state_dict, logger=logger, assign=True)

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            pooled_projections: torch.Tensor = None,
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
                hidden_states=hidden_states.to(dtype),
                encoder_hidden_states=encoder_hidden_states.to(dtype),
                pooled_projections=pooled_projections.to(dtype),
                timestep=timestep,
                **kwargs)
