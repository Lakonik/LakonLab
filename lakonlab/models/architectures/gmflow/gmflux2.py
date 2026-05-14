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
from .gm_output import GMFlowModelOutput
from lakonlab.utils import get_root_logger
from lakonlab.runner.checkpoint import _load_cached_checkpoint, load_full_state_dict


class _GMFlux2Transformer2DModel(Flux2Transformer2DModel):

    @register_to_config
    def __init__(
            self,
            num_gaussians=16,
            constant_logstd=None,
            logstd_inner_dim=1024,
            gm_num_logstd_layers=2,
            logweights_channels=1,
            in_channels: int = 128,
            out_channels: int | None = None,
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
            guidance_embeds: bool = True):
        super(Flux2Transformer2DModel, self).__init__()

        self.num_gaussians = num_gaussians
        self.logweights_channels = logweights_channels

        self.out_channels = out_channels or in_channels
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
        self.x_embedder = nn.Linear(in_channels, self.inner_dim, bias=False)
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
        self.proj_out_means = nn.Linear(self.inner_dim, self.num_gaussians * self.out_channels)
        self.proj_out_logweights = nn.Linear(self.inner_dim, self.num_gaussians * self.logweights_channels)
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
        #         xavier_init(m.to_empty(device='cpu'), distribution='uniform')
        #
        # # Zero-out adaLN modulation layers in DiT blocks
        # for m in self.modules():
        #     if isinstance(m, (AdaLayerNormZero, AdaLayerNormZeroSingle, AdaLayerNormContinuous)):
        #         constant_init(m.linear, val=0)

        # Output layers
        constant_init(self.proj_out_means.to_empty(device='cpu'), val=0)
        rand_noise = torch.randn((self.num_gaussians * self.out_channels // self.logweights_channels)) * 0.1
        self.proj_out_means.bias.data.copy_(rand_noise[:, None].expand(-1, self.logweights_channels).flatten())
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
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            timestep: torch.LongTensor = None,
            img_ids: torch.Tensor = None,
            txt_ids: torch.Tensor = None,
            guidance: torch.Tensor = None,
            joint_attention_kwargs: dict[str, Any] | None = None):
        # 0. Handle input arguments

        num_txt_tokens = encoder_hidden_states.shape[1]

        # 1. Calculate timestep embedding and modulation parameters
        timestep = timestep.to(hidden_states.dtype) * 1000

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = self.time_guidance_embed(timestep, guidance)

        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)

        # 2. Input projection for image (hidden_states) and conditioning text (encoder_hidden_states)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # 3. Calculate RoPE embeddings from image and text tokens
        # NOTE: the below logic means that we can't support batched inference with images of different resolutions or
        # text prompts of differents lengths. Is this a use case we want to support?
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        # 4. Double Stream Transformer Blocks
        for index_block, block in enumerate(self.transformer_blocks):
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
        # Concatenate text and image streams for single-block inference
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # 5. Single Stream Transformer Blocks
        for index_block, block in enumerate(self.single_transformer_blocks):
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
        # Remove text tokens from concatenated stream
        hidden_states = hidden_states[:, num_txt_tokens:, ...]

        # 6. Output layers
        hidden_states = self.norm_out(hidden_states, temb)

        bs, seq_len, _ = hidden_states.size()
        out_means = self.proj_out_means(hidden_states).reshape(
            bs, seq_len, self.num_gaussians, self.out_channels)
        out_logweights = self.proj_out_logweights(hidden_states).reshape(
            bs, seq_len, self.num_gaussians, self.logweights_channels).log_softmax(dim=-2)
        if self.constant_logstd is None:
            out_logstds = self.proj_out_logstds(temb.detach()).reshape(bs, 1, 1, 1)
        else:
            out_logstds = hidden_states.new_full((bs, 1, 1, 1), float(self.constant_logstd))

        return GMFlowModelOutput(
            means=out_means,
            logweights=out_logweights,
            logstds=out_logstds)


@MODULES.register_module()
class GMFlux2Transformer2DModel(_GMFlux2Transformer2DModel):

    def __init__(
            self,
            *args,
            patch_size=2,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            pretrained_adapter=None,
            torch_dtype='float32',
            autocast_dtype=None,
            freeze_exclude_fp32=True,
            freeze_exclude_autocast_dtype='float32',
            checkpointing=True,
            use_lora=False,
            lora_target_modules=None,
            lora_rank=16,
            lora_dropout=0.0,
            **kwargs):
        with init_empty_weights():
            super().__init__(*args, **kwargs)
        self.patch_size = patch_size
        assert self.patch_size * self.patch_size == self.logweights_channels

        self.init_weights(pretrained, pretrained_adapter)

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

    def init_weights(self, pretrained=None, pretrained_adapter=None):
        super().init_weights()
        if pretrained is not None:
            logger = get_root_logger()
            checkpoint = _load_cached_checkpoint(pretrained, map_location='cpu', logger=logger)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            # expand the output channels
            if 'proj_out.weight' in state_dict and state_dict['proj_out.weight'].size(0) == self.out_channels:
                state_dict['proj_out_means.weight'] = state_dict['proj_out.weight'][None].expand(
                    self.num_gaussians, -1, -1).reshape(self.num_gaussians * self.out_channels, -1)
                del state_dict['proj_out.weight']
            if self.constant_logstd is None and 'proj_out_means.weight' in state_dict:
                self.proj_out_logstds[-1].bias.data[:1] = math.log(0.05)  # reduce the initial logstd
            if pretrained_adapter is not None:
                adapter_state_dict = _load_cached_checkpoint(
                    pretrained_adapter, map_location='cpu', logger=logger)
                lora_state_dict = dict()
                for k, v in adapter_state_dict.items():
                    if 'lora' in k:
                        lora_state_dict[k] = v
                    else:
                        state_dict[k] = v
                load_full_state_dict(self, state_dict, logger=logger, assign=True)
                if len(lora_state_dict) > 0:
                    self.load_lora_adapter(lora_state_dict, prefix=None)
                    self.fuse_lora()
                    self.unload_lora()
            else:
                load_full_state_dict(self, state_dict, logger=logger, assign=True)

    @staticmethod
    def _prepare_latent_ids(latents):
        """
        Modified from Diffusers
        """
        batch_size, _, height, width = latents.shape

        t = torch.arange(1)  # [0] - time dimension
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)  # [0] - layer dimension

        # Create position IDs: (H*W, 4)
        latent_ids = torch.cartesian_prod(t, h, w, l)

        # Expand to batch: (B, H*W, 4)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        return latent_ids.to(device=latents.device)

    @staticmethod
    def _prepare_condition_latent_ids(
            image_latents: List[torch.Tensor],  # [(1, C, H, W), (1, C, H, W), ...]
            scale: int = 10):
        """
        Modified from Diffusers
        """
        if not isinstance(image_latents, list):
            raise ValueError(f"Expected `image_latents` to be a list, got {type(image_latents)}.")

        # create time offset for each reference image
        t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
        t_coords = [t.view(-1) for t in t_coords]

        image_latent_ids = []
        for x, t in zip(image_latents, t_coords):
            _, _, h, w = x.shape
            x_ids = torch.cartesian_prod(t, torch.arange(h), torch.arange(w), torch.arange(1))
            image_latent_ids.append(x_ids)

        image_latent_ids = torch.cat(image_latent_ids, dim=0)
        image_latent_ids = image_latent_ids.unsqueeze(0).expand(image_latents[0].size(0), -1, -1)

        return image_latent_ids.to(device=image_latents[0].device)

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

    def patchify(self, latents):
        if self.patch_size > 1:
            bs, c, h, w = latents.size()
            latents = latents.reshape(
                bs, c, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size
            ).permute(
                0, 1, 3, 5, 2, 4
            ).reshape(
                bs, c * self.patch_size * self.patch_size, h // self.patch_size, w // self.patch_size)
        return latents

    def unpatchify(self, gm):
        if self.patch_size > 1:
            bs, k, c, h, w = gm['means'].size()
            gm['means'] = gm['means'].reshape(
                bs, k, c // (self.patch_size * self.patch_size), self.patch_size, self.patch_size, h, w
            ).permute(
                0, 1, 2, 5, 3, 6, 4
            ).reshape(
                bs, k, c // (self.patch_size * self.patch_size), h * self.patch_size, w * self.patch_size)
            gm['logweights'] = gm['logweights'].reshape(
                bs, k, 1, self.patch_size, self.patch_size, h, w
            ).permute(
                0, 1, 2, 5, 3, 6, 4
            ).reshape(
                bs, k, 1, h * self.patch_size, w * self.patch_size)
        return gm

    @staticmethod
    def _pack_latents(latents):
        bs, c, h, w = latents.shape
        latents = latents.reshape(bs, c, h * w).permute(0, 2, 1)
        return latents

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            condition_latents: Optional[torch.Tensor] = None,
            **kwargs):
        hidden_states = self.patchify(hidden_states)
        img_ids = self._prepare_latent_ids(hidden_states)
        bs, c, h, w = hidden_states.size()
        if self.autocast_dtype is not None:
            dtype = getattr(torch, self.autocast_dtype)
        else:
            dtype = hidden_states.dtype
        hidden_states = self._pack_latents(hidden_states)
        txt_ids = self._prepare_text_ids(encoder_hidden_states)

        input_hidden_states = hidden_states.to(dtype)
        input_img_ids = img_ids
        if condition_latents is not None:
            condition_latents = [self.patchify(condition_latents)]  # currently only supports one condition image
            condition_latent_ids = self._prepare_condition_latent_ids(condition_latents)
            condition_latents = torch.cat([self._pack_latents(x).to(dtype) for x in condition_latents], dim=1)
            input_hidden_states = torch.cat([hidden_states, condition_latents], dim=1)
            input_img_ids = torch.cat([img_ids, condition_latent_ids], dim=1)

        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=dtype if self.autocast_dtype is not None else None):
            output = super().forward(
                hidden_states=input_hidden_states,
                encoder_hidden_states=encoder_hidden_states.to(dtype),
                timestep=timestep,
                img_ids=input_img_ids,
                txt_ids=txt_ids,
                **kwargs)

        output['means'] = output['means'][:, :hidden_states.size(1)].permute(0, 2, 3, 1).reshape(
            bs, self.num_gaussians, self.out_channels, h, w)
        output['logweights'] = output['logweights'][:, :hidden_states.size(1)].permute(0, 2, 3, 1).reshape(
            bs, self.num_gaussians, self.logweights_channels, h, w)
        output['logstds'] = output['logstds'].unsqueeze(-1)  # (bs, 1, 1, 1, 1)
        return self.unpatchify(output)
