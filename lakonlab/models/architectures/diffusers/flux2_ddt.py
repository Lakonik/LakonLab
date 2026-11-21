from typing import Optional, List, Any

import torch
import torch.nn as nn

from accelerate import init_empty_weights
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Transformer2DModel, Flux2PosEmbed, Flux2TransformerBlock, Flux2SingleTransformerBlock,
    Flux2TimestepGuidanceEmbeddings, Flux2Modulation)
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.utils import apply_lora_scale
from diffusers.configuration_utils import register_to_config
from peft import LoraConfig
from mmcv.cnn import constant_init, xavier_init

from lakonlab.models.builder import MODULES
from lakonlab.models.architectures.utils import flex_freeze
from lakonlab.utils import get_root_logger
from lakonlab.runner.checkpoint import _load_cached_checkpoint, load_full_state_dict


class DDTAdaLayerNormContinuous(AdaLayerNormContinuous):

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class _Flux2DDTTransformer2DModel(Flux2Transformer2DModel):

    @register_to_config
    def __init__(
            self,
            patch_size: int = 16,
            in_channels: int = 3,
            num_layers: int = 8,
            num_single_layers: int = 48,
            attention_head_dim: int = 128,
            num_attention_heads: int = 48,
            joint_attention_dim: int = 15360,
            timestep_guidance_channels: int = 256,
            mlp_ratio: float = 3.0,
            axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32),
            rope_theta: int = 2000,
            ddt_num_layers: int = 2,
            ddt_attention_head_dim: int = 192,
            ddt_num_attention_heads: int = 48,
            ddt_axes_dims_rope: tuple[int, ...] = (48, 48, 48, 48),
            eps: float = 1e-6,
            guidance_embeds: bool = True):
        super(Flux2Transformer2DModel, self).__init__()

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

        # 7. DDT
        self.ddt_inner_dim = ddt_num_attention_heads * ddt_attention_head_dim
        self.ddt_pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=ddt_axes_dims_rope)
        self.s_embedder = nn.Linear(in_channels * (patch_size ** 2), self.ddt_inner_dim, bias=False)
        self.s_projector = (
            nn.Linear(self.inner_dim, self.ddt_inner_dim)
            if self.inner_dim != self.ddt_inner_dim
            else nn.Identity()
        )
        self.ddt_modulation = Flux2Modulation(self.ddt_inner_dim, mod_param_sets=1, bias=False)
        self.ddt_transformer_blocks = nn.ModuleList(
            [
                Flux2SingleTransformerBlock(
                    dim=self.ddt_inner_dim,
                    num_attention_heads=ddt_num_attention_heads,
                    attention_head_dim=ddt_attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(ddt_num_layers)
            ]
        )

        # 8. Output layers
        self.norm_out = DDTAdaLayerNormContinuous(
            self.ddt_inner_dim, self.ddt_inner_dim, elementwise_affine=False, eps=eps, bias=False
        )
        self.proj_out = nn.Linear(
            self.ddt_inner_dim, self.out_channels * (patch_size ** 2), bias=False
        )

        self.gradient_checkpointing = False

    def init_weights(self):
        # DDT initialization
        xavier_init(self.s_embedder.to_empty(device='cpu'), distribution='uniform')
        if isinstance(self.s_projector, nn.Linear):
            xavier_init(self.s_projector.to_empty(device='cpu'), distribution='uniform')

        # DDT modulation should not be zero initialized
        xavier_init(self.ddt_modulation.linear.to_empty(device='cpu'), distribution='uniform')

        for block in self.ddt_transformer_blocks:
            block.to_empty(device='cpu')
            xavier_init(block.attn.to_qkv_mlp_proj, distribution='uniform')
            block.attn.norm_q.reset_parameters()
            block.attn.norm_k.reset_parameters()
            xavier_init(block.attn.to_out, distribution='uniform')

        # DDT modulation should not be zero initialized
        xavier_init(self.norm_out.linear.to_empty(device='cpu'), distribution='uniform')
        constant_init(self.proj_out.to_empty(device='cpu'), val=0.0)

    @apply_lora_scale("joint_attention_kwargs")
    def forward(
        self,
        x_t: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[dict[str, Any]] = None,
    ):
        # 0. Handle input arguments
        batch_size, num_txt_tokens = encoder_hidden_states.shape[:2]

        # 1. Calculate timestep embedding and modulation parameters
        timestep = timestep.to(x_t.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(x_t.dtype) * 1000

        temb = self.time_guidance_embed(timestep, guidance)

        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)

        # 2. Input projection
        hidden_states = self.x_embedder(x_t)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # 3. Calculate RoPE embeddings
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
        for block in self.single_transformer_blocks:
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

        # 6. DDT
        ddt_embed = self.s_projector(hidden_states)
        ddt_mod = self.ddt_modulation(ddt_embed)
        ddt_hidden_states = self.s_embedder(x_t)
        ddt_image_rotary_emb = self.ddt_pos_embed(img_ids)
        for block in self.ddt_transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                ddt_hidden_states = self._gradient_checkpointing_func(
                    block,
                    ddt_hidden_states,
                    None,
                    ddt_mod,
                    ddt_image_rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                ddt_hidden_states = block(
                    hidden_states=ddt_hidden_states,
                    encoder_hidden_states=None,
                    temb_mod=ddt_mod,
                    image_rotary_emb=ddt_image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

        # 7. Output layers
        ddt_hidden_states = self.norm_out(ddt_hidden_states, ddt_embed)
        output = self.proj_out(ddt_hidden_states)

        return output


@MODULES.register_module()
class Flux2DDTTransformer2DModel(_Flux2DDTTransformer2DModel):

    def __init__(
            self,
            *args,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            pretrained_adapter=None,
            pretrained_linear_proj=None,
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

        self.init_weights(pretrained, pretrained_adapter, pretrained_linear_proj)

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

    def init_weights(
            self,
            pretrained=None,
            pretrained_adapter=None,
            pretrained_linear_proj=None):
        super().init_weights()

        if pretrained is not None:
            logger = get_root_logger()
            checkpoint = _load_cached_checkpoint(pretrained, map_location='cpu', logger=logger)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            if pretrained_linear_proj is not None:
                linear_proj_state_dict = _load_cached_checkpoint(
                    pretrained_linear_proj, map_location='cpu', logger=logger)
                dtype = state_dict['x_embedder.weight'].dtype
                p = self.patch_size
                proj_mat = linear_proj_state_dict[f'proj_mat_p{p}']  # (in_channels * p2, basis_rank)
                if 'x_embedder.weight' in state_dict:
                    in_proj_weight = proj_mat.T.to(dtype)  # (basis_rank, in_channels * p2)
                    state_dict['x_embedder.weight'] = state_dict['x_embedder.weight'] @ in_proj_weight

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
                    self.fuse_lora(safe_fusing=True)
                    self.unload_lora()
                    self.requires_grad_(True)  # PEFT freezes the weights after fusing
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

    def unpatchify(self, latents):
        if self.patch_size > 1:
            bs, c, h, w = latents.size()
            latents = latents.reshape(
                bs, c // (self.patch_size * self.patch_size), self.patch_size, self.patch_size, h, w
            ).permute(
                0, 1, 4, 2, 5, 3
            ).reshape(
                bs, c // (self.patch_size * self.patch_size), h * self.patch_size, w * self.patch_size)
        return latents

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
        dtype = hidden_states.dtype
        hidden_states = self._pack_latents(hidden_states)
        txt_ids = self._prepare_text_ids(encoder_hidden_states)

        input_hidden_states = hidden_states
        input_img_ids = img_ids
        if condition_latents is not None:
            condition_latents = [self.patchify(condition_latents)]  # currently only supports one condition image
            condition_latent_ids = self._prepare_condition_latent_ids(condition_latents)
            condition_latents = torch.cat([self._pack_latents(x).to(dtype) for x in condition_latents], dim=1)
            input_hidden_states = torch.cat([hidden_states, condition_latents], dim=1)
            input_img_ids = torch.cat([img_ids, condition_latent_ids], dim=1)

        output = super().forward(
            x_t=input_hidden_states,
            encoder_hidden_states=encoder_hidden_states.to(dtype),
            timestep=timestep,
            img_ids=input_img_ids,
            txt_ids=txt_ids,
            **kwargs)

        output = output[:, :hidden_states.size(1)].permute(0, 2, 1).reshape(bs, c, h, w)
        return self.unpatchify(output)
