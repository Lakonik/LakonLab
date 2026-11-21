from typing import Optional, List, Any

import torch

from accelerate import init_empty_weights
from diffusers.configuration_utils import register_to_config
from diffusers.models import Flux2Transformer2DModel as _Flux2Transformer2DModel
from diffusers.models.transformers.transformer_flux2 import Flux2TimestepGuidanceEmbeddings, Flux2Modulation
from diffusers.utils import apply_lora_scale
from peft import LoraConfig

from ...builder import MODULES
from ..dwd_head import DWDHeadMixin
from ..utils import flex_freeze, fp32_module_forward
from lakonlab.ops import quantize_frozen_linears
from lakonlab.utils import get_root_logger
from lakonlab.runner.checkpoint import _load_cached_checkpoint, load_full_state_dict


class _Flux2Transformer2DModelCache(DWDHeadMixin, _Flux2Transformer2DModel):

    _cache_storage = dict()

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 128,
        out_channels: int | None = None,
        num_layers: int = 8,
        num_single_layers: int = 48,
        secondary_temb_start_single_layer: int | None = None,
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        joint_attention_dim: int = 15360,
        timestep_guidance_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 2000,
        eps: float = 1e-6,
        guidance_embeds: bool = True,
        dwd_head_config: dict | None = None,
    ):
        super().__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            timestep_guidance_channels=timestep_guidance_channels,
            mlp_ratio=mlp_ratio,
            axes_dims_rope=axes_dims_rope,
            rope_theta=rope_theta,
            eps=eps,
            guidance_embeds=guidance_embeds,
        )
        self.secondary_temb_start_single_layer = secondary_temb_start_single_layer
        if secondary_temb_start_single_layer is not None:
            assert 0 <= secondary_temb_start_single_layer < num_single_layers
            self.secondary_time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
                in_channels=timestep_guidance_channels,
                embedding_dim=self.inner_dim,
                bias=False,
                guidance_embeds=guidance_embeds,
            )
            self.secondary_single_stream_modulation = Flux2Modulation(self.inner_dim, mod_param_sets=1, bias=False)

        if dwd_head_config is not None:
            self.init_dwd_head(self.inner_dim, **dwd_head_config)

    def get_cache_hidden_states(self):
        return self._cache_storage['hidden_states']

    def clear_cache(self):
        self._cache_storage.clear()

    @apply_lora_scale("joint_attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[dict[str, Any]] = None,
        cache_mode: Optional[str] = None,
        cache_config: Optional[dict] = None,
        fp32_final_layer: bool = False,
        return_features: bool = False,
    ):
        # 0. Handle input arguments
        batch_size, num_txt_tokens = encoder_hidden_states.shape[:2]

        if cache_mode is not None:
            cache_after_single_block = cache_config['cache_after_single_block']
            if self.secondary_temb_start_single_layer is not None:
                assert cache_after_single_block < self.secondary_temb_start_single_layer

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        # When loading from cache, skip to single-stream block processing
        if cache_mode == 'load':
            hidden_states = self._cache_storage.pop('hidden_states')[-batch_size:]
            temb = self._cache_storage.pop('temb')[-batch_size:]
            single_stream_mod = self._cache_storage.pop('single_stream_mod')[-batch_size:]
            concat_rotary_emb = self._cache_storage.pop('concat_rotary_emb')

        else:
            # 1. Calculate timestep embedding and modulation parameters
            temb = self.time_guidance_embed(timestep, guidance)

            double_stream_mod_img = self.double_stream_modulation_img(temb)
            double_stream_mod_txt = self.double_stream_modulation_txt(temb)
            single_stream_mod = self.single_stream_modulation(temb)

            # 2. Input projection
            hidden_states = self.x_embedder(hidden_states)
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
        start_block = 0
        if cache_mode == 'load':
            start_block = cache_after_single_block + 1

        for index_block in range(start_block, len(self.single_transformer_blocks)):
            # Update temb and single_stream_mod when we reach the specified block
            if index_block == self.secondary_temb_start_single_layer:
                temb = self.secondary_time_guidance_embed(timestep, guidance)
                single_stream_mod = self.secondary_single_stream_modulation(temb)

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

        # Remove text tokens from concatenated stream
        hidden_states = hidden_states[:, num_txt_tokens:, ...]

        # 6. Output layers
        hidden_states = fp32_module_forward(
            self.norm_out, hidden_states, temb, enabled=fp32_final_layer)
        output = fp32_module_forward(self.proj_out, hidden_states, enabled=fp32_final_layer)

        return (output, hidden_states) if return_features else output


@MODULES.register_module()
class Flux2Transformer2DModel(_Flux2Transformer2DModelCache):

    def __init__(
            self,
            *args,
            patch_size=2,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            pretrained_lora=None,
            pretrained_lora_scale=1.0,
            pretrained_dwd_head=None,
            torch_dtype='float32',
            freeze_exclude_fp32=True,
            freeze_exclude_autocast_dtype=None,
            freeze_quant_config=None,
            checkpointing=True,
            compile_forward=False,
            compile_kwargs=dict(
                mode='default',
                fullgraph=True,
                dynamic=False),
            use_lora=False,
            lora_target_modules=None,
            lora_rank=16,
            lora_dropout=0.0,
            **kwargs):
        with init_empty_weights():
            super().__init__(patch_size=1, *args, **kwargs)
        self.patch_size = patch_size

        self.init_weights(pretrained, pretrained_lora, pretrained_lora_scale, pretrained_dwd_head)

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
                ['self.norm_out', 'self.proj_out', 'self.dwd_linear_head', 'self.first_frame_dwd_linear_head']
                + list(freeze_quant_config.get('exclude_keys', [])))
            quantize_frozen_linears(self, **freeze_quant_config)

        if checkpointing:
            self.enable_gradient_checkpointing()

        self._compiled_forward = None
        if compile_forward:
            self._compiled_forward = torch.compile(self._forward_impl, **compile_kwargs)

    def init_weights(
            self,
            pretrained=None,
            pretrained_lora=None,
            pretrained_lora_scale=1.0,
            pretrained_dwd_head=None):
        super().init_weights()

        if pretrained is not None:
            logger = get_root_logger()
            checkpoint = _load_cached_checkpoint(pretrained, map_location='cpu', logger=logger)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            if self.secondary_temb_start_single_layer is not None:
                secondary_temb_state_dict = dict()
                for k, v in state_dict.items():
                    if k.startswith(('time_guidance_embed', 'single_stream_modulation')):
                        secondary_k = 'secondary_' + k
                        if secondary_k not in state_dict:
                            secondary_temb_state_dict[secondary_k] = v.clone()
                state_dict.update(secondary_temb_state_dict)

            if pretrained_dwd_head is not None:
                state_dict.update(_load_cached_checkpoint(
                    pretrained_dwd_head, map_location='cpu', logger=logger))

            load_full_state_dict(self, state_dict, logger=logger, assign=True)
            if pretrained_lora is not None:
                if not isinstance(pretrained_lora, (list, tuple)):
                    assert isinstance(pretrained_lora, str)
                    pretrained_lora = [pretrained_lora]
                if not isinstance(pretrained_lora_scale, (list, tuple)):
                    assert isinstance(pretrained_lora_scale, (int, float))
                    pretrained_lora_scale = [pretrained_lora_scale]
                for pretrained_lora_single, pretrained_lora_scale_single in zip(pretrained_lora, pretrained_lora_scale):
                    lora_state_dict = _load_cached_checkpoint(
                        pretrained_lora_single, map_location='cpu', logger=logger)
                    self.load_lora_adapter(lora_state_dict)
                    self.fuse_lora(lora_scale=pretrained_lora_scale_single, safe_fusing=True)
                    self.unload_lora()
                    self.requires_grad_(True)  # PEFT freezes the weights after fusing

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

    def _forward_impl(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            condition_latents: Optional[torch.Tensor] = None,
            return_pixels: bool = False,
            return_features: bool = False,
            **kwargs):
        hidden_states = self.patchify(hidden_states)
        img_ids = self._prepare_latent_ids(hidden_states)
        bs, c, h, w = hidden_states.size()
        hidden_states = self._pack_latents(hidden_states)
        txt_ids = self._prepare_text_ids(encoder_hidden_states)

        input_hidden_states = hidden_states
        input_img_ids = img_ids
        if condition_latents is not None:
            condition_latents = [self.patchify(condition_latents)]  # currently only supports one condition image
            condition_latent_ids = self._prepare_condition_latent_ids(condition_latents)
            condition_latents = torch.cat([self._pack_latents(x) for x in condition_latents], dim=1)
            input_hidden_states = torch.cat([hidden_states, condition_latents], dim=1)
            input_img_ids = torch.cat([img_ids, condition_latent_ids], dim=1)

        output = super().forward(
            hidden_states=input_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=input_img_ids,
            txt_ids=txt_ids,
            return_features=return_pixels or return_features,
            **kwargs)
        output_len = hidden_states.size(1)
        auxiliary_output = None
        if return_pixels or return_features:
            output, auxiliary_output = output
            auxiliary_output = auxiliary_output[:, :output_len]

        output = self.unpatchify(
            output[:, :output_len].permute(0, 2, 1).reshape(bs, self.out_channels, h, w)
        )
        if return_pixels:
            auxiliary_output = self.forward_dwd_head(auxiliary_output, (h, w))
        return output, auxiliary_output

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            condition_latents: Optional[torch.Tensor] = None,
            return_pixels: bool = False,
            return_features: bool = False,
            **kwargs):
        encoder_hidden_states = encoder_hidden_states[-hidden_states.size(0):]
        dtype = hidden_states.dtype
        forward_fn = self._forward_impl if self._compiled_forward is None else self._compiled_forward
        output, auxiliary_output = forward_fn(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states.to(dtype),
            condition_latents=condition_latents.to(dtype) if condition_latents is not None else None,
            return_pixels=return_pixels,
            return_features=return_features,
            **kwargs)
        if return_pixels or return_features:
            return output, auxiliary_output
        else:
            return output
