from typing import Optional, List, Any

import torch

from accelerate import init_empty_weights
from diffusers.models import Flux2Transformer2DModel as _Flux2Transformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import apply_lora_scale
from peft import LoraConfig

from ...builder import MODULES
from ..utils import flex_freeze
from lakonlab.utils import get_root_logger
from lakonlab.runner.checkpoint import load_checkpoint, _load_cached_checkpoint


class _Flux2Transformer2DModelCache(_Flux2Transformer2DModel):

    _cache_storage = dict()

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
        return_dict: bool = True,
        cache_mode: Optional[str] = None,
        cache_config: Optional[dict] = None,
    ):
        # 0. Handle input arguments
        batch_size, num_txt_tokens = encoder_hidden_states.shape[:2]

        if cache_mode is not None:
            cache_after_single_block = cache_config['cache_after_single_block']

        # When loading from cache, skip to single-stream block processing
        if cache_mode == 'load':
            hidden_states = self._cache_storage.pop('hidden_states')[-batch_size:]
            temb = self._cache_storage.pop('temb')[-batch_size:]
            single_stream_mod = tuple(x[-batch_size:] for x in self._cache_storage.pop('single_stream_mod'))
            concat_rotary_emb = self._cache_storage.pop('concat_rotary_emb')

        else:
            # 1. Calculate timestep embedding and modulation parameters
            timestep = timestep.to(hidden_states.dtype) * 1000
            if guidance is not None:
                guidance = guidance.to(hidden_states.dtype) * 1000

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
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


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
            torch_dtype='float32',
            freeze_exclude_fp32=True,
            freeze_exclude_autocast_dtype='float32',
            checkpointing=True,
            use_lora=False,
            lora_target_modules=None,
            lora_rank=16,
            lora_dropout=0.0,
            **kwargs):
        with init_empty_weights():
            super().__init__(patch_size=1, *args, **kwargs)
        self.patch_size = patch_size

        self.init_weights(pretrained, pretrained_lora, pretrained_lora_scale)

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

    def init_weights(self, pretrained=None, pretrained_lora=None, pretrained_lora_scale=1.0):
        if pretrained is not None:
            logger = get_root_logger()
            load_checkpoint(
                self, pretrained,
                map_location='cpu', strict=False, logger=logger, assign=True, use_cache=True)
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
                    self.fuse_lora(lora_scale=pretrained_lora_scale_single)
                    self.unload_lora()

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
            hidden_states=input_hidden_states,
            encoder_hidden_states=encoder_hidden_states.to(dtype),
            timestep=timestep,
            img_ids=input_img_ids,
            txt_ids=txt_ids,
            return_dict=False,
            **kwargs)[0]

        output = output[:, :hidden_states.size(1)].permute(0, 2, 1).reshape(bs, self.out_channels, h, w)
        return self.unpatchify(output)
