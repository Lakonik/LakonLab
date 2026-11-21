import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import PatchEmbed
from mmcv.cnn import constant_init

from ...builder import MODULES
from ..diffusers.dit import LabelEmbeddingMod
from .modules import (
    DDTFinalLayer,
    GaussianFourierEmbedding,
    LightningDDTBlock,
    LightningDiTBlock,
    VisionRotaryEmbeddingFast,
)
from ..utils import flex_freeze, fp32_module_forward
from lakonlab.runner.checkpoint import _load_cached_checkpoint, load_full_state_dict
from lakonlab.utils import get_root_logger


class _LightningDDTIG(nn.Module):

    _cache_storage = dict()

    def __init__(
            self,
            input_size=16,
            in_channels=1024,
            patch_size=(1, 1),
            hidden_size=(1440, 2048),
            depth=(28, 2),
            num_heads=(20, 16),
            mlp_ratio=4.0,
            base_model_depth=8,
            num_classes=1000,
            num_t_tokens=4,
            num_c_tokens=8,
            use_internal_guidance=True,
            checkpointing=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.encoder_hidden_size = hidden_size[0]
        self.decoder_hidden_size = hidden_size[1]
        self.num_heads = [num_heads, num_heads] if isinstance(num_heads, int) else list(num_heads)
        self.num_encoder_blocks = depth[0]
        self.num_decoder_blocks = depth[1]
        self.num_blocks = self.num_encoder_blocks + self.num_decoder_blocks

        if isinstance(patch_size, (int, float)):
            patch_size = [patch_size, patch_size]
        if len(patch_size) != 2:
            raise ValueError(f'patch_size should contain two values, got {patch_size}')
        self.patch_size = patch_size
        self.s_patch_size = int(patch_size[0])
        self.x_patch_size = int(patch_size[1])
        self.s_num_patches = (input_size // self.s_patch_size) ** 2

        self.x_embedder = PatchEmbed(
            height=input_size,
            width=input_size,
            patch_size=self.x_patch_size,
            in_channels=self.in_channels,
            embed_dim=self.decoder_hidden_size,
            bias=True,
            pos_embed_type=None,
        )
        self.s_embedder = PatchEmbed(
            height=input_size,
            width=input_size,
            patch_size=self.s_patch_size,
            in_channels=self.in_channels,
            embed_dim=self.encoder_hidden_size,
            bias=True,
            pos_embed_type=None,
        )
        self.s_projector = (
            nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)
            if self.encoder_hidden_size != self.decoder_hidden_size
            else nn.Identity()
        )
        self.t_embedder = GaussianFourierEmbedding(self.encoder_hidden_size)
        self.y_embedder = LabelEmbeddingMod(
            num_classes,
            self.encoder_hidden_size,
            dropout_prob=0.0,
            use_cfg_embedding=True,
        )

        self.t_tokens = nn.Parameter(torch.normal(
            0, 1 / self.encoder_hidden_size**0.5, (num_t_tokens, self.encoder_hidden_size)))
        self.y_tokens = nn.Parameter(torch.normal(
            0, 1 / self.encoder_hidden_size**0.5, (num_c_tokens, self.encoder_hidden_size)))
        num_cond_tokens = num_t_tokens + num_c_tokens

        enc_num_heads = self.num_heads[0]
        dec_num_heads = self.num_heads[1]
        enc_half_head_dim = self.encoder_hidden_size // enc_num_heads // 2
        enc_hw_seq_len = input_size // self.s_patch_size
        self.enc_feat_rope = VisionRotaryEmbeddingFast(
            dim=enc_half_head_dim, pt_seq_len=enc_hw_seq_len, cond_len=num_cond_tokens)
        dec_half_head_dim = self.decoder_hidden_size // dec_num_heads // 2
        dec_hw_seq_len = input_size // self.x_patch_size
        self.dec_feat_rope = VisionRotaryEmbeddingFast(
            dim=dec_half_head_dim, pt_seq_len=dec_hw_seq_len)

        self.blocks = nn.ModuleList([
            LightningDiTBlock(
                self.encoder_hidden_size,
                enc_num_heads,
                mlp_ratio=mlp_ratio,
                use_qknorm=True,
                use_rmsnorm=True,
                use_swiglu=True,
                use_modulation=False,
                standalone_qkv=True,  # for Muon optimizer
                standalone_w1w2=True)  # for Muon optimizer
            for _ in range(self.num_encoder_blocks)
        ] + [
            LightningDDTBlock(
                self.decoder_hidden_size,
                dec_num_heads,
                mlp_ratio=mlp_ratio,
                use_qknorm=True,
                use_rmsnorm=True,
                use_swiglu=True,
                standalone_qkv=True,  # for Muon optimizer
                standalone_w1w2=True)  # for Muon optimizer
            for _ in range(self.num_decoder_blocks)
        ])
        self.final_layer = DDTFinalLayer(
            self.decoder_hidden_size, self.x_patch_size, self.out_channels, use_rmsnorm=True)

        self.use_internal_guidance = use_internal_guidance
        if self.use_internal_guidance:
            self.base_model_depth = base_model_depth
            self.base_final_layer = DDTFinalLayer(
                self.encoder_hidden_size, self.s_patch_size, self.out_channels, use_rmsnorm=True)
        else:
            self.base_final_depth = self.base_final_layer = None

        self.gradient_checkpointing = checkpointing

    def get_cache_hidden_states(self):
        return self._cache_storage['s']

    def clear_cache(self):
        self._cache_storage.clear()

    def init_weights(self):
        for embedder in [self.x_embedder, self.s_embedder]:
            w = embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks[self.num_encoder_blocks:]:
            constant_init(block.adaLN_modulation[-1], val=0)

        for final_layer in [self.final_layer, self.base_final_layer]:
            if final_layer is None:
                continue
            constant_init(final_layer.adaLN_modulation[-1], val=0)
            constant_init(final_layer.linear, val=0)

    def forward(
            self,
            x,
            timestep,
            class_labels=None,
            cache_mode=None,
            cache_config=None,
            unpatchify_output=True,
            fp32_final_layer=False):
        bs, _, h, w = x.size()
        s_height, s_width = h // self.s_patch_size, w // self.s_patch_size
        x_height, x_width = h // self.x_patch_size, w // self.x_patch_size
        class_labels = class_labels[-bs:]

        if cache_mode is not None:
            cache_after_block = cache_config['cache_after_block']

        if cache_mode == 'load':
            s = self._cache_storage.pop('s')[-bs:]
            t = self._cache_storage.pop('t')[-bs:]
            x_base = self._cache_storage.pop('x_base')
            if x_base is not None:
                x_base = x_base[-bs:]
            start_block = cache_after_block + 1
        else:
            t = self.t_embedder(timestep)
            y = self.y_embedder(class_labels)
            s = torch.cat(
                [self.s_embedder(x), self.t_tokens + t.unsqueeze(1), self.y_tokens + y.unsqueeze(1)],
                dim=1)
            x_base = None
            start_block = 0

        for i in range(start_block, self.num_encoder_blocks):
            block = self.blocks[i]
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                s = torch.utils.checkpoint.checkpoint(
                    block,
                    s, None, self.enc_feat_rope,
                    use_reentrant=False,
                )
            else:
                s = block(s, feat_rope=self.enc_feat_rope)
            if self.use_internal_guidance and (i + 1) == self.base_model_depth:
                x_base = s[:, :self.s_num_patches]
            if cache_mode == 'save' and i == cache_after_block:
                self._cache_storage['s'] = s
                self._cache_storage['t'] = t
                self._cache_storage['x_base'] = x_base
        s = self.s_projector(F.silu(t.unsqueeze(1) + s[:, :self.s_num_patches]))

        x = self.x_embedder(x)
        for i in range(self.num_encoder_blocks, self.num_blocks):
            block = self.blocks[i]
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x, s, self.dec_feat_rope,
                    use_reentrant=False,
                )
            else:
                x = block(x, s, feat_rope=self.dec_feat_rope)
        x = fp32_module_forward(self.final_layer, x, s, enabled=fp32_final_layer)
        if unpatchify_output:
            x = x.reshape(
                bs, x_height, x_width, self.x_patch_size, self.x_patch_size, self.out_channels
            ).permute(0, 5, 1, 3, 2, 4).reshape(
                bs, self.out_channels, h, w)

        if self.use_internal_guidance:
            x_base = F.silu(t.unsqueeze(1) + x_base)
            x_base = fp32_module_forward(self.base_final_layer, x_base, x_base, enabled=fp32_final_layer)
            if unpatchify_output:
                x_base = x_base.reshape(
                    bs, s_height, s_width, self.s_patch_size, self.s_patch_size, self.out_channels
                ).permute(0, 5, 1, 3, 2, 4).reshape(
                    bs, self.out_channels, h, w)
            x = torch.cat([x_base, x], dim=0)

        return x


@MODULES.register_module()
class LightningDDTIG(_LightningDDTIG):

    def __init__(
            self,
            *args,
            pretrained=None,
            torch_dtype='float32',
            compile_forward=False,
            compile_kwargs=dict(
                mode='reduce-overhead',
                fullgraph=True,
                dynamic=False),
            autocast_dtype=None,
            freeze=False,
            freeze_exclude=[],
            freeze_exclude_fp32=True,
            freeze_exclude_autocast_dtype=None,
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
                exclude_autocast_dtype=freeze_exclude_autocast_dtype,
            )

        self._compiled_forward = None
        if compile_forward:
            self._compiled_forward = torch.compile(self._forward_impl, **compile_kwargs)

    def init_weights(self, pretrained=None):
        super().init_weights()

        if pretrained is not None:
            logger = get_root_logger()
            state_dict = _load_cached_checkpoint(pretrained, map_location='cpu', logger=logger)
            load_full_state_dict(self, state_dict, logger=logger)

    def _forward_impl(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def forward(self, x, timestep, class_labels=None, **kwargs):
        if self.autocast_dtype is not None:
            dtype = getattr(torch, self.autocast_dtype)
        else:
            dtype = x.dtype
        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=dtype if self.autocast_dtype is not None else None):
            if self._compiled_forward is not None:
                return self._compiled_forward(x.to(dtype), timestep, class_labels=class_labels, **kwargs)
            return self._forward_impl(x.to(dtype), timestep, class_labels=class_labels, **kwargs)
