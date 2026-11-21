import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import PatchEmbed
from mmcv.cnn import constant_init

from ...builder import MODULES
from ..diffusers.dit import LabelEmbeddingMod
from ..utils import flex_freeze, fp32_module_forward
from .modules import (
    GaussianFourierEmbedding,
    VisionRotaryEmbeddingFast,
    LightningDDTBlock,
    DDTFinalLayer,
)
from lakonlab.utils import get_root_logger
from lakonlab.runner.checkpoint import _load_cached_checkpoint, load_full_state_dict


class _LightningDDT(nn.Module):

    def __init__(
            self,
            input_size=16,
            patch_size=1,
            in_channels=768,
            hidden_size=(1152, 2048),
            depth=(28, 2),
            num_heads=(16, 16),
            mlp_ratio=4.0,
            num_classes=1000,
            use_qknorm=False,
            use_swiglu=True,
            use_rope=True,
            use_rmsnorm=True,
            wo_shift=False,
            use_pos_embed=True,
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
        self.use_rope = use_rope
        self.use_pos_embed = use_pos_embed

        if isinstance(patch_size, (int, float)):
            patch_size = [patch_size, patch_size]
        if len(patch_size) != 2:
            raise ValueError(f'patch_size should contain two values, got {patch_size}')
        self.patch_size = patch_size
        self.s_patch_size = int(patch_size[0])
        self.x_patch_size = int(patch_size[1])

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
            pos_embed_type='sincos' if use_pos_embed else None,
            # forcing persistent pos_embed, since pretrained models may incorrectly use learnable pos_embed
            pos_embed_max_size=input_size // self.s_patch_size,
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

        enc_num_heads = self.num_heads[0]
        dec_num_heads = self.num_heads[1]
        if self.use_rope:
            enc_half_head_dim = self.encoder_hidden_size // enc_num_heads // 2
            enc_hw_seq_len = input_size // self.s_patch_size
            self.enc_feat_rope = VisionRotaryEmbeddingFast(dim=enc_half_head_dim, pt_seq_len=enc_hw_seq_len)
            dec_half_head_dim = self.decoder_hidden_size // dec_num_heads // 2
            dec_hw_seq_len = input_size // self.x_patch_size
            self.dec_feat_rope = VisionRotaryEmbeddingFast(dim=dec_half_head_dim, pt_seq_len=dec_hw_seq_len)
        else:
            self.enc_feat_rope = None
            self.dec_feat_rope = None

        self.blocks = nn.ModuleList([
            LightningDDTBlock(
                self.encoder_hidden_size if i < self.num_encoder_blocks else self.decoder_hidden_size,
                enc_num_heads if i < self.num_encoder_blocks else dec_num_heads,
                mlp_ratio=mlp_ratio,
                use_qknorm=use_qknorm,
                use_rmsnorm=use_rmsnorm,
                use_swiglu=use_swiglu,
                wo_shift=wo_shift,
            )
            for i in range(self.num_blocks)
        ])
        self.final_layer = DDTFinalLayer(
            self.decoder_hidden_size, self.x_patch_size, self.out_channels, use_rmsnorm=use_rmsnorm)

        self.gradient_checkpointing = checkpointing

    def init_weights(self):
        for embedder in [self.x_embedder, self.s_embedder]:
            w = embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            constant_init(block.adaLN_modulation[-1], val=0)

        constant_init(self.final_layer.adaLN_modulation[-1], val=0)
        constant_init(self.final_layer.linear, val=0)

    def _apply_blocks(self, x, block_indices, cond, feat_rope):
        for i in block_indices:
            block = self.blocks[i]
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x, cond, feat_rope,
                    use_reentrant=False,
                )
            else:
                x = block(x, cond, feat_rope=feat_rope)
        return x

    def forward(
            self,
            x,
            timestep=None,
            class_labels=None,
            unpatchify_output=True,
            fp32_final_layer=False):
        bs, _, h, w = x.size()
        height, width = h // self.x_patch_size, w // self.x_patch_size

        t = self.t_embedder(timestep)
        y = self.y_embedder(class_labels)
        # Modulation layers already have SiLU, this is redundant, but we keep it for consistency
        # with the original RAE implementation
        c = F.silu(t + y)

        s = self.s_embedder(x)
        s = self._apply_blocks(s, range(self.num_encoder_blocks), c, self.enc_feat_rope)
        s = self.s_projector(F.silu(t.unsqueeze(1) + s))

        x = self.x_embedder(x)
        x = self._apply_blocks(x, range(self.num_encoder_blocks, self.num_blocks), s, self.dec_feat_rope)
        x = fp32_module_forward(self.final_layer, x, s, enabled=fp32_final_layer)

        if unpatchify_output:
            x = x.reshape(
                bs, height, width, self.x_patch_size, self.x_patch_size, self.out_channels
            ).permute(0, 5, 1, 3, 2, 4).reshape(
                bs, self.out_channels, h, w)
        return x


@MODULES.register_module()
class LightningDDT(_LightningDDT):

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
        if pretrained is not None:
            logger = get_root_logger()
            state_dict = _load_cached_checkpoint(pretrained, map_location='cpu', logger=logger)
            # Convert original state dict to match diffusers' PatchEmbed
            if 'pos_embed' in state_dict and 's_embedder.pos_embed' not in state_dict:
                state_dict['s_embedder.pos_embed'] = state_dict.pop('pos_embed')
            load_full_state_dict(self, state_dict, logger=logger)

        else:
            super().init_weights()

    def _forward_impl(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def forward(self, x, timestep=None, class_labels=None, **kwargs):
        if self.autocast_dtype is not None:
            dtype = getattr(torch, self.autocast_dtype)
        else:
            dtype = x.dtype
        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=dtype if self.autocast_dtype is not None else None):
            if self._compiled_forward is not None:
                return self._compiled_forward(x.to(dtype), timestep=timestep, class_labels=class_labels, **kwargs)
            return self._forward_impl(x.to(dtype), timestep=timestep, class_labels=class_labels, **kwargs)
