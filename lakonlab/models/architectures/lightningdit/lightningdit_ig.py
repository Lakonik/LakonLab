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
    LightningDiTBlock,
    VisionRotaryEmbeddingFast,
)
from ..utils import flex_freeze, fp32_module_forward
from lakonlab.runner.checkpoint import _load_cached_checkpoint, load_full_state_dict
from lakonlab.utils import get_root_logger


class _LightningDiTIG(nn.Module):

    def __init__(
            self,
            input_size=16,
            in_channels=1024,
            patch_size=1,
            hidden_size=1440,
            depth=28,
            num_heads=20,
            mlp_ratio=4.0,
            base_model_depth=8,
            num_classes=1000,
            use_internal_guidance=True,
            checkpointing=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth

        self.patch_size = patch_size

        self.x_embedder = PatchEmbed(
            height=input_size,
            width=input_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.hidden_size,
            bias=True,
            pos_embed_type=None,
        )

        self.t_embedder = GaussianFourierEmbedding(self.hidden_size)
        self.y_embedder = LabelEmbeddingMod(
            num_classes,
            self.hidden_size,
            dropout_prob=0.0,
            use_cfg_embedding=True,
        )

        half_head_dim = self.hidden_size // num_heads // 2
        hw_seq_len = input_size // self.patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim, pt_seq_len=hw_seq_len)

        self.blocks = nn.ModuleList([
            LightningDiTBlock(
                self.hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                use_qknorm=True,
                use_rmsnorm=True,
                use_swiglu=True,
                use_modulation=True,
                standalone_qkv=True,  # for Muon optimizer
                standalone_w1w2=True)  # for Muon optimizer
            for _ in range(self.depth)
        ])
        self.final_layer = DDTFinalLayer(
            self.hidden_size, self.patch_size, self.out_channels, use_rmsnorm=True)

        self.use_internal_guidance = use_internal_guidance
        if self.use_internal_guidance:
            self.base_model_depth = base_model_depth
            self.base_final_layer = DDTFinalLayer(
                self.hidden_size, self.patch_size, self.out_channels, use_rmsnorm=True)
        else:
            self.base_final_depth = self.base_final_layer = None

        self.gradient_checkpointing = checkpointing

    def init_weights(self):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
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
            unpatchify_output=True,
            fp32_final_layer=False):
        bs, _, h, w = x.size()
        height, width = h // self.patch_size, w // self.patch_size
        class_labels = class_labels[-bs:]

        t = self.t_embedder(timestep)
        y = self.y_embedder(class_labels)
        c = t + y

        x = self.x_embedder(x)
        x_base = None
        for i in range(self.depth):
            block = self.blocks[i]
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x, c, self.feat_rope,
                    use_reentrant=False,
                )
            else:
                x = block(x, c, feat_rope=self.feat_rope)
            if self.use_internal_guidance and (i + 1) == self.base_model_depth:
                x_base = x

        x = fp32_module_forward(self.final_layer, x, c, enabled=fp32_final_layer)
        if unpatchify_output:
            x = x.reshape(
                bs, height, width, self.patch_size, self.patch_size, self.out_channels
            ).permute(0, 5, 1, 3, 2, 4).reshape(
                bs, self.out_channels, h, w)

        if self.use_internal_guidance:
            x_base = fp32_module_forward(self.base_final_layer, x_base, c, enabled=fp32_final_layer)
            if unpatchify_output:
                x_base = x_base.reshape(
                    bs, height, width, self.patch_size, self.patch_size, self.out_channels
                ).permute(0, 5, 1, 3, 2, 4).reshape(
                    bs, self.out_channels, h, w)
            return torch.cat([x_base, x], dim=0)
        else:
            return x


@MODULES.register_module()
class LightningDiTIG(_LightningDiTIG):

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
            load_full_state_dict(self, state_dict, logger=logger)

        else:
            super().init_weights()

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
