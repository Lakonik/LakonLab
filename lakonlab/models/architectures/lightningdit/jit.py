import torch
import torch.nn as nn
from mmcv.cnn import constant_init, xavier_init

from ..diffusers.dit import LabelEmbeddingMod
from .modules import (
    BottleneckPatchEmbed,
    FinalLayer,
    LightningDiTBlock,
    VisionRotaryEmbeddingFast,
    rotate_half,
    TimestepEmbedder,
)
from ...builder import MODULES
from ..dwd_head import DWDHeadMixin
from ..utils import flex_freeze, fp32_module_forward
from lakonlab.runner.checkpoint import _load_cached_checkpoint, load_full_state_dict
from lakonlab.utils import get_root_logger


class JiTVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim, pt_seq_len=16, ft_seq_len=None, num_prefix_tokens=0):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.base_rope = VisionRotaryEmbeddingFast(dim=dim, pt_seq_len=pt_seq_len, ft_seq_len=ft_seq_len)

    def forward(self, x):
        token_length = x.shape[-2]
        base_length, dim = self.base_rope.freqs_cos.shape
        image_token_length = token_length - self.num_prefix_tokens
        if image_token_length <= 0 or image_token_length % base_length != 0:
            raise ValueError(
                f'Unexpected token length {token_length} for base rope length {base_length} '
                f'with {self.num_prefix_tokens} prefix tokens.')

        repeat_factor = image_token_length // base_length
        freqs_cos = self.base_rope.freqs_cos
        freqs_sin = self.base_rope.freqs_sin
        if repeat_factor != 1:
            freqs_cos = freqs_cos.repeat_interleave(repeat_factor, dim=0)
            freqs_sin = freqs_sin.repeat_interleave(repeat_factor, dim=0)

        if self.num_prefix_tokens > 0:
            prefix_cos = torch.ones(
                self.num_prefix_tokens, dim, dtype=freqs_cos.dtype, device=freqs_cos.device)
            prefix_sin = torch.zeros(
                self.num_prefix_tokens, dim, dtype=freqs_sin.dtype, device=freqs_sin.device)
            freqs_cos = torch.cat([prefix_cos, freqs_cos], dim=0)
            freqs_sin = torch.cat([prefix_sin, freqs_sin], dim=0)

        return x * freqs_cos + rotate_half(x) * freqs_sin


class _JiT(DWDHeadMixin, nn.Module):

    _cache_storage = dict()

    def __init__(
            self,
            input_size=256,
            patch_size=16,
            in_channels=3,
            hidden_size=768,
            depth=12,
            secondary_emb_start_layer=None,
            dwd_head_config=None,
            num_heads=12,
            mlp_ratio=4.0,
            num_classes=1000,
            bottleneck_dim=128,
            in_context_len=32,
            in_context_start=4,
            attn_dropout=0.0,
            proj_dropout=0.0,
            upcast_attention=False,
            fused_attention=True,
            checkpointing=False):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.upcast_attention = upcast_attention
        self.fused_attention = fused_attention
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.y_embedder = LabelEmbeddingMod(
            self.num_classes,
            self.hidden_size,
            dropout_prob=0.0,
            use_cfg_embedding=True,
        )
        self.x_embedder = BottleneckPatchEmbed(
            height=self.input_size,
            width=self.input_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            bottleneck_dim=self.bottleneck_dim,
            embed_dim=self.hidden_size,
            bias=True,
            # forcing persistent pos_embed, since pretrained models may incorrectly use learnable pos_embed
            pos_embed_max_size=self.input_size // self.patch_size,
        )
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, self.hidden_size))
            torch.nn.init.normal_(self.in_context_posemb, std=0.02)
        else:
            self.in_context_posemb = None
        self.secondary_emb_start_layer = secondary_emb_start_layer
        if self.secondary_emb_start_layer is not None:
            assert 0 <= self.secondary_emb_start_layer < self.depth
            self.secondary_t_embedder = TimestepEmbedder(self.hidden_size)
            self.secondary_y_embedder = LabelEmbeddingMod(
                self.num_classes,
                self.hidden_size,
                dropout_prob=0.0,
                use_cfg_embedding=True,
            )

        head_dim = self.hidden_size // self.num_heads
        rope_dim = head_dim // 2
        feat_seq_len = self.input_size // self.patch_size
        self.feat_rope = JiTVisionRotaryEmbedding(dim=rope_dim, pt_seq_len=feat_seq_len, num_prefix_tokens=0)
        self.feat_rope_incontext = JiTVisionRotaryEmbedding(
            dim=rope_dim,
            pt_seq_len=feat_seq_len,
            num_prefix_tokens=self.in_context_len,
        )

        self.blocks = nn.ModuleList([
            LightningDiTBlock(
                self.hidden_size,
                self.num_heads,
                mlp_ratio=mlp_ratio,
                use_qknorm=True,
                use_swiglu=True,
                use_rmsnorm=True,
                wo_shift=False,
                attn_drop=attn_dropout if (self.depth // 4 <= block_id < (self.depth // 4) * 3) else 0.0,
                proj_drop=proj_dropout if (self.depth // 4 <= block_id < (self.depth // 4) * 3) else 0.0,
                upcast_attn=self.upcast_attention,
                fused_attn=self.fused_attention,
            )
            for block_id in range(self.depth)
        ])
        self.final_layer = FinalLayer(
            self.hidden_size, self.patch_size, self.out_channels, use_rmsnorm=True)

        # Optional DWD heads
        if dwd_head_config is not None:
            self.init_dwd_head(self.hidden_size, **dwd_head_config)

        self.gradient_checkpointing = checkpointing

    def get_cache_hidden_states(self):
        return self._cache_storage['x']

    def clear_cache(self):
        self._cache_storage.clear()

    def init_weights(self):
        super().init_weights()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')

        proj1_weight = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(proj1_weight.view(proj1_weight.shape[0], -1))
        proj2_weight = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(proj2_weight.view(proj2_weight.shape[0], -1))
        nn.init.zeros_(self.x_embedder.proj2.bias)

        for y_embedder in [self.y_embedder, getattr(self, 'secondary_y_embedder', None)]:
            if y_embedder is None:
                continue
            nn.init.normal_(y_embedder.embedding_table.weight, std=0.02)

        for t_embedder in [self.t_embedder, getattr(self, 'secondary_t_embedder', None)]:
            if t_embedder is None:
                continue
            nn.init.normal_(t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            constant_init(block.adaLN_modulation[-1], val=0)

        constant_init(self.final_layer.adaLN_modulation[-1], val=0)
        constant_init(self.final_layer.linear, val=0)

    def forward(
            self,
            x,
            timestep,
            class_labels=None,
            cache_mode=None,
            cache_config=None,
            unpatchify_output=True,
            return_features=False,
            fp32_final_layer=False):
        bs, _, h, w = x.size()
        height, width = h // self.patch_size, w // self.patch_size

        if cache_mode is not None:
            cache_after_block = cache_config['cache_after_block']
            if self.secondary_emb_start_layer is not None:
                assert cache_after_block < self.secondary_emb_start_layer

        if cache_mode == 'load':
            x = self._cache_storage.pop('x')[-bs:]
            label_emb = self._cache_storage.pop('label_emb')[-bs:]
            cond = self._cache_storage.pop('cond')[-bs:]
            start_block = cache_after_block + 1
        else:
            x = self.x_embedder(x)
            timestep_emb = self.t_embedder(timestep)
            label_emb = self.y_embedder(class_labels)
            cond = timestep_emb + label_emb
            start_block = 0

        for index_block in range(start_block, len(self.blocks)):
            # Update emb when we reach the specified block
            if index_block == self.secondary_emb_start_layer:
                timestep_emb = self.secondary_t_embedder(timestep)
                label_emb = self.secondary_y_embedder(class_labels)
                cond = timestep_emb + label_emb

            block = self.blocks[index_block]

            if self.in_context_len > 0 and index_block == self.in_context_start:
                in_context_tokens = label_emb.unsqueeze(1).expand(-1, self.in_context_len, -1)
                in_context_tokens = in_context_tokens + self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)
            rope = self.feat_rope if index_block < self.in_context_start else self.feat_rope_incontext

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x, cond, rope,
                    use_reentrant=False,
                )
            else:
                x = block(x, cond, rope)

            if cache_mode == 'save' and index_block == cache_after_block:
                self._cache_storage['x'] = x
                self._cache_storage['label_emb'] = label_emb
                self._cache_storage['cond'] = cond

        if self.in_context_len > 0:
            x = x[:, self.in_context_len:]

        output = fp32_module_forward(
            self.final_layer, x, cond, return_features=return_features, enabled=fp32_final_layer)
        if return_features:
            output, features = output

        if unpatchify_output:
            output = output.reshape(
                bs, height, width, self.patch_size, self.patch_size, self.out_channels
            ).permute(0, 5, 1, 3, 2, 4).reshape(
                bs, self.out_channels, height * self.patch_size, width * self.patch_size)

        return (output, features) if return_features else output


@MODULES.register_module()
class JiT(_JiT):
    def __init__(
            self,
            *args,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            pretrained_dwd_head=None,
            torch_dtype='float32',
            compile_forward=False,
            compile_kwargs=dict(
                mode='reduce-overhead',
                fullgraph=True,
                dynamic=False),
            autocast_dtype=None,
            freeze_exclude_fp32=True,
            freeze_exclude_autocast_dtype=None,
            **kwargs):
        super().__init__(*args, **kwargs)

        self.init_weights(pretrained, pretrained_dwd_head)

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

    def init_weights(self, pretrained=None, pretrained_dwd_head=None):
        super().init_weights()

        if pretrained is not None:
            logger = get_root_logger()
            checkpoint = _load_cached_checkpoint(pretrained, map_location='cpu', logger=logger)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            if self.secondary_emb_start_layer is not None:
                secondary_emb_state_dict = dict()
                for k, v in state_dict.items():
                    if k.startswith(('t_embedder', 'y_embedder')):
                        secondary_k = 'secondary_' + k
                        if secondary_k not in state_dict:
                            secondary_emb_state_dict[secondary_k] = v  # no need to clone since assign=False
                state_dict.update(secondary_emb_state_dict)

            if pretrained_dwd_head is not None:
                state_dict.update(_load_cached_checkpoint(
                    pretrained_dwd_head, map_location='cpu', logger=logger))

            load_full_state_dict(self, state_dict, logger=logger)

    def _forward_impl(self, x, return_pixels=False, return_features=False, **kwargs):
        output = super().forward(x, return_features=return_pixels or return_features, **kwargs)
        auxiliary_output = None
        if return_pixels or return_features:
            output, auxiliary_output = output
        if return_pixels:
            h, w = x.shape[-2:]
            auxiliary_output = self.forward_dwd_head(
                auxiliary_output, (h // self.patch_size, w // self.patch_size))
        return output, auxiliary_output

    def forward(self, x, timestep, class_labels=None,
                return_pixels=False, return_features=False, **kwargs):
        class_labels = class_labels[-x.size(0):]
        if self.autocast_dtype is not None:
            dtype = getattr(torch, self.autocast_dtype)
        else:
            dtype = x.dtype
        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=dtype if self.autocast_dtype is not None else None):
            forward_fn = self._forward_impl if self._compiled_forward is None else self._compiled_forward
            output, auxiliary_output = forward_fn(
                x.to(dtype), timestep=timestep, class_labels=class_labels,
                return_pixels=return_pixels, return_features=return_features, **kwargs)
        if return_pixels or return_features:
            return output, auxiliary_output
        else:
            return output
