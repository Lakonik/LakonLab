from math import log

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import PatchEmbed
from mmcv.cnn import constant_init, xavier_init

from ...builder import MODULES
from ..diffusers.dit import LabelEmbeddingMod
from ..lightningdit.modules import (
    RMSNorm,
    GaussianFourierEmbedding,
    VisionRotaryEmbeddingFast,
    LightningDDTBlock,
    ddt_modulate,
)
from ..lightningdit.lightningddt import _LightningDDT
from ..utils import flex_freeze
from .gm_output import GMFlowModelOutput
from lakonlab.utils import get_root_logger
from lakonlab.runner.checkpoint import _load_cached_checkpoint, load_full_state_dict


class GMDDTFinalLayer(nn.Module):
    def __init__(
            self,
            encoder_hidden_size,
            decoder_hidden_size,
            patch_size,
            out_channels,
            use_rmsnorm=False,
            num_gaussians=8,
            constant_logstd=None,
            logstd_inner_dim=1024,
            gm_num_logstd_layers=2):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.num_gaussians = num_gaussians

        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(decoder_hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(decoder_hidden_size)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(decoder_hidden_size, 2 * decoder_hidden_size, bias=True),
        )

        self.proj_out_means = nn.Linear(
            decoder_hidden_size,
            patch_size * patch_size * num_gaussians * out_channels)
        self.proj_out_logweights = nn.Linear(
            decoder_hidden_size,
            patch_size * patch_size * num_gaussians)
        self.constant_logstd = constant_logstd

        if self.constant_logstd is None:
            assert gm_num_logstd_layers >= 1
            in_dim = encoder_hidden_size
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

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')
        constant_init(self.proj_out_means, val=0)
        rand_noise = torch.randn((self.num_gaussians * self.out_channels)) * 0.1
        self.proj_out_means.bias.data.copy_(rand_noise[None, :].expand(
            self.patch_size * self.patch_size, -1).flatten())
        constant_init(self.proj_out_logweights, val=0)
        if self.constant_logstd is None:
            constant_init(self.proj_out_logstds[-1], val=0)

    def forward(self, x, s, c):
        if len(s.shape) < len(x.shape):
            s = s.unsqueeze(1)
        shift, scale = self.adaLN_modulation(s).chunk(2, dim=-1)
        x = ddt_modulate(self.norm_final(x), shift, scale)
        out_means = self.proj_out_means(x)
        out_logweights = self.proj_out_logweights(x)
        if self.constant_logstd is None:
            out_logstds = self.proj_out_logstds(c.detach())
        else:
            out_logstds = x.new_full((x.shape[0], 1), float(self.constant_logstd))
        return out_means, out_logweights, out_logstds


class _GMLightningDDT(_LightningDDT):

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
            checkpointing=True,
            num_gaussians=8,
            constant_logstd=None,
            logstd_inner_dim=1024,
            gm_num_logstd_layers=2):
        super(_LightningDDT, self).__init__()
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
        self.num_gaussians = num_gaussians
        self.final_layer = GMDDTFinalLayer(
            self.encoder_hidden_size,
            self.decoder_hidden_size,
            self.x_patch_size,
            self.out_channels,
            use_rmsnorm=use_rmsnorm,
            num_gaussians=num_gaussians,
            constant_logstd=constant_logstd,
            logstd_inner_dim=logstd_inner_dim,
            gm_num_logstd_layers=gm_num_logstd_layers)

        self.gradient_checkpointing = checkpointing

    def init_weights(self, init_backbone=True):
        if init_backbone:
            for embedder in [self.x_embedder, self.s_embedder]:
                w = embedder.proj.weight.data
                nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
                nn.init.constant_(embedder.proj.bias, 0)

            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

            for block in self.blocks:
                constant_init(block.adaLN_modulation[-1], val=0)

        self.final_layer.init_weights()

    def forward(self, x, timestep=None, class_labels=None):
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
        out_means, out_logweights, out_logstds = self.final_layer(x, s, c)

        out_means = out_means.reshape(
            bs, height, width, self.x_patch_size, self.x_patch_size, self.num_gaussians * self.out_channels
        ).permute(0, 5, 1, 3, 2, 4).reshape(
            bs, self.num_gaussians, self.out_channels, height * self.x_patch_size, width * self.x_patch_size)
        out_logweights = out_logweights.reshape(
            bs, height, width, self.x_patch_size, self.x_patch_size, self.num_gaussians
        ).permute(0, 5, 1, 3, 2, 4).reshape(
            bs, self.num_gaussians, 1, height * self.x_patch_size, width * self.x_patch_size
        ).log_softmax(dim=1)
        out_logstds = out_logstds.reshape(bs, 1, 1, 1, 1)

        return GMFlowModelOutput(
            means=out_means,
            logweights=out_logweights,
            logstds=out_logstds)


@MODULES.register_module()
class GMLightningDDT(_GMLightningDDT):

    def __init__(
            self,
            *args,
            pretrained=None,
            pretrained_linear_proj=None,
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

        self.init_weights(pretrained, pretrained_linear_proj)

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

    @staticmethod
    def _project_patch_embed_weight(patch_embed_weight, in_proj_weight, patch_size, channels):
        # Rewrite PatchEmbed as explicit patchify(p, q, c) + linear, apply the
        # pretrained projector in that basis, then convert the linear weight back
        # to conv storage (c, p, q).
        linear_weight = patch_embed_weight.permute(0, 2, 3, 1).reshape(patch_embed_weight.size(0), -1)
        expected_shape = (linear_weight.size(1), channels * patch_size * patch_size)
        if tuple(in_proj_weight.shape) != expected_shape:
            raise RuntimeError(
                f'in_proj shape mismatch: expected {expected_shape}, got {tuple(in_proj_weight.shape)}. '
                'The pretrained linear projector likely does not match the latent/pixel patch sizes for this model.'
            )
        return (linear_weight @ in_proj_weight).reshape(
            patch_embed_weight.size(0), patch_size, patch_size, channels
        ).permute(
            0, 3, 1, 2
        )

    def init_weights(self, pretrained, pretrained_linear_proj):
        super().init_weights(init_backbone=pretrained is None)
        if pretrained is not None:
            logger = get_root_logger()
            state_dict = _load_cached_checkpoint(pretrained, map_location='cpu', logger=logger)

            # Convert original state dict to match diffusers' PatchEmbed
            if 'pos_embed' in state_dict and 's_embedder.pos_embed' not in state_dict:
                state_dict['s_embedder.pos_embed'] = state_dict.pop('pos_embed')

            # load the pretrained linear projection weights
            if pretrained_linear_proj is not None:
                linear_proj_state_dict = _load_cached_checkpoint(
                    pretrained_linear_proj, map_location='cpu', logger=logger)
                in_proj_weight = linear_proj_state_dict['in_proj'].to(state_dict['x_embedder.proj.weight'].dtype)
                out_proj_weight = linear_proj_state_dict['out_proj'].to(state_dict['x_embedder.proj.weight'].dtype)
                if 'x_embedder.proj.weight' in state_dict:
                    state_dict['x_embedder.proj.weight'] = self._project_patch_embed_weight(
                        state_dict['x_embedder.proj.weight'],
                        in_proj_weight,
                        self.x_patch_size,
                        self.in_channels)
                if 's_embedder.proj.weight' in state_dict:
                    state_dict['s_embedder.proj.weight'] = self._project_patch_embed_weight(
                        state_dict['s_embedder.proj.weight'],
                        in_proj_weight,
                        self.s_patch_size,
                        self.in_channels)
                if 'final_layer.linear.weight' in state_dict and 'final_layer.linear.bias' in state_dict:
                    state_dict['final_layer.linear.bias'] = (
                        out_proj_weight @ state_dict['final_layer.linear.bias'].unsqueeze(-1)
                    ).squeeze(-1)
                    state_dict['final_layer.linear.weight'] = (
                        out_proj_weight @ state_dict['final_layer.linear.weight']
                    )

            # expand the output channels
            p2 = self.final_layer.patch_size * self.final_layer.patch_size
            ori_out_channels = p2 * self.out_channels
            if ('final_layer.linear.weight' in state_dict
                    and state_dict['final_layer.linear.weight'].size(0) == ori_out_channels):
                state_dict['final_layer.proj_out_means.weight'] = state_dict['final_layer.linear.weight'].reshape(
                    p2, 1, self.out_channels, -1
                ).expand(-1, self.num_gaussians, -1, -1).reshape(
                    self.num_gaussians * ori_out_channels, -1)
                del state_dict['final_layer.linear.weight']
            if ('final_layer.linear.bias' in state_dict
                    and state_dict['final_layer.linear.bias'].size(0) == ori_out_channels):
                state_dict['final_layer.proj_out_means.bias'] = state_dict['final_layer.linear.bias'].reshape(
                    p2, 1, self.out_channels
                ).expand(-1, self.num_gaussians, -1).reshape(
                    self.num_gaussians * ori_out_channels)
                rand_noise = torch.randn(
                    (self.num_gaussians * self.out_channels),
                    dtype=state_dict['final_layer.proj_out_means.bias'].dtype,
                    device=state_dict['final_layer.proj_out_means.bias'].device) * 0.05
                state_dict['final_layer.proj_out_means.bias'] += rand_noise[None, :].expand(p2, -1).flatten()
                del state_dict['final_layer.linear.bias']
            if (self.final_layer.constant_logstd is None
                    and 'final_layer.proj_out_means.weight' in state_dict
                    and 'final_layer.proj_out_means.bias' in state_dict):
                self.final_layer.proj_out_logstds[-1].bias.data[:1] = log(0.05)  # reduce the initial logstd

            load_full_state_dict(self, state_dict)

    def _forward_impl(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def forward(self, x, timestep=None, class_labels=None):
        if self.autocast_dtype is not None:
            dtype = getattr(torch, self.autocast_dtype)
        else:
            dtype = x.dtype
        x = x.to(dtype)
        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=dtype if self.autocast_dtype is not None else None):
            if self._compiled_forward is not None:
                return self._compiled_forward(x, timestep=timestep, class_labels=class_labels)
            return self._forward_impl(x, timestep=timestep, class_labels=class_labels)
