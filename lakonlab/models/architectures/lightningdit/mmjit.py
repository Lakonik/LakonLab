from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux2 import Flux2PosEmbed
from mmcv.cnn import constant_init, xavier_init

from .modules import (
    BottleneckPatchEmbed,
    FinalLayer,
    LightningDiTBlock,
    RMSNorm,
    SwiGLUFFN,
    TimestepEmbedder,
    modulate,
)
from ...builder import MODULES
from lakonlab.ops import quantize_frozen_linears
from ..dwd_head import DWDHeadMixin
from ..utils import flex_freeze, fp32_module_forward
from lakonlab.runner.checkpoint import _load_cached_checkpoint, load_full_state_dict
from lakonlab.utils import get_root_logger


class MMAttention(nn.Module):
    """Joint attention over concatenated text and image tokens, with per-modality qkv
    and output projections and shared QK norms (MM-DiT style).
    """

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=0.0,
            proj_drop=0.0,
            txt_out=True):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.txt_out = txt_out
        self.txt_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.img_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        if txt_out:
            self.txt_proj = nn.Linear(dim, dim)
        self.img_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _qkv(self, x, qkv_layer):
        batch, seq_len, _ = x.shape
        qkv = qkv_layer(x).reshape(
            batch, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        return qkv.unbind(0)

    def forward(self, x_txt, x_img, txt_rope=None, feat_rope=None):
        batch, txt_len, channels = x_txt.shape
        q_t, k_t, v_t = self._qkv(x_txt, self.txt_qkv)
        q_i, k_i, v_i = self._qkv(x_img, self.img_qkv)
        q_t, k_t = self.q_norm(q_t), self.k_norm(k_t)
        q_i, k_i = self.q_norm(q_i), self.k_norm(k_i)
        if txt_rope is not None:
            q_t, k_t = txt_rope(q_t), txt_rope(k_t)
        if feat_rope is not None:
            q_i, k_i = feat_rope(q_i), feat_rope(k_i)
        q = torch.cat([q_t, q_i], dim=2)
        k = torch.cat([k_t, k_i], dim=2)
        v = torch.cat([v_t, v_i], dim=2)

        x = F.scaled_dot_product_attention(
            q.to(v.dtype),
            k.to(v.dtype),
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0)

        x = x.transpose(1, 2).reshape(batch, -1, channels)
        x_img = self.proj_drop(self.img_proj(x[:, txt_len:]))
        if not self.txt_out:
            return None, x_img
        return self.proj_drop(self.txt_proj(x[:, :txt_len])), x_img


class MMJiTBlock(nn.Module):
    """Dual-stream block: per-modality norms and FFNs around one joint attention.

    Modulation is passed in already projected and shared across blocks (Flux2 style); each
    block only owns a `(1, 6, hidden_size)` offset table (Wan style), so per-block cost is
    6 * hidden_size instead of a 6 * hidden_size ** 2 projection.

    `txt_out=False` drops the text-stream update after attention, for the last block whose
    text output nothing reads. Upstream computes and discards it, which leaves those
    parameters without gradient.
    """

    def __init__(
            self,
            hidden_size,
            num_heads,
            mlp_hidden_features,
            use_modulation=False,
            attn_drop=0.0,
            proj_drop=0.0,
            txt_out=True):
        super().__init__()
        self.attn = MMAttention(
            hidden_size,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            txt_out=txt_out)
        self.txt_norm1 = RMSNorm(hidden_size)
        self.img_norm1 = RMSNorm(hidden_size)
        self.img_norm2 = RMSNorm(hidden_size)
        self.img_mlp = SwiGLUFFN(hidden_size, mlp_hidden_features, drop=proj_drop, bias=False)
        self.txt_out = txt_out
        if txt_out:
            self.txt_norm2 = RMSNorm(hidden_size)
            self.txt_mlp = SwiGLUFFN(hidden_size, mlp_hidden_features, drop=proj_drop, bias=False)

        self.use_modulation = use_modulation
        if use_modulation:
            self.txt_scale_shift_table = nn.Parameter(torch.zeros(1, 6, hidden_size))
            self.img_scale_shift_table = nn.Parameter(torch.zeros(1, 6, hidden_size))

    def forward(self, x_txt, x_img, txt_mod=None, img_mod=None, txt_rope=None, feat_rope=None):
        if not self.use_modulation:
            attn_txt, attn_img = self.attn(
                self.txt_norm1(x_txt), self.img_norm1(x_img), txt_rope=txt_rope, feat_rope=feat_rope)
            x_img = x_img + attn_img
            x_img = x_img + self.img_mlp(self.img_norm2(x_img))
            if self.txt_out:
                x_txt = x_txt + attn_txt
                x_txt = x_txt + self.txt_mlp(self.txt_norm2(x_txt))
            return x_txt, x_img

        (t_shift_msa, t_scale_msa, t_gate_msa,
         t_shift_mlp, t_scale_mlp, t_gate_mlp) = (
            self.txt_scale_shift_table + txt_mod.unflatten(-1, (6, -1))).unbind(dim=1)
        (i_shift_msa, i_scale_msa, i_gate_msa,
         i_shift_mlp, i_scale_mlp, i_gate_mlp) = (
            self.img_scale_shift_table + img_mod.unflatten(-1, (6, -1))).unbind(dim=1)
        attn_txt, attn_img = self.attn(
            modulate(self.txt_norm1(x_txt), t_shift_msa, t_scale_msa),
            modulate(self.img_norm1(x_img), i_shift_msa, i_scale_msa),
            txt_rope=txt_rope, feat_rope=feat_rope)
        x_img = x_img + i_gate_msa.unsqueeze(1) * attn_img
        x_img = x_img + i_gate_mlp.unsqueeze(1) * self.img_mlp(
            modulate(self.img_norm2(x_img), i_shift_mlp, i_scale_mlp))
        if self.txt_out:
            x_txt = x_txt + t_gate_msa.unsqueeze(1) * attn_txt
            x_txt = x_txt + t_gate_mlp.unsqueeze(1) * self.txt_mlp(
                modulate(self.txt_norm2(x_txt), t_shift_mlp, t_scale_mlp))
        return x_txt, x_img


class _MMJiT(DWDHeadMixin, nn.Module):
    """Pixel-space text-to-image MM-DiT (MMJiT), the text-conditioned counterpart of JiT.

    The original MiniT2I recipe drops timestep conditioning entirely, relying on the
    magnitude of `x_t` to carry the noise level; `use_modulation=False` reproduces that
    and is what the converted MiniT2I checkpoint expects. `use_modulation=True` adds
    zero-initialized adaLN timestep modulation for training from scratch, with the
    projection shared across all blocks (Flux2) and a per-block offset table (Wan), which
    costs ~2% of the parameters instead of the ~50% a per-block projection would.
    """

    _cache_storage = dict()

    def __init__(
            self,
            input_size=512,
            patch_size=16,
            in_channels=3,
            hidden_size=768,
            depth=17,
            secondary_emb_start_layer=None,
            dwd_head_config=None,
            txt_preamble_depth=2,
            num_heads=12,
            mlp_ratio=8 / 3,
            bottleneck_dim=128,
            text_embed_dim=1024,
            text_seq_len=256,
            use_pos_embed=True,
            use_modulation=False,
            attn_dropout=0.0,
            proj_dropout=0.0,
            checkpointing=False,
            out_channels=None):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.txt_preamble_depth = txt_preamble_depth
        self.num_heads = num_heads
        self.text_embed_dim = text_embed_dim
        self.text_seq_len = text_seq_len
        self.use_pos_embed = use_pos_embed
        self.use_modulation = use_modulation

        self.x_embedder = BottleneckPatchEmbed(
            height=self.input_size,
            width=self.input_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            bottleneck_dim=self.bottleneck_dim,
            embed_dim=self.hidden_size,
            bias=True,
            pos_embed_type='sincos' if use_pos_embed else None,
        )
        self.txt_in = nn.Linear(self.text_embed_dim, self.hidden_size, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.text_embed_dim))
        if self.use_modulation:
            self.t_embedder = TimestepEmbedder(self.hidden_size, bias=False)
            # one projection per stream for the whole model; blocks own only an offset table
            self.txt_adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=False),
            )
            self.img_adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=False),
            )

        self.secondary_emb_start_layer = secondary_emb_start_layer
        if self.secondary_emb_start_layer is not None:
            assert self.use_modulation
            assert 0 <= self.secondary_emb_start_layer < self.depth
            self.secondary_t_embedder = deepcopy(self.t_embedder)
            self.secondary_txt_adaLN_modulation = deepcopy(self.txt_adaLN_modulation)
            self.secondary_img_adaLN_modulation = deepcopy(self.img_adaLN_modulation)

        head_dim = self.hidden_size // self.num_heads
        # MiniT2I ropes the two streams differently: text over every channel, image with
        # the channels split between the row and column axes
        self.txt_pos_embed = Flux2PosEmbed(theta=10000, axes_dim=[head_dim])
        self.feat_pos_embed = Flux2PosEmbed(theta=10000, axes_dim=[head_dim // 2, head_dim // 2])

        # MiniT2I rounds the SwiGLU inner width up to a multiple of 8
        mlp_hidden_features = (int(self.hidden_size * mlp_ratio) + 7) // 8 * 8
        self.txt_preamble_blocks = nn.ModuleList([
            LightningDiTBlock(
                self.hidden_size,
                self.num_heads,
                use_qknorm=True,
                use_swiglu=True,
                use_rmsnorm=True,
                use_modulation=False,  # a text adapter has no reason to see the noise level
                mlp_hidden_features=mlp_hidden_features,
                mlp_bias=False,
            )
            for _ in range(self.txt_preamble_depth)
        ])
        self.blocks = nn.ModuleList([
            MMJiTBlock(
                self.hidden_size,
                self.num_heads,
                mlp_hidden_features,
                use_modulation=self.use_modulation,
                attn_drop=attn_dropout if (self.depth // 4 <= block_id < (self.depth // 4) * 3) else 0.0,
                proj_drop=proj_dropout if (self.depth // 4 <= block_id < (self.depth // 4) * 3) else 0.0,
                txt_out=block_id < self.depth - 1,
            )
            for block_id in range(self.depth)
        ])
        self.final_layer = FinalLayer(
            self.hidden_size, self.patch_size, self.out_channels,
            use_rmsnorm=True, use_modulation=self.use_modulation, use_scale_shift_table=True)

        if dwd_head_config is not None:
            self.init_dwd_head(self.hidden_size, **dwd_head_config)

        self.gradient_checkpointing = checkpointing

    def get_cache_hidden_states(self):
        return self._cache_storage['x_img']

    def clear_cache(self):
        self._cache_storage.clear()

    def flag_muon_params(self, output_weights=None):
        """Tell an orthogonalizing optimizer how to treat this model's matrices.

        Muon is for hidden layers, so the text input projection, the output head and the timestep
        MLP that embeds sinusoidal features are left to AdamW. The fused matrices are marked with
        the number of functionally separate blocks they concatenate, so that each is orthogonalized
        on its own rather than coupled through a joint Gram matrix.

        Subclasses can supply `output_weights` when they replace the final output head.
        Call this after any checkpoint load, since loading with `assign=True` replaces the
        parameter objects.
        """
        for name, param in self.named_parameters():
            if name.endswith('qkv.weight'):
                param.lakonlab_muon_chunks = 3
            elif name.endswith('w12.weight'):  # SwiGLU gate and up
                param.lakonlab_muon_chunks = 2
            elif name.endswith('adaLN_modulation.1.weight'):
                param.lakonlab_muon_chunks = 6

        if output_weights is None:
            output_weights = [self.final_layer.linear.weight]
        no_muon = [self.txt_in.weight] + output_weights
        if self.use_modulation:
            no_muon.append(self.t_embedder.mlp[0].weight)
        if self.secondary_emb_start_layer is not None:
            no_muon.append(self.secondary_t_embedder.mlp[0].weight)
        if getattr(self, 'dwd_linear_head', None) is not None:
            no_muon.append(self.dwd_linear_head.weight)
        for param in no_muon:
            param.lakonlab_no_muon = True

    def init_weights(self):
        super().init_weights()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m.to_empty(device='cpu'), distribution='uniform')
            elif isinstance(m, RMSNorm):
                nn.init.ones_(m.to_empty(device='cpu').weight)

        proj1_weight = self.x_embedder.proj1.to_empty(device='cpu').weight.data
        nn.init.xavier_uniform_(proj1_weight.view(proj1_weight.shape[0], -1))
        proj2_weight = self.x_embedder.proj2.to_empty(device='cpu').weight.data
        nn.init.xavier_uniform_(proj2_weight.view(proj2_weight.shape[0], -1))
        nn.init.zeros_(self.x_embedder.proj2.bias)

        self.mask_token = nn.Parameter(torch.empty_like(self.mask_token, device='cpu'))
        nn.init.normal_(self.mask_token, std=0.02)

        if self.use_modulation:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
            constant_init(self.txt_adaLN_modulation[-1], val=0)
            constant_init(self.img_adaLN_modulation[-1], val=0)
            if self.secondary_emb_start_layer is not None:
                nn.init.normal_(self.secondary_t_embedder.mlp[0].weight, std=0.02)
                nn.init.normal_(self.secondary_t_embedder.mlp[2].weight, std=0.02)
                constant_init(self.secondary_txt_adaLN_modulation[-1], val=0)
                constant_init(self.secondary_img_adaLN_modulation[-1], val=0)
            for m in self.modules():
                for key in ('txt_scale_shift_table', 'img_scale_shift_table', 'scale_shift_table'):
                    if hasattr(m, key):
                        setattr(m, key, nn.Parameter(
                            torch.zeros_like(getattr(m, key), device='cpu')))

        constant_init(self.final_layer.linear, val=0)

    def forward(
            self,
            x,
            timestep,
            encoder_hidden_states=None,
            encoder_hidden_states_mask=None,
            secondary_t=None,
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
            x_img = self._cache_storage.pop('x_img')[-bs:]
            x_txt = self._cache_storage.pop('x_txt')[-bs:]
            txt_rotary_emb = self._cache_storage.pop('txt_rotary_emb')
            feat_rotary_emb = self._cache_storage.pop('feat_rotary_emb')
            txt_rope = partial(apply_rotary_emb, freqs_cis=txt_rotary_emb)
            feat_rope = partial(apply_rotary_emb, freqs_cis=feat_rotary_emb)
            if self.use_modulation:
                cond = self._cache_storage.pop('cond')[-bs:]
                txt_mod = self._cache_storage.pop('txt_mod')[-bs:]
                img_mod = self._cache_storage.pop('img_mod')[-bs:]
            else:
                cond = txt_mod = img_mod = None
            start_block = cache_after_block + 1

        else:
            # Replace every masked text position with one learned `mask_token`; those positions
            # remain visible to attention.
            if encoder_hidden_states_mask is not None:
                encoder_hidden_states = torch.where(
                    encoder_hidden_states_mask[:, :, None] > 0.5,
                    encoder_hidden_states,
                    self.mask_token.to(dtype=encoder_hidden_states.dtype))

            x_img = self.x_embedder(x)
            x_txt = self.txt_in(encoder_hidden_states)
            if self.use_modulation:
                cond = self.t_embedder(timestep)
                txt_mod = self.txt_adaLN_modulation(cond)
                img_mod = self.img_adaLN_modulation(cond)
            else:
                cond = txt_mod = img_mod = None

            txt_rotary_emb = self.txt_pos_embed(
                torch.arange(x_txt.shape[1], device=x.device)[:, None])
            feat_rotary_emb = self.feat_pos_embed(torch.stack([
                torch.arange(height, device=x.device).repeat_interleave(width),
                torch.arange(width, device=x.device).repeat(height)], dim=-1))
            txt_rope = partial(apply_rotary_emb, freqs_cis=txt_rotary_emb)
            feat_rope = partial(apply_rotary_emb, freqs_cis=feat_rotary_emb)

            for block in self.txt_preamble_blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    x_txt = torch.utils.checkpoint.checkpoint(
                        block, x_txt, None, txt_rope, use_reentrant=False)
                else:
                    x_txt = block(x_txt, None, txt_rope)

            start_block = 0

        for index_block in range(start_block, len(self.blocks)):
            if index_block == self.secondary_emb_start_layer:
                if secondary_t is None:
                    secondary_t = timestep
                cond = self.secondary_t_embedder(secondary_t)
                txt_mod = self.secondary_txt_adaLN_modulation(cond)
                img_mod = self.secondary_img_adaLN_modulation(cond)

            block = self.blocks[index_block]
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x_txt, x_img = torch.utils.checkpoint.checkpoint(
                    block, x_txt, x_img, txt_mod, img_mod, txt_rope, feat_rope,
                    use_reentrant=False)
            else:
                x_txt, x_img = block(x_txt, x_img, txt_mod, img_mod, txt_rope, feat_rope)

            if cache_mode == 'save' and index_block == cache_after_block:
                self._cache_storage['x_img'] = x_img
                self._cache_storage['x_txt'] = x_txt
                self._cache_storage['txt_rotary_emb'] = txt_rotary_emb
                self._cache_storage['feat_rotary_emb'] = feat_rotary_emb
                if self.use_modulation:
                    self._cache_storage['cond'] = cond
                    self._cache_storage['txt_mod'] = txt_mod
                    self._cache_storage['img_mod'] = img_mod

        output = fp32_module_forward(
            self.final_layer, x_img, cond, return_features=return_features, enabled=fp32_final_layer)
        if return_features:
            output, features = output

        if unpatchify_output:
            output = output.reshape(
                bs, height, width, self.patch_size, self.patch_size, self.out_channels
            ).permute(0, 5, 1, 3, 2, 4).reshape(
                bs, self.out_channels, height * self.patch_size, width * self.patch_size)

        return (output, features) if return_features else output


@MODULES.register_module()
class MMJiT(_MMJiT):
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
                mode='default',
                fullgraph=True,
                dynamic=False),
            autocast_dtype=None,
            freeze_exclude_fp32=True,
            freeze_exclude_autocast_dtype=None,
            freeze_quant_config=None,
            **kwargs):
        with init_empty_weights():
            super().__init__(*args, **kwargs)

        self.init_weights(pretrained, pretrained_dwd_head)
        self.flag_muon_params()

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

        if freeze_quant_config is not None:
            assert torch_dtype in ('bfloat16', 'float16'), \
                'freeze_quant_config requires torch_dtype bfloat16 or float16'
            freeze_quant_config['exclude_keys'] = (
                ['self.final_layer', 'self.dwd_linear_head', 'self.first_frame_dwd_linear_head']
                + list(freeze_quant_config.get('exclude_keys', [])))
            quantize_frozen_linears(self, **freeze_quant_config)

        self._compiled_forward = None
        if compile_forward:
            self._compiled_forward = torch.compile(self._forward_impl, **compile_kwargs)

    def init_weights(self, pretrained=None, pretrained_dwd_head=None):
        if pretrained is not None:
            super(_MMJiT, self).init_weights()

            logger = get_root_logger()
            checkpoint = _load_cached_checkpoint(pretrained, map_location='cpu', logger=logger)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

            if self.secondary_emb_start_layer is not None:
                secondary_emb_state_dict = dict()
                for k, v in state_dict.items():
                    if k.startswith(('t_embedder', 'txt_adaLN_modulation', 'img_adaLN_modulation')):
                        secondary_k = 'secondary_' + k
                        if secondary_k not in state_dict:
                            secondary_emb_state_dict[secondary_k] = v.clone()
                state_dict.update(secondary_emb_state_dict)

            if pretrained_dwd_head is not None:
                state_dict.update(_load_cached_checkpoint(
                    pretrained_dwd_head, map_location='cpu', logger=logger))

            load_full_state_dict(self, state_dict, logger=logger, assign=True)

        else:
            super().init_weights()

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

    def forward(self, x, timestep, encoder_hidden_states=None, encoder_hidden_states_mask=None,
                return_pixels=False, return_features=False, **kwargs):
        encoder_hidden_states = encoder_hidden_states[-x.size(0):]
        if encoder_hidden_states_mask is not None:
            encoder_hidden_states_mask = encoder_hidden_states_mask[-x.size(0):]
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
                x.to(dtype), timestep=timestep, encoder_hidden_states=encoder_hidden_states.to(dtype),
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                return_pixels=return_pixels, return_features=return_features, **kwargs)
        if return_pixels or return_features:
            return output, auxiliary_output
        else:
            return output
