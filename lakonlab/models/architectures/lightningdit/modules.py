import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_from_grid


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(len(t.shape) for t in tensors)
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*[list(t.shape) for t in tensors]))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(len(set(t[1])) <= 2 for t in expandable_dims), 'invalid dimensions for broadcastable concatentation'
    max_dims = [(t[0], max(t[1])) for t in expandable_dims]
    expanded_dims = [(t[0], (t[1],) * num_tensors) for t in max_dims]
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*[t[1] for t in expanded_dims]))
    tensors = [t[0].expand(*t[1]) for t in zip(tensors, expandable_shapes)]
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.reshape(*x.shape[:-2], -1)


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
            self,
            dim,
            pt_seq_len=16,
            ft_seq_len=None,
            custom_freqs=None,
            freqs_for='lang',
            theta=10000,
            max_freq=10,
            num_freqs=1,
            cond_len=0):
        super().__init__()
        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * math.pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len
        freqs = torch.einsum('..., f -> ... f', t, freqs)
        freqs = freqs.repeat_interleave(2, dim=-1)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        freqs = freqs.view(-1, freqs.shape[-1])
        if cond_len > 0:
            freqs = torch.cat([freqs, freqs.new_zeros(cond_len, freqs.shape[-1])], dim=0)
        freqs_cos = freqs.cos()
        freqs_sin = freqs.sin()
        self.register_buffer('freqs_cos', freqs_cos)
        self.register_buffer('freqs_sin', freqs_sin)

    def forward(self, t):
        _, _, token_length, _ = t.shape
        base_length, _ = self.freqs_cos.shape
        repeat_factor = token_length // base_length
        freqs_cos = self.freqs_cos
        freqs_sin = self.freqs_sin
        if repeat_factor != 1:
            freqs_cos = freqs_cos.repeat_interleave(repeat_factor, dim=0)
            freqs_sin = freqs_sin.repeat_interleave(repeat_factor, dim=0)
        return t * freqs_cos + rotate_half(t) * freqs_sin


class BottleneckPatchEmbed(nn.Module):
    """2D image to patch embedding through a low-rank bottleneck.

    Follows `diffusers.models.embeddings.PatchEmbed`, except that without `pos_embed_max_size`
    an input size other than the configured one embeds the token ids at unit pitch, instead of
    rescaling each axis into the base extent.
    The single patch conv is split into `proj1` (to the bottleneck, no bias) and `proj2`
    (1x1 to the model width); those names are load-bearing for existing
    JiT/AnisoJiT/MMJiT checkpoints.
    """

    def __init__(
        self,
        height=256,
        width=256,
        patch_size=16,
        in_channels=3,
        bottleneck_dim=128,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=1,
        pos_embed_type='sincos',
        pos_embed_max_size=None,  # For SD3 cropping
    ):
        super().__init__()

        num_patches = (height // patch_size) * (width // patch_size)
        self.flatten = flatten
        self.layer_norm = layer_norm
        self.pos_embed_max_size = pos_embed_max_size

        self.proj1 = nn.Conv2d(
            in_channels, bottleneck_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(bottleneck_dim, embed_dim, kernel_size=1, stride=1, bias=bias)
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

        self.patch_size = patch_size
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size
        self.interpolation_scale = interpolation_scale

        if pos_embed_max_size:
            grid_size = pos_embed_max_size
        else:
            grid_size = int(num_patches ** 0.5)

        if pos_embed_type is None:
            self.pos_embed = None
        elif pos_embed_type == 'sincos':
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim,
                grid_size,
                base_size=self.base_size,
                interpolation_scale=self.interpolation_scale,
                output_type='pt')
            persistent = True if pos_embed_max_size else False
            self.register_buffer('pos_embed', pos_embed.float().unsqueeze(0), persistent=persistent)
        else:
            raise ValueError(f'Unsupported pos_embed_type: {pos_embed_type}')

    def cropped_pos_embed(self, height, width):
        """Crops positional embeddings for SD3 compatibility."""
        if self.pos_embed_max_size is None:
            raise ValueError('`pos_embed_max_size` must be set for cropping.')

        height = height // self.patch_size
        width = width // self.patch_size
        if height > self.pos_embed_max_size or width > self.pos_embed_max_size:
            raise ValueError(
                f'Size ({height}, {width}) cannot be greater than '
                f'`pos_embed_max_size`: {self.pos_embed_max_size}.')

        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        spatial_pos_embed = self.pos_embed.reshape(
            1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        spatial_pos_embed = spatial_pos_embed[:, top:top + height, left:left + width, :]
        return spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])

    def forward(self, latent):
        if self.pos_embed_max_size is not None:
            height, width = latent.shape[-2:]
        else:
            height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size
        latent = self.proj2(self.proj1(latent))
        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if self.layer_norm:
            latent = self.norm(latent)
        if self.pos_embed is None:
            return latent.to(latent.dtype)

        if self.pos_embed_max_size:
            pos_embed = self.cropped_pos_embed(height, width)
        elif self.height != height or self.width != width:
            grid = torch.meshgrid(  # w first, as in get_2d_sincos_pos_embed
                torch.arange(width, device=latent.device), torch.arange(height, device=latent.device), indexing='xy')
            pos_embed = get_2d_sincos_pos_embed_from_grid(
                self.pos_embed.shape[-1], torch.stack(grid).float() / self.interpolation_scale,
                output_type='pt').float().unsqueeze(0)
        else:
            pos_embed = self.pos_embed

        return (latent + pos_embed).to(latent.dtype)


class GaussianFourierEmbedding(nn.Module):
    def __init__(self, hidden_size, embedding_size=256, scale=1.0):
        super().__init__()
        self.register_buffer('W', torch.normal(0, scale, (embedding_size,)))
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @property
    def dtype(self):
        return next(self.mlp.parameters()).dtype

    def forward(self, t):
        t = 2 * torch.pi * t[:, None] * self.W[None, :]
        t_embed = torch.cat([torch.sin(t), torch.cos(t)], dim=-1).to(self.dtype)
        t_embed = self.mlp(t_embed)
        return t_embed


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            drop=0.0,
            bias=True,
            standalone_w1w2=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.standalone_w1w2 = standalone_w1w2
        if standalone_w1w2:
            self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
            self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        else:
            self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        if self.standalone_w1w2:
            x1, x2 = self.w1(x), self.w2(x)
        else:
            x12 = self.w12(x)
            x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.0,
            proj_drop=0.0,
            norm_layer=nn.LayerNorm,
            fused_attn=True,
            upcast_attn=False,
            use_rmsnorm=False,
            standalone_qkv=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn
        self.upcast_attn = upcast_attn
        self.standalone_qkv = standalone_qkv
        if use_rmsnorm:
            norm_layer = RMSNorm
        if standalone_qkv:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.k = nn.Linear(dim, dim, bias=qkv_bias)
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope=None):
        batch, seq_len, channels = x.shape
        if self.standalone_qkv:
            q = self.q(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            qkv = self.qkv(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q = rope(q)
            k = rope(k)
        if self.upcast_attn:
            with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=False):
                if self.fused_attn:
                    x = F.scaled_dot_product_attention(
                        q.float(),
                        k.float(),
                        v.float(),
                        dropout_p=self.attn_drop.p if self.training else 0.0
                    ).to(v.dtype)
                else:
                    attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale
                    attn = attn.softmax(dim=-1)
                    attn = self.attn_drop(attn)
                    x = (attn @ v.float()).to(v.dtype)
        else:
            if self.fused_attn:
                x = F.scaled_dot_product_attention(
                    q.to(v.dtype),
                    k.to(v.dtype),
                    v,
                    dropout_p=self.attn_drop.p if self.training else 0.0)
            else:
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v
        x = x.transpose(1, 2).reshape(batch, seq_len, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def modulate(x, shift, scale):
    if x.dim() == shift.dim() + 1:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


class LightningDiTBlock(nn.Module):
    def __init__(
            self,
            hidden_size,
            num_heads,
            mlp_ratio=4.0,
            use_qknorm=False,
            use_swiglu=False,
            use_rmsnorm=False,
            wo_shift=False,
            use_modulation=True,
            standalone_qkv=False,
            standalone_w1w2=False,
            mlp_hidden_features=None,
            mlp_bias=True,
            **block_kwargs):
        super().__init__()

        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)

        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            standalone_qkv=standalone_qkv,
            **block_kwargs,
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        mlp_drop = block_kwargs.get('proj_drop', 0.0)
        if use_swiglu:
            self.mlp = SwiGLUFFN(
                hidden_size,
                int(2 / 3 * mlp_hidden_dim) if mlp_hidden_features is None else mlp_hidden_features,
                drop=mlp_drop,
                bias=mlp_bias,
                standalone_w1w2=standalone_w1w2)
        else:
            raise NotImplementedError

        self.use_modulation = use_modulation
        self.wo_shift = wo_shift
        if use_modulation:
            if self.wo_shift:
                self.adaLN_modulation = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(hidden_size, 4 * hidden_size, bias=True),
                )
            else:
                self.adaLN_modulation = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(hidden_size, 6 * hidden_size, bias=True),
                )

    def forward(self, x, c=None, feat_rope=None):
        if self.use_modulation:
            if self.wo_shift:
                scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
                shift_msa = None
                shift_mlp = None
            else:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            x = x + self.attn(self.norm1(x), rope=feat_rope)
            x = x + self.mlp(self.norm2(x))
        return x


def ddt_modulate(x, shift, scale):
    _, len_x, _ = x.shape
    _, len_mod, _ = shift.shape
    if len_x % len_mod != 0:
        raise ValueError(f'L_x ({len_x}) must be divisible by L ({len_mod})')
    repeat_factor = len_x // len_mod
    if repeat_factor != 1:
        shift = shift.repeat_interleave(repeat_factor, dim=1)
        scale = scale.repeat_interleave(repeat_factor, dim=1)
    return x * (1 + scale) + shift


def ddt_gate(x, gate):
    _, len_x, _ = x.shape
    _, len_gate, _ = gate.shape
    if len_x % len_gate != 0:
        raise ValueError(f'L_x ({len_x}) must be divisible by L ({len_gate})')
    repeat_factor = len_x // len_gate
    if repeat_factor != 1:
        gate = gate.repeat_interleave(repeat_factor, dim=1)
    return x * gate


class LightningDDTBlock(LightningDiTBlock):

    def forward(self, x, c, feat_rope=None):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=-1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + ddt_gate(self.attn(ddt_modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope), gate_msa)
        x = x + ddt_gate(self.mlp(ddt_modulate(self.norm2(x), shift_mlp, scale_mlp)), gate_mlp)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False, use_modulation=True,
                 use_scale_shift_table=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.use_modulation = use_modulation
        self.use_scale_shift_table = use_scale_shift_table
        if use_modulation:
            if use_scale_shift_table:  # Wan style: offset table instead of a projection
                self.scale_shift_table = nn.Parameter(torch.zeros(1, 2, hidden_size))
            else:
                self.adaLN_modulation = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(hidden_size, 2 * hidden_size, bias=True),
                )

    def forward(self, x, c=None, return_features=False):
        if not self.use_modulation:
            x = self.norm_final(x)
        else:
            if self.use_scale_shift_table:
                shift, scale = (self.scale_shift_table + F.silu(c).unsqueeze(1)).unbind(dim=1)
            else:
                shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale)
        return (self.linear(x), x) if return_features else self.linear(x)


class DDTFinalLayer(FinalLayer):

    def forward(self, x, c, return_features=False):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = ddt_modulate(self.norm_final(x), shift, scale)
        return (self.linear(x), x) if return_features else self.linear(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, bias=True):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=bias),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=bias),
        )

    @property
    def dtype(self):
        return next(self.mlp.parameters()).dtype

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
        return self.mlp(t_freq)
