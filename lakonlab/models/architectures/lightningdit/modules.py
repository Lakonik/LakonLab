import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            num_freqs=1):
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

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])
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
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
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
            use_rmsnorm=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn
        self.upcast_attn = upcast_attn
        if use_rmsnorm:
            norm_layer = RMSNorm
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope=None):
        batch, seq_len, channels = x.shape
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
            **block_kwargs,
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        mlp_drop = block_kwargs.get('proj_drop', 0.0)
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim), drop=mlp_drop)
        else:
            raise NotImplementedError

        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 4 * hidden_size, bias=True),
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True),
            )
        self.wo_shift = wo_shift

    def forward(self, x, c, feat_rope=None):
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
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
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DDTFinalLayer(FinalLayer):

    def forward(self, x, c):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = ddt_modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
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
