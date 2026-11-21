# Copyright (c) 2026 Hansheng Chen
# Adapted from the Apache-2.0 Sol-H3-VAE reference decoder:
# https://huggingface.co/Efficient-Large-Model/Sol-H3-VAE/blob/6a8a33fc61e092281f32bf28cde236760e2caaa4/code/sol_h3_vae/model.py

import math

import torch
from torch import nn
from torch.nn import functional as F
from diffusers.models.attention import AttentionModuleMixin
from diffusers.models.attention_dispatch import (
    AttentionBackendName, _AttentionBackendRegistry, dispatch_attention_fn)

from lakonlab.ops import swiglu_linear

# Sol-H3-VAE decoder constants (MiniMax-H3 vae/config.json and the Sol reference decoder)
NUM_HEADS = 32
HEAD_DIM = 64
DIM = NUM_HEADS * HEAD_DIM
FFN_MULT = 4
NUM_LAYERS = 24
NORM_EPS = 1e-5
NUM_REGISTER_TOKENS = 4
ROPE_THETA = 100.0
ROPE_DIM = int(HEAD_DIM * 0.75)
WINDOW = 16  # spatial window in tokens; odd layers shift it by half
BLOCK = 5  # native latent frames per block (17 video frames)
T_SHIFT = 2  # temporal window shift of odd layers
TIME_NORM = 5
PATCH, PATCH_T, OUT_CH = 16, 4, 3
FRAMES_TO_TRIM = 3  # frames dropped from the first latent of every block


def _rope(T, H, W, s, st, device):
    """cos/sin (1, T*H*W, 1, ROPE_DIM // 2) for tokens in (T, H, W) order with window-local time and space ids
    for shifts (st, s); both halves of the rotary dims share them."""
    lt = ((torch.arange(T, device=device) + st) % BLOCK).float()
    ly = ((torch.arange(H, device=device) + s) % WINDOW).float()
    lx = ((torch.arange(W, device=device) + s) % WINDOW).float()
    t = (lt + 0.5) / TIME_NORM * 2 - 1
    hy = (ly + 0.5) / WINDOW * 2 - 1
    wx = (lx + 0.5) / WINDOW * 2 - 1
    grid = torch.stack(torch.meshgrid(t, hy, wx, indexing='ij'), -1).reshape(1, -1, 3)
    # diffusers MiniMaxH3VideoRotaryPosEmbed
    inv_freq = 1.0 / ROPE_THETA ** torch.arange(0, 1, 6 / ROPE_DIM, dtype=torch.float32, device=device)
    angles = 2.0 * math.pi * grid[:, :, :, None] * inv_freq[None, None, None, :]
    angles = angles.flatten(2, 3).unsqueeze(2)
    return angles.cos(), angles.sin()


def _rotate_(x, emb):
    """Apply rotary embeddings in place to (B, L, heads, head_dim) x; rounds like the out-of-place
    ``x * cos + rotate_half(x) * sin`` in x's dtype."""
    cos, sin = (t.to(x.dtype) for t in emb)
    half = cos.size(-1)
    first, second = x[..., :half], x[..., half:2 * half]
    first_sin, second_sin = first * sin, second * sin
    first.mul_(cos).sub_(second_sin)
    second.mul_(cos).add_(first_sin)


def _partition(x, s, st):
    """x: (B, T, H, W, ...) -> windows (B*nW, BLOCK*WINDOW*WINDOW, ...), valid (nW, BLOCK*WINDOW*WINDOW), info.
    Window order is (time window, row, col)."""
    B, T, H, W = x.shape[:4]
    ws, tb = WINDOW, BLOCK
    rest = x.shape[4:]
    Tp = ((T + st + tb - 1) // tb) * tb
    Hp = ((H + s + ws - 1) // ws) * ws
    Wp = ((W + s + ws - 1) // ws) * ws
    xp = x.new_zeros(B, Tp, Hp, Wp, *rest)
    xp[:, st:st + T, s:s + H, s:s + W] = x
    valid = torch.zeros(Tp, Hp, Wp, dtype=torch.bool, device=x.device)
    valid[st:st + T, s:s + H, s:s + W] = True
    nt, nh, nw = Tp // tb, Hp // ws, Wp // ws
    r = len(rest)
    xw = xp.view(B, nt, tb, nh, ws, nw, ws, *rest).permute(0, 1, 3, 5, 2, 4, 6, *range(7, 7 + r))
    xw = xw.reshape(B * nt * nh * nw, tb * ws * ws, *rest)
    vw = valid.view(nt, tb, nh, ws, nw, ws).permute(0, 2, 4, 1, 3, 5).reshape(nt * nh * nw, tb * ws * ws)
    return xw, vw, (B, T, H, W, Tp, Hp, Wp, nt, nh, nw)


def _pack(rows, valid, total):
    """The ``total`` valid entries of (B, n, L) padded rows in order, and the cu_seqlens of the B*n sequences."""
    valid = valid.expand(rows.shape).flatten(0, 1)
    order = valid.logical_not().flatten().to(torch.int8).argsort(stable=True)[:total]
    return rows.flatten()[order], F.pad(valid.sum(1, dtype=torch.int32).cumsum(0, dtype=torch.int32), (1, 0))


def _window_rows(B, T, H, W, nb, S, s, st, device, padded=False):
    """Window attention indices over the B*T*H*W token rows and the (B, nb, S) global-token rows after them.

    Windows run (batch, time window, row, col) and keep their tokens in (t, y, x) order; the keys of a window
    append the global tokens of the native blocks it overlaps. For masked SDPA, return padded query and key rows,
    a key mask and the gather that restores token order. For FlashAttention varlen kernels, return packed rows
    with their cu_seqlens.
    """
    N = T * H * W
    tokens, valid, info = _partition(torch.arange(1, N + 1, device=device).view(1, T, H, W), s, st)
    Tp, Hp, Wp, nt, nh, nw = info[4:]
    # native blocks of the first and last latent of temporal window j, which covers [j*BLOCK - st, (j+1)*BLOCK - st)
    blocks = [(max(j * BLOCK - st, 0) // BLOCK, (min((j + 1) * BLOCK - st, T) - 1) // BLOCK) for j in range(nt)]
    global_rows = torch.tensor([[b * S + i for b in pair for i in range(S)] for pair in blocks], device=device)
    global_valid = torch.tensor([[True] * S + [b1 != b0] * S for b0, b1 in blocks], device=device)
    global_rows, global_valid = (
        t[:, None, None].expand(nt, nh, nw, 2 * S).reshape(-1, 2 * S) for t in (global_rows, global_valid))
    batch = torch.arange(B, device=device)[:, None, None]
    token_rows = batch * N + tokens - 1
    key_rows = torch.cat([token_rows, B * N + batch * nb * S + global_rows], -1)
    key_valid = torch.cat([valid, global_valid], -1)
    if padded:
        # Undo the window permutation and trim the padding to get one output row per input token.
        output_rows = torch.arange(token_rows.numel(), device=device).view(
            B, nt, nh, nw, BLOCK, WINDOW, WINDOW).permute(0, 1, 4, 2, 5, 3, 6).reshape(B, Tp, Hp, Wp)
        output_rows = output_rows[:, st:st + T, s:s + H, s:s + W].reshape(-1)
        return (token_rows.masked_fill(~valid, 0).flatten(0, 1),
                key_rows.masked_fill(~key_valid, 0).flatten(0, 1),
                key_valid.expand_as(key_rows).flatten(0, 1).contiguous(), output_rows)
    q_rows, cu_seqlens_q = _pack(token_rows, valid, B * N)
    k_rows, cu_seqlens_k = _pack(
        key_rows, key_valid, B * (N + nh * nw * S * sum(1 + (b1 != b0) for b0, b1 in blocks)))
    return q_rows, k_rows, cu_seqlens_q, cu_seqlens_k


def _global_rows(B, T, H, W, nb, S, device, padded=False):
    """Keys of the (B, nb, S) global tokens: every token of their native block, then the block's global tokens.
    Returns padded key rows and a mask for masked SDPA, or packed key rows and the query and key cu_seqlens for
    FlashAttention varlen kernels."""
    N = T * H * W
    tokens = torch.arange(nb * BLOCK * H * W, device=device).view(nb, -1)
    batch = torch.arange(B, device=device)[:, None, None]
    globals_ = torch.arange(nb * S, device=device).view(nb, S)
    rows = torch.cat([batch * N + tokens, B * N + batch * nb * S + globals_], -1)
    valid = torch.cat([tokens < N, torch.ones(nb, S, dtype=torch.bool, device=device)], -1)
    if padded:
        return rows.masked_fill(~valid, 0).flatten(0, 1), valid.expand_as(rows).flatten(0, 1).contiguous()
    rows, cu_seqlens_k = _pack(rows, valid, B * (N + nb * S))
    cu_seqlens_q = torch.arange(B * nb + 1, dtype=torch.int32, device=device) * S
    return rows, cu_seqlens_q, cu_seqlens_k


def _padded(attn, device, dtype):
    """Whether attention takes padded windows with a key mask: SDPA where PyTorch's FlashAttention 2 kernels are
    unavailable (CPU, below SM80, not 16-bit). FlashAttention 3 and PyTorch's FlashAttention 2 take packed varlen
    rows."""
    return attn.attention_backend == AttentionBackendName.NATIVE and not (
        device.type == 'cuda' and dtype in (torch.bfloat16, torch.float16)
        and torch.backends.cuda.is_flash_attention_available() and torch.cuda.get_device_capability(device) >= (8, 0))


class SolH3AttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
            self, attn, query, key, value, cu_seqlens_q=None, cu_seqlens_k=None,
            max_seqlen_q=None, max_seqlen_k=None, attn_mask=None):
        backend = attn.attention_backend
        if backend == AttentionBackendName.NATIVE and attn_mask is None:
            return torch.ops.aten._flash_attention_forward(
                query, key, value, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, 0.0, False, False)[0]
        if backend == AttentionBackendName.NATIVE:
            return dispatch_attention_fn(query, key, value, attn_mask=attn_mask, backend=backend)
        if backend != AttentionBackendName._FLASH_VARLEN_3:
            raise ValueError(f'Sol window padding is not supported by attention backend {backend!r}')
        return dispatch_attention_fn(
            query[None], key[None], value[None],
            attention_kwargs=dict(
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k),
            backend=backend)[0]


class SolH3Attention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = SolH3AttnProcessor
    _available_processors = [SolH3AttnProcessor]
    _supports_qkv_fusion = False

    def __init__(self):
        super().__init__()
        self.heads, self.dim_head = NUM_HEADS, HEAD_DIM
        self.to_qkv = nn.Linear(DIM, 3 * DIM)
        self.to_out = nn.Linear(DIM, DIM)
        self.norm_q = nn.RMSNorm(HEAD_DIM, eps=NORM_EPS, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(HEAD_DIM, eps=NORM_EPS, elementwise_affine=False)
        self.set_processor(SolH3AttnProcessor())

    @property
    def attention_backend(self):
        return self.processor._attention_backend or _AttentionBackendRegistry.get_active_backend()[0]

    def forward(self, query, key, value, *args, **kwargs):
        return self.processor(self, query, key, value, *args, **kwargs)


class SolH3SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, 2 * hidden)
        self.w2 = nn.Linear(hidden, dim)

    def forward(self, x):
        x = self.w1(x)
        if x.requires_grad:  # autograd keeps both factors of the product
            gate, up = x[..., :x.size(-1) // 2], x[..., x.size(-1) // 2:]  # slices, unlike chunk, allow in-place ops
            return self.w2(F.silu(gate, inplace=True) * up)
        return swiglu_linear(self.w2, x)


class SolH3Block(nn.Module):
    """One Sol decoder layer on the tokens x (B, T, H, W, D) and the per-block global tokens g (B, nb, S, D)."""

    def __init__(self):
        super().__init__()
        self.norm1 = nn.RMSNorm(DIM, eps=NORM_EPS)
        self.norm2 = nn.RMSNorm(DIM, eps=NORM_EPS)
        self.scale1 = nn.Parameter(torch.empty(DIM))
        self.scale2 = nn.Parameter(torch.empty(DIM))
        self.attn = SolH3Attention()
        self.ff = SolH3SwiGLU(DIM, DIM * FFN_MULT)

    def forward(self, x, g, shifted, global_indices=None, window_indices=None):
        # attention temporaries die with _attention, before the FFN allocates its hidden activations
        x, g = self._attention(x, g, shifted, global_indices, window_indices)
        x = x + self.ff(self.norm2(x.float()).to(x.dtype)) * self.scale2
        g = g + self.ff(self.norm2(g.float()).to(g.dtype)) * self.scale2
        return x, g

    def _attention(self, x, g, shifted, global_indices, window_indices):
        """Window attention of the tokens (each window also sees the global tokens of the native blocks it
        overlaps) and global attention of the global tokens over their native block, with the residual."""
        B, T, H, W, _ = x.shape
        nb, S = g.shape[1:3]
        N = B * T * H * W
        heads, dh = NUM_HEADS, HEAD_DIM
        s, st = (WINDOW // 2, T_SHIFT) if shifted else (0, 0)
        # token rows, then global-token rows: one gather packs the queries or keys of either attention
        qkv = self.attn.to_qkv(self.norm1(torch.cat([x.flatten(0, 3), g.flatten(0, 2)]).float()).to(x.dtype))
        qkv = qkv.view(-1, heads, 3, dh)
        q = self.attn.norm_q(qkv[:, :, 0].float()).to(x.dtype)
        k = self.attn.norm_k(qkv[:, :, 1].float()).to(x.dtype)
        v = qkv[:, :, 2]
        rope = _rope(T, H, W, s, st, x.device)
        _rotate_(q[:N].view(B, -1, heads, dh), rope)
        _rotate_(k[:N].view(B, -1, heads, dh), rope)
        padded = _padded(self.attn, x.device, x.dtype)
        if global_indices is None:
            global_indices = _global_rows(B, T, H, W, nb, S, x.device, padded=padded)
        if padded:
            rows, key_mask = global_indices
            og = self.attn(q[N:].unflatten(0, (B * nb, S)), k[rows], v[rows], attn_mask=key_mask[:, None, None, :])
            og = og.flatten(0, 1)
        else:
            rows, cu_seqlens_q, cu_seqlens_k = global_indices
            og = self.attn(q[N:], k[rows], v[rows], cu_seqlens_q, cu_seqlens_k, S, BLOCK * H * W + S)
        if window_indices is None:
            window_indices = _window_rows(B, T, H, W, nb, S, s, st, x.device, padded=padded)
        q_rows, k_rows, *_ = window_indices
        q = q[q_rows]
        k = k[k_rows]
        v = v[k_rows]
        del qkv
        if padded:
            _, _, key_mask, output_rows = window_indices
            o = self.attn(q, k, v, attn_mask=key_mask[:, None, None, :]).flatten(0, 1)
        else:
            _, _, cu_seqlens_q, cu_seqlens_k = window_indices
            o = self.attn(q, k, v, cu_seqlens_q, cu_seqlens_k, BLOCK * WINDOW ** 2, BLOCK * WINDOW ** 2 + 2 * S)
        del q, k, v
        if padded:
            o = o[output_rows]
        else:
            o = torch.empty_like(o).index_copy_(0, q_rows, o)  # back to token order
        x = x + self.attn.to_out(o.flatten(1)).view_as(x) * self.scale1
        g = g + self.attn.to_out(og.flatten(1)).view_as(g) * self.scale1
        return x, g
