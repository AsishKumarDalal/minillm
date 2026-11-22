"""Llama-2-style building blocks for the weight-space student.

Self-contained: does not import from the project's ``model/`` package.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Norm and activation
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Llama-2 style)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * out).to(x.dtype)


class SwiGLU(nn.Module):
    """Llama-2 MLP: down(silu(gate(x)) * up(x)). No biases."""

    def __init__(self, cfg: dict):
        super().__init__()
        d = int(cfg["emb_dim"])
        hidden = int(cfg.get("intermediate_size", 4 * d))
        self.gate_proj = nn.Linear(d, hidden, bias=False)
        self.up_proj = nn.Linear(d, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Rotary positional embeddings
# ---------------------------------------------------------------------------
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Precomputes RoPE cos/sin. Contains no learned weights."""

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 1024,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.head_dim = int(head_dim)
        self.max_seq_len = int(max_seq_len)
        inv_freq = 1.0 / (
            float(theta)
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _build(self, seq_len: int, offset: int, device, dtype):
        t = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._build(seq_len, offset, q.device, q.dtype)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return q, k


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------
class LayerKVCache:
    """Plain (non-module) KV cache, one per decoder layer."""

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.batch_size = int(batch_size)
        self.seq_len = 0
        self.k = torch.zeros(
            batch_size, num_heads, max_seq_len, head_dim, device=device, dtype=dtype
        )
        self.v = torch.zeros_like(self.k)

    def update(self, k: torch.Tensor, v: torch.Tensor):
        t_new = k.size(2)
        end = self.seq_len + t_new
        if end > self.k.size(2):
            raise ValueError(f"KV cache overflow: need {end}, have {self.k.size(2)}")
        self.k[:, :, self.seq_len:end].copy_(k)
        self.v[:, :, self.seq_len:end].copy_(v)
        self.seq_len = end
        return self.k[:, :, :end], self.v[:, :, :end]

    def reset(self):
        self.seq_len = 0


# ---------------------------------------------------------------------------
# Attention and decoder layer
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = int(cfg["emb_dim"])
        h = int(cfg["n_heads"])
        if d % h != 0:
            raise ValueError("emb_dim must be divisible by n_heads")
        self.n_heads = h
        self.head_dim = d // h
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        ctx = int(cfg["context_length"])
        self.register_buffer(
            "mask", torch.triu(torch.ones(ctx, ctx, dtype=torch.bool), diagonal=1)
        )

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        cache: Optional[LayerKVCache] = None,
        offset: int = 0,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        h, hd = self.n_heads, self.head_dim

        q = self.q_proj(x).view(b, t, h, hd).transpose(1, 2)
        k = self.k_proj(x).view(b, t, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(b, t, h, hd).transpose(1, 2)

        q, k = rope(q, k, seq_len=t, offset=offset)

        if cache is None:
            key_len = t
            mask_bool = self.mask[:t, :t]
        else:
            if k.dtype != cache.k.dtype:
                k = k.to(cache.k.dtype)
                v = v.to(cache.v.dtype)
            k, v = cache.update(k, v)
            key_len = offset + t
            mask_bool = self.mask[offset : offset + t, :key_len]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        scores = scores.masked_fill(mask_bool, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, t, h * hd)
        return self.o_proj(out)


class DecoderLayer(nn.Module):
    """Pre-norm Llama block with RMSNorm and SwiGLU."""

    def __init__(self, cfg: dict):
        super().__init__()
        d = int(cfg["emb_dim"])
        self.input_layernorm = RMSNorm(d)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(d)
        self.mlp = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        cache: Optional[LayerKVCache] = None,
        offset: int = 0,
    ) -> torch.Tensor:
        h = x + self.self_attn(self.input_layernorm(x), rope, cache, offset)
        return h + self.mlp(self.post_attention_layernorm(h))
