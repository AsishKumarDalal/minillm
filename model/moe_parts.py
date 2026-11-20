"""Core layers for the Mixture-of-Experts (MoE) GPT.

This is a corrected port of the ``gptoss`` model
(https://github.com/AsishKumarDalal/gptoss) into the minillm layout.

Bugs fixed relative to the reference implementation
---------------------------------------------------
* Cached prefill was not causal in ``gptoss``: the causal mask was only
  applied when ``kv_cache is None``, so a prompt processed with
  ``use_cache=True`` could attend to future tokens.  The mask is now always
  applied using the absolute positions of the current query chunk.
* The reference ``attention_bias`` was a dense ``(1, H, S, S)`` trainable
  parameter (134M parameters at H=32, S=2048) that was also sliced
  incorrectly for cached decode.  It is replaced by an optional parameter-free
  ALiBi bias and optional per-head attention sinks.
* Expert capacity did not account for ``top_k`` and dropped tokens in
  arbitrary order.  Capacity is now ``capacity_factor * top_k * N / E`` and
  the lowest-gate tokens are dropped first.
* KV-cache reset zeroed the whole buffer on every request.  It now only
  resets the fill pointer; stale bytes past ``seq_len`` are never read.
* The router softmax is computed in float32 for stable routing under AMP,
  and an optional router z-loss is available.

The KV cache is deliberately a plain object, not an ``nn.Module`` or a
registered buffer, so it never enters ``state_dict()`` and never changes
checkpoint keys.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------
class LayerKVCache:
    """Preallocated key/value storage for one transformer block.

    Shapes
    ------
    k_cache, v_cache : (batch, num_kv_heads, max_seq_len, head_dim)
    seq_len          : number of valid positions currently stored
    """

    def __init__(
        self,
        batch_size: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("KV cache batch_size must be > 0")
        if num_kv_heads <= 0:
            raise ValueError("KV cache num_kv_heads must be > 0")
        if max_seq_len <= 0:
            raise ValueError("KV cache max_seq_len must be > 0")
        if head_dim <= 0:
            raise ValueError("KV cache head_dim must be > 0")

        self.batch_size = int(batch_size)
        self.num_kv_heads = int(num_kv_heads)
        self.max_seq_len = int(max_seq_len)
        self.head_dim = int(head_dim)
        self.seq_len = 0

        self.k_cache = torch.zeros(
            self.batch_size,
            self.num_kv_heads,
            self.max_seq_len,
            self.head_dim,
            device=device,
            dtype=dtype,
        )
        self.v_cache = torch.zeros_like(self.k_cache)

    def update(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V and return views over all valid cache positions."""
        if new_k.shape != new_v.shape:
            raise ValueError(
                f"K/V shape mismatch: {tuple(new_k.shape)} vs "
                f"{tuple(new_v.shape)}"
            )
        expected = (
            self.batch_size,
            self.num_kv_heads,
            new_k.size(2),
            self.head_dim,
        )
        if new_k.size(0) != self.batch_size:
            raise ValueError(
                f"KV cache batch mismatch: cache={self.batch_size}, "
                f"new={new_k.size(0)}"
            )
        if new_k.size(1) != self.num_kv_heads:
            raise ValueError(
                f"KV cache head mismatch: cache={self.num_kv_heads}, "
                f"new={new_k.size(1)}"
            )
        if new_k.size(-1) != self.head_dim:
            raise ValueError(
                f"KV cache head_dim mismatch: cache={self.head_dim}, "
                f"new={new_k.size(-1)}"
            )
        if new_k.size(2) <= 0:
            raise ValueError("KV cache update needs at least one token")

        t_new = int(new_k.size(2))
        end = self.seq_len + t_new
        if end > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: need positions [0, {end}) but "
                f"max_seq_len={self.max_seq_len}. Call reset() between "
                f"requests or increase context_length."
            )

        self.k_cache[:, :, self.seq_len : end].copy_(new_k)
        self.v_cache[:, :, self.seq_len : end].copy_(new_v)
        self.seq_len = end
        return self.k_cache[:, :, :end], self.v_cache[:, :, :end]

    def reset(self) -> None:
        """Mark the cache empty. Stale bytes past seq_len are never read."""
        self.seq_len = 0


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root-mean-square LayerNorm with a learnable per-channel scale."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("RMSNorm dim must be > 0")
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        # Normalise in float32 for stability under fp16/bf16 autocast.
        x_f = x.float()
        x_f = x_f * torch.rsqrt(
            x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (x_f * self.weight.float()).to(dtype)


# ---------------------------------------------------------------------------
# Grouped-query attention + RoPE + KV cache
# ---------------------------------------------------------------------------
class GroupedQueryAttentionWithRoPE(nn.Module):
    """Causal GQA with rotate-half RoPE, optional ALiBi and attention sinks."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        d_model = int(cfg["emb_dim"])
        num_heads = int(cfg["n_heads"])
        num_kv_heads = int(cfg["n_kv_heads"])
        context_length = int(cfg["context_length"])

        if d_model % num_heads != 0:
            raise ValueError("emb_dim must be divisible by n_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.head_dim = d_model // num_heads
        self.context_length = context_length
        self.use_rope = bool(cfg.get("use_rope", True))
        self.rope_theta = float(cfg.get("rope_theta", 10_000.0))
        self.use_alibi = bool(
            cfg.get("use_alibi_bias", cfg.get("use_attention_bias", False))
        )
        self.use_attention_sinks = bool(
            cfg.get("use_attention_sinks", False)
        )
        self.qkv_bias = bool(cfg.get("qkv_bias", False))
        self.out_proj_bias = bool(cfg.get("out_proj_bias", False))

        if self.head_dim % 2 != 0:
            raise ValueError(
                "head_dim must be even for RoPE "
                f"(got emb_dim/n_heads={self.head_dim})"
            )

        kv_dim = self.num_kv_heads * self.head_dim
        self.W_q = nn.Linear(d_model, d_model, bias=self.qkv_bias)
        self.W_k = nn.Linear(d_model, kv_dim, bias=self.qkv_bias)
        self.W_v = nn.Linear(d_model, kv_dim, bias=self.qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=self.out_proj_bias)
        self.dropout = nn.Dropout(float(cfg.get("drop_rate", 0.0)))

        # A single boolean mask shared by all layers; plain buffers do not
        # become parameters and survive .to(device).
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(
                    context_length, context_length, dtype=torch.bool
                ),
                diagonal=1,
            ),
            persistent=False,
        )

        if self.use_rope:
            cos, sin = self._precompute_rope(
                context_length, self.head_dim, self.rope_theta
            )
            self.register_buffer("cos_cached", cos, persistent=False)
            self.register_buffer("sin_cached", sin, persistent=False)
        else:
            self.cos_cached = None
            self.sin_cached = None

        if self.use_alibi:
            slopes = self._alibi_slopes(num_heads)
            self.register_buffer("alibi_slopes", slopes, persistent=False)
        else:
            self.alibi_slopes = None

        if self.use_attention_sinks:
            # One learned logit per query head.  It participates in the
            # softmax but has a value vector of zero, so it only rescales
            # attention weights (the "no-op" sink).
            self.attention_sinks = nn.Parameter(torch.zeros(num_heads))
        else:
            self.register_buffer(
                "attention_sinks", torch.empty(0), persistent=False
            )

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _precompute_rope(
        seq_len: int, dim: int, theta: float = 10_000.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin tables of shape ``(seq_len, dim)``."""
        if dim % 2 != 0:
            raise ValueError("RoPE head_dim must be even")
        if theta <= 0:
            raise ValueError("rope_theta must be > 0")
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    @staticmethod
    def _alibi_slopes(num_heads: int) -> torch.Tensor:
        """Geometric ALiBi slopes, one per head."""
        return torch.tensor(
            [2.0 ** (-8.0 * (i + 1) / num_heads) for i in range(num_heads)],
            dtype=torch.float32,
        )

    @staticmethod
    def _apply_rotary_emb(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate-half RoPE; cos/sin are sliced to the current positions."""
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        c = cos[..., :half]
        s = sin[..., :half]
        return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)

    def _repeat_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """Expand ``(b, Hkv, S, D)`` to ``(b, Hq, S, D)``."""
        if self.num_groups == 1:
            return kv
        b, hkv, s, d = kv.shape
        kv = kv[:, :, None, :, :].expand(b, hkv, self.num_groups, s, d)
        return kv.reshape(b, hkv * self.num_groups, s, d)

    def _causal_mask(
        self, query_len: int, key_len: int, offset: int, device: torch.device
    ) -> torch.Tensor:
        """Mask where a query cannot attend to a future key.

        ``offset`` is the absolute position of the first query.  This works
        for cacheless training, cached prefill and one-token decode.
        """
        if offset + query_len > self.context_length or key_len > self.context_length:
            # Can only happen if a caller bypasses the model's checks.
            q_pos = torch.arange(offset, offset + query_len, device=device)
            k_pos = torch.arange(key_len, device=device)
            return k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
        return self.causal_mask[
            offset : offset + query_len, :key_len
        ]

    def _alibi_bias(
        self,
        query_len: int,
        key_len: int,
        offset: int,
        device: torch.device,
    ) -> torch.Tensor:
        q_pos = torch.arange(offset, offset + query_len, device=device)
        k_pos = torch.arange(key_len, device=device)
        distance = (k_pos.unsqueeze(0) - q_pos.unsqueeze(1)).clamp(min=0)
        slopes = self.alibi_slopes.to(device=device, dtype=torch.float32)
        bias = -slopes.view(1, -1, 1, 1) * distance.view(
            1, 1, query_len, key_len
        )
        return bias

    # -- forward -----------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        if t <= 0:
            raise ValueError("attention needs at least one token")

        q = self.W_q(x).view(
            b, t, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.W_k(x).view(
            b, t, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = self.W_v(x).view(
            b, t, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        offset = kv_cache.seq_len if kv_cache is not None else 0

        if self.use_rope:
            max_pos = self.cos_cached.shape[0]
            if offset + t > max_pos:
                raise ValueError(
                    f"RoPE table overflow: offset ({offset}) + T ({t}) = "
                    f"{offset + t} > context_length ({max_pos})."
                )
            cos = self.cos_cached[offset : offset + t].unsqueeze(0).unsqueeze(0)
            sin = self.sin_cached[offset : offset + t].unsqueeze(0).unsqueeze(0)
            q = self._apply_rotary_emb(q, cos, sin)
            k = self._apply_rotary_emb(k, cos, sin)

        if kv_cache is not None:
            k_full, v_full = kv_cache.update(k, v)
        else:
            k_full, v_full = k, v

        key_len = int(k_full.size(2))
        k_full = self._repeat_kv(k_full)
        v_full = self._repeat_kv(v_full)

        # Keep the matmul in a common dtype under autocast.
        if k_full.dtype != q.dtype:
            k_full = k_full.to(q.dtype)
            v_full = v_full.to(q.dtype)

        attn = torch.matmul(q, k_full.transpose(-2, -1)) * (
            self.head_dim ** -0.5
        )

        if self.use_alibi:
            bias = self._alibi_bias(t, key_len, offset, x.device)
            attn = attn + bias.to(attn.dtype)

        # Always apply causality: this is the important cached-prefill fix.
        mask = self._causal_mask(t, key_len, offset, x.device)
        attn = attn.masked_fill(mask, float("-inf"))

        sink_weights = None
        if self.use_attention_sinks:
            sink_logits = self.attention_sinks.to(attn.dtype).view(
                1, self.num_heads, 1, 1
            )
            sink_logits = sink_logits.expand(b, self.num_heads, t, 1)
            attn = torch.cat((attn, sink_logits), dim=-1)

        attn = F.softmax(attn, dim=-1)
        if sink_weights is not None or self.use_attention_sinks:
            # Drop the sink column; its value vector is implicit zeros.
            attn = attn[..., :-1]
        attn = self.dropout(attn)

        context = torch.matmul(attn, v_full)
        context = context.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.out_proj(context)


# ---------------------------------------------------------------------------
# Experts / router / MoE layer
# ---------------------------------------------------------------------------
class Expert(nn.Module):
    """Two-layer GELU feed-forward expert."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        d_model = int(cfg["emb_dim"])
        d_ff = int(cfg["d_ff"])
        dropout = float(cfg.get("drop_rate", 0.0))
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class TopKRouter(nn.Module):
    """Token router: top-k experts with renormalised gate weights."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.num_experts = int(cfg["num_experts"])
        self.top_k = int(cfg["top_k"])
        self.jitter = float(cfg.get("router_jitter", 0.0))
        self.W_r = nn.Linear(int(cfg["emb_dim"]), self.num_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.W_r(x)
        if self.training and self.jitter > 0:
            logits = logits + torch.randn_like(logits) * self.jitter
        # Router probabilities in float32: fp16 softmax over many experts is
        # a common source of routing collapse / NaNs.
        probs = F.softmax(logits.float(), dim=-1)
        gate_weights, expert_idx = torch.topk(probs, self.top_k, dim=-1)
        gate_weights = gate_weights / gate_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-9)
        return (
            gate_weights.to(x.dtype),
            expert_idx,
            probs,
            logits.float(),
        )


def load_balance_loss(
    router_probs: torch.Tensor,
    expert_idx: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Switch-style load-balancing auxiliary loss.

    ``f`` is the fraction of tokens routed to each expert and ``P`` is the
    mean router probability for that expert.  Minimised when both are uniform.
    """
    if router_probs.ndim != 2:
        raise ValueError("router_probs must be (tokens, experts)")
    one_hot = F.one_hot(expert_idx, num_classes=num_experts).float()
    token_mask = one_hot.sum(dim=1).clamp(max=1.0)
    f = token_mask.mean(dim=0)
    p = router_probs.mean(dim=0)
    return num_experts * (f * p).sum()


def router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """Stabilise router logits by penalising their log-sum-exp magnitude."""
    log_z = torch.logsumexp(router_logits, dim=-1)
    return (log_z ** 2).mean()


class MoELayer(nn.Module):
    """Sparse top-k MoE feed-forward layer with optional capacity dropping."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = dict(cfg)
        self.num_experts = int(cfg["num_experts"])
        self.top_k = int(cfg["top_k"])
        self.capacity_factor = cfg.get("capacity_factor")
        self.aux_loss_coef = float(cfg.get("aux_loss_coef", 0.01))
        self.router_z_loss_coef = float(cfg.get("router_z_loss_coef", 0.0))
        self.router = TopKRouter(cfg)
        self.experts = nn.ModuleList(
            [Expert(cfg) for _ in range(self.num_experts)]
        )

    def _expert_capacity(self, num_tokens: int) -> int:
        if self.capacity_factor is None or not self.training:
            return num_tokens
        factor = max(0.0, float(self.capacity_factor))
        if factor <= 0:
            raise ValueError("capacity_factor must be > 0 or null")
        # Standard top-k capacity: each token creates top_k assignments.
        return max(
            int(factor * self.top_k * num_tokens / self.num_experts), 1
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, d_model = x.shape
        num_tokens = b * t
        x_flat = x.reshape(num_tokens, d_model)

        gate_weights, expert_idx, router_probs, router_logits = self.router(
            x_flat
        )

        aux_loss = self.aux_loss_coef * load_balance_loss(
            router_probs, expert_idx, self.num_experts
        )
        if self.router_z_loss_coef > 0:
            aux_loss = aux_loss + self.router_z_loss_coef * router_z_loss(
                router_logits
            )

        capacity = self._expert_capacity(num_tokens)
        out_flat = torch.zeros_like(x_flat)

        for expert_id, expert in enumerate(self.experts):
            token_idx, slot_idx = torch.where(expert_idx == expert_id)
            if token_idx.numel() == 0:
                continue

            weights = gate_weights[token_idx, slot_idx]
            if token_idx.numel() > capacity:
                # Keep the highest-gate tokens; dropping arbitrary tokens
                # biases whichever tokens happen to come first.
                keep = torch.topk(weights, capacity, largest=True).indices
                token_idx = token_idx[keep]
                slot_idx = slot_idx[keep]
                weights = weights[keep]

            expert_out = expert(x_flat[token_idx])
            out_flat.index_add_(
                0, token_idx, expert_out * weights.unsqueeze(-1)
            )

        return out_flat.view(b, t, d_model), aux_loss


class MoETransformerBlock(nn.Module):
    """Pre-norm transformer block: GQA attention followed by a MoE FFN."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        d_model = int(cfg["emb_dim"])
        self.norm1 = RMSNorm(d_model)
        self.attn = GroupedQueryAttentionWithRoPE(cfg)
        self.norm2 = RMSNorm(d_model)
        self.moe = MoELayer(cfg)
        self.residual_dropout = nn.Dropout(float(cfg.get("drop_rate", 0.0)))

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.attn(self.norm1(x), kv_cache=kv_cache)
        x = residual + self.residual_dropout(x)

        residual = x
        moe_out, aux_loss = self.moe(self.norm2(x))
        x = residual + self.residual_dropout(moe_out)
        return x, aux_loss
