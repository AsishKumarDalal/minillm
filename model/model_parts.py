import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class LayerKVCache:
    """Preallocated key/value storage for one transformer block.

    Shapes:
        k_cache, v_cache: (batch, num_heads, max_seq_len, head_dim)
        seq_len: number of valid positions currently stored

    This is intentionally a plain object, not an ``nn.Module`` or buffer, so
    it never appears in ``state_dict()`` and does not change checkpoint keys.
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.batch_size = int(batch_size)
        self.num_heads = int(num_heads)
        self.max_seq_len = int(max_seq_len)
        self.head_dim = int(head_dim)
        self.seq_len = 0

        self.k_cache = torch.zeros(
            self.batch_size,
            self.num_heads,
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
        if new_k.size(0) != self.batch_size:
            raise ValueError(
                f"KV cache batch mismatch: cache={self.batch_size}, "
                f"new={new_k.size(0)}"
            )

        t_new = new_k.size(2)
        end = self.seq_len + t_new
        if end > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: need positions [0, {end}) but "
                f"max_seq_len={self.max_seq_len}. Reset between requests "
                f"or increase the cache length."
            )

        self.k_cache[:, :, self.seq_len:end].copy_(new_k)
        self.v_cache[:, :, self.seq_len:end].copy_(new_v)
        self.seq_len = end
        return self.k_cache[:, :, :end], self.v_cache[:, :, :end]

    def reset(self) -> None:
        """Mark the cache empty. Stale bytes past seq_len are never read."""
        self.seq_len = 0

class LayerNorm(nn.Module):
  def __init__(self,emb_dim,bias=True):
    super().__init__()
    self.eps=1e-5
    self.scale=nn.Parameter(torch.ones(emb_dim))
    # Keep shift a Parameter even when bias=False so it is registered in
    # state_dict and moves with .to(device).
    self.shift=nn.Parameter(torch.zeros(emb_dim), requires_grad=bias)
  def forward(self,x):
     mean=x.mean(dim=-1, keepdim=True)
     variance=x.var(dim=-1, keepdim=True, unbiased=False)
     norm_x = (x - mean) / torch.sqrt(variance + self.eps)
     return self.scale * norm_x + self.shift
 
class MultiHeadAttention(nn.Module):# multihead attention
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(context_length, context_length, dtype=torch.bool),
                diagonal=1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        if kv_cache is None:
            key_len = num_tokens
            mask_bool = self.mask[:num_tokens, :num_tokens]
        else:
            # Queries in this chunk correspond to global positions
            # offset .. offset+num_tokens-1.  Keys are all cached+new
            # positions 0 .. key_len-1.
            offset = kv_cache.seq_len
            keys, values = kv_cache.update(keys, values)
            key_len = offset + num_tokens
            mask_bool = self.mask[offset : offset + num_tokens, :key_len]

        # Mixed precision / autocast can make the cache dtype differ from
        # the query dtype; use one dtype for the matmul.
        if keys.dtype != queries.dtype:
            keys = keys.to(queries.dtype)
            values = values.to(queries.dtype)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
        


class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers=nn.Sequential(
          nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),
          GELU(),
          nn.Linear(4*cfg["emb_dim"],cfg["emb_dim"])
        )
    def forward(self,x):
        return self.layers(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff= FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        placeholder = x
        x = self.norm1(x)
        x = self.att(x, kv_cache=kv_cache)
        x = placeholder + x
        placeholder2 = x
        x = self.norm2(x)
        x = self.ff(x)
        x = placeholder2 + x
        return x