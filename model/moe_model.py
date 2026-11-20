"""Mixture-of-Experts decoder-only language model.

The model keeps the minillm checkpoint conventions (plain ``state_dict`` with
``model_config`` stored alongside it) while using the gptoss architecture:
GQA, RoPE, RMSNorm, sparse top-k experts and a native KV cache.

Parameter accounting
--------------------
``parameter_breakdown()`` reports total parameters, parameters active for one
token (embeddings + attention + router + ``top_k`` experts per layer), and
the expert/attention split.  Training scripts print this at startup so a
config mistake is visible before the first step.

Generation
----------
``generate`` uses a prefill + one-token decode loop through the KV cache.
Because RoPE is relative, a full cache uses a reset-and-reprefill sliding
window fallback; cached positions are never silently reused with wrong
absolute tables.  With ``use_rope: false`` the model uses learned absolute
position embeddings and the same sliding-window fallback as the dense model.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe_parts import LayerKVCache, MoETransformerBlock, RMSNorm

EOT_TOKEN_ID = 50256

# Keys that define checkpoint compatibility.  Changing any of these requires
# a fresh checkpoint.
MOE_ARCH_KEYS = (
    "vocab_size",
    "context_length",
    "emb_dim",
    "n_layers",
    "n_heads",
    "n_kv_heads",
    "d_ff",
    "num_experts",
    "top_k",
    "qkv_bias",
    "out_proj_bias",
    "tie_weights",
    "use_rope",
    "use_alibi_bias",
    "use_attention_sinks",
)

MOE_CONFIG_DEFAULTS = {
    "vocab_size": 50257,
    "context_length": 128,
    "emb_dim": 768,
    "n_layers": 6,
    "n_heads": 12,
    "n_kv_heads": 4,
    "d_ff": 1536,
    "num_experts": 8,
    "top_k": 2,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "out_proj_bias": True,
    "head_bias": False,
    "tie_weights": True,
    "use_rope": True,
    "rope_theta": 10_000.0,
    "use_alibi_bias": False,
    "use_attention_sinks": True,
    "aux_loss_coef": 0.01,
    "router_z_loss_coef": 0.001,
    "capacity_factor": 1.25,
    "router_jitter": 0.0,
    "init_std": 0.02,
    # Legacy gptoss key: mapped to ALiBi instead of the reference's dense
    # (1, H, S, S) trainable bias tensor.
    "use_attention_bias": False,
}


def normalize_moe_config(cfg: dict) -> dict:
    """Return a complete MoE config dict with defaults filled in."""
    if not isinstance(cfg, dict):
        raise TypeError("MoE model config must be a dict")
    out = dict(MOE_CONFIG_DEFAULTS)
    out.update(cfg)
    # Aliases accepted from gptoss configs.
    if "max_seq_len" in cfg and "context_length" not in cfg:
        out["context_length"] = cfg["max_seq_len"]
    if "dropout" in cfg and "drop_rate" not in cfg:
        out["drop_rate"] = cfg["dropout"]
    if "vocabulary_size" in cfg and "vocab_size" not in cfg:
        out["vocab_size"] = cfg["vocabulary_size"]
    if "transformer_blocks" in cfg and "n_layers" not in cfg:
        out["n_layers"] = cfg["transformer_blocks"]
    if "d_model" in cfg and "emb_dim" not in cfg:
        out["emb_dim"] = cfg["d_model"]
    return out


def validate_moe_config(cfg: dict) -> dict:
    """Validate architecture invariants and return the normalised config."""
    cfg = normalize_moe_config(cfg)

    int_keys = (
        "vocab_size",
        "context_length",
        "emb_dim",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "d_ff",
        "num_experts",
        "top_k",
    )
    for key in int_keys:
        value = cfg.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer (got {value!r})")
        if value <= 0:
            raise ValueError(f"{key} must be > 0 (got {value})")

    if cfg["emb_dim"] % cfg["n_heads"] != 0:
        raise ValueError(
            f"emb_dim ({cfg['emb_dim']}) must be divisible by "
            f"n_heads ({cfg['n_heads']})"
        )
    if cfg["n_heads"] % cfg["n_kv_heads"] != 0:
        raise ValueError(
            f"n_heads ({cfg['n_heads']}) must be divisible by "
            f"n_kv_heads ({cfg['n_kv_heads']})"
        )
    head_dim = cfg["emb_dim"] // cfg["n_heads"]
    if cfg["use_rope"] and head_dim % 2 != 0:
        raise ValueError(
            f"head_dim ({head_dim}) must be even when use_rope is true"
        )
    if cfg["top_k"] > cfg["num_experts"]:
        raise ValueError(
            f"top_k ({cfg['top_k']}) cannot exceed num_experts "
            f"({cfg['num_experts']})"
        )

    for key in ("drop_rate", "init_std"):
        value = float(cfg.get(key, 0.0))
        if value < 0:
            raise ValueError(f"{key} must be >= 0")
    if not 0.0 <= float(cfg.get("drop_rate", 0.0)) < 1.0:
        raise ValueError("drop_rate must be in [0, 1)")
    if float(cfg.get("init_std", 0.0)) <= 0:
        raise ValueError("init_std must be > 0")
    if float(cfg.get("rope_theta", 10_000.0)) <= 0:
        raise ValueError("rope_theta must be > 0")
    if float(cfg.get("aux_loss_coef", 0.0)) < 0:
        raise ValueError("aux_loss_coef must be >= 0")
    if float(cfg.get("router_z_loss_coef", 0.0)) < 0:
        raise ValueError("router_z_loss_coef must be >= 0")
    if float(cfg.get("router_jitter", 0.0)) < 0:
        raise ValueError("router_jitter must be >= 0")

    capacity = cfg.get("capacity_factor")
    if capacity is not None and float(capacity) <= 0:
        raise ValueError("capacity_factor must be > 0 or null")

    if cfg["use_alibi_bias"] and cfg["use_rope"]:
        # Both are legal, but ALiBi + RoPE is unusual.  Warn once at model
        # construction rather than silently producing a double position bias.
        pass
    return cfg


class MoELanguageModel(nn.Module):
    """Decoder-only MoE language model with KV-cached generation."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        cfg = validate_moe_config(cfg)
        self.cfg = cfg
        self.vocab_size = int(cfg["vocab_size"])
        self.context_length = int(cfg["context_length"])
        self.n_layers = int(cfg["n_layers"])
        self.n_heads = int(cfg["n_heads"])
        self.n_kv_heads = int(cfg["n_kv_heads"])
        self.head_dim = int(cfg["emb_dim"]) // self.n_heads
        self.use_rope = bool(cfg["use_rope"])
        self.tie_weights = bool(cfg.get("tie_weights", True))

        self.tok_emb = nn.Embedding(self.vocab_size, int(cfg["emb_dim"]))
        if not self.use_rope:
            self.pos_emb = nn.Embedding(self.context_length, int(cfg["emb_dim"]))
        else:
            self.pos_emb = None
        self.drop_emb = nn.Dropout(float(cfg.get("drop_rate", 0.0)))

        self.layers = nn.ModuleList(
            [MoETransformerBlock(cfg) for _ in range(self.n_layers)]
        )
        self.norm = RMSNorm(int(cfg["emb_dim"]))
        self.out_head = nn.Linear(
            int(cfg["emb_dim"]),
            self.vocab_size,
            bias=bool(cfg.get("head_bias", False)),
        )

        self.apply(
            lambda module: self._init_weights(module, float(cfg["init_std"]))
        )
        if self.tie_weights:
            self.out_head.weight = self.tok_emb.weight

        # GPT-2 residual scaling: keep the residual stream variance bounded.
        residual_std = float(cfg["init_std"]) / math.sqrt(2 * self.n_layers)
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith(
                "fc2.weight"
            ):
                nn.init.normal_(param, mean=0.0, std=residual_std)

        # Router weights start small: uniform routing at step 0.
        for module in self.modules():
            if module.__class__.__name__ == "TopKRouter":
                nn.init.normal_(module.W_r.weight, mean=0.0, std=0.02)

        # Plain attribute: cache tensors never enter state_dict().
        self.kv_caches: Optional[list[LayerKVCache]] = None

    @staticmethod
    def _init_weights(module: nn.Module, std: float) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    # -- parameter accounting ---------------------------------------------
    def parameter_breakdown(self) -> dict:
        """Total/active parameter counts for this architecture."""
        total = sum(p.numel() for p in self.parameters())
        expert_total = sum(
            p.numel()
            for name, p in self.named_parameters()
            if ".experts." in name
        )
        per_expert = (
            expert_total // (self.n_layers * int(self.cfg["num_experts"]))
            if self.n_layers and int(self.cfg["num_experts"])
            else 0
        )
        inactive_experts = (
            int(self.cfg["num_experts"]) - int(self.cfg["top_k"])
        ) * self.n_layers
        active = total - inactive_experts * per_expert
        router_total = sum(
            p.numel()
            for name, p in self.named_parameters()
            if ".router." in name
        )
        return {
            "total": int(total),
            "active": int(active),
            "expert_total": int(expert_total),
            "expert_active": int(per_expert * int(self.cfg["top_k"]) * self.n_layers),
            "router": int(router_total),
            "embedding": int(self.tok_emb.weight.numel()),
        }

    # -- KV cache lifecycle ------------------------------------------------
    def build_cache(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_seq_len: Optional[int] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if device is None:
            device = self.tok_emb.weight.device
        if dtype is None:
            dtype = self.tok_emb.weight.dtype
        if max_seq_len is None:
            max_seq_len = self.context_length
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")

        self.kv_caches = [
            LayerKVCache(
                batch_size=batch_size,
                num_kv_heads=self.n_kv_heads,
                max_seq_len=max_seq_len,
                head_dim=self.head_dim,
                device=device,
                dtype=dtype,
            )
            for _ in range(self.n_layers)
        ]

    def reset_cache(self) -> None:
        if self.kv_caches is not None:
            for cache in self.kv_caches:
                cache.reset()

    def clear_cache(self) -> None:
        self.kv_caches = None

    def _cache_matches(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> bool:
        if not self.kv_caches or len(self.kv_caches) != self.n_layers:
            return False
        first = self.kv_caches[0]
        return (
            first.batch_size == batch_size
            and first.num_kv_heads == self.n_kv_heads
            and first.head_dim == self.head_dim
            and first.max_seq_len == self.context_length
            and first.k_cache.device == device
            and first.k_cache.dtype == dtype
        )

    def _ensure_cache(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        if not self._cache_matches(batch_size, device, dtype):
            self.build_cache(batch_size, device=device, dtype=dtype)

    # -- forward -----------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        ignore_index: int = -100,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, aux_loss)``.

        ``labels`` is accepted for API symmetry but the loss is not returned
        here; use :meth:`forward_loss` when both CE and the auxiliary routing
        loss are needed.  ``use_cache=True`` is inference-only.
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be (batch, sequence)")
        batch_size, seq_len = input_ids.shape
        if seq_len <= 0:
            raise ValueError("input_ids must contain at least one token")
        if labels is not None and use_cache:
            raise ValueError(
                "use_cache=True is inference-only; labels are not supported"
            )

        if use_cache:
            self._ensure_cache(
                batch_size, input_ids.device, self.tok_emb.weight.dtype
            )
            assert self.kv_caches is not None
            past_len = self.kv_caches[0].seq_len
            if any(cache.seq_len != past_len for cache in self.kv_caches):
                raise RuntimeError(
                    "Layer KV caches are out of sync; call reset_cache()"
                )
            total_len = past_len + seq_len
            if total_len > self.context_length:
                raise ValueError(
                    f"KV-cached forward needs positions [0, {total_len}) but "
                    f"context_length={self.context_length}"
                )
            positions = torch.arange(
                past_len, total_len, device=input_ids.device
            )
            caches = self.kv_caches
        else:
            if seq_len > self.context_length:
                raise ValueError(
                    f"sequence length {seq_len} exceeds context_length "
                    f"{self.context_length}"
                )
            positions = torch.arange(seq_len, device=input_ids.device)
            caches = [None] * self.n_layers

        x = self.tok_emb(input_ids)
        if self.pos_emb is not None:
            x = x + self.pos_emb(positions)
        x = self.drop_emb(x)

        total_aux = torch.zeros((), device=input_ids.device, dtype=torch.float32)
        for layer, cache in zip(self.layers, caches):
            x, aux = layer(x, kv_cache=cache)
            total_aux = total_aux + aux.float()

        x = self.norm(x)
        logits = self.out_head(x)
        return logits, total_aux

    def forward_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(total_loss, ce_loss, aux_loss, logits)``."""
        if labels.shape != input_ids.shape:
            raise ValueError(
                f"labels shape {tuple(labels.shape)} does not match input_ids "
                f"{tuple(input_ids.shape)}"
            )
        logits, aux_loss = self(input_ids)
        ce_loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            labels.reshape(-1).long(),
            ignore_index=ignore_index,
        )
        total_loss = ce_loss + aux_loss.to(ce_loss.dtype)
        return total_loss, ce_loss, aux_loss, logits

    # -- generation --------------------------------------------------------
    def _sample_next(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
    ) -> torch.Tensor:
        if top_k is not None:
            k = min(int(top_k), logits.size(-1))
            if k <= 0:
                raise ValueError("top_k must be > 0")
            top_values, _ = torch.topk(logits, k)
            threshold = top_values[:, -1]
            logits = logits.masked_fill(
                logits < threshold.unsqueeze(-1), float("-inf")
            )
        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return torch.argmax(logits, dim=-1, keepdim=True)

    @staticmethod
    def _apply_eos(
        idx_next: torch.Tensor,
        finished: torch.Tensor,
        eos_id: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        if eos_id is None:
            return idx_next, finished, False
        finished = finished | (idx_next.squeeze(-1) == eos_id)
        if finished.any():
            idx_next = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(idx_next, int(eos_id)),
                idx_next,
            )
        return idx_next, finished, bool(finished.all())

    def _generate_uncached(
        self,
        generated: torch.Tensor,
        num_tokens: int,
        context_size: int,
        temperature: float,
        top_k: Optional[int],
        eos_id: Optional[int],
        finished: torch.Tensor,
    ) -> torch.Tensor:
        for _ in range(num_tokens):
            idx_cond = generated[:, -context_size:]
            logits, _ = self(idx_cond)
            next_logits = logits[:, -1, :]
            idx_next = self._sample_next(next_logits, temperature, top_k)
            idx_next, finished, all_done = self._apply_eos(
                idx_next, finished, eos_id
            )
            generated = torch.cat((generated, idx_next), dim=1)
            if all_done:
                break
        return generated

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 250,
        context_size: Optional[int] = None,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        eos_id: Optional[int] = EOT_TOKEN_ID,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressively extend ``idx`` with optional KV caching."""
        if idx.ndim != 2:
            raise ValueError("idx must be (batch, sequence)")
        if int(max_new_tokens) <= 0:
            return idx
        if temperature < 0:
            raise ValueError("temperature must be >= 0")

        was_training = self.training
        self.eval()
        try:
            if context_size is None:
                context_size = self.context_length
            context_size = min(int(context_size), self.context_length)
            if context_size <= 0:
                raise ValueError("context_size must be > 0")

            generated = idx
            finished = torch.zeros(
                idx.size(0), dtype=torch.bool, device=idx.device
            )
            if not use_cache:
                return self._generate_uncached(
                    generated,
                    int(max_new_tokens),
                    context_size,
                    temperature,
                    top_k,
                    eos_id,
                    finished,
                )

            cond = idx[:, -context_size:]
            self._ensure_cache(
                cond.size(0), cond.device, self.tok_emb.weight.dtype
            )
            self.reset_cache()
            logits, _ = self(cond, use_cache=True)
            next_logits = logits[:, -1, :]
            remaining = int(max_new_tokens)

            for step in range(remaining):
                idx_next = self._sample_next(next_logits, temperature, top_k)
                idx_next, finished, all_done = self._apply_eos(
                    idx_next, finished, eos_id
                )
                generated = torch.cat((generated, idx_next), dim=1)
                if all_done or step == remaining - 1:
                    break

                # Cached K/V is already rotated at its absolute position, so
                # the window cannot slide in place.  Reset and reprefill the
                # tail when it is full.
                assert self.kv_caches is not None
                if self.kv_caches[0].seq_len + 1 > context_size:
                    self.reset_cache()
                    return self._generate_uncached(
                        generated,
                        remaining - (step + 1),
                        context_size,
                        temperature,
                        top_k,
                        eos_id,
                        finished,
                    )

                logits, _ = self(idx_next, use_cache=True)
                next_logits = logits[:, -1, :]
            return generated
        finally:
            self.train(was_training)
