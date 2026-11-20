"""Tiny experiment model: real attention/norm, swappable MoE layer.

Reuses model.moe_parts (read-only) so benchmarks differ ONLY in the expert
implementation. Returns (logits, aux_loss) like the real MoELanguageModel.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.moe_parts import (  # noqa: E402
    GroupedQueryAttentionWithRoPE,
    RMSNorm,
)
from model.moe_model import validate_moe_config  # noqa: E402

from moe_variants import build_layer  # noqa: E402


class VariantBlock(nn.Module):
    def __init__(self, cfg: dict, variant: str) -> None:
        super().__init__()
        d = int(cfg["emb_dim"])
        self.norm1 = RMSNorm(d)
        self.attn = GroupedQueryAttentionWithRoPE(cfg)
        self.norm2 = RMSNorm(d)
        self.moe = build_layer(variant, cfg)
        self.drop = nn.Dropout(float(cfg.get("drop_rate", 0.0)))

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        y, aux = self.moe(self.norm2(x))
        return x + self.drop(y), aux


class VariantMoELanguageModel(nn.Module):
    def __init__(self, cfg: dict, variant: str = "loop") -> None:
        super().__init__()
        self.cfg = validate_moe_config(cfg)
        self.variant = variant
        d = int(self.cfg["emb_dim"])
        self.vocab_size = int(self.cfg["vocab_size"])
        self.context_length = int(self.cfg["context_length"])

        self.tok_emb = nn.Embedding(self.vocab_size, d)
        self.pos_emb = (
            None
            if self.cfg["use_rope"]
            else nn.Embedding(self.context_length, d)
        )
        self.drop_emb = nn.Dropout(float(self.cfg.get("drop_rate", 0.0)))
        self.layers = nn.ModuleList(
            [VariantBlock(self.cfg, variant) for _ in range(int(self.cfg["n_layers"]))]
        )
        self.norm = RMSNorm(d)
        self.out_head = nn.Linear(d, self.vocab_size, bias=False)
        self.out_head.weight = self.tok_emb.weight

        self.apply(
            lambda m: self._init(m, float(self.cfg.get("init_std", 0.02)))
        )
        self.out_head.weight = self.tok_emb.weight

    @staticmethod
    def _init(module: nn.Module, std: float) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx: torch.Tensor):
        b, t = idx.shape
        if t > self.context_length:
            raise ValueError(f"sequence {t} > context {self.context_length}")
        pos = torch.arange(t, device=idx.device)
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(pos)
        x = self.drop_emb(x)
        aux_total = torch.zeros((), device=idx.device)
        for layer in self.layers:
            x, aux = layer(x)
            aux_total = aux_total + aux.float()
        return self.out_head(self.norm(x)), aux_total

    def parameter_breakdown(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        expert = sum(
            p.numel()
            for n, p in self.named_parameters()
            if ".moe." in n and ".router." not in n
        )
        return {
            "total": total,
            "expert": expert,
            "non_expert": total - expert,
        }
