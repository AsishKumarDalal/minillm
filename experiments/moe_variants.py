"""Experimental MoE layer implementations (does not touch the real model).

Four variants with identical maths, inputs/outputs and auxiliary loss:

  loop        - the current implementation from model.moe_parts.MoELayer
                (baseline, imported directly, not copied)
  fused_loop  - HF Mixtral / Qwen3 style storage: experts held as fused 3D
                weight tensors, loop over experts but skip empty ones
  grouped_mm  - static-shape sort + torch._grouped_mm (no data-dependent
                shapes, no device synchronisation, capacity via a mask)
  dense_bmm   - compute every expert for every token as ONE batched matmul
                (8x the routed FLOPs at E=8, but zero dispatch cost)

All use the same fc1 -> GELU -> fc2 expert so outputs are directly comparable.
Every variant returns (output, aux_loss), matching MoELayer.forward.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.moe_parts import (  # noqa: E402
    MoELayer,
    TopKRouter,
    load_balance_loss,
    router_z_loss,
)

HAS_GROUPED_MM = hasattr(torch, "_grouped_mm")


class _VariantBase(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.num_experts = int(cfg["num_experts"])
        self.top_k = int(cfg["top_k"])
        self.capacity_factor = cfg.get("capacity_factor")
        self.aux_loss_coef = float(cfg.get("aux_loss_coef", 0.01))
        self.z_coef = float(cfg.get("router_z_loss_coef", 0.0))
        self.router = TopKRouter(cfg)
        self.std = float(cfg.get("init_std", 0.02))

    def _aux(self, probs, expert_idx, logits) -> torch.Tensor:
        loss = self.aux_loss_coef * load_balance_loss(
            probs, expert_idx, self.num_experts
        )
        if self.z_coef > 0:
            loss = loss + self.z_coef * router_z_loss(logits)
        return loss

    def _capacity(self, num_tokens: int) -> int:
        if self.capacity_factor is None or not self.training:
            return num_tokens
        return max(
            int(
                max(0.0, float(self.capacity_factor))
                * self.top_k
                * num_tokens
                / self.num_experts
            ),
            1,
        )


class FusedLoopMoELayer(_VariantBase):
    """Fused 3D expert weights + loop over experts, skipping empty ones."""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        d, h, e = int(cfg["emb_dim"]), int(cfg["d_ff"]), self.num_experts
        self.fc1_w = nn.Parameter(torch.empty(e, h, d))
        self.fc2_w = nn.Parameter(torch.empty(e, d, h))
        with torch.no_grad():
            nn.init.normal_(self.fc1_w, std=self.std)
            nn.init.normal_(self.fc2_w, std=self.std)

    def forward(self, x: torch.Tensor):
        b, t, d = x.shape
        flat = x.reshape(-1, d)
        w, idx, probs, logits = self.router(flat)
        aux = self._aux(probs, idx, logits)
        cap = self._capacity(flat.size(0))
        out = torch.zeros_like(flat)

        with torch.no_grad():
            hit = torch.unique(idx.reshape(-1))

        for expert_id in hit.tolist():
            token_idx, slot_idx = torch.where(idx == expert_id)
            weights = w[token_idx, slot_idx]
            if token_idx.numel() > cap:
                keep = torch.topk(weights, cap, largest=True).indices
                token_idx, slot_idx, weights = (
                    token_idx[keep], slot_idx[keep], weights[keep]
                )
            w1 = self.fc1_w[expert_id].to(flat.dtype)
            w2 = self.fc2_w[expert_id].to(flat.dtype)
            hidden = F.gelu(F.linear(flat[token_idx], w1))
            y = F.linear(hidden, w2)
            out.index_add_(0, token_idx, y * weights.to(y.dtype).unsqueeze(-1))

        return out.view(b, t, d), aux


class GroupedMmMoELayer(_VariantBase):
    """Static-shape routing + torch._grouped_mm. CUDA / bf16 recommended."""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        d, h, e = int(cfg["emb_dim"]), int(cfg["d_ff"]), self.num_experts
        self.fc1_w = nn.Parameter(torch.empty(e, d, h))
        self.fc2_w = nn.Parameter(torch.empty(e, h, d))
        with torch.no_grad():
            nn.init.normal_(self.fc1_w, std=self.std)
            nn.init.normal_(self.fc2_w, std=self.std)

    def _dense_fallback(self, flat, flat_idx, flat_w, order, token_of):
        """Same maths via bmm; return per-assignment outputs in sorted order."""
        w1 = self.fc1_w.to(flat.dtype)
        w2 = self.fc2_w.to(flat.dtype)
        hidden = torch.bmm(
            flat.unsqueeze(0).expand(self.num_experts, -1, -1), w1
        )
        y = torch.bmm(F.gelu(hidden), w2)  # (E, n, d)
        expert_of = flat_idx[order]
        return y[expert_of, token_of]  # (n*top_k, d)

    def forward(self, x: torch.Tensor):
        b, t, d = x.shape
        flat = x.reshape(-1, d)
        n = flat.size(0)
        w, idx, probs, logits = self.router(flat)
        aux = self._aux(probs, idx, logits)
        cap = self._capacity(n)

        flat_idx = idx.reshape(-1)
        flat_w = w.reshape(-1).to(flat.dtype)

        # Sort by gate weight DESC, then stably by expert. Groups tokens by
        # expert with the highest-gate ones first, so capacity is a static
        # mask (no topk, no dynamic shapes, no device synchronisation).
        by_weight = torch.argsort(-flat_w, stable=True)
        by_expert = torch.argsort(flat_idx[by_weight], stable=True)
        order = by_weight[by_expert]
        token_of = order // self.top_k

        counts = torch.bincount(flat_idx, minlength=self.num_experts)
        starts = torch.cumsum(counts, 0) - counts
        rank = torch.arange(n * self.top_k, device=flat.device) - starts[flat_idx[order]]
        keep = (rank < cap).to(flat.dtype)
        flat_w = flat_w[order] * keep

        if HAS_GROUPED_MM and flat.is_cuda:
            try:
                offs = torch.cumsum(counts, 0, dtype=torch.int32)
                xs = flat.index_select(0, token_of)
                hidden = torch._grouped_mm(
                    xs, self.fc1_w.to(flat.dtype), offs=offs
                )
                y = torch._grouped_mm(
                    F.gelu(hidden), self.fc2_w.to(flat.dtype), offs=offs
                )
            except (RuntimeError, AssertionError):
                y = self._dense_fallback(flat, flat_idx, flat_w, order, token_of)
        else:
            y = self._dense_fallback(flat, flat_idx, flat_w, order, token_of)

        out = torch.zeros_like(flat).index_add_(
            0, token_of, y * flat_w.unsqueeze(-1)
        )
        return out.view(b, t, d), aux


class DenseBmmMoELayer(_VariantBase):
    """One batched matmul over ALL experts; routing applied as a sparse mask."""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        d, h, e = int(cfg["emb_dim"]), int(cfg["d_ff"]), self.num_experts
        self.fc1_w = nn.Parameter(torch.empty(e, d, h))
        self.fc2_w = nn.Parameter(torch.empty(e, h, d))
        with torch.no_grad():
            nn.init.normal_(self.fc1_w, std=self.std)
            nn.init.normal_(self.fc2_w, std=self.std)

    def forward(self, x: torch.Tensor):
        b, t, d = x.shape
        flat = x.reshape(-1, d)
        n = flat.size(0)
        w, idx, probs, logits = self.router(flat)
        aux = self._aux(probs, idx, logits)

        gate_w = w.reshape(n, self.top_k).to(flat.dtype)
        mask = torch.zeros(n, self.num_experts, device=flat.device, dtype=flat.dtype)
        mask.scatter_(1, idx.reshape(n, self.top_k), gate_w)

        w1 = self.fc1_w.to(flat.dtype)
        w2 = self.fc2_w.to(flat.dtype)
        hidden = torch.bmm(
            flat.unsqueeze(0).expand(self.num_experts, -1, -1), w1
        )
        y = torch.bmm(F.gelu(hidden), w2)
        out = (y * mask.permute(1, 0).unsqueeze(-1)).sum(0)
        return out.view(b, t, d), aux


VARIANTS = {
    "loop": MoELayer,
    "fused_loop": FusedLoopMoELayer,
    "grouped_mm": GroupedMmMoELayer,
    "dense_bmm": DenseBmmMoELayer,
}


def copy_weights_from_baseline(variant_layer, baseline_layer) -> None:
    """Copy baseline expert/router weights into a variant for equivalence tests."""
    def put(target, source):
        if tuple(target.shape) == tuple(source.shape):
            target.copy_(source)
        else:
            target.copy_(source.t())

    with torch.no_grad():
        variant_layer.router.load_state_dict(baseline_layer.router.state_dict())
        for i, expert in enumerate(baseline_layer.experts):
            put(variant_layer.fc1_w[i], expert.fc1.weight)
            put(variant_layer.fc2_w[i], expert.fc2.weight)


def build_layer(name: str, cfg: dict) -> nn.Module:
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; choose from {list(VARIANTS)}")
    return VARIANTS[name](cfg)
