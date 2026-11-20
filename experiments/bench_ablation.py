"""Ablate non-expert overhead in the MoE model (attention sinks, RoPE, aux
loss, capacity dropping, DataParallel tuple gather).

    python experiments/bench_ablation.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pretrain.moe_train import load_moe_model_config  # noqa: E402

from variant_model import VariantMoELanguageModel  # noqa: E402


def bench(model, x, y, accum, iters):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)

    def step():
        for _ in range(accum):
            opt.zero_grad(set_to_none=True)
            logits, aux = model(x)
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)), y.reshape(-1)
            )
            (loss + aux.mean().to(loss.dtype)).backward()

    for _ in range(4):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base = load_moe_model_config(ROOT / "configs" / "moe_config.yaml")
    gpus = min(2, torch.cuda.device_count()) if device.type == "cuda" else 0
    iters, seq, batch, accum = 10, 128, 4, 2

    cases = {
        "baseline (top_k=4, sinks, rope, aux, cap)": {},
        "no attention sinks": {"use_attention_sinks": False},
        "no RoPE (learned pos)": {"use_rope": False},
        "no aux losses": {"aux_loss_coef": 0.0, "router_z_loss_coef": 0.0},
        "no capacity drop": {"capacity_factor": None},
        "no sinks+rope+aux+cap": {
            "use_attention_sinks": False,
            "use_rope": False,
            "aux_loss_coef": 0.0,
            "router_z_loss_coef": 0.0,
            "capacity_factor": None,
        },
    }
    print(f"device={device} gpus={gpus} micro_batch={batch} accum={accum} seq={seq}\n")
    print(f"{'variant':44s} {'ms/opt-step':>12s} {'tok/s':>9s}")
    for name, overrides in cases.items():
        cfg = dict(base, **overrides)
        model = VariantMoELanguageModel(cfg, "loop").to(device=device, dtype=dtype)
        if gpus > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(gpus)))
        x = torch.randint(0, cfg["vocab_size"], (batch, seq), device=device)
        y = torch.randint(0, cfg["vocab_size"], (batch, seq), device=device)
        dt = bench(model, x, y, accum, iters)
        print(f"{name:44s} {dt:12.1f} {batch*seq*accum/(dt/1000):9.0f}")


if __name__ == "__main__":
    main()
