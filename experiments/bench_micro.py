"""Microbenchmark: MoE expert layer only.

Checks output equivalence with the real baseline on CPU/fp32 (deterministic),
then times forward and forward+backward on the chosen device/dtype.

    python experiments/bench_micro.py
    python experiments/bench_micro.py --device cpu --tokens 256
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moe_variants import VARIANTS, build_layer, copy_weights_from_baseline  # noqa: E402

BASE_CFG = {
    "vocab_size": 512,
    "context_length": 128,
    "emb_dim": 768,
    "n_layers": 6,
    "n_heads": 12,
    "n_kv_heads": 4,
    "d_ff": 1536,
    "num_experts": 8,
    "top_k": 4,
    "drop_rate": 0.0,
    "capacity_factor": None,  # exact equivalence with the baseline
    "aux_loss_coef": 0.01,
    "router_z_loss_coef": 0.001,
    "init_std": 0.02,
}


def timeit(fn, iters: int) -> float:
    for _ in range(5):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def check_equivalence() -> None:
    print("== equivalence vs baseline (CPU fp32, capacity_factor=None) ==")
    torch.manual_seed(0)
    cfg = dict(BASE_CFG)
    baseline = build_layer("loop", cfg).eval()
    x = torch.randn(1, 64, cfg["emb_dim"])
    with torch.no_grad():
        ref, ref_aux = baseline(x)
        for name in VARIANTS:
            layer = baseline if name == "loop" else build_layer(name, cfg).eval()
            if name != "loop":
                copy_weights_from_baseline(layer, baseline)
            out, aux = layer(x)
            err = float((out - ref).abs().max())
            aux_err = float((aux - ref_aux).abs().max())
            print(f"  {name:12s} out_max_err={err:.6f}  aux_max_err={aux_err:.6f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tokens", type=int, nargs="+", default=[128, 256, 512, 1024])
    p.add_argument("--iters", type=int, default=30)
    args = p.parse_args()

    check_equivalence()

    torch.manual_seed(0)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    cfg = dict(BASE_CFG)

    layers = {
        name: build_layer(name, cfg).to(device=device, dtype=dtype)
        for name in VARIANTS
    }

    for n in args.tokens:
        print(f"\n== tokens={n}  dtype={dtype}  device={device} ==")
        x = torch.randn(1, n, cfg["emb_dim"], device=device, dtype=dtype, requires_grad=True)
        print(f"  {'variant':12s} {'fwd ms':>9s} {'fwd+bwd ms':>11s}")
        for name, layer in layers.items():
            layer.train()

            def forward():
                return layer(x)

            def forward_backward():
                layer.zero_grad(set_to_none=True)
                out, aux = layer(x)
                (out.sum() + aux).backward()

            tf = timeit(forward, args.iters)
            tb = timeit(forward_backward, args.iters)
            print(f"  {name:12s} {tf:9.2f} {tb:11.2f}")
            layer.eval()


if __name__ == "__main__":
    main()
