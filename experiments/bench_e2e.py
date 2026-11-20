"""End-to-end benchmark: real dense GPT vs MoE variants under trainer settings.

Mirrors configs/pretrain.yaml (dense) and configs/moe_train.yaml (MoE):
  dense  : batch_size=8, grad_accum=1, 2 GPUs
  MoE    : batch_size=4, grad_accum=2, 2 GPUs  (1024 tokens/opt-step both)

    python experiments/bench_e2e.py
    python experiments/bench_e2e.py --gpus 1 --micro-batch 4 --accum 1 --iters 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model import GPT  # noqa: E402
from pretrain.moe_train import load_moe_model_config  # noqa: E402

from moe_variants import VARIANTS  # noqa: E402
from variant_model import VariantMoELanguageModel  # noqa: E402


def bench_model(model, x, y, accum, iters, use_dp):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)

    def step():
        for _ in range(accum):
            opt.zero_grad(set_to_none=True)
            out = model(x)
            if isinstance(out, tuple):
                logits, aux = out
            else:
                logits, aux = out, None
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)), y.reshape(-1)
            )
            if aux is not None:
                loss = loss + aux.mean().to(loss.dtype)
            loss.backward()

    for _ in range(4):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    tokens = x.size(0) * x.size(1) * accum
    return dt, tokens / dt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", type=int, default=2)
    p.add_argument("--micro-batch", type=int, default=None)
    p.add_argument("--accum", type=int, default=None)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--skip-dense", action="store_true")
    args = p.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    gpus = min(args.gpus, torch.cuda.device_count()) if device.type == "cuda" else 0

    dense_cfg = yaml.safe_load((ROOT / "configs" / "model.yaml").read_text())
    moe_cfg = load_moe_model_config(ROOT / "configs" / "moe_config.yaml")
    vocab = dense_cfg["vocab_size"]

    def wrap(model):
        model = model.to(device=device, dtype=dtype)
        if gpus > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(gpus)))
        return model

    cases = []
    if not args.skip_dense:
        mb = args.micro_batch or 8
        acc = args.accum or 1
        cases.append(("dense", wrap(GPT(dense_cfg)), mb, acc))
    mb = args.micro_batch or 4
    acc = args.accum or 2
    for name in args.variants:
        cases.append((name, wrap(VariantMoELanguageModel(moe_cfg, name)), mb, acc))

    print(f"device={device} dtype={dtype} gpus={gpus} seq={args.seq} iters={args.iters}")
    print(f"micro_batch={mb} accum={acc} -> {mb*args.seq*acc} tokens/opt-step\n")
    print(f"{'model':14s} {'ms/opt-step':>12s} {'tok/s':>9s} {'params':>13s}")
    for name, model, batch, accum in cases:
        # DataParallel splits `batch` across GPUs; pass the full batch.
        x = torch.randint(0, vocab, (batch, args.seq), device=device)
        y = torch.randint(0, vocab, (batch, args.seq), device=device)
        dt, rate = bench_model(model, x, y, accum, args.iters, gpus > 1)
        nparams = sum(p.numel() for p in model.parameters())
        print(f"{name:14s} {dt*1000:12.1f} {rate:9.0f} {nparams:13,}")


if __name__ == "__main__":
    main()
