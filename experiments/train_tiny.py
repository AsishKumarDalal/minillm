"""Tiny training run for every MoE variant (isolated experiment, no effect on
the real model or checkpoints).

Same seed, same batches and same optimizer for all variants, so the only
difference is the expert implementation. Prints final loss and tok/s.

    python experiments/train_tiny.py
    python experiments/train_tiny.py --steps 300 --device cpu
    python experiments/train_tiny.py --real-data --max-docs 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pretrain.data import OpenWebTextTokenStream  # noqa: E402

from moe_variants import VARIANTS  # noqa: E402
from variant_model import VariantMoELanguageModel  # noqa: E402

TINY_CFG = {
    "vocab_size": 512,
    "context_length": 32,
    "emb_dim": 128,
    "n_layers": 2,
    "n_heads": 4,
    "n_kv_heads": 2,
    "d_ff": 256,
    "num_experts": 8,
    "top_k": 4,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "out_proj_bias": False,
    "tie_weights": True,
    "use_rope": True,
    "use_attention_sinks": False,
    "use_alibi_bias": False,
    "capacity_factor": 1.25,
    "aux_loss_coef": 0.01,
    "router_z_loss_coef": 0.001,
    "init_std": 0.02,
}


def make_batches(steps, batch_size, seq_len, vocab, real_data, seed=0):
    if real_data:
        stream = OpenWebTextTokenStream(
            block_size=seq_len,
            dataset_name="Skylion007/openwebtext",
            shuffle_buffer=0,
            seed=seed,
            max_docs=max(64, steps * batch_size // 4),
        )
        gen = stream.batches(batch_size=batch_size, drop_last=True)
        batches = []
        for i, (x, y) in enumerate(gen):
            if i >= steps:
                break
            batches.append((x.astype("int64"), y.astype("int64")))
        if len(batches) >= steps // 2:
            return batches
        print("real data unavailable; falling back to synthetic")

    # Deterministic next-token task the model can memorise.
    base = (torch.arange(seq_len) % (vocab - 1)).long()
    x = base.unsqueeze(0).expand(batch_size, -1).clone()
    y = (x + 1) % vocab
    return [(x.clone(), y.clone()) for _ in range(steps)]


def train_one(name, cfg, batches, device, dtype, lr, log_every):
    torch.manual_seed(0)
    model = VariantMoELanguageModel(cfg, name).to(device=device, dtype=dtype)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    losses = []
    start = time.perf_counter()
    for step, (x, y) in enumerate(batches, 1):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        if not torch.is_tensor(y):
            y = torch.from_numpy(y)
        x = x.long().to(device)
        y = y.long().to(device)
        logits, aux = model(x)
        ce = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), y.reshape(-1)
        )
        loss = ce + aux.to(ce.dtype)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if step % log_every == 0 or step == len(batches):
            print(f"  {name:12s} step {step:4d} loss {losses[-1]:.4f}")
    elapsed = time.perf_counter() - start
    ntok = sum(int(x.shape[0]) * int(x.shape[1]) for x, _ in batches)
    return losses, ntok / elapsed, elapsed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--real-data", action="store_true")
    p.add_argument("--max-docs", type=int, default=None)
    args = p.parse_args()

    torch.set_num_threads(1)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    cfg = dict(TINY_CFG, context_length=args.seq)
    if args.max_docs is not None:
        cfg["_max_docs"] = args.max_docs

    print(f"device={device} dtype={dtype} steps={args.steps} "
          f"batch={args.batch_size} seq={args.seq}")
    batches = make_batches(
        args.steps, args.batch_size, args.seq, cfg["vocab_size"], args.real_data
    )
    results = {}
    for name in args.variants:
        losses, rate, elapsed = train_one(
            name, cfg, batches, device, dtype, args.lr, args.log_every
        )
        results[name] = (losses, rate, elapsed)

    print("\n== summary ==")
    print(f"{'variant':12s} {'first loss':>11s} {'last loss':>10s} {'tok/s':>9s} {'seconds':>9s}")
    for name, (losses, rate, elapsed) in results.items():
        print(f"{name:12s} {losses[0]:11.4f} {losses[-1]:10.4f} {rate:9.0f} {elapsed:9.2f}")


if __name__ == "__main__":
    main()
