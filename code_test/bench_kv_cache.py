"""KV-cache correctness + speed benchmark on a small CPU model.

Trains nothing: builds a fresh random ~92k-parameter GPT, generates the same
tokens twice (``use_cache=False`` and ``use_cache=True``) and checks that

1. the cached and uncached outputs are bit-identical, and
2. the cached path is actually faster in wall-clock time.

Run from the project root:

    .venv/bin/python code_test/bench_kv_cache.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model import GPT  # noqa: E402

# ~92k parameters (well under the 0.1M budget): enough context to make the
# per-step O(T) recompute visible on CPU, small enough to run in seconds.
SMALL_CFG = {
    "vocab_size": 512,
    "context_length": 256,
    "emb_dim": 48,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "tie_weights": True,
}

PROMPT_LEN = 8
NEW_TOKENS = 128
BATCH = 2
REPEATS = 5


def make_model(seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    model = GPT(SMALL_CFG)
    model.eval()
    return model


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def timed_generate(model: GPT, idx: torch.Tensor, use_cache: bool) -> tuple[torch.Tensor, float]:
    """Run greedy generation once and return (output, seconds)."""
    start = time.perf_counter()
    out = model.generate(
        idx,
        max_new_tokens=NEW_TOKENS,
        temperature=0.0,
        use_cache=use_cache,
    )
    return out, time.perf_counter() - start


def bench(
    model: GPT,
    idx: torch.Tensor,
    use_cache: bool,
    warmup: int = 2,
    repeats: int = REPEATS,
) -> tuple[torch.Tensor, float]:
    """Best-of-N wall time after warmup, plus the last generated output."""
    out = None
    for _ in range(warmup):
        out, _ = timed_generate(model, idx, use_cache)

    times = []
    for _ in range(repeats):
        out, elapsed = timed_generate(model, idx, use_cache)
        times.append(elapsed)
    return out, min(times)


def check_equivalence(label: str, idx: torch.Tensor) -> None:
    model = make_model(seed=1)
    plain, _ = timed_generate(model, idx, use_cache=False)
    cached, _ = timed_generate(model, idx, use_cache=True)

    if not torch.equal(cached, plain):
        diff = (cached != plain).sum().item()
        raise SystemExit(
            f"{label}: cached and uncached generation differ "
            f"({diff} of {plain.numel()} tokens)"
        )
    assert cached[:, : idx.size(1)].tolist() == idx.tolist()
    print(f"{label}: cached == uncached ({cached.size(0)}x{cached.size(1)} tokens) OK")


def main() -> None:
    torch.set_num_threads(1)

    model = make_model(seed=0)
    n_params = count_params(model)
    print(
        f"model: {n_params:,} params | context={SMALL_CFG['context_length']} "
        f"| emb={SMALL_CFG['emb_dim']} | layers={SMALL_CFG['n_layers']}"
    )

    idx = torch.randint(0, SMALL_CFG["vocab_size"], (BATCH, PROMPT_LEN))
    torch.manual_seed(2)

    # 1. Correctness: greedy decoding must be identical both ways.
    check_equivalence(f"batch={BATCH} prompt={PROMPT_LEN}", idx)
    check_equivalence(
        f"batch=1 prompt={PROMPT_LEN}",
        idx[:1],
    )

    # 2. Speed: cached decode must beat full recompute.
    plain_out, plain_time = bench(model, idx, use_cache=False)
    cached_out, cached_time = bench(model, idx, use_cache=True)
    assert torch.equal(cached_out, plain_out), "benchmark outputs diverged"

    speedup = plain_time / cached_time
    tok_per_sec_plain = BATCH * NEW_TOKENS / plain_time
    tok_per_sec_cached = BATCH * NEW_TOKENS / cached_time

    print()
    print(f"generated {NEW_TOKENS} tokens x batch {BATCH}, best of {REPEATS}:")
    print(f"  no cache : {plain_time * 1e3:8.1f} ms  ({tok_per_sec_plain:7.1f} tok/s)")
    print(f"  kv cache : {cached_time * 1e3:8.1f} ms  ({tok_per_sec_cached:7.1f} tok/s)")
    print(f"  speedup  : {speedup:8.2f}x")

    if speedup <= 1.0:
        raise SystemExit(
            f"KV cache did not speed up generation ({speedup:.2f}x)"
        )

    # 3. Longer context should widen the gap (the recompute cost grows with T).
    long_prompt = SMALL_CFG["context_length"] - 16
    idx_long = torch.randint(0, SMALL_CFG["vocab_size"], (1, long_prompt))
    plain_long, plain_long_time = bench(model, idx_long, use_cache=False, repeats=3)
    cached_long, cached_long_time = bench(model, idx_long, use_cache=True, repeats=3)
    assert torch.equal(cached_long, plain_long), "long-prompt outputs diverged"
    print(
        f"  prompt={long_prompt}, {NEW_TOKENS} new: no cache "
        f"{plain_long_time * 1e3:.1f} ms vs kv cache "
        f"{cached_long_time * 1e3:.1f} ms -> "
        f"{plain_long_time / cached_long_time:.2f}x"
    )

    print("\nall KV-cache checks passed")


if __name__ == "__main__":
    main()
