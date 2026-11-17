"""Real OpenWebText training-curve test on CPU.

The one-step smoke test cannot show a trend: its loss is just the initial
uniform-ish loss. This test takes a few real OpenWebText batches, repeats
them, and runs enough optimizer steps for the loss to fall. It also checks
that ``tokens_seen`` is strictly increasing and that the plots/metrics are
written.

Run from the project root:
    .venv/bin/python code_test/test_train_curves_real.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model import GPT  # noqa: E402
from pretrain.data import OpenWebTextTokenStream  # noqa: E402
from pretrain.train import (  # noqa: E402
    Trainer,
    load_env_file,
    select_device,
)

# A small but trainable model with the real GPT-2 vocabulary (~1.6M tied
# parameters). This is still tiny enough for a CPU curve test.
MODEL_CFG = {
    "vocab_size": 50257,
    "context_length": 16,
    "emb_dim": 32,
    "n_heads": 4,
    "n_layers": 1,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "tie_weights": True,
}

NUM_BATCHES = 8
STEPS = 60


def main() -> None:
    load_env_file(ROOT / ".env")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        stream = OpenWebTextTokenStream(
            block_size=MODEL_CFG["context_length"],
            dataset_name="Skylion007/openwebtext",
            split="train",
            shuffle_buffer=0,
            seed=0,
            max_docs=20,
            min_chars=1,
        )
        iterator = stream.batches(batch_size=4, drop_last=True)
        fixed_batches = [next(iterator) for _ in range(NUM_BATCHES)]
        try:
            iterator.close()
        except Exception:
            pass

        def repeated_batches():
            for i in range(STEPS):
                yield fixed_batches[i % len(fixed_batches)]

        cfg = {
            "seed": 0,
            "device": "cpu",
            "num_gpus": 0,
            "amp": False,
            "batch_size": 4,
            "grad_accum_steps": 1,
            "max_steps": STEPS,
            "learning_rate": 1e-3,
            "min_lr": 1e-4,
            "warmup_steps": 0,
            "weight_decay": 0.0,
            "betas": [0.9, 0.95],
            "grad_clip": 1.0,
            "log_interval": 5,
            "checkpoint_interval_minutes": 60,
            "checkpoint_dir": str(tmp / "checkpoints"),
            "out_dir": str(tmp / "out"),
            "save_optimizer": False,
            "progress": True,
        }

        device, _ = select_device(cfg)
        model = GPT(MODEL_CFG)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["learning_rate"]
        )
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=device,
            model_cfg=MODEL_CFG,
            train_cfg=cfg,
            hf_cfg=None,
        )
        trainer.train(repeated_batches())

        history = trainer.history
        assert len(history) >= 10, history
        tokens = [point["tokens_seen"] for point in history]
        losses = [point["loss"] for point in history]
        perplexities = [point["perplexity"] for point in history]
        assert all(a < b for a, b in zip(tokens, tokens[1:])), tokens
        assert losses[-1] < losses[0], (losses[0], losses[-1])
        assert perplexities[-1] < perplexities[0], (
            perplexities[0],
            perplexities[-1],
        )

        for name in (
            "metrics.json",
            "token_seen_vs_loss.png",
            "token_seen_vs_perplexity.png",
            "training_curves.png",
        ):
            path = Path(cfg["out_dir"]) / name
            assert path.exists() and path.stat().st_size > 0, path

        print(
            "real-data training curve OK: "
            f"tokens {tokens[0]:,} -> {tokens[-1]:,}, "
            f"loss {losses[0]:.3f} -> {losses[-1]:.3f}, "
            f"ppl {perplexities[0]:,.0f} -> {perplexities[-1]:,.0f}"
        )
        print(f"plots: {Path(cfg['out_dir'])}")


if __name__ == "__main__":
    main()
