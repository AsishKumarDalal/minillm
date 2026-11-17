"""Real OpenWebText -> tiny GPT training smoke test on CPU.

This uses the real streaming data pipeline and the real ``run_training``
entry point, but a ~50k-parameter model with vocab=50257 (so GPT-2 token ids
are valid). It reads only a few documents, runs one optimizer step, saves a
local checkpoint, and never contacts HuggingFace Hub.

Run from the project root:

    .venv/bin/python code_test/smoke_train_openwebtext_tiny.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pretrain.train import run_training  # noqa: E402

# emb_dim=1 with the GPT-2 vocab is ~50k tied parameters; still a valid GPT.
MODEL_CFG = {
    "vocab_size": 50257,
    "context_length": 8,
    "emb_dim": 1,
    "n_heads": 1,
    "n_layers": 1,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "tie_weights": True,
}


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        train_cfg = {
            "seed": 0,
            "device": "cpu",
            "num_gpus": 0,
            "amp": False,
            "batch_size": 1,
            "grad_accum_steps": 1,
            "max_steps": 1,
            "learning_rate": 1e-3,
            "min_lr": 1e-4,
            "warmup_steps": 0,
            "weight_decay": 0.0,
            "betas": [0.9, 0.95],
            "grad_clip": 1.0,
            "log_interval": 1,
            "checkpoint_interval_minutes": 60,
            "checkpoint_dir": tmpdir,
            "out_dir": str(Path(tmpdir) / "out"),
            "save_optimizer": False,
            "progress": True,
            "data": {
                "dataset_name": "Skylion007/openwebtext",
                "split": "train",
                "block_size": 8,
                "shuffle_buffer": 0,
                "max_docs": 3,
                "min_chars": 1,
            },
            "hf": {
                "repo_id": "",
                "token_env": "HF_TOKEN",
                "filename": "minillm_latest.pt",
                "private": True,
            },
        }

        run_training(train_cfg, MODEL_CFG, no_hf=True)
        checkpoint = Path(tmpdir) / "minillm_latest.pt"
        assert checkpoint.exists(), checkpoint
        print(
            f"tiny real-data training smoke test passed; "
            f"checkpoint={checkpoint.stat().st_size / 1024:.1f} KB "
            f"(1 step; see test_train_curves_real.py for the loss/token trend)"
        )


if __name__ == "__main__":
    main()
