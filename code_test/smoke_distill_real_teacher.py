"""Real-teacher distillation smoke test (small HF model, ~2 steps on CPU).

This downloads ``hf-internal-testing/tiny-random-gpt2`` (a few MB), builds a
tiny local student checkpoint, and runs the real ``run_distillation`` path on
a couple of OpenWebText documents. It never touches the production student
checkpoint or HuggingFace uploads.

Run from the project root:
    python3 code_test/smoke_distill_real_teacher.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import distill.distill as distill_mod  # noqa: E402
from distill.distill import DEFAULT_DISTILL_CONFIG, load_yaml, run_distillation  # noqa: E402
from model.model import GPT  # noqa: E402
from pretrain.data import OpenWebTextTokenStream  # noqa: E402


def _local_stream(**kwargs):
    """Avoid OpenWebText network calls; exercise distillation with local text."""
    return OpenWebTextTokenStream(
        block_size=int(kwargs["block_size"]),
        documents=["The quick brown fox jumps over the lazy dog. "] * 200,
        shuffle_buffer=0,
        seed=int(kwargs.get("seed", 0)),
        min_chars=1,
    )

STUDENT_CFG = {
    "vocab_size": 50257,
    "context_length": 8,
    "emb_dim": 32,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "tie_weights": True,
}


def main() -> None:
    distill_mod.OpenWebTextTokenStream = _local_stream
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        model_cfg_path = tmpdir / "tiny_student.yaml"
        model_cfg_path.write_text(yaml.safe_dump(STUDENT_CFG))

        checkpoint_dir = tmpdir / "distill_checkpoints"
        checkpoint_dir.mkdir()
        student_path = checkpoint_dir / "student.pt"
        torch.save(
            {
                "format_version": 1,
                "model": GPT(STUDENT_CFG).state_dict(),
                "model_config": STUDENT_CFG,
                "step": 0,
                "tokens_seen": 0,
            },
            student_path,
        )

        cfg = load_yaml(DEFAULT_DISTILL_CONFIG)
        cfg["model_config"] = str(model_cfg_path)
        cfg["seed"] = 0
        cfg["device"] = "cpu"
        cfg["teacher"] = {
            "name": "sshleifer/tiny-gpt2",
            "device": "cpu",
            "dtype": "fp32",
        }
        cfg["student"] = {
            "source": "local",
            "local_path": str(student_path),
        }
        cfg["alpha"] = 0.4
        cfg["temperature"] = 1.0
        cfg["amp"] = False
        cfg["batch_size"] = 2
        cfg["max_steps"] = 2
        cfg["log_first_steps"] = 2
        cfg["log_interval"] = 1
        cfg["checkpoint_dir"] = str(checkpoint_dir)
        cfg["out_dir"] = str(tmpdir / "distilled_outs")
        cfg["checkpoint_interval_minutes"] = 60
        cfg["progress"] = False
        cfg["hf"] = {"enabled": False}
        cfg["data"] = {
            "dataset_name": "Skylion007/openwebtext",
            "split": "train",
            "block_size": 8,
            "shuffle_buffer": 0,
            "max_docs": 3,
            "min_chars": 1,
        }

        run_distillation(cfg)

        out = checkpoint_dir / "distilled_model.pt"
        metrics = tmpdir / "distilled_outs" / "metrics.json"
        assert out.exists(), out
        assert metrics.exists(), metrics
        state = torch.load(out, map_location="cpu", weights_only=True)
        assert state["step"] == 2
        print("real-teacher distillation smoke OK: step", state["step"])


if __name__ == "__main__":
    main()
