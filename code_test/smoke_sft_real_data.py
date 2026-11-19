"""Smoke test: real streamed Dolly data, tiny random student, 2 steps."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sft.tuning as tuning  # noqa: E402

MODEL_CFG = {
    "vocab_size": 50257,
    "context_length": 128,
    "emb_dim": 64,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "tie_weights": True,
}


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        model_cfg = tmpdir / "model.yaml"
        model_cfg.write_text(yaml.safe_dump(MODEL_CFG))
        cfg = tuning.load_yaml(tuning.DEFAULT_SFT_CONFIG)
        cfg["model_config"] = str(model_cfg)
        cfg["seed"] = 0
        cfg["device"] = "cpu"
        cfg["num_gpus"] = 0
        cfg["batch_size"] = 2
        cfg["steps"] = 2
        cfg["epochs"] = 1
        cfg["learning_rate"] = 1e-3
        cfg["amp"] = False
        cfg["progress"] = False
        cfg["sample_interval"] = 0
        cfg["data"] = {
            "dataset_name": "databricks/databricks-dolly-15k",
            "split": "train",
            "block_size": 128,
            "shuffle_buffer": 0,
            "max_docs": 12,
            "min_chars": 1,
            "tokenizer_name": "gpt2",
        }
        cfg["checkpoint_dir"] = str(tmpdir / "sft_checkpoints")
        cfg["out_dir"] = str(tmpdir / "sft_outs")
        cfg["hf"] = {"enabled": False}

        tuning.run_sft(cfg, full_new=True)

        checkpoint = tmpdir / "sft_checkpoints" / "sft_model.pt"
        assert checkpoint.exists()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert state["step"] == 2
        print("real-data SFT smoke OK: step", state["step"], "stats", state["data_stats"])


if __name__ == "__main__":
    main()
