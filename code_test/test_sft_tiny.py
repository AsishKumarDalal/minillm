"""Tiny CPU tests for the SFT pipeline (no Dolly download)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sft.tuning as tuning  # noqa: E402
from model.model import GPT  # noqa: E402
from sft.sft_data import SFTData  # noqa: E402

MODEL_CFG = {
    "vocab_size": 50257,
    "context_length": 32,
    "emb_dim": 16,
    "n_heads": 2,
    "n_layers": 1,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "tie_weights": True,
}


def local_documents():
    short = {"instruction": "Hi", "context": "there", "response": "ok"}
    no_context = {
        "instruction": "Hi",
        "context": "word " * 100,
        "response": "ok",
    }
    too_long = {
        "instruction": "Hi",
        "context": "x",
        "response": "word " * 100,
    }
    return [short, no_context, too_long] + [short] * 10


def test_sft_data_triage() -> None:
    data = SFTData(block_size=32, shuffle_buffer=0, documents=local_documents())
    examples = list(data.examples(epochs=1))
    triage = [example["triage"] for example in examples]
    assert "full" in triage
    assert "answer_only" in triage
    assert data.stats["dropped"] >= 1
    for example in examples:
        assert len(example["input_ids"]) <= 32
        assert -100 in example["labels"]
        assert example["labels"][-1] == data.eot_token_id
    print("SFTData runtime triage OK:", data.stats)


def test_sft_pipeline_end_to_end(tmpdir: str) -> None:
    tmp = Path(tmpdir)
    model_cfg_path = tmp / "model.yaml"
    model_cfg_path.write_text(yaml.safe_dump(MODEL_CFG))

    cfg = tuning.load_yaml(tuning.DEFAULT_SFT_CONFIG)
    cfg["model_config"] = str(model_cfg_path)
    cfg["seed"] = 0
    cfg["device"] = "cpu"
    cfg["num_gpus"] = 0
    cfg["batch_size"] = 2
    cfg["steps"] = 3
    cfg["epochs"] = 1
    cfg["learning_rate"] = 1e-3
    cfg["amp"] = False
    cfg["progress"] = False
    cfg["log_first_steps"] = 1
    cfg["log_interval"] = 1
    cfg["sample_interval"] = 2
    cfg["sample_prompt"] = "A"
    cfg["sample_max_new_tokens"] = 4
    cfg["data"] = {"block_size": 32, "shuffle_buffer": 0, "tokenizer_name": "gpt2"}
    cfg["checkpoint_dir"] = str(tmp / "sft_checkpoints")
    cfg["out_dir"] = str(tmp / "sft_outs")
    cfg["checkpoint_interval_minutes"] = 60
    cfg["hf"] = {"enabled": False}

    def local_factory(**kwargs):
        return SFTData(
            block_size=32,
            shuffle_buffer=0,
            documents=local_documents(),
            tokenizer_name="gpt2",
        )

    with patch.object(tuning, "SFTData", side_effect=local_factory):
        tuning.run_sft(cfg, full_new=True)

    checkpoint = tmp / "sft_checkpoints" / "sft_model.pt"
    metrics = tmp / "sft_outs" / "metrics.json"
    assert checkpoint.exists(), checkpoint
    assert metrics.exists(), metrics
    for name in (
        "token_seen_vs_loss.png",
        "token_seen_vs_perplexity.png",
        "step_vs_loss.png",
    ):
        plot = tmp / "sft_outs" / name
        assert plot.exists() and plot.stat().st_size > 0, plot

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["step"] == 3
    assert state["model_config"]["vocab_size"] == 50257
    import json

    history = json.loads(metrics.read_text())
    assert len(history) >= 2
    assert all(entry["loss"] > 0 for entry in history)
    print("SFT pipeline end-to-end OK")


def test_sft_resume_and_cli() -> None:
    checkpoint_dir = Path(tempfile.mkdtemp())
    model = GPT(MODEL_CFG)
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": MODEL_CFG,
            "step": 9,
            "tokens_seen": 123,
            "epoch": 1,
        },
        checkpoint_dir / "sft_model.pt",
    )
    cfg = {"hf": {"filename": "sft_model.pt", "repo_id": ""}}
    resumed = tuning.load_sft_state(
        cfg, MODEL_CFG, torch.device("cpu"), checkpoint_dir
    )
    assert resumed is not None
    assert resumed[1]["step"] == 9

    args = tuning.build_arg_parser().parse_args(
        ["--epochs", "2", "--steps", "50", "--upload", "--fullnew"]
    )
    overrides = {}
    tuning.apply_cli_overrides(overrides, args)
    assert overrides["epochs"] == 2
    assert overrides["steps"] == 50
    assert overrides["hf"]["enabled"] is True
    print("SFT resume and CLI overrides OK")


def main() -> None:
    test_sft_data_triage()
    with tempfile.TemporaryDirectory() as tmpdir:
        test_sft_pipeline_end_to_end(tmpdir)
    test_sft_resume_and_cli()
    print("all tiny SFT tests passed")


if __name__ == "__main__":
    main()
