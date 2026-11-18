"""Tiny CPU test for pretrain/train.py.

This does not load the 100M model, does not touch OpenWebText, and does
not upload to HuggingFace. It trains a ~4.5k-parameter GPT for a few steps
on synthetic token batches, then tests local checkpoint save/resume.

Run from the project root:
    .venv/bin/python code_test/test_train_tiny.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model import GPT  # noqa: E402
from pretrain.train import (  # noqa: E402
    Trainer,
    get_hf_settings,
    load_env_file,
    lr_for_step,
    model_config_mismatches,
    select_device,
    upload_checkpoint,
    upload_outputs,
)

MODEL_CFG = {
    "vocab_size": 64,
    "context_length": 8,
    "emb_dim": 16,
    "n_heads": 2,
    "n_layers": 1,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "tie_weights": True,
}

TRAIN_CFG = {
    "seed": 0,
    "device": "cpu",
    "num_gpus": 0,
    "amp": False,
    "batch_size": 2,
    "grad_accum_steps": 1,
    "max_steps": 3,
    "learning_rate": 1e-3,
    "min_lr": 1e-4,
    "warmup_steps": 1,
    "weight_decay": 0.01,
    "betas": [0.9, 0.95],
    "grad_clip": 1.0,
    "log_first_steps": 10,
    "log_interval": 1,
    "sample_interval": 0,
    "checkpoint_interval_minutes": 60,
    "checkpoint_dir": "",
    "save_optimizer": True,
    "progress": False,
    "data": {"block_size": 8},
    "hf": {
        "repo_id": "",
        "token_env": "HF_TOKEN",
        "filename": "minillm_latest.pt",
        "private": True,
    },
}


def make_batches(num_steps: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    batch, block, vocab = 2, 8, MODEL_CFG["vocab_size"]

    def _gen():
        for _ in range(num_steps):
            x = rng.integers(0, vocab, size=(batch, block), dtype=np.int32)
            y = rng.integers(0, vocab, size=(batch, block), dtype=np.int32)
            yield x, y

    return _gen()


def make_fixed_batches(
    num_steps: int,
    batch_size: int = 4,
    block_size: int = 8,
    vocab_size: int = 32,
):
    """A tiny deterministic next-token task the model can memorize."""
    base = np.arange(block_size, dtype=np.int32)
    x = np.tile(base, (batch_size, 1))
    y = (x + 1) % vocab_size

    def _gen():
        for _ in range(num_steps):
            yield x.copy(), y.copy()

    return _gen()


def build_trainer(checkpoint_dir: str, max_steps: int = 3) -> Trainer:
    cfg = dict(TRAIN_CFG)
    cfg["checkpoint_dir"] = checkpoint_dir
    cfg["out_dir"] = str(Path(checkpoint_dir) / "out")
    cfg["max_steps"] = max_steps

    device, num_gpus = select_device(cfg)
    assert device.type == "cpu" and num_gpus == 0

    model = GPT(MODEL_CFG)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        betas=tuple(cfg["betas"]),
        weight_decay=cfg["weight_decay"],
    )
    return Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        model_cfg=MODEL_CFG,
        train_cfg=cfg,
        hf_cfg=None,
    )


def test_tiny_train_and_resume(tmpdir: str) -> None:
    torch.set_num_threads(1)

    model = GPT(MODEL_CFG)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 100_000, n_params
    print(f"tiny model: {n_params:,} parameters")

    # Regression guard for GPT-2 init: a fresh model must start near
    # ln(vocab_size), not with explosive logits/loss.
    with torch.no_grad():
        x0 = torch.randint(0, MODEL_CFG["vocab_size"], (1, 8))
        y0 = torch.randint(0, MODEL_CFG["vocab_size"], (1, 8))
        _, initial_loss = model(x0, y0)
    assert initial_loss.item() < 20.0, initial_loss.item()

    trainer = build_trainer(tmpdir, max_steps=3)
    assert not trainer.resume(), "should start fresh with an empty checkpoint dir"
    trainer.train(make_batches(num_steps=3, seed=0))

    assert trainer.step == 3, trainer.step
    assert trainer.tokens_seen == 3 * 2 * 8, trainer.tokens_seen
    assert trainer.checkpoint_path.exists(), trainer.checkpoint_path
    assert len(trainer.history) == 3, trainer.history
    hist_tokens = [point["tokens_seen"] for point in trainer.history]
    assert hist_tokens == sorted(hist_tokens), hist_tokens
    assert len(set(hist_tokens)) == len(hist_tokens), hist_tokens
    assert hist_tokens[-1] == trainer.tokens_seen
    assert (trainer.out_dir / "metrics.json").exists()
    assert (trainer.out_dir / "token_seen_vs_loss.png").exists()
    assert (trainer.out_dir / "token_seen_vs_perplexity.png").exists()
    assert (trainer.out_dir / "training_curves.png").exists()
    print(
        f"trained 3 steps, {trainer.tokens_seen} tokens, "
        f"checkpoint={trainer.checkpoint_path.name}, "
        f"plots={trainer.out_dir.name}"
    )

    trainer2 = build_trainer(tmpdir, max_steps=4)
    assert trainer2.resume(), "local checkpoint should resume"
    assert trainer2.step == 3, trainer2.step
    assert trainer2.tokens_seen == 48, trainer2.tokens_seen
    assert len(trainer2.history) == 3, trainer2.history

    trainer2.train(make_batches(num_steps=1, seed=99))
    assert trainer2.step == 4, trainer2.step
    assert trainer2.tokens_seen == 64, trainer2.tokens_seen
    assert len(trainer2.history) == 4, trainer2.history
    print("resumed from local checkpoint, restored metrics, and trained 1 more step")


def test_grad_accumulation(tmpdir: str) -> None:
    cfg = dict(TRAIN_CFG)
    cfg.update(
        {
            "checkpoint_dir": tmpdir,
            "out_dir": str(Path(tmpdir) / "out"),
            "max_steps": 1,
            "grad_accum_steps": 2,
            "progress": False,
        }
    )
    device, _ = select_device(cfg)
    model = GPT(MODEL_CFG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        model_cfg=MODEL_CFG,
        train_cfg=cfg,
        hf_cfg=None,
    )
    trainer.train(make_batches(num_steps=2, seed=7))
    assert trainer.step == 1, trainer.step
    assert trainer.tokens_seen == 2 * 2 * 8, trainer.tokens_seen
    print("gradient accumulation OK")


def test_sparse_log_schedule(tmpdir: str) -> None:
    cfg = dict(TRAIN_CFG)
    cfg.update(
        {
            "checkpoint_dir": str(Path(tmpdir) / "checkpoints"),
            "out_dir": str(Path(tmpdir) / "out"),
            "max_steps": 25,
            "log_first_steps": 3,
            "log_interval": 10,
            "sample_interval": 0,
            "progress": False,
        }
    )
    device, _ = select_device(cfg)
    model = GPT(MODEL_CFG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        model_cfg=MODEL_CFG,
        train_cfg=cfg,
        hf_cfg=None,
    )
    trainer.train(make_batches(num_steps=25, seed=11))

    # First 3 steps, then every 10; the final save adds the last step.
    steps = [point["step"] for point in trainer.history]
    assert steps == [1, 2, 3, 10, 20, 25], steps
    print(f"sparse log schedule OK: {steps}")


def test_sample_generation(tmpdir: str) -> None:
    cfg = dict(TRAIN_CFG)
    cfg.update(
        {
            "checkpoint_dir": str(Path(tmpdir) / "checkpoints"),
            "out_dir": str(Path(tmpdir) / "out"),
            "max_steps": 2,
            "log_first_steps": 0,
            "log_interval": 100,
            "sample_interval": 1,
            "sample_prompt": "A",  # token id 32, inside the tiny vocab
            "sample_max_new_tokens": 2,
            "sample_temperature": 0.0,
            "sample_top_k": None,
            "progress": False,
        }
    )
    device, _ = select_device(cfg)
    model = GPT(MODEL_CFG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        model_cfg=MODEL_CFG,
        train_cfg=cfg,
        hf_cfg=None,
    )
    trainer.train(make_batches(num_steps=2, seed=3))
    assert trainer.last_sample_text is not None
    assert trainer.last_sample_text.startswith("A"), trainer.last_sample_text
    print(f"sample generation OK: {trainer.last_sample_text!r}")


def test_resume_rejects_nonfinite_checkpoint(tmpdir: str) -> None:
    trainer = build_trainer(tmpdir, max_steps=3)

    state = {
        "format_version": 1,
        "model": {
            name: tensor.clone()
            for name, tensor in trainer.model.state_dict().items()
        },
        "step": 10,
        "tokens_seen": 100,
        "model_config": dict(MODEL_CFG),
        "metrics_history": [],
    }
    first = next(iter(state["model"].values()))
    first.view(-1)[0] = float("nan")
    torch.save(state, trainer.checkpoint_path)

    assert not trainer.resume(), "NaN checkpoint must be rejected"

    # Saving NaN weights must be refused as well.
    with torch.no_grad():
        next(trainer.model.parameters()).view(-1)[0] = float("nan")
    assert not trainer.save_checkpoint(), "NaN weights must not be saved"
    print("non-finite checkpoint reject + save guard OK")


def test_loss_and_perplexity_decrease(tmpdir: str) -> None:
    cfg = dict(TRAIN_CFG)
    cfg.update(
        {
            "checkpoint_dir": str(Path(tmpdir) / "checkpoints"),
            "out_dir": str(Path(tmpdir) / "out"),
            "max_steps": 80,
            "batch_size": 4,
            "log_interval": 5,
            "learning_rate": 3e-3,
            "warmup_steps": 0,
            "weight_decay": 0.0,
            "progress": False,
        }
    )
    device, _ = select_device(cfg)
    model = GPT(MODEL_CFG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        model_cfg=MODEL_CFG,
        train_cfg=cfg,
        hf_cfg=None,
    )

    trainer.train(make_fixed_batches(num_steps=80))
    tokens = [point["tokens_seen"] for point in trainer.history]
    losses = [point["loss"] for point in trainer.history]
    perplexities = [point["perplexity"] for point in trainer.history]

    assert len(tokens) >= 10, (len(tokens), trainer.history)
    assert all(a < b for a, b in zip(tokens, tokens[1:])), tokens
    assert losses[-1] < losses[0], (losses[0], losses[-1])
    assert perplexities[-1] < perplexities[0], (
        perplexities[0],
        perplexities[-1],
    )
    assert (trainer.out_dir / "token_seen_vs_loss.png").exists()
    assert (trainer.out_dir / "token_seen_vs_perplexity.png").exists()
    print(
        f"loss decreased {losses[0]:.3f} -> {losses[-1]:.3f}; "
        f"perplexity {perplexities[0]:,.1f} -> {perplexities[-1]:,.1f}; "
        f"tokens {tokens[0]} -> {tokens[-1]}"
    )


def test_time_limit_stops_saves_and_uploads(tmpdir: str) -> None:
    import huggingface_hub

    cfg = dict(TRAIN_CFG)
    cfg.update(
        {
            "checkpoint_dir": str(Path(tmpdir) / "checkpoints"),
            "out_dir": str(Path(tmpdir) / "out"),
            "max_steps": 100,
            "max_hours": 1e-9,
            "log_interval": 1,
            "progress": False,
        }
    )
    hf_cfg = {
        "repo_id": "someone/minillm",
        "token": "test-token",
        "filename": "minillm_latest.pt",
        "out_dir": "out",
        "private": True,
    }

    calls: dict = {}

    class FakeApi:
        def __init__(self, token=None):
            calls["token"] = token

        def create_repo(self, **kwargs):
            calls["create_repo"] = kwargs

        def upload_file(self, **kwargs):
            calls["upload_file"] = kwargs

        def upload_folder(self, **kwargs):
            calls["upload_folder"] = kwargs

    device, _ = select_device(cfg)
    model = GPT(MODEL_CFG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        model_cfg=MODEL_CFG,
        train_cfg=cfg,
        hf_cfg=hf_cfg,
    )

    original = huggingface_hub.HfApi
    huggingface_hub.HfApi = FakeApi
    try:
        trainer.train(make_batches(num_steps=100, seed=13))
    finally:
        huggingface_hub.HfApi = original

    assert trainer.step == 1, trainer.step
    assert trainer.train_start_time is not None
    assert trainer.max_train_seconds == 1e-9 * 3600
    assert trainer.checkpoint_path.exists()
    assert len(trainer.history) == 1
    assert (trainer.out_dir / "metrics.json").exists()
    assert "upload_file" in calls, calls
    assert "upload_folder" in calls, calls
    assert calls["upload_file"]["path_in_repo"] == "minillm_latest.pt"
    assert calls["upload_folder"]["path_in_repo"] == "out"
    print("time limit stopped training and saved/uploaded checkpoint")


def test_lr_schedule() -> None:
    assert abs(lr_for_step(0, TRAIN_CFG) - 1e-3) < 1e-12
    assert abs(lr_for_step(1, TRAIN_CFG) - 1e-3) < 1e-12
    # On the last step of a 3-step cosine schedule, LR reaches min_lr.
    assert abs(lr_for_step(3, TRAIN_CFG) - 1e-4) < 1e-12
    print("lr schedule warmup/decay OK")


def test_env_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / ".env"
        path.write_text(
            "# comment\n"
            "TINY_TEST_TOKEN=abc123\n"
            "export TINY_TEST_OTHER='quoted'\n"
        )
        os.environ.pop("TINY_TEST_TOKEN", None)
        os.environ.pop("TINY_TEST_OTHER", None)
        load_env_file(path)
        assert os.environ["TINY_TEST_TOKEN"] == "abc123"
        assert os.environ["TINY_TEST_OTHER"] == "quoted"
        os.environ.pop("TINY_TEST_TOKEN", None)
        os.environ.pop("TINY_TEST_OTHER", None)
    print("env loader OK")


def test_hf_settings_warnings() -> None:
    cfg = {"hf": {"repo_id": "", "token_env": "TINY_MISSING_TOKEN"}}
    os.environ.pop("TINY_MISSING_TOKEN", None)
    assert get_hf_settings(cfg) is None

    cfg = {
        "hf": {
            "repo_id": "someone/minillm",
            "token_env": "TINY_MISSING_TOKEN",
        }
    }
    assert get_hf_settings(cfg) is None

    os.environ["TINY_PRESENT_TOKEN"] = "test-token"
    cfg = {
        "hf": {
            "repo_id": "someone/minillm",
            "token_env": "TINY_PRESENT_TOKEN",
            "filename": "minillm_latest.pt",
        }
    }
    settings = get_hf_settings(cfg)
    assert settings is not None
    assert settings["token"] == "test-token"
    assert settings["repo_id"] == "someone/minillm"
    os.environ.pop("TINY_PRESENT_TOKEN", None)
    print("missing/present HF settings OK")


def test_hf_upload_mock(tmpdir: str) -> None:
    import huggingface_hub

    path = Path(tmpdir) / "minillm_latest.pt"
    path.write_bytes(b"tiny fake checkpoint")

    calls = {}

    class FakeApi:
        def __init__(self, token=None):
            calls["token"] = token

        def create_repo(self, **kwargs):
            calls["create_repo"] = kwargs

        def upload_file(self, **kwargs):
            calls["upload_file"] = kwargs

        def upload_folder(self, **kwargs):
            calls["upload_folder"] = kwargs

    hf = {
        "repo_id": "someone/minillm",
        "token": "secret-token",
        "filename": "minillm_latest.pt",
        "out_dir": "out",
        "private": True,
    }

    original = huggingface_hub.HfApi
    huggingface_hub.HfApi = FakeApi
    try:
        ok = upload_checkpoint(path, hf, step=7)
        out_dir = Path(tmpdir) / "out"
        out_dir.mkdir()
        (out_dir / "metrics.json").write_text("[]")
        (out_dir / "token_seen_vs_loss.png").write_bytes(b"png")
        ok_outputs = upload_outputs(out_dir, hf, step=7)
    finally:
        huggingface_hub.HfApi = original

    assert ok and ok_outputs
    assert calls["token"] == "secret-token"
    assert calls["create_repo"]["repo_id"] == "someone/minillm"
    assert calls["create_repo"]["exist_ok"] is True
    assert calls["upload_file"]["path_in_repo"] == "minillm_latest.pt"
    assert calls["upload_file"]["repo_type"] == "model"
    assert "step 7" in calls["upload_file"]["commit_message"]
    assert calls["upload_folder"]["path_in_repo"] == "out"
    assert calls["upload_folder"]["repo_type"] == "model"
    assert "step 7" in calls["upload_folder"]["commit_message"]
    print("mock HF checkpoint + output upload OK")


def test_model_config_mismatch() -> None:
    saved = dict(MODEL_CFG)
    saved["emb_dim"] = 32
    mismatches = model_config_mismatches(saved, MODEL_CFG)
    assert any("emb_dim" in item for item in mismatches)
    assert not model_config_mismatches(dict(MODEL_CFG), MODEL_CFG)
    print("model config mismatch detection OK")


def test_run_training_end_to_end(tmpdir: str) -> None:
    """Exercise run_training() with a fake stream; no network, no model 100M."""
    import pretrain.train as train_mod

    cfg = dict(TRAIN_CFG)
    cfg.update(
        {
            "checkpoint_dir": tmpdir,
            "out_dir": str(Path(tmpdir) / "out"),
            "max_steps": 2,
            "batch_size": 2,
            "progress": False,
            "data": {
                "dataset_name": "fake",
                "split": "train",
                "block_size": 8,
                "shuffle_buffer": 0,
                "max_docs": None,
                "min_chars": 1,
            },
        }
    )

    class FakeStream:
        def __init__(self, **kwargs):
            pass

        def batches(self, batch_size=2, drop_last=True):
            return make_batches(2, seed=5)

    original = train_mod.OpenWebTextTokenStream
    train_mod.OpenWebTextTokenStream = FakeStream
    try:
        train_mod.run_training(cfg, MODEL_CFG, no_hf=True)
    finally:
        train_mod.OpenWebTextTokenStream = original

    assert (Path(tmpdir) / "minillm_latest.pt").exists()
    print("run_training end-to-end tiny OK")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        test_tiny_train_and_resume(tmpdir)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_run_training_end_to_end(tmpdir)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_grad_accumulation(tmpdir)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_sparse_log_schedule(tmpdir)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_sample_generation(tmpdir)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_resume_rejects_nonfinite_checkpoint(tmpdir)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_loss_and_perplexity_decrease(tmpdir)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_time_limit_stops_saves_and_uploads(tmpdir)
    test_lr_schedule()
    test_env_file()
    test_hf_settings_warnings()
    with tempfile.TemporaryDirectory() as tmpdir:
        test_hf_upload_mock(tmpdir)
    test_model_config_mismatch()
    print("all tiny training tests passed")


if __name__ == "__main__":
    main()
