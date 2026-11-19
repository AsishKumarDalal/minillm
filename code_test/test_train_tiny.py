"""Tiny CPU test for pretrain/train.py.

This does not load the 100M model, does not touch OpenWebText, and does
not upload to HuggingFace. It trains a ~4.5k-parameter GPT for a few steps
on synthetic token batches, then tests local checkpoint save/resume.

Run from the project root:
    .venv/bin/python code_test/test_train_tiny.py
"""

from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model import GPT  # noqa: E402
import pretrain.train as train_mod  # noqa: E402
from pretrain.train import (  # noqa: E402
    Trainer,
    get_hf_settings,
    load_env_file,
    lr_for_step,
    model_config_mismatches,
    select_device,
    stop_training,
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

    messages = []

    class Capture(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    capture = Capture()
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(capture)
    try:
        trainer.train(make_batches(num_steps=25, seed=11))
    finally:
        root_logger.removeHandler(capture)
        root_logger.setLevel(previous_level)

    # First 3 steps, then every 10; the final save adds the last step.
    steps = [point["step"] for point in trainer.history]
    assert steps == [1, 2, 3, 10, 20, 25], steps

    # Full INFO step lines stop after log_first_steps; the bar carries the
    # live metrics from then on.
    step_lines = [
        message
        for message in messages
        if re.match(r"^step \d+ \| tokens_seen ", message)
    ]
    logged_steps = [int(re.match(r"^step (\d+)", m).group(1)) for m in step_lines]
    assert logged_steps == [1, 2, 3], logged_steps
    print(f"sparse log schedule OK: {steps}; INFO lines only for {logged_steps}")


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


def test_progress_bar_percentage(tmpdir: str) -> None:
    """The tqdm bar gets a real total/percentage in every training mode."""
    import pretrain.train as train_mod

    bars = []

    class FakeTqdm:
        def __init__(self, total=None, initial=0, desc="", unit="step", **kwargs):
            self.total = total
            self.initial = initial
            self.desc = desc
            self.unit = unit
            self.n = initial
            self.postfix = {}
            self.writes = []
            self.resets = 0
            self.closed = False
            self.bar_format = kwargs.get("bar_format", "")
            self.colour = kwargs.get("colour")
            bars.append(self)

        def update(self, n=1.0):
            self.n += float(n)

        def set_postfix(self, postfix=None, **kwargs):
            self.postfix = dict(postfix or {})
            self.postfix.update(kwargs)

        def set_postfix_str(self, s="", refresh=True):
            self.postfix = str(s)

        def write(self, s, end="\n", **kwargs):
            self.writes.append(str(s))

        def set_description(self, desc):
            self.desc = desc

        def reset(self, total=None):
            self.resets += 1
            if total is not None:
                self.total = total
            self.n = 0
            self.initial = 0

        def close(self):
            self.closed = True

    def run_case(name: str, overrides: dict, num_batches: int = 1) -> None:
        cfg = dict(TRAIN_CFG)
        cfg.update(
            {
                "checkpoint_dir": str(Path(tmpdir) / name),
                "out_dir": str(Path(tmpdir) / name / "out"),
                "progress": True,
            }
        )
        cfg.update(overrides)
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
        trainer.train(make_batches(num_steps=num_batches, seed=17))
        return trainer

    original = train_mod.tqdm
    train_mod.tqdm = FakeTqdm
    try:
        run_case(
            "steps",
            {"max_steps": 2, "max_hours": None},
            num_batches=2,
        )
        bar = bars[-1]
        assert bar.total == 2 and bar.unit == "step", (bar.total, bar.unit)
        assert bar.desc == "pretrain", bar.desc
        assert bar.n == 2, bar.n
        assert "{percentage" in bar.bar_format, bar.bar_format
        assert "{postfix}" in bar.bar_format, bar.bar_format
        assert "{bar}" in bar.bar_format, bar.bar_format
        assert "{bar:" not in bar.bar_format, bar.bar_format
        assert bar.colour == "green", bar.colour
        for field in ("t/s", "tok", "ppl", "loss"):
            assert field in bar.postfix, (field, bar.postfix)

        assert train_mod.format_compact(5480.3) == "5.5k"
        assert train_mod.format_compact(82_944) == "82.9k"
        assert train_mod.format_compact(2_740_224) == "2.74M"
        assert train_mod.format_compact(float("nan")) == "nan"

        class FakeLayout:
            flex = None
            width = None
            min_width = None
            overflow = None

        class FakeWidget:
            def __init__(self):
                self.layout = FakeLayout()

        fake_desc, fake_bar, fake_stats = FakeWidget(), FakeWidget(), FakeWidget()
        fake_bar.bar_style = "info"

        class FakeContainer:
            children = (fake_desc, fake_bar, fake_stats)

            def __init__(self):
                self.layout = FakeLayout()

        class FakePbar:
            def __init__(self):
                self.container = FakeContainer()

        train_mod.style_progress_bar(FakePbar())
        assert fake_bar.layout.flex == "0 0 15%", fake_bar.layout.flex
        assert fake_bar.layout.width == "15%", fake_bar.layout.width
        assert fake_stats.layout.flex == "1 1 auto", fake_stats.layout.flex
        assert fake_stats.layout.min_width == "max-content", fake_stats.layout.min_width
        assert fake_stats.layout.overflow == "visible", fake_stats.layout.overflow
        assert fake_bar.bar_style == "success", fake_bar.bar_style

        run_case(
            "hours",
            {"max_steps": None, "max_hours": 1, "checkpoint_interval_minutes": 60},
        )
        bar = bars[-1]
        assert bar.total == 3600.0 and bar.unit == "s", (bar.total, bar.unit)
        assert "time budget" in bar.desc, bar.desc
        assert bar.n > 0, bar.n

        run_case(
            "save",
            {"max_steps": None, "max_hours": None, "checkpoint_interval_minutes": 60},
        )
        bar = bars[-1]
        assert bar.total == 3600.0 and bar.unit == "s", (bar.total, bar.unit)
        assert "to next save" in bar.desc, bar.desc
        assert "#" in bar.postfix and "loss" in bar.postfix, bar.postfix

        trainer = run_case(
            "budget",
            {
                "max_steps": None,
                "max_hours": None,
                "token_budget": 200,
                "checkpoint_interval_minutes": 60,
            },
            num_batches=3,
        )
        bar = bars[-1]
        assert bar.total == 200 and bar.unit == "token", (bar.total, bar.unit)
        assert "token budget" in bar.desc, bar.desc
        assert bar.n == trainer.tokens_seen == 48, (bar.n, trainer.tokens_seen)

        trainer = run_case(
            "budget-stop",
            {
                "max_steps": None,
                "max_hours": None,
                "token_budget": 16,
                "checkpoint_interval_minutes": 60,
            },
            num_batches=5,
        )
        assert trainer.step == 1, trainer.step
        assert trainer.tokens_seen == 16, trainer.tokens_seen

        run_case(
            "estimate",
            {
                "max_steps": None,
                "max_hours": None,
                "estimated_total_tokens": 1000,
                "checkpoint_interval_minutes": 60,
            },
            num_batches=3,
        )
        bar = bars[-1]
        assert bar.total == 1000 and bar.unit == "token", (bar.total, bar.unit)
        assert bar.desc == "pretrain", bar.desc  # short label keeps stats visible
        assert bar.n == 48, bar.n

        run_case(
            "reset",
            {
                "max_steps": None,
                "max_hours": None,
                "checkpoint_interval_minutes": 1e-6,
            },
        )
        bar = bars[-1]
        assert bar.resets >= 1, "bar must reset after a periodic checkpoint save"

        before = len(bars)
        run_case(
            "sample-bar",
            {
                "max_steps": 3,
                "max_hours": None,
                "sample_interval": 1,
                "sample_prompt": "A",
                "sample_max_new_tokens": 2,
            },
            num_batches=3,
        )
        bar = bars[-1]
        assert len(bars) - before == 1, "sample generation must not create a new bar"
        assert len(bar.writes) == 3, bar.writes
        assert all("sample @ step" in message for message in bar.writes), bar.writes
        assert bar.n == 3, bar.n
        assert "#3" in bar.postfix and "loss" in bar.postfix, bar.postfix
    finally:
        train_mod.tqdm = original
    print("progress bar percentage modes + single bar with samples OK")


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


def test_new_and_no_hf_flags(tmpdir: str) -> None:
    """--new keeps HF sync but ignores checkpoints; --no-hf also disables HF."""
    import pretrain.train as train_mod

    class FakeStream:
        def __init__(self, **kwargs):
            pass

        def batches(self, batch_size=2, drop_last=True):
            return make_batches(6, seed=13)

    cfg = dict(TRAIN_CFG)
    cfg.update(
        {
            "checkpoint_dir": str(Path(tmpdir) / "ckpt"),
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
    checkpoint = Path(cfg["checkpoint_dir"]) / "minillm_latest.pt"

    resume_calls = []
    hf_disable_args = []
    original_resume = train_mod.Trainer.resume
    original_stream = train_mod.OpenWebTextTokenStream
    original_hf = train_mod.get_hf_settings

    def counting_resume(self):
        resume_calls.append(True)
        return original_resume(self)

    def fake_hf_settings(pretrain_cfg, disable=False):
        hf_disable_args.append(disable)
        return None

    train_mod.Trainer.resume = counting_resume
    train_mod.OpenWebTextTokenStream = FakeStream
    train_mod.get_hf_settings = fake_hf_settings
    try:
        train_mod.run_training(cfg, MODEL_CFG)
        assert len(resume_calls) == 1
        assert hf_disable_args[-1] is False
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert saved["step"] == 2

        cfg["max_steps"] = 1
        train_mod.run_training(cfg, MODEL_CFG, new_run=True)
        assert len(resume_calls) == 1, "--new must not load any checkpoint"
        assert hf_disable_args[-1] is False, "--new must keep HF upload enabled"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert saved["step"] == 1, "--new must restart from step 0"

        train_mod.run_training(cfg, MODEL_CFG, no_hf=True)
        assert len(resume_calls) == 1, "--no-hf must not load any checkpoint"
        assert hf_disable_args[-1] is True, "--no-hf must disable HF"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert saved["step"] == 1, "--no-hf must restart from step 0"
    finally:
        train_mod.Trainer.resume = original_resume
        train_mod.OpenWebTextTokenStream = original_stream
        train_mod.get_hf_settings = original_hf

    parser = train_mod.build_arg_parser()
    args = parser.parse_args(["--new"])
    assert args.new and not args.no_hf
    args = parser.parse_args(["--no-hf"])
    assert args.no_hf and not args.new
    args = parser.parse_args(["--new", "--no-hf"])
    assert args.new and args.no_hf
    args = parser.parse_args(["--budget", "15000000"])
    assert args.token_budget == 15_000_000, args.token_budget
    args = parser.parse_args(["--hours", "6"])
    assert args.max_hours == 6.0, args.max_hours
    print("--new/--no-hf/--budget/--hours flags OK")


def test_final_sync_uploads_on_interrupt(tmpdir: str) -> None:
    """Ctrl+C / SIGTERM still end with a finite checkpoint on HF."""
    import pretrain.train as train_mod

    cfg = dict(TRAIN_CFG)
    cfg.update(
        {
            "checkpoint_dir": str(Path(tmpdir) / "ckpt"),
            "out_dir": str(Path(tmpdir) / "out"),
            "max_steps": 1,
            "progress": False,
        }
    )
    device, _ = select_device(cfg)
    model = GPT(MODEL_CFG)
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        device=device,
        model_cfg=MODEL_CFG,
        train_cfg=cfg,
        hf_cfg=None,
    )
    # Train without HF first so the test never touches the real Hub.
    trainer.train(make_batches(num_steps=1, seed=1))
    trainer.hf_cfg = {
        "repo_id": "someone/minillm",
        "token": "secret-token",
        "filename": "minillm_latest.pt",
        "out_dir": "out",
        "private": True,
    }
    # Pretend the periodic save/upload never happened (as on an interrupt).
    trainer.last_saved_step = 0
    trainer.last_uploaded_step = 0
    trainer.last_outputs_uploaded_step = 0
    uploaded = []
    original = train_mod.upload_checkpoint
    original_outputs = train_mod.upload_outputs

    def fake_upload(path, hf, step):
        uploaded.append(step)
        return True

    train_mod.upload_checkpoint = fake_upload
    train_mod.upload_outputs = lambda *args, **kwargs: True
    try:
        trainer.final_sync()
    finally:
        train_mod.upload_checkpoint = original
        train_mod.upload_outputs = original_outputs

    assert uploaded == [1], uploaded
    assert trainer.last_saved_step == 1
    assert trainer.last_uploaded_step == 1
    assert trainer.last_outputs_uploaded_step == 1
    print("final sync uploads on interrupt OK")


def test_output_upload_retried_after_checkpoint_success(tmpdir: str) -> None:
    """A failed out/ upload is retried even when the model upload worked."""
    import pretrain.train as train_mod

    cfg = dict(TRAIN_CFG)
    cfg.update(
        {
            "checkpoint_dir": str(Path(tmpdir) / "ckpt"),
            "out_dir": str(Path(tmpdir) / "out"),
            "max_steps": 1,
            "progress": False,
        }
    )
    device, _ = select_device(cfg)
    model = GPT(MODEL_CFG)
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        device=device,
        model_cfg=MODEL_CFG,
        train_cfg=cfg,
        hf_cfg={
            "repo_id": "someone/minillm",
            "token": "secret-token",
            "filename": "minillm_latest.pt",
            "out_dir": "out",
            "private": True,
            "upload_retries": 1,
            "upload_retry_backoff": 0.0,
        },
    )
    trainer.step = 3
    trainer.tokens_seen = 48
    trainer.last_loss = 2.0
    trainer.last_lr = 1e-3
    trainer._record_metric(2.0, 1e-3)

    checkpoint_calls: list[int] = []
    output_calls: list[int] = []

    def fake_checkpoint(path, hf, step):
        checkpoint_calls.append(step)
        return True

    def fake_outputs(out_dir, hf, step):
        output_calls.append(step)
        return len(output_calls) > 1  # first upload fails, retry succeeds

    original_checkpoint = train_mod.upload_checkpoint
    original_outputs = train_mod.upload_outputs
    train_mod.upload_checkpoint = fake_checkpoint
    train_mod.upload_outputs = fake_outputs
    try:
        assert trainer.save_checkpoint()
        assert trainer.last_uploaded_step == 3
        assert trainer.last_outputs_uploaded_step == 0, (
            "failed out/ upload must not be marked as uploaded"
        )

        trainer.final_sync()
    finally:
        train_mod.upload_checkpoint = original_checkpoint
        train_mod.upload_outputs = original_outputs

    assert output_calls == [3, 3], output_calls
    assert checkpoint_calls == [3], checkpoint_calls
    assert trainer.last_outputs_uploaded_step == 3
    print("out/ upload retried independently of the model upload OK")


def test_final_sync_defers_repeated_interrupts(tmpdir: str) -> None:
    """Ctrl+C while the final save/upload runs must not abort the sync."""
    import pretrain.train as train_mod

    cfg = dict(TRAIN_CFG)
    cfg.update(
        {
            "checkpoint_dir": str(Path(tmpdir) / "ckpt"),
            "out_dir": str(Path(tmpdir) / "out"),
            "max_steps": 1,
            "progress": False,
        }
    )
    device, _ = select_device(cfg)
    model = GPT(MODEL_CFG)
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        device=device,
        model_cfg=MODEL_CFG,
        train_cfg=cfg,
        hf_cfg=None,
    )
    trainer.train(make_batches(num_steps=1, seed=1))
    trainer.hf_cfg = {
        "repo_id": "someone/minillm",
        "token": "secret-token",
        "filename": "minillm_latest.pt",
        "out_dir": "out",
        "private": True,
    }
    # Pretend the periodic save/upload never happened (as on an interrupt).
    trainer.last_saved_step = 0
    trainer.last_uploaded_step = 0
    trainer.last_outputs_uploaded_step = 0

    events: list[str] = []

    def fake_outputs(out_dir, hf, step):
        events.append("out")
        signal.raise_signal(signal.SIGINT)
        return True

    def fake_checkpoint(path, hf, step):
        events.append("model")
        signal.raise_signal(signal.SIGINT)
        return True

    previous_int = signal.getsignal(signal.SIGINT)
    previous_term = signal.getsignal(signal.SIGTERM)
    original_checkpoint = train_mod.upload_checkpoint
    original_outputs = train_mod.upload_outputs
    train_mod.upload_checkpoint = fake_checkpoint
    train_mod.upload_outputs = fake_outputs
    try:
        trainer.final_sync()
    finally:
        train_mod.upload_checkpoint = original_checkpoint
        train_mod.upload_outputs = original_outputs

    assert events == ["out", "model"], events
    assert trainer.last_saved_step == 1
    assert trainer.last_uploaded_step == 1
    assert trainer.last_outputs_uploaded_step == 1
    assert signal.getsignal(signal.SIGINT) is previous_int
    assert signal.getsignal(signal.SIGTERM) is previous_term
    print("final sync survives repeated Ctrl+C and restores handlers OK")


def test_atomic_checkpoint_save_removes_stale_tmp(tmpdir: str) -> None:
    """An interrupted save must not leave a .tmp file that confuses resume."""
    trainer = build_trainer(tmpdir, max_steps=1)
    trainer.step = 1
    trainer.tokens_seen = 8
    trainer.last_loss = 2.0
    trainer.last_lr = 1e-3
    trainer._record_metric(2.0, 1e-3)

    stale = trainer.checkpoint_path.with_name(trainer.checkpoint_path.name + ".tmp")
    stale.write_bytes(b"partial checkpoint")
    pid_tmp = trainer.checkpoint_path.with_name(
        f"{trainer.checkpoint_path.name}.tmp.{os.getpid()}"
    )
    pid_tmp.write_bytes(b"another partial checkpoint")

    assert trainer.save_checkpoint(upload=False)
    assert trainer.checkpoint_path.exists()
    assert not stale.exists(), stale
    assert not pid_tmp.exists(), pid_tmp

    saved = torch.load(trainer.checkpoint_path, map_location="cpu", weights_only=True)
    assert saved["step"] == 1
    assert saved["tokens_seen"] == 8
    print("atomic checkpoint save removes stale .tmp files OK")


def test_upload_retries_transient_errors(tmpdir: str) -> None:
    import huggingface_hub

    path = Path(tmpdir) / "minillm_latest.pt"
    path.write_bytes(b"tiny fake checkpoint")

    class FlakyApi:
        attempts = 0

        def __init__(self, token=None):
            pass

        def create_repo(self, **kwargs):
            pass

        def upload_file(self, **kwargs):
            FlakyApi.attempts += 1
            if FlakyApi.attempts == 1:
                raise RuntimeError("temporary network error")

    hf = {
        "repo_id": "someone/minillm",
        "token": "secret-token",
        "filename": "minillm_latest.pt",
        "out_dir": "out",
        "private": True,
        "upload_retries": 2,
        "upload_retry_backoff": 0.0,
    }

    original = huggingface_hub.HfApi
    huggingface_hub.HfApi = FlakyApi
    try:
        assert upload_checkpoint(path, hf, step=1)
    finally:
        huggingface_hub.HfApi = original

    assert FlakyApi.attempts == 2, FlakyApi.attempts
    print("transient HF upload error retried OK")


def test_detached_stop_signal(tmpdir: str) -> None:
    if not sys.platform.startswith("linux"):
        print("detached --stop test skipped (needs /proc)")
        return

    cfg = {"checkpoint_dir": tmpdir}
    pid_file = Path(tmpdir) / "train.pid"
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", "pretrain.train"],
    )
    try:
        time.sleep(0.2)
        pid_file.write_text(str(proc.pid))
        assert stop_training(cfg, timeout=5.0) == 0
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    # A PID file left by a dead process is stale: --stop removes it.
    pid_file.write_text(str(proc.pid))
    assert stop_training(cfg, timeout=1.0) == 1
    assert not pid_file.exists()
    print("detached --stop sends SIGTERM and cleans stale PID OK")


def test_periodic_checkpoint_upload(tmpdir: str) -> None:
    interval_seconds = 2.0
    cfg = dict(TRAIN_CFG)
    cfg["checkpoint_dir"] = tmpdir
    cfg["out_dir"] = str(Path(tmpdir) / "out")
    cfg["max_steps"] = None
    cfg["checkpoint_interval_minutes"] = interval_seconds / 60.0
    cfg["progress"] = False
    cfg["log_interval"] = 100
    cfg["save_optimizer"] = False

    device, num_gpus = select_device(cfg)
    assert device.type == "cpu" and num_gpus == 0
    model = GPT(MODEL_CFG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    hf = {
        "repo_id": "someone/minillm",
        "token": "not-used",
        "token_env": "HF_TOKEN",
        "filename": "minillm_latest.pt",
        "out_dir": "out",
        "private": True,
        "upload_retries": 1,
        "upload_retry_backoff": 0.0,
    }
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        model_cfg=MODEL_CFG,
        train_cfg=cfg,
        hf_cfg=hf,
    )

    uploads: dict[str, list[tuple[float, int]]] = {"model": [], "out": []}
    original_model = train_mod.upload_checkpoint
    original_out = train_mod.upload_outputs

    def fake_model(path, hf_cfg, step):
        uploads["model"].append((time.monotonic(), step))
        return True

    def fake_out(output_dir, hf_cfg, step):
        uploads["out"].append((time.monotonic(), step))
        return True

    deadline = time.monotonic() + 35.0

    def batches():
        rng = np.random.default_rng(7)
        shape = (2, 8)
        while len(uploads["model"]) < 2 and time.monotonic() < deadline:
            x = rng.integers(
                0, MODEL_CFG["vocab_size"], size=shape, dtype=np.int32
            )
            y = rng.integers(
                0, MODEL_CFG["vocab_size"], size=shape, dtype=np.int32
            )
            yield x, y

    train_mod.upload_checkpoint = fake_model
    train_mod.upload_outputs = fake_out
    try:
        trainer.train(batches())
    finally:
        train_mod.upload_checkpoint = original_model
        train_mod.upload_outputs = original_out

    assert len(uploads["model"]) >= 2, uploads
    assert len(uploads["out"]) >= 2, uploads
    first, second = uploads["model"][:2]
    gap = second[0] - first[0]
    assert 1.5 <= gap <= 4.0, gap
    for (output_time, _), (model_time, _) in zip(
        uploads["out"][:2], uploads["model"][:2]
    ):
        assert output_time <= model_time
    print(
        f"periodic checkpoint upload every {interval_seconds:.0f}s OK "
        f"(gap {gap:.1f}s)"
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        test_tiny_train_and_resume(tmpdir)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_run_training_end_to_end(tmpdir)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_new_and_no_hf_flags(tmpdir)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_grad_accumulation(tmpdir)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_sparse_log_schedule(tmpdir)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_sample_generation(tmpdir)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_progress_bar_percentage(tmpdir)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_resume_rejects_nonfinite_checkpoint(tmpdir)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_loss_and_perplexity_decrease(tmpdir)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_time_limit_stops_saves_and_uploads(tmpdir)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_final_sync_uploads_on_interrupt(tmpdir)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_output_upload_retried_after_checkpoint_success(tmpdir)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_final_sync_defers_repeated_interrupts(tmpdir)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_atomic_checkpoint_save_removes_stale_tmp(tmpdir)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_upload_retries_transient_errors(tmpdir)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_detached_stop_signal(tmpdir)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_periodic_checkpoint_upload(tmpdir)
    test_lr_schedule()
    test_env_file()
    test_hf_settings_warnings()
    with tempfile.TemporaryDirectory() as tmpdir:
        test_hf_upload_mock(tmpdir)
    test_model_config_mismatch()
    print("all tiny training tests passed")


if __name__ == "__main__":
    main()
