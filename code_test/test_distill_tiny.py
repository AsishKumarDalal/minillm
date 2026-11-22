"""Tiny CPU tests for distill/distill.py (no large model downloads).

Run from the project root:
    python3 code_test/test_distill_tiny.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distill import distill as distill_mod  # noqa: E402
from distill.distill import (  # noqa: E402
    DEFAULT_DISTILL_CONFIG,
    Distiller,
    apply_cli_overrides,
    build_arg_parser,
    distillation_loss,
    load_resume_state,
    load_student_model,
    load_yaml,
    prepare_student,
)
from model.model import GPT  # noqa: E402

VOCAB = 64
STUDENT_CFG = {
    "vocab_size": VOCAB,
    "context_length": 8,
    "emb_dim": 16,
    "n_heads": 2,
    "n_layers": 1,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "tie_weights": True,
}


def make_teacher() -> GPT2LMHeadModel:
    cfg = GPT2Config(
        vocab_size=VOCAB,
        n_positions=8,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=0,
        eos_token_id=1,
    )
    model = GPT2LMHeadModel(cfg)
    model.eval()
    model.requires_grad_(False)
    return model


def make_batches(num_steps: int, seed: int = 0):
    rng = np.random.default_rng(seed)

    def _gen():
        for _ in range(num_steps):
            x = rng.integers(0, VOCAB, size=(2, 8), dtype=np.int32)
            y = rng.integers(0, VOCAB, size=(2, 8), dtype=np.int32)
            yield x, y

    return _gen()


def test_distillation_loss_weighting() -> None:
    torch.manual_seed(0)
    student = torch.randn(2, 3, VOCAB)
    teacher = torch.randn(2, 3, VOCAB)
    labels = torch.randint(0, VOCAB, (2, 3))

    total_ce, ce, kl = distillation_loss(student, teacher, labels, alpha=0.0)
    assert torch.allclose(total_ce, ce)
    assert kl.item() > 0

    total_kl, ce2, kl2 = distillation_loss(student, teacher, labels, alpha=1.0)
    assert torch.allclose(total_kl, kl2)
    assert torch.allclose(ce, ce2)

    total_mix, ce3, kl3 = distillation_loss(student, teacher, labels, alpha=0.4)
    expected = 0.6 * ce3 + 0.4 * kl3
    assert torch.allclose(total_mix, expected)
    assert torch.allclose(ce, ce3)
    assert torch.allclose(kl, kl3)

    _, _, kl_hot = distillation_loss(
        student, teacher, labels, alpha=0.4, temperature=0.5
    )
    assert torch.isfinite(kl_hot)
    assert abs(float(kl_hot) - float(kl)) > 1e-6
    print("distillation loss CE/KL weighting OK")


def test_distiller_end_to_end(tmpdir: str) -> None:
    torch.manual_seed(1)
    student = GPT(STUDENT_CFG)
    teacher = make_teacher()
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    train_cfg = {
        "alpha": 0.4,
        "temperature": 1.0,
        "device": "cpu",
        "amp": False,
        "batch_size": 2,
        "grad_accum_steps": 1,
        "max_steps": 3,
        "log_first_steps": 1,
        "log_interval": 1,
        "sample_interval": 2,
        "sample_prompt": "A",
        "sample_max_new_tokens": 4,
        "sample_temperature": 0.8,
        "sample_top_k": 10,
        "checkpoint_interval_minutes": 60,
        "checkpoint_dir": tmpdir,
        "out_dir": str(Path(tmpdir) / "distilled_outs"),
        "save_optimizer": False,
        "progress": False,
    }
    distiller = Distiller(
        model=student,
        teacher=teacher,
        optimizer=optimizer,
        device=torch.device("cpu"),
        teacher_device=torch.device("cpu"),
        teacher_dtype=torch.float32,
        student_cfg=STUDENT_CFG,
        train_cfg=train_cfg,
        hf_cfg=None,
    )
    distiller.train(make_batches(3))
    assert distiller.last_sample_text is not None
    assert distiller.last_sample_text.startswith("A")

    checkpoint = Path(tmpdir) / "distilled_model.pt"
    metrics = Path(tmpdir) / "distilled_outs" / "metrics.json"
    assert checkpoint.exists(), checkpoint
    assert metrics.exists(), metrics
    for name in (
        "token_seen_vs_loss.png",
        "token_seen_vs_perplexity.png",
        "token_seen_vs_ce.png",
        "token_seen_vs_kl.png",
    ):
        plot = Path(tmpdir) / "distilled_outs" / name
        assert plot.exists() and plot.stat().st_size > 0, plot

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["step"] == 3
    assert state["model_config"]["vocab_size"] == VOCAB
    assert abs(float(state["distill_alpha"]) - 0.4) < 1e-9

    fresh = GPT(STUDENT_CFG)
    fresh.load_state_dict(state["model"], strict=True)
    for key in state["model"]:
        assert torch.allclose(fresh.state_dict()[key], state["model"][key])

    import json

    history = json.loads(metrics.read_text())
    assert len(history) >= 1
    assert all(0.0 <= entry["alpha"] <= 1.0 for entry in history)
    assert all(np.isfinite(entry["ce"]) and np.isfinite(entry["kl"]) for entry in history)
    print("distiller end-to-end save/load OK")


def test_config_and_overrides() -> None:
    cfg = load_yaml(DEFAULT_DISTILL_CONFIG)
    assert cfg["teacher"]["name"] == "openai-community/gpt2-xl"
    assert cfg["student"]["repo_id"] == "sddfhy546y/minillm"
    assert cfg["student"]["filename"] == "dense/minillm_latest.pt"
    assert cfg["hf"]["filename"] == "distilled_model.pt"
    assert cfg["hf"]["path_in_repo"] == "dense/distilled_model.pt"
    assert cfg["hf"]["out_dir"] == "dense/distill_outs"
    assert cfg["out_dir"] == "distilled_outs"
    assert int(cfg["num_gpus"]) == 2
    assert 0.0 <= float(cfg["alpha"]) <= 1.0

    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--teacher",
            "hf-internal-testing/tiny-random-gpt2",
            "--alpha",
            "0.6",
            "--checkpoint-seconds",
            "10",
            "--upload",
            "--new",
            "--fullnew",
        ]
    )
    apply_cli_overrides(cfg, args)
    assert cfg["teacher"]["name"] == "hf-internal-testing/tiny-random-gpt2"
    assert cfg["alpha"] == 0.6
    assert abs(cfg["checkpoint_interval_minutes"] - 10.0 / 60.0) < 1e-9
    assert cfg["hf"]["enabled"] is True
    assert args.new is True
    assert args.fullnew is True
    print("distill config and CLI overrides OK")


def test_resume_from_distilled_checkpoint(tmpdir: str) -> None:
    checkpoint_dir = Path(tmpdir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "distilled_model.pt"
    model = GPT(STUDENT_CFG)
    torch.save(
        {
            "format_version": 1,
            "model": model.state_dict(),
            "model_config": STUDENT_CFG,
            "step": 7,
            "tokens_seen": 896,
            "metrics_history": [{"step": 7, "loss": 1.0, "ce": 0.8, "kl": 1.2}],
            "distill_alpha": 0.6,
            "distill_temperature": 1.0,
        },
        checkpoint,
    )
    cfg = {
        "checkpoint_dir": str(checkpoint_dir),
        "hf": {"filename": "distilled_model.pt", "repo_id": ""},
    }
    resumed = load_resume_state(
        cfg, STUDENT_CFG, torch.device("cpu"), checkpoint_dir
    )
    assert resumed is not None
    student, state = resumed
    assert state["step"] == 7

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    distiller = Distiller(
        model=student,
        teacher=make_teacher(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        teacher_device=torch.device("cpu"),
        teacher_dtype=torch.float32,
        student_cfg=STUDENT_CFG,
        train_cfg={"checkpoint_dir": str(checkpoint_dir), "save_optimizer": False},
        hf_cfg=None,
    )
    distiller.restore_state(state)
    assert distiller.step == 7
    assert distiller.tokens_seen == 896
    assert distiller.alpha == 0.6
    assert len(distiller.history) == 1
    print("distilled checkpoint resume OK")


def test_fullnew_random_init(tmpdir: str) -> None:
    # A deliberately invalid student path/repo proves no checkpoint is read.
    cfg = {
        "checkpoint_dir": tmpdir,
        "student": {
            "source": "local",
            "local_path": str(Path(tmpdir) / "does-not-exist.pt"),
        },
        "hf": {"repo_id": "invalid/resume", "filename": "distilled_model.pt"},
    }
    student, state, source = prepare_student(
        cfg, STUDENT_CFG, torch.device("cpu"), Path(tmpdir), full_new=True
    )
    assert source == "random init"
    assert state is None
    assert isinstance(student, GPT)
    print("fullnew random student OK")


def test_dense_hf_paths(tmpdir: str) -> None:
    calls = []
    original = distill_mod.download_checkpoint

    def fake_download(hf, checkpoint_dir):
        calls.append((dict(hf), Path(checkpoint_dir)))
        return None

    distill_mod.download_checkpoint = fake_download
    try:
        cfg = {
            "student": {
                "source": "hf",
                "repo_id": "sddfhy546y/minillm",
                "filename": "dense/minillm_latest.pt",
                "local_path": str(Path(tmpdir) / "checkpoints" / "minillm_latest.pt"),
                "prefer_local": False,
            }
        }
        try:
            load_student_model(
                cfg, STUDENT_CFG, torch.device("cpu"), Path(tmpdir) / "checkpoints"
            )
        except SystemExit:
            pass
        else:
            raise AssertionError("missing student download should exit")
        assert calls[-1][0]["path_in_repo"] == "dense/minillm_latest.pt"
        assert calls[-1][0]["filename"] == "minillm_latest.pt"

        cfg = {
            "hf": {
                "enabled": True,
                "repo_id": "sddfhy546y/minillm",
                "filename": "distilled_model.pt",
                "path_in_repo": "dense/distilled_model.pt",
            }
        }
        load_resume_state(
            cfg, STUDENT_CFG, torch.device("cpu"), Path(tmpdir) / "distill_checkpoints"
        )
        assert calls[-1][0]["path_in_repo"] == "dense/distilled_model.pt"
        assert calls[-1][0]["filename"] == "distilled_model.pt"
    finally:
        distill_mod.download_checkpoint = original
    print("distill dense/ HF paths OK")


def main() -> None:
    test_distillation_loss_weighting()
    with tempfile.TemporaryDirectory() as tmpdir:
        test_distiller_end_to_end(tmpdir)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_resume_from_distilled_checkpoint(tmpdir)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_fullnew_random_init(tmpdir)
    test_config_and_overrides()
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dense_hf_paths(tmpdir)
    print("all tiny distillation tests passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
