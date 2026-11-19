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

from distill.distill import (  # noqa: E402
    DEFAULT_DISTILL_CONFIG,
    Distiller,
    apply_cli_overrides,
    build_arg_parser,
    distillation_loss,
    load_yaml,
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

    checkpoint = Path(tmpdir) / "distilled_model.pt"
    metrics = Path(tmpdir) / "distilled_outs" / "metrics.json"
    assert checkpoint.exists(), checkpoint
    assert metrics.exists(), metrics

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
    assert cfg["hf"]["filename"] == "distilled_model.pt"
    assert cfg["hf"]["out_dir"] == "distilled_outs"
    assert cfg["out_dir"] == "distilled_outs"
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
        ]
    )
    apply_cli_overrides(cfg, args)
    assert cfg["teacher"]["name"] == "hf-internal-testing/tiny-random-gpt2"
    assert cfg["alpha"] == 0.6
    assert abs(cfg["checkpoint_interval_minutes"] - 10.0 / 60.0) < 1e-9
    assert cfg["hf"]["enabled"] is True
    print("distill config and CLI overrides OK")


def main() -> None:
    test_distillation_loss_weighting()
    with tempfile.TemporaryDirectory() as tmpdir:
        test_distiller_end_to_end(tmpdir)
    test_config_and_overrides()
    print("all tiny distillation tests passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
