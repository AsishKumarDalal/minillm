"""Tiny CPU tests for the Mixture-of-Experts model and pipelines.

This does not load the ~100M-parameter config, does not touch OpenWebText or
Dolly, and does not upload to HuggingFace. It trains a ~50k-parameter MoE for
a few steps on synthetic batches and checks checkpoint/resume, KV-cache
correctness, generation and the HF ``moe/`` path handling.

Run from the project root:
    .venv/bin/python code_test/test_moe.py
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

from model.moe_model import (  # noqa: E402
    MOE_ARCH_KEYS,
    MoELanguageModel,
    validate_moe_config,
)
from model.moe_parts import (  # noqa: E402
    LayerKVCache,
    load_balance_loss,
    router_z_loss,
)
from pretrain.moe_train import MoETrainer  # noqa: E402
from pretrain.train import (  # noqa: E402
    download_checkpoint,
    get_hf_settings,
    model_config_mismatches,
    select_device,
    upload_checkpoint,
)

MODEL_CFG = {
    "vocab_size": 64,
    "context_length": 16,
    "emb_dim": 32,
    "n_layers": 2,
    "n_heads": 4,
    "n_kv_heads": 2,
    "d_ff": 64,
    "num_experts": 4,
    "top_k": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "out_proj_bias": True,
    "tie_weights": True,
    "use_rope": True,
    "use_attention_sinks": True,
    "use_alibi_bias": True,
    "capacity_factor": 1.25,
    "aux_loss_coef": 0.01,
    "router_z_loss_coef": 0.001,
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
    "data": {"block_size": 16},
    "hf": {
        "repo_id": "",
        "token_env": "HF_TOKEN",
        "filename": "moe_model_latest.pt",
        "private": True,
    },
}

ATOL = 1e-5
RTOL = 1e-4


def make_model(seed: int = 0) -> MoELanguageModel:
    torch.manual_seed(seed)
    model = MoELanguageModel(MODEL_CFG)
    model.eval()
    return model


def assert_close(actual, expected, label: str) -> None:
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL, msg=label)


# ---------------------------------------------------------------------------
# Config / routing helpers
# ---------------------------------------------------------------------------
def test_config_validation() -> None:
    cfg = validate_moe_config(MODEL_CFG)
    assert cfg["vocab_size"] == 64

    bad = dict(MODEL_CFG, emb_dim=30)  # not divisible by n_heads=4
    try:
        validate_moe_config(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("emb_dim/n_heads mismatch must raise")

    bad = dict(MODEL_CFG, top_k=8)  # top_k > num_experts
    try:
        validate_moe_config(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("top_k > num_experts must raise")

    # Legacy gptoss aliases are accepted.
    aliased = validate_moe_config(
        {
            "vocab_size": 64,
            "max_seq_len": 8,
            "d_model": 32,
            "transformer_blocks": 1,
            "n_heads": 4,
            "n_kv_heads": 2,
            "d_ff": 64,
            "num_experts": 4,
            "top_k": 2,
        }
    )
    assert aliased["context_length"] == 8
    assert aliased["emb_dim"] == 32
    assert aliased["n_layers"] == 1
    print("MoE config validation OK")


def test_aux_losses() -> None:
    probs = torch.tensor(
        [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]], dtype=torch.float32
    )
    idx = torch.tensor([[0, 0], [1, 1]])
    aux = load_balance_loss(probs, idx, 4)
    assert aux.ndim == 0 and float(aux) > 0

    z = router_z_loss(torch.randn(8, 4))
    assert z.ndim == 0 and float(z) >= 0
    # z-loss penalises large logits more than small ones.
    assert float(router_z_loss(torch.ones(8, 4) * 5)) > float(
        router_z_loss(torch.ones(8, 4) * 0.1)
    )
    print("router aux losses OK")


def test_layer_kv_cache() -> None:
    cache = LayerKVCache(1, 2, 4, 8, torch.device("cpu"))
    assert cache.seq_len == 0
    k1 = torch.randn(1, 2, 2, 8)
    v1 = torch.randn(1, 2, 2, 8)
    k_full, v_full = cache.update(k1, v1)
    assert k_full.shape == (1, 2, 2, 8) and cache.seq_len == 2
    k2 = torch.randn(1, 2, 1, 8)
    v2 = torch.randn(1, 2, 1, 8)
    k_full, _ = cache.update(k2, v2)
    assert k_full.shape == (1, 2, 3, 8)
    assert torch.equal(k_full[:, :, 2], k2[:, :, 0])
    try:
        cache.update(torch.randn(1, 2, 2, 8), torch.randn(1, 2, 2, 8))
    except ValueError:  # 3 + 2 > 4
        pass
    else:
        raise AssertionError("KV cache overflow must raise")
    cache.reset()
    assert cache.seq_len == 0
    print("LayerKVCache update/overflow/reset OK")


# ---------------------------------------------------------------------------
# Model forward / cache / generation
# ---------------------------------------------------------------------------
def test_forward_backward() -> None:
    torch.set_num_threads(1)
    model = make_model()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 200_000, n_params
    breakdown = model.parameter_breakdown()
    assert breakdown["active"] <= breakdown["total"]
    assert breakdown["expert_total"] > 0

    x = torch.randint(0, MODEL_CFG["vocab_size"], (2, 10))
    y = torch.randint(0, MODEL_CFG["vocab_size"], (2, 10))
    total, ce, aux, logits = model.forward_loss(x, y)
    assert logits.shape == (2, 10, MODEL_CFG["vocab_size"])
    assert total.ndim == 0 and ce.ndim == 0 and aux.ndim == 0
    assert abs(float(total.detach()) - float(ce.detach()) - float(aux.detach())) < 1e-5
    total.backward()
    assert any(p.grad is not None for p in model.parameters())
    assert model.kv_caches is None
    assert not any("kv_cache" in key for key in model.state_dict())
    print(f"MoE forward/backward OK ({n_params:,} parameters)")


def test_cached_forward_matches() -> None:
    model = make_model()
    x = torch.randint(0, MODEL_CFG["vocab_size"], (2, 12))
    with torch.no_grad():
        ref, _ = model(x)

        model.reset_cache()
        got, _ = model(x, use_cache=True)
        assert_close(got, ref, "cached prefill")
        assert model.kv_caches[0].seq_len == x.size(1)

        model.reset_cache()
        _ = model(x[:, :5], use_cache=True)
        chunk, _ = model(x[:, 5:], use_cache=True)
        assert_close(chunk, ref[:, 5:], "chunked decode")

        model.reset_cache()
        pieces = []
        for i in range(x.size(1)):
            out, _ = model(x[:, i : i + 1], use_cache=True)
            pieces.append(out)
        assert_close(torch.cat(pieces, dim=1), ref, "token-by-token decode")

        # A cached forward must not change state when use_cache=False.
        before = model.kv_caches[0].seq_len
        model(x)
        assert model.kv_caches[0].seq_len == before
    print("cached prefill/chunk/token decode OK")


def test_generate_cached_matches_uncached() -> None:
    model = make_model(seed=7)
    prompt = torch.randint(0, MODEL_CFG["vocab_size"], (2, 5))

    with torch.no_grad():
        torch.manual_seed(11)
        cached = model.generate(
            prompt, max_new_tokens=6, temperature=0.0, use_cache=True
        )
        torch.manual_seed(11)
        plain = model.generate(
            prompt, max_new_tokens=6, temperature=0.0, use_cache=False
        )
        assert torch.equal(cached, plain)
        assert torch.equal(cached[:, : prompt.size(1)], prompt)

        # Sliding-window fallback when generation exceeds the context.
        long_prompt = torch.randint(
            0, MODEL_CFG["vocab_size"], (1, MODEL_CFG["context_length"] + 2)
        )
        torch.manual_seed(13)
        cached_long = model.generate(
            long_prompt,
            max_new_tokens=MODEL_CFG["context_length"] + 2,
            temperature=0.0,
            use_cache=True,
        )
        torch.manual_seed(13)
        plain_long = model.generate(
            long_prompt,
            max_new_tokens=MODEL_CFG["context_length"] + 2,
            temperature=0.0,
            use_cache=False,
        )
        assert torch.equal(cached_long, plain_long)
    assert model.training is False
    print("cached generate (incl. sliding window) matches uncached OK")


# ---------------------------------------------------------------------------
# Trainer / checkpoint
# ---------------------------------------------------------------------------
def _make_batches(num_steps: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    batch, block, vocab = 2, 16, MODEL_CFG["vocab_size"]

    def _gen():
        for _ in range(num_steps):
            x = rng.integers(0, vocab, size=(batch, block), dtype=np.int32)
            y = rng.integers(0, vocab, size=(batch, block), dtype=np.int32)
            yield x, y

    return _gen()


def _build_trainer(checkpoint_dir: str, max_steps: int = 3) -> MoETrainer:
    cfg = dict(TRAIN_CFG)
    cfg["checkpoint_dir"] = checkpoint_dir
    cfg["out_dir"] = str(Path(checkpoint_dir) / "out")
    cfg["max_steps"] = max_steps
    device, num_gpus = select_device(cfg)
    assert device.type == "cpu" and num_gpus == 0
    model = MoELanguageModel(MODEL_CFG)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        betas=tuple(cfg["betas"]),
        weight_decay=cfg["weight_decay"],
    )
    return MoETrainer(model, optimizer, device, MODEL_CFG, cfg, None)


def test_trainer_train_and_resume(tmpdir: str) -> None:
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _build_trainer(tmpdir, max_steps=3)
        assert not trainer.resume()
        trainer.train(_make_batches(3))
        assert trainer.step == 3
        assert trainer.tokens_seen == 3 * 2 * 16
        assert trainer.checkpoint_path.exists()
        assert (trainer.out_dir / "metrics.json").exists()
        assert any("aux" in entry for entry in trainer.history)
        assert any("ce" in entry for entry in trainer.history)
        print("MoE trainer trained 3 steps with aux metrics")

        trainer2 = _build_trainer(tmpdir, max_steps=4)
        assert trainer2.resume()
        assert trainer2.step == 3
        assert trainer2.tokens_seen == 96
        trainer2.train(_make_batches(1, seed=99))
        assert trainer2.step == 4
        assert trainer2.tokens_seen == 128
        print("MoE trainer resumed and trained 1 more step")


def test_trainer_arch_keys_and_arch_mismatch() -> None:
    assert "num_experts" in MOE_ARCH_KEYS
    assert "top_k" in MOE_ARCH_KEYS
    saved = dict(MODEL_CFG)
    saved["num_experts"] = 8
    mismatches = model_config_mismatches(saved, MODEL_CFG, MOE_ARCH_KEYS)
    assert any("num_experts" in item for item in mismatches)
    assert not model_config_mismatches(dict(MODEL_CFG), MODEL_CFG, MOE_ARCH_KEYS)
    print("MoE architecture mismatch detection OK")


# ---------------------------------------------------------------------------
# HF path handling
# ---------------------------------------------------------------------------
def test_hf_moe_paths(tmpdir: str) -> None:
    import huggingface_hub

    cfg = {
        "hf": {
            "repo_id": "someone/minillm",
            "token_env": "TINY_MOE_HF_TOKEN",
            "filename": "moe_model_latest.pt",
            "path_in_repo": "moe/moe_model_latest.pt",
            "out_dir": "moe/out",
        }
    }
    os.environ["TINY_MOE_HF_TOKEN"] = "test-token"
    hf = get_hf_settings(cfg)
    os.environ.pop("TINY_MOE_HF_TOKEN", None)
    assert hf is not None
    assert hf["filename"] == "moe_model_latest.pt"
    assert hf["path_in_repo"] == "moe/moe_model_latest.pt"
    assert hf["out_dir"] == "moe/out"

    calls: dict = {}

    class FakeApi:
        def __init__(self, token=None):
            calls["token"] = token

        def create_repo(self, **kwargs):
            calls["create_repo"] = kwargs

        def upload_file(self, **kwargs):
            calls["upload_file"] = kwargs

    path = Path(tmpdir) / "moe_model_latest.pt"
    path.write_bytes(b"fake checkpoint")
    hf["token"] = "test-token"
    original = huggingface_hub.HfApi
    huggingface_hub.HfApi = FakeApi
    try:
        assert upload_checkpoint(path, hf, step=7)
    finally:
        huggingface_hub.HfApi = original
    assert calls["upload_file"]["path_in_repo"] == "moe/moe_model_latest.pt"
    print("HF moe/ path_in_repo upload OK")

    # download_checkpoint maps the remote moe/ path back to a flat local name.
    downloaded: dict = {}

    def fake_hf_hub_download(**kwargs):
        downloaded.update(kwargs)
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"remote checkpoint")
        return str(target)

    original_dl = huggingface_hub.hf_hub_download
    huggingface_hub.hf_hub_download = fake_hf_hub_download
    try:
        result = download_checkpoint(hf, Path(tmpdir) / "ckpt")
    finally:
        huggingface_hub.hf_hub_download = original_dl
    assert downloaded["filename"] == "moe/moe_model_latest.pt"
    assert result is not None and result.name == "moe_model_latest.pt"
    assert result.exists()
    print("HF moe/ path_in_repo download OK")


def main() -> None:
    test_config_validation()
    test_aux_losses()
    test_layer_kv_cache()
    test_forward_backward()
    test_cached_forward_matches()
    test_generate_cached_matches_uncached()
    with tempfile.TemporaryDirectory() as tmpdir:
        test_trainer_train_and_resume(tmpdir)
    test_trainer_arch_keys_and_arch_mismatch()
    with tempfile.TemporaryDirectory() as tmpdir:
        test_hf_moe_paths(tmpdir)
    print("all MoE tiny tests passed")


if __name__ == "__main__":
    main()
