#!/usr/bin/env python3
"""Tiny CPU end-to-end test for weight-space programming.

No downloads, no 13B model, no GPU. Builds a random source state dict with
Llama naming, projects it into a small student, and checks that the resulting
student can do a forward pass and short generation.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

try:
    from .model import LlamaForCausalLM
    from .weight_space import DictProvider, build_student, select_layers
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import LlamaForCausalLM
    from weight_space import DictProvider, build_student, select_layers

import yaml


def make_tiny_source(d=64, i=128, layers=6, vocab=100):
    state = {}
    g = torch.Generator().manual_seed(1337)
    randn = lambda *shape: torch.randn(*shape, generator=g) * 0.02

    state["model.embed_tokens.weight"] = randn(vocab, d)
    state["lm_head.weight"] = randn(vocab, d)
    state["model.norm.weight"] = torch.ones(d)

    for layer in range(layers):
        p = f"model.layers.{layer}."
        state[p + "input_layernorm.weight"] = torch.ones(d)
        for kind in ("q", "k", "v", "o"):
            state[p + f"self_attn.{kind}_proj.weight"] = randn(d, d)
        state[p + "post_attention_layernorm.weight"] = torch.ones(d)
        for kind in ("gate", "up"):
            state[p + f"mlp.{kind}_proj.weight"] = randn(i, d)
        state[p + "mlp.down_proj.weight"] = randn(d, i)
    return state


def main():
    torch.manual_seed(0)
    src_dim, src_inter, src_layers, vocab = 64, 128, 6, 100
    dst_dim, dst_inter, dst_layers, heads = 16, 32, 3, 2

    assert select_layers(src_layers, dst_layers) == sorted(
        set(select_layers(src_layers, dst_layers))
    )

    provider = DictProvider(make_tiny_source(src_dim, src_inter, src_layers, vocab))

    cfg = {
        "source": {
            "path": "<tiny in-memory>",
            "hidden_size": src_dim,
            "intermediate_size": src_inter,
            "num_hidden_layers": src_layers,
            "vocab_size": vocab,
            "rope_theta": 10000.0,
            "eos_token_id": 2,
        },
        "projection": {"method": "svd", "profile_layers": 3},
        "student": {
            "vocab_size": vocab,
            "context_length": 32,
            "emb_dim": dst_dim,
            "intermediate_size": dst_inter,
            "n_heads": heads,
            "n_layers": dst_layers,
            "dtype": "float32",
        },
        "output": {},
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg_path = tmp / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        ckpt = build_student(
            str(cfg_path),
            provider=provider,
            output_dir=str(tmp / "checkpoints"),
            filename="tiny.pt",
            verbose=True,
        )

        model = LlamaForCausalLM.from_checkpoint(ckpt)
        model.eval()

        params = sum(p.numel() for p in model.parameters())
        print(f"student params (tiny test): {params:,}")

        ids = torch.randint(0, vocab, (2, 5))
        logits = model(ids)
        assert logits.shape == (2, 5, vocab), logits.shape
        print("forward OK:", tuple(logits.shape))

        out = model.generate(ids[:1, :3], max_new_tokens=4, temperature=0.0)
        assert out.shape[0] == 1 and out.shape[1] <= 7
        print("generate OK:", tuple(out.shape))

    print("ALL GOOD")


if __name__ == "__main__":
    main()
