"""Tiny GPU/CPU tests for the tool_calling pipeline (no HF, no teacher)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tool_calling.data_format import (  # noqa: E402
    encode_example,
    from_openai_row,
    make_prompt,
    parse_call,
    read_jsonl,
    write_jsonl,
)
from tool_calling.runtime import demo_registry  # noqa: E402
from tool_calling.tool_calling_distill import build_registry  # noqa: E402
from tool_calling.tool_calling_sft_data import ToolCallingSFTData  # noqa: E402

TINY_CFG = {
    "vocab_size": 50257,
    "context_length": 64,
    "emb_dim": 16,
    "n_heads": 2,
    "n_layers": 1,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "tie_weights": True,
}


def test_format_and_parser() -> None:
    line = 'CALL weather {"city":"Paris"}'
    assert parse_call(line) == ("weather", {"city": "Paris"})
    assert parse_call("It is 22C.") is None
    assert parse_call("CALL weather {bad") is None

    from pretrain.data import get_encoder
    enc = get_encoder("gpt2")
    row = {
        "query": "weather in Paris?",
        "tools": [{"function": {"name": "weather",
                                "parameters": {"properties": {"city": {}}}}}],
        "answers": [{"name": "weather", "arguments": {"city": "Paris"}}],
        "answer": "22C",
    }
    ex = from_openai_row(row, tool_result={"temp_c": 22}, final_answer="It is 22C.")
    assert ex is not None and ex.call == line
    encoded = encode_example(ex, enc, 64, enc.eot_token)
    assert encoded is not None
    ids, labels = encoded
    assert len(ids) == len(labels) and -100 in labels
    assert labels[-1] == enc.eot_token
    # prompt and result tokens are masked
    assert labels[: len(enc.encode(make_prompt(ex.tools, ex.user)))] == [-100] * len(
        enc.encode(make_prompt(ex.tools, ex.user))
    )
    print("tool format/parser OK")


def test_registry() -> None:
    reg = demo_registry()
    assert reg.header == "weather(city),calc(expr)"
    out = reg.try_call('CALL calc {"expr":"2+3"}')
    assert out is not None and out[0] == "calc"
    assert reg.try_call('CALL nope {"x":1}') is None
    specs = [{"name": "foo", "params": ["x"], "stub_result": {"ok": True}}]
    built = build_registry(specs)
    assert built.header == "foo(x)"
    print("tool registry OK")


def test_sft_and_dpo(tmpdir: str) -> None:
    tmp = Path(tmpdir)
    model_yaml = tmp / "model.yaml"
    model_yaml.write_text(yaml.safe_dump(TINY_CFG))

    traces = tmp / "tool_traces.jsonl"
    write_jsonl(list(read_jsonl(ROOT / "data" / "tool_traces.jsonl")), traces)

    # GPU 0 is often busy with the local Inkling server; use GPU 1 when available.
    if torch.cuda.device_count() > 1:
        device = "cuda:1"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    from tool_calling.tool_calling_sft import run_tool_sft

    cfg = {
        "seed": 0,
        "device": device,
        "num_gpus": 1 if device.startswith("cuda") else 0,
        "model_config": str(model_yaml),
        "student": {"source": "local", "local_path": str(tmp / "none.pt")},
        "epochs": 1,
        "steps": 2,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "betas": [0.9, 0.95],
        "grad_clip": 1.0,
        "amp": False,
        "progress": False,
        "log_first_steps": 1,
        "log_interval": 1,
        "sample_interval": 0,
        "checkpoint_interval_minutes": 60,
        "checkpoint_dir": str(tmp / "sft_ckpt"),
        "out_dir": str(tmp / "sft_out"),
        "hf": {"enabled": False, "filename": "tool_sft_model.pt"},
        "data": {"path": str(traces), "block_size": 64,
                 "tokenizer_name": "gpt2", "max_docs": 4},
    }
    # full_new avoids needing a pretrained checkpoint
    run_tool_sft(cfg, full_new=True)
    ckpt = Path(cfg["checkpoint_dir"]) / "tool_sft_model.pt"
    assert ckpt.exists(), ckpt

    # data masking
    data = ToolCallingSFTData.from_jsonl(traces, block_size=64)
    first = next(data.examples_iter())
    assert first["labels"][-1] == 50256
    print("tool SFT tiny run OK on", device)

    # negative sampling + DPO
    from tool_calling.negative_sampling import collect_preferences, run_dpo

    prefs = tmp / "prefs.jsonl"
    with open(prefs, "w", encoding="utf-8") as fh:
        for _ in range(4):
            fh.write(json.dumps({
                "prompt": "TOOLS: weather(city)\nUSER: weather in Paris?\nASSISTANT: ",
                "chosen": 'CALL weather {"city":"Paris"}',
                "rejected": "I think it might be warm.",
            }) + "\n")

    collected = tmp / "collected.jsonl"
    neg_cfg = {
        "seed": 0,
        "device": device,
        "num_gpus": 1 if device.startswith("cuda") else 0,
        "model_config": str(model_yaml),
        "student": {"source": "local", "local_path": str(ckpt)},
        "checkpoint_dir": str(tmp / "neg_ckpt"),
        "out_dir": str(tmp / "neg_out"),
        "data": {"path": str(traces), "preferences_path": str(collected),
                 "tokenizer_name": "gpt2"},
        "tools": [{"name": "weather", "params": ["city"], "stub_result": {"temp_c": 20}}],
        "num_candidates": 2,
        "max_new_tokens": 8,
        "temperature": 0.8,
        "top_k": None,
        "beta": 0.1,
        "learning_rate": 1e-4,
        "betas": [0.9, 0.95],
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "batch_size": 2,
        "steps": 1,
        "log_interval": 1,
        "hf": {"enabled": False, "filename": "negative_model.pt"},
    }
    collect_preferences(neg_cfg)  # writes its own file; harmless
    neg_cfg["data"]["preferences_path"] = str(prefs)
    run_dpo(neg_cfg)
    assert (Path(neg_cfg["checkpoint_dir"]) / "negative_model.pt").exists()
    print("negative sampling + DPO tiny run OK on", device)


def main() -> None:
    test_format_and_parser()
    test_registry()
    with tempfile.TemporaryDirectory() as tmpdir:
        test_sft_and_dpo(tmpdir)
    print("all tool-calling tiny tests passed")


if __name__ == "__main__":
    main()
