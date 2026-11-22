"""Negative sampling + DPO for tool calling (replacement for RL).

Two stages:

1. ``collect_preferences`` samples N candidate continuations per prompt from
   the SFT model.  A candidate that parses and executes is a **chosen** trace;
   a malformed / unknown-tool / wrong-argument candidate is a **rejected** trace.
   This is rejection sampling; it can be used on its own by re-training on the
   chosen traces (RFT).

2. ``run_dpo`` fine-tunes the policy with Direct Preference Optimization on
   (chosen, rejected) pairs against a frozen copy of the SFT model.  No reward
   model, no PPO, no GRPO.

HF artifacts go to ``tool_calling/negative_model.pt`` and
``tool_calling/negative_outs/``.

Usage
-----
::

    python -m tool_calling.negative_sampling --config configs/negative_sampling.yaml
    python -m tool_calling.negative_sampling --collect-only --no-hf
    python -m tool_calling.negative_sampling --train-only --steps 100
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pretrain.data import get_encoder  # noqa: E402
from pretrain.train import (  # noqa: E402
    atomic_torch_save,
    atomic_write_text,
    get_hf_settings,
    load_env_file,
    load_yaml,
    select_device,
    setup_logging,
    unwrap_model,
    upload_checkpoint,
    upload_outputs,
)
from sft.tuning import load_model_cfg, load_student_model  # noqa: E402
from .data_format import (  # noqa: E402
    ToolExample,
    make_prompt,
    parse_call,
    read_jsonl,
    write_jsonl,
)
from .tool_calling_distill import build_registry  # noqa: E402

DEFAULT_CONFIG = ROOT_DIR / "configs" / "negative_sampling.yaml"


# ---------------------------------------------------------------------------
# Stage 1: rejection sampling
# ---------------------------------------------------------------------------
def _load_model(cfg: dict, model_cfg: dict, device: torch.device):
    checkpoint_dir = Path(cfg.get("checkpoint_dir", "tool_calling_checkpoints"))
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = ROOT_DIR / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = load_student_model(cfg, model_cfg, device, checkpoint_dir)
    return model, checkpoint_dir


def collect_preferences(cfg: dict) -> int:
    model_cfg = load_model_cfg(cfg)
    device, _ = select_device(cfg)
    model, _ = _load_model(cfg, model_cfg, device)
    model.eval()

    enc = get_encoder(str((cfg.get("data") or {}).get("tokenizer_name", "gpt2")))
    registry = build_registry(cfg.get("tools") or [])

    source_path = ROOT_DIR / str((cfg.get("data") or {}).get("path", "data/tool_traces.jsonl"))
    examples = list(read_jsonl(source_path))
    limit = cfg.get("limit")
    if limit is not None:
        examples = examples[: int(limit)]

    n_candidates = int(cfg.get("num_candidates", 8))
    max_new = int(cfg.get("max_new_tokens", 48))
    temperature = float(cfg.get("temperature", 0.8))
    top_k = cfg.get("top_k")
    top_k = None if top_k in (None, 0) else int(top_k)

    rows = []
    for ex in examples:
        prompt = make_prompt(ex.tools, ex.user)
        ids = enc.encode(prompt)
        if not ids:
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        candidates = []
        for _ in range(n_candidates):
            with torch.no_grad():
                out = model.generate(x, max_new_tokens=max_new,
                                     temperature=temperature, top_k=top_k,
                                     use_cache=True)
            candidates.append(
                enc.decode(out[0, len(ids):].tolist()).split("\n")[0].strip()
            )

        chosen = None
        rejected = None
        for cand in candidates:
            parsed = registry.try_call(cand) if cand else None
            if parsed is not None and chosen is None:
                chosen = cand
            elif parsed is None and rejected is None and cand:
                rejected = cand
        if ex.call is None:
            # No-tool prompt: the correct behaviour is the plain answer.
            chosen = ex.answer.strip()
        if chosen is None or rejected is None or chosen == rejected:
            continue
        rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected,
                     "user": ex.user, "tools": ex.tools})

    out_path = ROOT_DIR / str((cfg.get("data") or {}).get(
        "preferences_path", "data/negative_preferences.jsonl"
    ))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    logging.info("Collected %d chosen/rejected pairs -> %s", len(rows), out_path)
    return len(rows)


# ---------------------------------------------------------------------------
# Stage 2: DPO
# ---------------------------------------------------------------------------
def _encode_pair(row: dict, enc, block_size: int, eot: int):
    p = enc.encode(row["prompt"])
    c = enc.encode(row["chosen"]) + [eot]
    r = enc.encode(row["rejected"]) + [eot]
    if len(p) + max(len(c), len(r)) > block_size:
        return None
    return p, c, r


def _collate(pairs, eot: int):
    seqs = []
    for p, c, r in pairs:
        seqs.append((p + c, [-100] * len(p) + c))
        seqs.append((p + r, [-100] * len(p) + r))
    max_len = max(len(s) for s, _ in seqs)
    x = np.full((len(seqs), max_len), eot, dtype=np.int64)
    y = np.full((len(seqs), max_len), -100, dtype=np.int64)
    for i, (ids, labels) in enumerate(seqs):
        x[i, : len(ids)] = ids
        y[i, : len(labels)] = labels
    return torch.from_numpy(x), torch.from_numpy(y)


def _sequence_logprob(logits: torch.Tensor, labels: torch.Tensor):
    """Sum log p(y_t | context) over non-masked completion tokens."""
    logp = F.log_softmax(logits.float(), dim=-1)
    targets = labels[:, 1:]
    lp = logp[:, :-1].gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = targets != -100
    return (lp * mask).sum(-1), mask.sum(-1)


def run_dpo(cfg: dict) -> None:
    model_cfg = load_model_cfg(cfg)
    device, _ = select_device(cfg)
    policy, checkpoint_dir = _load_model(cfg, model_cfg, device)
    policy.train()

    reference = copy.deepcopy(unwrap_model(policy)).to(device)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)

    enc = get_encoder(str((cfg.get("data") or {}).get("tokenizer_name", "gpt2")))
    eot = enc.eot_token
    block_size = int(model_cfg["context_length"])
    pref_path = ROOT_DIR / str((cfg.get("data") or {}).get(
        "preferences_path", "data/negative_preferences.jsonl"
    ))
    pairs = []
    with open(pref_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            encoded = _encode_pair(json.loads(line), enc, block_size, eot)
            if encoded is not None:
                pairs.append(encoded)
    if len(pairs) < 2:
        raise SystemExit("DPO needs at least 2 valid preference pairs")

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(cfg.get("learning_rate", 5e-7)),
        betas=tuple(cfg.get("betas", [0.9, 0.95])),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )
    beta = float(cfg.get("beta", 0.1))
    batch_size = max(1, int(cfg.get("batch_size", 2)))
    max_steps = int(cfg.get("steps", 100))
    log_interval = max(1, int(cfg.get("log_interval", 10)))

    ckpt_dir = Path(cfg.get("checkpoint_dir", "tool_calling_checkpoints"))
    if not ckpt_dir.is_absolute():
        ckpt_dir = ROOT_DIR / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(cfg.get("out_dir", "tool_calling_outs"))
    if not out_dir.is_absolute():
        out_dir = ROOT_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_cfg = get_hf_settings(cfg, disable=not bool((cfg.get("hf") or {}).get("enabled", False)))
    filename = str((cfg.get("hf") or {}).get("filename", "negative_model.pt"))

    history = []
    rng = random.Random(int(cfg.get("seed", 1337)))
    for step in range(1, max_steps + 1):
        batch = rng.sample(pairs, min(batch_size * 2, len(pairs)))
        batch = batch[: batch_size * 2]
        x, y = _collate(batch, eot)
        x, y = x.to(device), y.to(device)

        logits = policy(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        with torch.no_grad():
            ref_logits = reference(x)

        lp, ntok = _sequence_logprob(logits, y)
        ref_lp, _ = _sequence_logprob(ref_logits, y)
        logits_dpo = lp - ref_lp
        chosen, reject = logits_dpo[0::2], logits_dpo[1::2]
        loss = -F.logsigmoid(beta * (chosen - reject)).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), float(cfg.get("grad_clip", 1.0)))
        optimizer.step()

        value = float(loss.detach())
        if step <= 5 or step % log_interval == 0:
            logging.info("dpo step %d/%d loss %.4f acc %.3f", step, max_steps,
                         value, float((chosen > reject).float().mean()))
        history.append({"step": step, "loss": round(value, 6)})

    state = {
        "format_version": 1,
        "model": unwrap_model(policy).state_dict(),
        "model_config": model_cfg,
        "train_config": cfg,
        "step": max_steps,
        "metrics_history": history,
    }
    path = ckpt_dir / filename
    atomic_torch_save(state, path)
    atomic_write_text(out_dir / "metrics.json", json.dumps(history, indent=2))
    logging.info("Saved DPO model to %s", path)
    if hf_cfg:
        upload_checkpoint(path, hf_cfg, max_steps)
        upload_outputs(out_dir, hf_cfg, max_steps)


def main(argv: Optional[list[str]] = None) -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Negative sampling and DPO for tool calls.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--no-hf", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    load_env_file(ROOT_DIR / ".env")
    if args.limit is not None:
        cfg["limit"] = args.limit
    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.no_hf:
        cfg.setdefault("hf", {})["enabled"] = False
    if args.upload:
        cfg.setdefault("hf", {})["enabled"] = True

    if not args.train_only:
        collect_preferences(cfg)
    if not args.collect_only:
        run_dpo(cfg)


if __name__ == "__main__":
    main()
