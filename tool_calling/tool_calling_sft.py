"""Supervised fine-tuning for tool calling.

Follows the same conventions as ``sft/tuning.py``:

* student loaded from a local/HF checkpoint (or a fresh/resumed SFT state),
* AdamW + cosine-free constant LR from config, optional AMP, DataParallel,
* periodic local checkpoints and plots,
* HF upload into the ``tool_calling/`` folder of the configured repo,
* ``--new`` / ``--fullnew`` / ``--no-hf`` / ``--upload`` flags.

The only difference is the dataset: ``ToolCallingSFTData`` reads compact tool
traces instead of Dolly rows, and loss is computed on assistant lines only.

Usage
-----
::

    python -m tool_calling.tool_calling_sft --config configs/tool_calling_sft.yaml
    python -m tool_calling.tool_calling_sft --steps 500 --no-hf
    python -m tool_calling.tool_calling_sft --fullnew --device cpu --steps 3
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import signal
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.model import GPT  # noqa: E402
from pretrain.train import (  # noqa: E402
    get_hf_settings,
    load_env_file,
    load_yaml,
    select_device,
    setup_logging,
)
from sft.tuning import (  # noqa: E402
    SFTTrainer,
    load_model_cfg,
    load_sft_state,
    load_student_model,
)
from .tool_calling_sft_data import ToolCallingSFTData  # noqa: E402

DEFAULT_CONFIG = ROOT_DIR / "configs" / "tool_calling_sft.yaml"


def prepare_student(cfg: dict, model_cfg: dict, device: torch.device,
                    checkpoint_dir: Path, new_run: bool = False,
                    full_new: bool = False):
    """Pick student weights: random, previous tool SFT, or prior checkpoint."""
    if full_new:
        model = GPT(model_cfg).to(device)
        model.train()
        logging.info("Full new run: student randomly initialized.")
        return model, None, "random init"
    if not new_run:
        resumed = load_sft_state(cfg, model_cfg, device, checkpoint_dir)
        if resumed is not None:
            model, state = resumed
            return model, state, "previous tool SFT checkpoint"
    model = load_student_model(cfg, model_cfg, device, checkpoint_dir)
    return model, None, "configured student checkpoint"


def run_tool_sft(cfg: dict, new_run: bool = False, full_new: bool = False) -> None:
    load_env_file(ROOT_DIR / ".env")
    seed = int(cfg.get("seed", 1337))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device, num_gpus = select_device(cfg)
    model_cfg = load_model_cfg(cfg)

    data_cfg = cfg.get("data") or {}
    block_size = int(data_cfg.get("block_size") or model_cfg["context_length"])
    if block_size > int(model_cfg["context_length"]):
        raise SystemExit(
            f"data.block_size={block_size} exceeds "
            f"context_length={model_cfg['context_length']}"
        )
    path = data_cfg.get("path")
    if not path:
        raise SystemExit("tool SFT needs data.path pointing to tool_traces.jsonl")

    checkpoint_dir = Path(cfg.get("checkpoint_dir", "tool_calling_checkpoints"))
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = ROOT_DIR / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model, resume_state, source = prepare_student(
        cfg, model_cfg, device, checkpoint_dir, new_run, full_new
    )
    logging.info("Tool SFT student source: %s", source)
    if num_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 2e-5)),
        betas=tuple(cfg.get("betas", [0.9, 0.95])),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )

    hf_section = cfg.get("hf") or {}
    hf_cfg = get_hf_settings(cfg, disable=not bool(hf_section.get("enabled", False)))
    trainer = SFTTrainer(model, optimizer, device, model_cfg, cfg, hf_cfg)
    if resume_state is not None:
        trainer.restore_state(resume_state)

    data = ToolCallingSFTData.from_jsonl(
        path,
        block_size=block_size,
        tokenizer_name=str(data_cfg.get("tokenizer_name", "gpt2")),
        max_docs=data_cfg.get("max_docs"),
    )
    trainer.data_stats = data.stats

    epochs = int(cfg.get("epochs", 1) or 1)
    batches = data.batches(
        batch_size=int(cfg.get("batch_size", 4)),
        epochs=epochs,
        return_epoch=True,
    )

    def _terminate(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")

    previous_sigterm = None
    try:
        previous_sigterm = signal.signal(signal.SIGTERM, _terminate)
    except (ValueError, OSError):
        pass
    try:
        trainer.train(batches)
    finally:
        logging.info("Tool SFT data stats: %s", data.stats)
        try:
            trainer.final_sync()
        finally:
            if previous_sigterm is not None:
                try:
                    signal.signal(signal.SIGTERM, previous_sigterm)
                except (ValueError, OSError):
                    pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SFT minillm on tool traces.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--new", action="store_true", help="Ignore previous tool SFT checkpoint.")
    parser.add_argument("--fullnew", action="store_true", help="Random init; ignore all checkpoints.")
    parser.add_argument("--no-hf", action="store_true")
    parser.add_argument("--upload", action="store_true")
    return parser


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> None:
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.device is not None:
        cfg["device"] = args.device
    if args.num_gpus is not None:
        cfg["num_gpus"] = args.num_gpus
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["learning_rate"] = args.lr
    if args.checkpoint_dir is not None:
        cfg["checkpoint_dir"] = args.checkpoint_dir
    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir
    if args.no_hf:
        cfg.setdefault("hf", {})["enabled"] = False
    if args.upload:
        cfg.setdefault("hf", {})["enabled"] = True


def main(argv: Optional[list[str]] = None) -> None:
    setup_logging()
    args = build_arg_parser().parse_args(argv)
    cfg = load_yaml(Path(args.config))
    apply_cli_overrides(cfg, args)
    try:
        run_tool_sft(cfg, new_run=args.new, full_new=args.fullnew)
    except KeyboardInterrupt:
        logging.info("Tool SFT interrupted by user.")


if __name__ == "__main__":
    main()
