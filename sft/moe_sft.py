"""MoE supervised fine-tuning on streamed Dolly data.

The data pipeline, triage rules and output plots are the same as
``sft/tuning.py``.  This module only swaps the dense GPT for the MoE model
and adds the router auxiliary loss plus held-out validation.

Local outputs
-------------
``moe_sft_checkpoints/moe_model_sft.pt``
``moe_sft_outs/metrics.json`` and loss/perplexity/validation plots

HF outputs
----------
``moe/moe_model_sft.pt``
``moe/sft_outs/...``

Usage
-----
::

    python -m sft.moe_sft
    python -m sft.moe_sft --epochs 1 --steps 500
    python -m sft.moe_sft --new
    python -m sft.moe_sft --fullnew --no-hf
    python -m sft.moe_sft --check-hf
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import signal
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.moe_model import (  # noqa: E402
    MOE_ARCH_KEYS,
    MoELanguageModel,
    validate_moe_config,
)
from pretrain.moe_train import load_moe_model_config  # noqa: E402
from pretrain.train import (  # noqa: E402
    check_hf,
    count_parameters,
    download_checkpoint,
    get_hf_settings,
    load_env_file,
    load_yaml,
    model_config_mismatches,
    resolve_path,
    select_device,
    setup_logging,
    unwrap_model,
)
from sft.sft_data import SFTData  # noqa: E402
from sft.tuning import SFTTrainer  # noqa: E402

DEFAULT_CONFIG = ROOT_DIR / "configs" / "moe_sft.yaml"
DEFAULT_MODEL_CONFIG = ROOT_DIR / "configs" / "moe_config.yaml"


# ---------------------------------------------------------------------------
# Student loading
# ---------------------------------------------------------------------------
def _load_state(path: Path) -> dict:
    try:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise SystemExit(f"Could not read checkpoint {path}: {exc}") from exc
    if not isinstance(state, dict) or "model" not in state:
        raise SystemExit(f"Checkpoint {path} has no 'model' state dict.")
    if not isinstance(state.get("model_config"), dict):
        raise SystemExit(f"Checkpoint {path} has no model_config.")
    return state


def _load_moe_from_state(
    state: dict,
    model_cfg: dict,
    device: torch.device,
    path: Path,
) -> MoELanguageModel:
    mismatches = model_config_mismatches(
        state["model_config"], model_cfg, MOE_ARCH_KEYS
    )
    if mismatches:
        raise SystemExit(
            f"Checkpoint {path} architecture mismatch: {'; '.join(mismatches)}"
        )
    model = MoELanguageModel(model_cfg).to(device)
    try:
        model.load_state_dict(state["model"], strict=True)
    except Exception as exc:
        raise SystemExit(f"Could not load MoE weights from {path}: {exc}") from exc
    return model


def load_moe_student(
    cfg: dict,
    model_cfg: dict,
    device: torch.device,
) -> MoELanguageModel:
    section = cfg.get("student") or {}
    source = str(section.get("source", "hf")).strip().lower()
    local_path = resolve_path(
        section.get("local_path", "moe_checkpoints/moe_model_latest.pt")
    )
    token_env = str(section.get("token_env", "HF_TOKEN"))

    path: Optional[Path] = None
    if source == "local":
        if not local_path.exists():
            raise SystemExit(
                f"student.source=local but {local_path} does not exist."
            )
        path = local_path
    elif source == "hf":
        if bool(section.get("prefer_local", True)) and local_path.exists():
            path = local_path
        else:
            repo_id = str(section.get("repo_id", "")).strip()
            if not repo_id:
                raise SystemExit("student.repo_id is empty in moe_sft.yaml")
            remote = str(
                section.get("filename", "moe/moe_model_latest.pt")
            ).strip()
            path = download_checkpoint(
                {
                    "repo_id": repo_id,
                    "filename": local_path.name,
                    "path_in_repo": remote,
                    "token": os.environ.get(token_env, "").strip(),
                },
                local_path.parent,
            )
        if path is None:
            raise SystemExit(
                "No pretrained MoE checkpoint found. Put it at "
                f"{local_path}, or set HF_TOKEN so it can be downloaded "
                f"from {section.get('repo_id')}."
            )
    else:
        raise SystemExit(f"unknown student.source={source!r}; use hf or local")

    state = _load_state(path)
    model = _load_moe_from_state(state, model_cfg, device, path)
    model.train()
    logging.info("MoE student loaded from %s.", path)
    return model


def load_moe_sft_state(
    cfg: dict,
    model_cfg: dict,
    device: torch.device,
    checkpoint_dir: Path,
) -> Optional[tuple[MoELanguageModel, dict]]:
    hf = cfg.get("hf") or {}
    local_name = str(hf.get("filename", "moe_model_sft.pt")).strip()
    local_path = checkpoint_dir / local_name
    repo_id = str(
        hf.get("repo_id") or (cfg.get("student") or {}).get("repo_id", "")
    ).strip()
    token_env = str(hf.get("token_env", "HF_TOKEN"))

    path: Optional[Path] = local_path if local_path.exists() else None
    if path is None and repo_id and bool(hf.get("enabled", True)):
        path = download_checkpoint(
            {
                "repo_id": repo_id,
                "filename": local_name,
                "path_in_repo": str(
                    hf.get("path_in_repo", f"moe/{local_name}")
                ),
                "token": os.environ.get(token_env, "").strip(),
            },
            checkpoint_dir,
        )
    if path is None:
        return None

    try:
        state = _load_state(path)
        model = _load_moe_from_state(state, model_cfg, device, path)
    except SystemExit as exc:
        logging.warning("Ignoring SFT checkpoint %s: %s", path, exc)
        return None
    model.train()
    logging.info(
        "Resuming MoE SFT from %s at step %d, epoch %s.",
        path,
        int(state.get("step", 0)),
        state.get("epoch", 0),
    )
    return model, state


def prepare_moe_student(
    cfg: dict,
    model_cfg: dict,
    device: torch.device,
    checkpoint_dir: Path,
    new_run: bool = False,
    full_new: bool = False,
) -> tuple[MoELanguageModel, Optional[dict], str]:
    if full_new:
        model = MoELanguageModel(model_cfg).to(device)
        model.train()
        logging.info("Full new run: random MoE student; no checkpoint loaded.")
        return model, None, "random init"
    if not new_run:
        resumed = load_moe_sft_state(cfg, model_cfg, device, checkpoint_dir)
        if resumed is not None:
            model, state = resumed
            return model, state, "previous MoE SFT checkpoint"
    return load_moe_student(cfg, model_cfg, device), None, "pretrained MoE"


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class MoESFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.val_batches: list[tuple[np.ndarray, np.ndarray]] = []
        self._last_ce: Optional[float] = None
        self._last_aux: Optional[float] = None
        self.last_val: Optional[dict] = None

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits, aux_loss = self.model(x)
        ce_loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=-100,
        )
        aux = aux_loss.mean()  # DataParallel gathers tuple outputs per replica
        self._last_ce = float(ce_loss.detach())
        self._last_aux = float(aux.detach())
        return ce_loss + aux.to(ce_loss.dtype)

    def extra_stats(self) -> str:
        ce = "?" if self._last_ce is None else f"{self._last_ce:.3f}"
        aux = "?" if self._last_aux is None else f"{self._last_aux:.3f}"
        return f" | ce {ce} | aux {aux}"

    def validate(self) -> Optional[dict]:
        if not self.val_batches:
            return None
        was_training = self.model.training
        self.model.eval()
        totals = {"loss": 0.0, "ce": 0.0, "aux": 0.0}
        with torch.no_grad():
            try:
                for x_np, y_np in self.val_batches:
                    x = torch.from_numpy(x_np).long().to(self.device)
                    y = torch.from_numpy(y_np).long().to(self.device)
                    with self._autocast():
                        logits, aux_loss = self.model(x)
                        ce_loss = F.cross_entropy(
                            logits.float().view(-1, logits.size(-1)),
                            y.view(-1),
                            ignore_index=-100,
                        )
                    aux = aux_loss.mean()
                    totals["ce"] += float(ce_loss)
                    totals["aux"] += float(aux)
                    totals["loss"] += float(
                        ce_loss + aux.to(ce_loss.dtype)
                    )
            finally:
                self.model.train(was_training)
        if not self.val_batches:
            return None
        stats = {
            key: value / len(self.val_batches) for key, value in totals.items()
        }
        stats["perplexity"] = (
            math.exp(min(stats["ce"], 50.0))
            if math.isfinite(stats["ce"])
            else float("nan")
        )
        stats["step"] = self.step
        stats["tokens_seen"] = self.tokens_seen
        self.last_val = stats
        return stats

    def _autocast(self):
        import contextlib

        if self.use_amp:
            return torch.amp.autocast(
                device_type=self.device.type, dtype=self.amp_dtype
            )
        return contextlib.nullcontext()

    def save_checkpoint(self, upload: bool = True) -> bool:
        if self.step > 0 and self.val_batches:
            stats = self.validate()
            if stats is not None:
                if self.history and self.history[-1].get("step") == self.step:
                    self.history[-1]["val_loss"] = round(
                        float(stats["loss"]), 6
                    )
                    self.history[-1]["val_ce"] = round(float(stats["ce"]), 6)
                    self.history[-1]["val_aux"] = round(float(stats["aux"]), 6)
                    self.history[-1]["val_perplexity"] = round(
                        float(stats["perplexity"]), 3
                    )
                logging.info(
                    "validation @ step %d | val_loss %.4f | val_ce %.4f | "
                    "val_aux %.4f | val_ppl %.1f",
                    self.step,
                    stats["loss"],
                    stats["ce"],
                    stats["aux"],
                    stats["perplexity"],
                )
        return super().save_checkpoint(upload=upload)

    def extra_checkpoint_state(self) -> dict:
        extra: dict = {}
        if self.last_val is not None:
            extra["last_val"] = dict(self.last_val)
        return extra


# ---------------------------------------------------------------------------
# Run setup
# ---------------------------------------------------------------------------
def build_validation_batches(
    cfg: dict, block_size: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    data_cfg = cfg.get("data") or {}
    val_docs = int(data_cfg.get("val_docs", 0) or 0)
    val_batches = int(data_cfg.get("val_batches", 0) or 0)
    if val_docs <= 0 or val_batches <= 0:
        return []
    val_batch_size = int(
        data_cfg.get("val_batch_size") or cfg.get("batch_size", 4)
    )
    data = SFTData(
        block_size=block_size,
        dataset_name=str(
            data_cfg.get("dataset_name", "databricks/databricks-dolly-15k")
        ),
        split=str(data_cfg.get("split", "train")),
        shuffle_buffer=0,
        seed=seed + 7919,
        max_docs=val_docs,
        min_chars=int(data_cfg.get("min_chars", 1)),
        tokenizer_name=str(data_cfg.get("tokenizer_name", "gpt2")),
    )
    batches: list[tuple[np.ndarray, np.ndarray]] = []
    for x, y in data.batches(batch_size=val_batch_size, epochs=1):
        batches.append((x, y))
        if len(batches) >= val_batches:
            break
    logging.info(
        "SFT validation: %d cached batch(es) from %d docs (%s).",
        len(batches),
        val_docs,
        data.stats,
    )
    return batches


def run_sft(
    cfg: dict,
    model_cfg: dict,
    no_hf: bool = False,
    new_run: bool = False,
    full_new: bool = False,
    require_hf: bool = False,
) -> None:
    load_env_file(ROOT_DIR / ".env")
    seed = int(cfg.get("seed", 1337))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device, num_gpus = select_device(cfg)
    logging.info("Student device: %s | visible GPUs: %d", device, num_gpus)

    data_cfg = cfg.get("data") or {}
    block_size = int(data_cfg.get("block_size") or model_cfg["context_length"])
    if block_size > int(model_cfg["context_length"]):
        raise SystemExit(
            f"data.block_size={block_size} exceeds context_length="
            f"{model_cfg['context_length']}"
        )

    checkpoint_dir = resolve_path(
        cfg.get("checkpoint_dir", "moe_sft_checkpoints")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model, resume_state, source = prepare_moe_student(
        cfg, model_cfg, device, checkpoint_dir, new_run, full_new
    )
    logging.info("Student source: %s.", source)
    breakdown = model.parameter_breakdown()
    logging.info(
        "MoE parameters: total %s | active/token %s | experts %s",
        f"{breakdown['total']:,}",
        f"{breakdown['active']:,}",
        f"{breakdown['expert_total']:,}",
    )

    if num_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
        logging.info("Student wrapped in DataParallel across %d GPUs.", num_gpus)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 2e-5)),
        betas=tuple(cfg.get("betas", [0.9, 0.95])),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )

    hf_section = cfg.get("hf") or {}
    hf_cfg = get_hf_settings(
        cfg, disable=no_hf or not bool(hf_section.get("enabled", True))
    )
    if require_hf and hf_cfg is None:
        raise SystemExit(
            "HF sync was required but is not configured. Run --check-hf."
        )
    if hf_cfg:
        logging.info(
            "HF sync: ENABLED | repo=%s | model=%s | out=%s",
            hf_cfg["repo_id"],
            hf_cfg["path_in_repo"],
            hf_cfg["out_dir"],
        )
    else:
        logging.warning("HF sync: DISABLED; local outputs only.")

    trainer = MoESFTTrainer(model, optimizer, device, model_cfg, cfg, hf_cfg)
    if resume_state is not None:
        trainer.restore_state(resume_state)
    trainer.val_batches = build_validation_batches(cfg, block_size, seed)

    data = SFTData(
        block_size=block_size,
        dataset_name=str(
            data_cfg.get("dataset_name", "databricks/databricks-dolly-15k")
        ),
        split=str(data_cfg.get("split", "train")),
        shuffle_buffer=int(data_cfg.get("shuffle_buffer", 100)),
        seed=seed,
        max_docs=data_cfg.get("max_docs"),
        min_chars=int(data_cfg.get("min_chars", 1)),
        tokenizer_name=str(data_cfg.get("tokenizer_name", "gpt2")),
    )
    trainer.data_stats = data.stats
    epochs = int(cfg.get("epochs", 1) or 1)
    if epochs < 1:
        raise SystemExit("epochs must be >= 1")
    logging.info(
        "MoE SFT: epochs=%d, steps=%s, block_size=%d, batch=%d, lr=%.2e.",
        epochs,
        cfg.get("steps"),
        block_size,
        int(cfg.get("batch_size", 4)),
        float(cfg.get("learning_rate", 2e-5)),
    )
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
        logging.info("MoE SFT data stats: %s", data.stats)
        try:
            trainer.final_sync()
        finally:
            if previous_sigterm is not None:
                try:
                    signal.signal(signal.SIGTERM, previous_sigterm)
                except (ValueError, OSError):
                    pass
    logging.info("MoE SFT finished.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SFT the Mixture-of-Experts GPT on streamed Dolly data."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--model-config", type=Path, default=DEFAULT_MODEL_CONFIG
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--checkpoint-seconds", type=float, default=None)
    parser.add_argument("--new", action="store_true")
    parser.add_argument("--fullnew", action="store_true")
    parser.add_argument("--no-hf", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--check-hf", action="store_true")
    parser.add_argument("--require-hf", action="store_true")
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
    if args.checkpoint_seconds is not None:
        if args.checkpoint_seconds <= 0:
            raise ValueError("--checkpoint-seconds must be > 0")
        cfg["checkpoint_interval_minutes"] = args.checkpoint_seconds / 60.0
    if args.no_hf:
        cfg.setdefault("hf", {})["enabled"] = False
    if args.upload:
        cfg.setdefault("hf", {})["enabled"] = True


def main(argv: Optional[list[str]] = None) -> None:
    setup_logging()
    args = build_arg_parser().parse_args(argv)
    cfg = load_yaml(Path(args.config))
    apply_cli_overrides(cfg, args)
    model_cfg = load_moe_model_config(args.model_config)
    if not isinstance(model_cfg, dict):
        raise SystemExit("Invalid MoE model config")
    load_env_file(ROOT_DIR / ".env")

    if args.check_hf:
        raise SystemExit(check_hf(cfg, no_hf=args.no_hf))

    try:
        run_sft(
            cfg,
            model_cfg,
            no_hf=args.no_hf,
            new_run=args.new,
            full_new=args.fullnew,
            require_hf=args.require_hf,
        )
    except KeyboardInterrupt:
        logging.info("MoE SFT interrupted by user.")


if __name__ == "__main__":
    main()
