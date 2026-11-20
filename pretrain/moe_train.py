"""Pretrain the Mixture-of-Experts GPT on streamed OpenWebText.

This is the MoE counterpart of ``pretrain/train.py``.  It reuses the dense
trainer's checkpoint/resume, tqdm, plotting, signal handling and HuggingFace
sync code, and adds:

* MoE construction from ``configs/moe_config.yaml`` with strict architecture
  validation before a single step runs;
* total + active parameter accounting and checkpoint architecture checks;
* the router load-balancing / z-loss term in the training objective;
* held-out validation from the same dataset, cached once at startup and
  evaluated before every checkpoint save;
* uploads to ``moe/moe_model_latest.pt`` and ``moe/out/`` in the HF repo.

Usage
-----
::

    python -m pretrain.moe_train
    python -m pretrain.moe_train --max-steps 2000
    python -m pretrain.moe_train --hours 12
    python -m pretrain.moe_train --budget 20000000
    python -m pretrain.moe_train --new --no-hf
    python -m pretrain.moe_train --check-hf
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.moe_model import (  # noqa: E402
    MOE_ARCH_KEYS,
    MoELanguageModel,
    normalize_moe_config,
    validate_moe_config,
)
from pretrain.data import OpenWebTextTokenStream, get_encoder  # noqa: E402
from pretrain.train import (  # noqa: E402
    Trainer,
    atomic_savefig,
    check_hf,
    count_parameters,
    get_hf_settings,
    load_env_file,
    load_yaml,
    resolve_path,
    select_device,
    setup_logging,
)

DEFAULT_TRAIN_CONFIG = ROOT_DIR / "configs" / "moe_train.yaml"
DEFAULT_MODEL_CONFIG = ROOT_DIR / "configs" / "moe_config.yaml"


# ---------------------------------------------------------------------------
# Config / model checks
# ---------------------------------------------------------------------------
def load_moe_model_config(path: str | Path) -> dict:
    model_cfg = load_yaml(Path(path))
    try:
        model_cfg = validate_moe_config(model_cfg)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid MoE model config {path}: {exc}") from exc

    try:
        tokenizer_vocab = int(get_encoder("gpt2").n_vocab)
    except Exception as exc:
        raise SystemExit(f"Could not load the GPT-2 tokenizer: {exc}") from exc
    if int(model_cfg["vocab_size"]) != tokenizer_vocab:
        raise SystemExit(
            f"moe_config.yaml vocab_size={model_cfg['vocab_size']} does not "
            f"match the GPT-2 tokenizer ({tokenizer_vocab}). The data and the "
            "distillation teacher both use GPT-2 BPE, so the vocabulary must "
            "stay at 50257."
        )
    return model_cfg


def _format_params(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1e6:.1f}M"
    if value >= 1_000:
        return f"{value / 1e3:.1f}K"
    return str(value)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class MoETrainer(Trainer):
    """Dense trainer + MoE auxiliary loss, validation and router metrics."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.arch_keys = MOE_ARCH_KEYS
        self.val_batches: list[tuple[np.ndarray, np.ndarray]] = []
        self._last_ce: Optional[float] = None
        self._last_aux: Optional[float] = None
        self.last_val: Optional[dict] = None

    def set_validation_batches(
        self, batches: list[tuple[np.ndarray, np.ndarray]]
    ) -> None:
        self.val_batches = list(batches)

    # -- loss --------------------------------------------------------------
    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits, aux_loss = self.model(x)
        ce_loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            y.reshape(-1),
        )
        # DataParallel gathers the tuple into a 1-D aux tensor (one entry per
        # replica); mean() is the correct global load-balance term.
        aux = aux_loss.mean()
        self._last_ce = float(ce_loss.detach())
        self._last_aux = float(aux.detach())
        return ce_loss + aux.to(ce_loss.dtype)

    def _record_metric(self, loss: float, lr: float) -> None:
        # The base class calls this name from train()/save_checkpoint().
        super()._record_metric(loss, lr)
        if self.history and self.history[-1].get("step") == self.step:
            if self._last_ce is not None:
                self.history[-1]["ce"] = round(self._last_ce, 6)
            if self._last_aux is not None:
                self.history[-1]["aux"] = round(self._last_aux, 6)

    def extra_stats(self) -> str:
        ce = "?" if self._last_ce is None else f"{self._last_ce:.3f}"
        aux = "?" if self._last_aux is None else f"{self._last_aux:.3f}"
        return f" | ce {ce} | aux {aux}"

    # -- validation --------------------------------------------------------
    def _autocast(self):
        if self.use_amp:
            return torch.amp.autocast(
                device_type=self.device.type, dtype=self.amp_dtype
            )
        return contextlib.nullcontext()

    @torch.no_grad()
    def validate(self) -> Optional[dict]:
        if not self.val_batches:
            return None
        was_training = self.model.training
        self.model.eval()
        totals = {"loss": 0.0, "ce": 0.0, "aux": 0.0}
        batches = 0
        try:
            for x_np, y_np in self.val_batches:
                x = torch.from_numpy(x_np).long().to(self.device)
                y = torch.from_numpy(y_np).long().to(self.device)
                with self._autocast():
                    logits, aux_loss = self.model(x)
                    ce_loss = F.cross_entropy(
                        logits.float().view(-1, logits.size(-1)),
                        y.reshape(-1),
                    )
                aux = aux_loss.mean()
                totals["ce"] += float(ce_loss)
                totals["aux"] += float(aux)
                totals["loss"] += float(ce_loss + aux.to(ce_loss.dtype))
                batches += 1
        finally:
            self.model.train(was_training)

        if batches == 0:
            return None
        stats = {key: value / batches for key, value in totals.items()}
        stats["perplexity"] = (
            math.exp(min(stats["ce"], 50.0))
            if math.isfinite(stats["ce"])
            else float("nan")
        )
        stats["step"] = self.step
        stats["tokens_seen"] = self.tokens_seen
        self.last_val = stats
        return stats

    def _record_validation(self, stats: dict) -> None:
        if not self.history or self.history[-1].get("step") != self.step:
            self._record_metric(self.last_loss or 0.0, self.last_lr or 0.0)
        if not self.history or self.history[-1].get("step") != self.step:
            return
        entry = self.history[-1]
        entry["val_loss"] = round(float(stats["loss"]), 6)
        entry["val_ce"] = round(float(stats["ce"]), 6)
        entry["val_aux"] = round(float(stats["aux"]), 6)
        entry["val_perplexity"] = round(float(stats["perplexity"]), 3)

    def save_checkpoint(self, upload: bool = True) -> bool:
        if self.step > 0 and self.val_batches:
            stats = self.validate()
            if stats is not None:
                self._record_validation(stats)
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

    # -- plots -------------------------------------------------------------
    def _plot_extra(self) -> list[Path]:
        if len(self.history) < 2:
            return []
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return []

        files: list[Path] = []

        def plot(
            points: list[tuple[float, float]],
            filename: str,
            title: str,
            ylabel: str,
            log_scale: bool = False,
        ) -> None:
            if len(points) < 2:
                return
            points = sorted(points)
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=120)
            ax.plot(
                xs, ys, color="#2563eb", linewidth=1.8, marker="o",
                markersize=2.5,
            )
            ax.set_xlabel("Tokens seen")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            if log_scale:
                ax.set_yscale("log")
            fig.tight_layout()
            path = self.out_dir / filename
            atomic_savefig(fig, path, bbox_inches="tight")
            plt.close(fig)
            files.append(path)

        val_points = [
            (float(entry.get("tokens_seen", 0)), float(entry["val_loss"]))
            for entry in self.history
            if entry.get("val_loss") is not None
        ]
        plot(
            val_points,
            "token_seen_vs_val_loss.png",
            "Validation loss vs. tokens seen",
            "Validation loss",
        )
        aux_points = [
            (float(entry.get("tokens_seen", 0)), float(entry["aux"]))
            for entry in self.history
            if entry.get("aux") is not None
        ]
        plot(
            aux_points,
            "token_seen_vs_router_aux.png",
            "Router auxiliary loss vs. tokens seen",
            "Aux loss",
            log_scale=True,
        )
        return files

    def _plot_metrics(self) -> list[Path]:
        files = super()._plot_metrics()
        try:
            files.extend(self._plot_extra())
        except Exception as exc:
            logging.warning("Could not write MoE validation/router plots: %s", exc)
        return files


# ---------------------------------------------------------------------------
# Run setup
# ---------------------------------------------------------------------------
def build_validation_batches(
    train_cfg: dict,
    block_size: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    data_cfg = train_cfg.get("data") or {}
    val_docs = int(data_cfg.get("val_docs", 0) or 0)
    if val_docs <= 0:
        return []
    val_batches = int(data_cfg.get("val_batches", 8) or 0)
    if val_batches <= 0:
        return []
    val_batch_size = int(
        data_cfg.get("val_batch_size") or train_cfg.get("batch_size", 4)
    )

    stream = OpenWebTextTokenStream(
        block_size=block_size,
        dataset_name=str(
            data_cfg.get("dataset_name", "Skylion007/openwebtext")
        ),
        split=str(data_cfg.get("split", "train")),
        shuffle_buffer=0,
        seed=int(seed) + 7919,
        max_docs=val_docs,
        min_chars=int(data_cfg.get("min_chars", 1)),
    )
    batches: list[tuple[np.ndarray, np.ndarray]] = []
    for x, y in stream.batches(batch_size=val_batch_size, drop_last=False):
        batches.append((x, y))
        if len(batches) >= val_batches:
            break
    return batches


def run_training(
    train_cfg: dict,
    model_cfg: dict,
    no_hf: bool = False,
    new_run: bool = False,
    require_hf: bool = False,
) -> None:
    load_env_file(ROOT_DIR / ".env")
    seed = int(train_cfg.get("seed", 1337))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device, num_gpus = select_device(train_cfg)
    logging.info("Device: %s | GPUs used: %d", device, num_gpus)

    hf_section = train_cfg.get("hf") or {}
    hf_cfg = get_hf_settings(
        train_cfg,
        disable=no_hf or not bool(hf_section.get("enabled", True)),
    )
    if require_hf and hf_cfg is None:
        raise SystemExit(
            "HF sync was required (--require-hf) but it is not configured. "
            "Set HF_TOKEN and hf.repo_id, then run --check-hf."
        )

    model = MoELanguageModel(model_cfg)
    breakdown = model.parameter_breakdown()
    model.to(device)
    logging.info(
        "MoE parameters: total %s | active/token %s (%.1f%%) | experts %s "
        "(active %s) | router %s | embedding %s",
        f"{breakdown['total']:,}",
        f"{breakdown['active']:,}",
        100.0 * breakdown["active"] / max(1, breakdown["total"]),
        f"{breakdown['expert_total']:,}",
        f"{breakdown['expert_active']:,}",
        f"{breakdown['router']:,}",
        f"{breakdown['embedding']:,}",
    )
    logging.info(
        "MoE architecture: layers=%d heads=%d kv_heads=%d experts=%d top_k=%d "
        "d_ff=%d context=%d capacity=%s",
        model_cfg["n_layers"],
        model_cfg["n_heads"],
        model_cfg["n_kv_heads"],
        model_cfg["num_experts"],
        model_cfg["top_k"],
        model_cfg["d_ff"],
        model_cfg["context_length"],
        model_cfg.get("capacity_factor"),
    )

    if num_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
        logging.info("Using nn.DataParallel across %d GPUs.", num_gpus)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 3e-4)),
        betas=tuple(train_cfg.get("betas", [0.9, 0.95])),
        weight_decay=float(train_cfg.get("weight_decay", 0.1)),
    )

    data_cfg = train_cfg.get("data") or {}
    block_size = int(data_cfg.get("block_size") or model_cfg["context_length"])
    if block_size > int(model_cfg["context_length"]):
        raise SystemExit(
            f"data.block_size={block_size} exceeds context_length="
            f"{model_cfg['context_length']}"
        )
    batch_size = int(train_cfg.get("batch_size", 4))
    grad_accum = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    logging.info(
        "batch_size=%d block_size=%d grad_accum=%d -> %s tokens per optimizer step",
        batch_size,
        block_size,
        grad_accum,
        f"{batch_size * block_size * grad_accum:,}",
    )

    val_batches = build_validation_batches(train_cfg, block_size, seed)
    if val_batches:
        first = val_batches[0]
        logging.info(
            "Validation: %d cached batch(es) of shape %s from val_docs=%s.",
            len(val_batches),
            first[0].shape,
            data_cfg.get("val_docs"),
        )
    else:
        logging.info("Validation disabled (data.val_docs=0 or val_batches=0).")

    trainer = MoETrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        hf_cfg=hf_cfg,
    )
    trainer.set_validation_batches(val_batches)

    if new_run or no_hf:
        reason = "--new" if new_run and not no_hf else "--no-hf"
        logging.warning(
            "Starting from a random MoE model (%s): checkpoints are ignored.",
            reason,
        )
    else:
        trainer.resume()

    if hf_cfg:
        logging.info(
            "HF sync: ENABLED | repo=%s | model=%s | out=%s",
            hf_cfg["repo_id"],
            hf_cfg["path_in_repo"],
            hf_cfg["out_dir"],
        )
    else:
        logging.warning(
            "HF sync: DISABLED. Local checkpoints/plots are still written."
        )

    stream = OpenWebTextTokenStream(
        block_size=block_size,
        dataset_name=str(
            data_cfg.get("dataset_name", "Skylion007/openwebtext")
        ),
        split=str(data_cfg.get("split", "train")),
        shuffle_buffer=int(data_cfg.get("shuffle_buffer", 1000)),
        seed=seed,
        max_docs=data_cfg.get("max_docs"),
        min_chars=int(data_cfg.get("min_chars", 1)),
        skip_docs=int(data_cfg.get("val_docs", 0) or 0),
    )
    batches = stream.batches(batch_size=batch_size, drop_last=True)

    def _terminate(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")

    previous_sigterm = None
    try:
        previous_sigterm = signal.signal(signal.SIGTERM, _terminate)
    except (ValueError, OSError):
        pass

    try:
        trainer.train(batches)
    except KeyboardInterrupt:
        logging.info("Interrupted; saving the final MoE checkpoint before exit.")
        raise
    except Exception:
        logging.exception(
            "MoE training crashed; attempting a final finite checkpoint save."
        )
        raise
    finally:
        try:
            trainer.final_sync()
        finally:
            if previous_sigterm is not None:
                try:
                    signal.signal(signal.SIGTERM, previous_sigterm)
                except (ValueError, OSError):
                    pass
    logging.info("MoE training finished.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretrain the Mixture-of-Experts GPT on OpenWebText."
    )
    parser.add_argument(
        "--config", "--train-config", dest="config", type=Path,
        default=DEFAULT_TRAIN_CONFIG,
    )
    parser.add_argument(
        "--model-config", type=Path, default=DEFAULT_MODEL_CONFIG
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--budget", "--token-budget", type=int, default=None,
        dest="token_budget",
    )
    parser.add_argument(
        "--max-hours", "--hours", type=float, default=None, dest="max_hours"
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--checkpoint-seconds", type=float, default=None)
    parser.add_argument(
        "--new", action="store_true",
        help="Ignore local/HF checkpoints and start from a random model.",
    )
    parser.add_argument(
        "--no-hf", action="store_true",
        help="Start fresh and disable every HF download/upload.",
    )
    parser.add_argument(
        "--check-hf", action="store_true",
        help="Check HF_TOKEN/repo access, then exit.",
    )
    parser.add_argument(
        "--require-hf", action="store_true",
        help="Abort if HF sync is not configured.",
    )
    return parser


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> None:
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.token_budget is not None:
        cfg["token_budget"] = args.token_budget
    if args.max_hours is not None:
        cfg["max_hours"] = args.max_hours
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.device is not None:
        cfg["device"] = args.device
    if args.num_gpus is not None:
        cfg["num_gpus"] = args.num_gpus
    if args.checkpoint_dir is not None:
        cfg["checkpoint_dir"] = args.checkpoint_dir
    if args.checkpoint_seconds is not None:
        if args.checkpoint_seconds <= 0:
            raise ValueError("--checkpoint-seconds must be > 0")
        cfg["checkpoint_interval_minutes"] = args.checkpoint_seconds / 60.0


def main(argv: Optional[list[str]] = None) -> None:
    setup_logging()
    args = build_arg_parser().parse_args(argv)
    train_cfg = load_yaml(Path(args.config))
    apply_cli_overrides(train_cfg, args)
    model_cfg = load_moe_model_config(args.model_config)
    load_env_file(ROOT_DIR / ".env")

    if args.check_hf:
        raise SystemExit(check_hf(train_cfg, no_hf=args.no_hf))

    try:
        run_training(
            train_cfg,
            model_cfg,
            no_hf=args.no_hf,
            new_run=args.new,
            require_hf=args.require_hf,
        )
    except KeyboardInterrupt:
        logging.info("MoE training interrupted by user.")


if __name__ == "__main__":
    main()
