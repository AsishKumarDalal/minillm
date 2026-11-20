"""Supervised fine-tuning (SFT) of the mini GPT on streamed Dolly data.

The student is loaded from ``sddfhy546y/minillm`` by default. Data comes from
``sft/sft_data.py`` and is streamed, never downloaded in full. Runtime triage
keeps ``instruction + context + response`` when it fits in the context window,
otherwise ``instruction + response``, otherwise drops the example.

Outputs mirror the other training pipelines: ``sft_checkpoints/sft_model.pt``
and ``sft_outs/`` (metrics + loss/perplexity plots), uploaded every 60 minutes
and on clean stop when `--upload` is set.

Usage
-----
::

    python -m sft.tuning
    python -m sft.tuning --epochs 2 --steps 500 --upload
    python -m sft.tuning --detach --fullnew
    python -m sft.tuning --status
    python -m sft.tuning --stop
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import json
import logging
import math
import os
import random
import signal
import subprocess
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

from model.model import GPT  # noqa: E402
from pretrain.data import get_encoder  # noqa: E402
from pretrain.train import (  # noqa: E402
    atomic_savefig,
    atomic_torch_save,
    atomic_write_text,
    choose_amp_dtype,
    defer_signals,
    download_checkpoint,
    get_hf_settings,
    load_env_file,
    load_yaml,
    make_grad_scaler,
    model_config_mismatches,
    resolve_path,
    select_device,
    setup_logging,
    style_progress_bar,
    unwrap_model,
    upload_checkpoint,
    upload_outputs,
)
from sft.sft_data import SFTData  # noqa: E402

DEFAULT_SFT_CONFIG = ROOT_DIR / "configs" / "sft_config.yaml"
DEFAULT_MODEL_CONFIG = ROOT_DIR / "configs" / "model.yaml"


# ---------------------------------------------------------------------------
# Student loading
# ---------------------------------------------------------------------------
def _token_from_env(token_env: str) -> str:
    return os.environ.get(token_env, "").strip()


def load_model_cfg(cfg: dict) -> dict:
    path = resolve_path(cfg.get("model_config", "configs/model.yaml"))
    model_cfg = load_yaml(Path(path))
    if not isinstance(model_cfg, dict) or "vocab_size" not in model_cfg:
        raise SystemExit(f"Invalid student model config: {path}")
    return model_cfg


def load_student_model(
    cfg: dict,
    model_cfg: dict,
    device: torch.device,
    checkpoint_dir: Path,
):
    section = cfg.get("student") or {}
    source = str(section.get("source", "hf")).strip().lower()
    local_path = resolve_path(
        section.get("local_path", "checkpoints/minillm_latest.pt")
    )
    token_env = str(section.get("token_env", "HF_TOKEN"))

    path: Optional[Path] = None
    if source == "local":
        path = local_path
        if not path.exists():
            raise SystemExit(f"student.source=local but {path} does not exist.")
    elif source == "hf":
        if bool(section.get("prefer_local", True)) and local_path.exists():
            path = local_path
        else:
            hf = {
                "repo_id": str(section.get("repo_id", "")).strip(),
                "filename": str(section.get("filename", "minillm_latest.pt")).strip(),
                "token": _token_from_env(token_env),
            }
            if not hf["repo_id"]:
                raise SystemExit("student.repo_id is empty in configs/sft_config.yaml")
            path = download_checkpoint(hf, checkpoint_dir)
        if path is None:
            raise SystemExit(
                "No student checkpoint found. Put it at "
                f"{local_path}, or set HF_TOKEN to download it from "
                f"{section.get('repo_id')}."
            )
    else:
        raise SystemExit(f"unknown student.source={source!r}; use hf or local")

    try:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise SystemExit(f"Could not read student checkpoint {path}: {exc}") from exc
    if not isinstance(state, dict) or "model" not in state:
        raise SystemExit(f"Student checkpoint {path} has no 'model' state dict.")
    saved_cfg = state.get("model_config")
    if not isinstance(saved_cfg, dict):
        raise SystemExit(f"Student checkpoint {path} has no model_config.")
    mismatches = model_config_mismatches(saved_cfg, model_cfg)
    if mismatches:
        raise SystemExit(
            f"Student checkpoint architecture mismatch: {'; '.join(mismatches)}"
        )

    model = GPT(model_cfg).to(device)
    try:
        model.load_state_dict(state["model"], strict=True)
    except Exception as exc:
        raise SystemExit(f"Could not load student weights from {path}: {exc}") from exc
    model.train()
    logging.info("Student loaded from %s.", path)
    return model


def load_sft_state(
    cfg: dict,
    model_cfg: dict,
    device: torch.device,
    checkpoint_dir: Path,
) -> Optional[tuple[nn.Module, dict]]:
    hf = cfg.get("hf") or {}
    filename = str(hf.get("filename", "sft_model.pt")).strip()
    local_path = checkpoint_dir / filename
    repo_id = str(
        hf.get("repo_id") or (cfg.get("student") or {}).get("repo_id", "")
    ).strip()

    path: Optional[Path] = local_path if local_path.exists() else None
    if path is None and repo_id:
        path = download_checkpoint(
            {
                "repo_id": repo_id,
                "filename": filename,
                "token": _token_from_env(str(hf.get("token_env", "HF_TOKEN"))),
            },
            checkpoint_dir,
        )
    if path is None:
        return None
    try:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
    except Exception as exc:
        logging.warning("Could not read SFT checkpoint %s: %s", path, exc)
        return None
    if not isinstance(state, dict) or "model" not in state:
        logging.warning("SFT checkpoint %s has no model state dict.", path)
        return None
    saved_cfg = state.get("model_config")
    if not isinstance(saved_cfg, dict) or model_config_mismatches(saved_cfg, model_cfg):
        logging.warning("SFT checkpoint %s architecture mismatch; ignoring.", path)
        return None
    model = GPT(model_cfg).to(device)
    try:
        model.load_state_dict(state["model"], strict=True)
    except Exception as exc:
        logging.warning("Could not load SFT weights from %s: %s", path, exc)
        return None
    model.train()
    logging.info(
        "Resuming SFT from %s at step %d, epoch %s.",
        path,
        int(state.get("step", 0)),
        state.get("epoch", 0),
    )
    return model, state


def prepare_student(
    cfg: dict,
    model_cfg: dict,
    device: torch.device,
    checkpoint_dir: Path,
    new_run: bool = False,
    full_new: bool = False,
) -> tuple[nn.Module, Optional[dict], str]:
    """Pick student weights: random, original HF student, or previous SFT."""
    if full_new:
        model = GPT(model_cfg).to(device)
        model.train()
        logging.info("Full new run: student randomly initialized; no checkpoint loaded.")
        return model, None, "random init"
    if not new_run:
        resumed = load_sft_state(cfg, model_cfg, device, checkpoint_dir)
        if resumed is not None:
            model, state = resumed
            return model, state, "previous SFT checkpoint"
    model = load_student_model(cfg, model_cfg, device, checkpoint_dir)
    return model, None, "original student checkpoint"


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _plot_series(
    out_dir: Path, xs: list[float], ys: list[float],
    filename: str, title: str, ylabel: str, xlabel: str = "Tokens seen",
) -> Optional[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=120)
    ax.plot(xs, ys, color="#2563eb", linewidth=1.8, marker="o", markersize=2.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / filename
    atomic_savefig(fig, path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_sft_plots(history: list[dict], out_dir: Path) -> list[Path]:
    if len(history) < 2:
        return []
    tokens = [float(entry.get("tokens_seen", 0)) for entry in history]
    steps = [float(entry.get("step", 0)) for entry in history]
    losses = [float(entry.get("loss", float("nan"))) for entry in history]
    ppls = [
        math.exp(min(loss, 50.0))
        if math.isfinite(loss) and loss > 0
        else float("nan")
        for loss in losses
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    specs = (
        ("token_seen_vs_loss.png", "SFT loss vs. tokens seen", "Loss", losses, tokens, "Tokens seen"),
        ("token_seen_vs_perplexity.png", "Perplexity vs. tokens seen", "Perplexity", ppls, tokens, "Tokens seen"),
        ("step_vs_loss.png", "SFT loss vs. step", "Loss", losses, steps, "Step"),
    )
    if any("val_loss" in entry for entry in history):
        val_points = [
            (float(entry.get("tokens_seen", 0)), float(entry["val_loss"]))
            for entry in history
            if "val_loss" in entry and entry["val_loss"] is not None
        ]
        if len(val_points) >= 2:
            specs = specs + (
                (
                    "token_seen_vs_val_loss.png",
                    "Validation loss vs. tokens seen",
                    "Val loss",
                    [point[1] for point in val_points],
                    [point[0] for point in val_points],
                    "Tokens seen",
                ),
            )
    for filename, title, ylabel, ys, xs, xlabel in specs:
        try:
            path = _plot_series(out_dir, xs, ys, filename, title, ylabel, xlabel)
            if path is not None:
                files.append(path)
        except Exception as exc:
            logging.warning("Could not write %s: %s", filename, exc)
    return files


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class SFTTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        model_cfg: dict,
        train_cfg: dict,
        hf_cfg: Optional[dict] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.hf_cfg = hf_cfg

        self.checkpoint_dir = resolve_path(
            train_cfg.get("checkpoint_dir", "sft_checkpoints")
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        filename = (hf_cfg or train_cfg.get("hf") or {}).get(
            "filename", "sft_model.pt"
        )
        self.checkpoint_path = self.checkpoint_dir / filename

        self.out_dir = resolve_path(train_cfg.get("out_dir", "sft_outs"))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.out_dir / "metrics.json"

        self.interval_seconds = max(
            0.0,
            float(train_cfg.get("checkpoint_interval_minutes", 60)) * 60.0,
        )
        self.last_checkpoint_time = time.monotonic()
        self.step = 0
        self.tokens_seen = 0
        self.epoch = 0
        self.history: list[dict] = []
        self.last_loss: Optional[float] = None
        self.last_sample_text: Optional[str] = None
        self.last_saved_step = 0
        self.last_uploaded_step = 0
        self.last_outputs_uploaded_step = 0
        self.data_stats: dict = {}

        self.amp_dtype = choose_amp_dtype(device, train_cfg)
        self.use_amp = self.amp_dtype is not None
        self.scaler = make_grad_scaler(device, self.use_amp, self.amp_dtype)

    # -- model/loss hooks ---------------------------------------------------
    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Default dense SFT loss. MoE SFT overrides this."""
        logits = self.model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        return F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=-100,
        )

    def extra_stats(self) -> str:
        return ""

    def extra_checkpoint_state(self) -> dict:
        return {}

    def restore_state(self, state: dict) -> None:
        self.step = int(state.get("step", 0))
        self.tokens_seen = int(state.get("tokens_seen", 0))
        self.epoch = int(state.get("epoch", 0))
        history = state.get("metrics_history")
        if isinstance(history, list):
            self.history = [dict(entry) for entry in history if isinstance(entry, dict)]
        stats = state.get("data_stats")
        if isinstance(stats, dict):
            self.data_stats = dict(stats)
        if bool(self.train_cfg.get("save_optimizer", False)):
            optimizer_state = state.get("optimizer")
            if isinstance(optimizer_state, dict):
                try:
                    self.optimizer.load_state_dict(optimizer_state)
                except Exception as exc:
                    logging.warning("Could not restore optimizer state: %s", exc)
        self.last_saved_step = self.step
        self.last_uploaded_step = self.step
        self.last_outputs_uploaded_step = self.step
        self.last_checkpoint_time = time.monotonic()

    def _record_metric(self, loss: float) -> None:
        ppl = math.exp(min(loss, 50.0)) if math.isfinite(loss) else float("nan")
        self.history.append(
            {
                "step": self.step,
                "epoch": self.epoch,
                "tokens_seen": self.tokens_seen,
                "loss": round(float(loss), 6),
                "perplexity": round(float(ppl), 3) if math.isfinite(ppl) else None,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
        self.last_loss = float(loss)

    def write_outputs(self) -> list[Path]:
        atomic_write_text(self.metrics_path, json.dumps(self.history, indent=2))
        files: list[Path] = [self.metrics_path]
        try:
            files.extend(save_sft_plots(self.history, self.out_dir))
        except Exception as exc:
            logging.warning("Could not write SFT plots: %s", exc)
        return files

    def sync_to_hf(self, *, checkpoint: bool = True, outputs: bool = True) -> None:
        if not self.hf_cfg:
            return
        if outputs and self.last_outputs_uploaded_step != self.step:
            if upload_outputs(self.out_dir, self.hf_cfg, self.step):
                self.last_outputs_uploaded_step = self.step
        if checkpoint and self.last_uploaded_step != self.step:
            if upload_checkpoint(self.checkpoint_path, self.hf_cfg, self.step):
                self.last_uploaded_step = self.step

    def save_checkpoint(self, upload: bool = True) -> bool:
        model_state = unwrap_model(self.model).state_dict()
        bad = [
            name
            for name, value in model_state.items()
            if torch.is_tensor(value)
            and value.is_floating_point()
            and not bool(torch.isfinite(value).all().item())
        ]
        if bad:
            logging.warning("Refusing to save non-finite SFT weights in %s.", bad)
            return False
        if self.last_loss is not None and (
            not self.history or self.history[-1].get("step") != self.step
        ):
            self._record_metric(self.last_loss)

        state = {
            "format_version": 1,
            "model": model_state,
            "model_config": self.model_cfg,
            "train_config": self.train_cfg,
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "epoch": self.epoch,
            "metrics_history": self.history,
            "data_stats": self.data_stats,
            "rng_state": torch.get_rng_state(),
        }
        if bool(self.train_cfg.get("save_optimizer", False)):
            state["optimizer"] = self.optimizer.state_dict()
        if torch.cuda.is_available():
            state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        state.update(self.extra_checkpoint_state())
        try:
            atomic_torch_save(state, self.checkpoint_path)
        except Exception as exc:
            logging.warning("Could not save SFT checkpoint: %s", exc)
            return False
        self.write_outputs()
        logging.info(
            "Saved SFT checkpoint %s (%.1f MB) at step %d, epoch %d (%s tokens).",
            self.checkpoint_path,
            self.checkpoint_path.stat().st_size / (1024 * 1024),
            self.step,
            self.epoch,
            f"{self.tokens_seen:,}",
        )
        self.last_saved_step = self.step
        if upload and self.hf_cfg:
            self.sync_to_hf()
        self.last_checkpoint_time = time.monotonic()
        return True

    def final_sync(self) -> None:
        if self.step <= 0:
            return
        with defer_signals(signal.SIGINT, signal.SIGTERM, raise_pending=False):
            if self.last_saved_step != self.step:
                self.save_checkpoint(upload=False)
            if self.hf_cfg:
                self.sync_to_hf()
            logging.info(
                "Final SFT sync at step %d: model %s, out/ %s.",
                self.step,
                "uploaded" if self.last_uploaded_step == self.step else "not uploaded",
                "uploaded"
                if self.last_outputs_uploaded_step == self.step
                else "not uploaded",
            )

    def _log_sample(self, pbar=None) -> Optional[str]:
        prompt = str(self.train_cfg.get("sample_prompt", "Albert Einstein is a")).strip()
        if not prompt:
            return None
        model = unwrap_model(self.model)
        vocab_size = int(self.model_cfg.get("vocab_size", 0))
        max_new_tokens = max(1, int(self.train_cfg.get("sample_max_new_tokens", 60)))
        temperature = float(self.train_cfg.get("sample_temperature", 0.8))
        top_k = self.train_cfg.get("sample_top_k", 50)
        top_k = None if top_k in (None, 0) else int(top_k)
        try:
            prompt_ids = get_encoder().encode(prompt)
        except Exception as exc:
            logging.warning("Could not tokenize sample prompt: %s", exc)
            return None
        if not prompt_ids or max(prompt_ids) >= vocab_size:
            return None
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        try:
            devices = (
                [self.device.index]
                if self.device.type == "cuda" and self.device.index is not None
                else []
            )
            with torch.no_grad(), torch.random.fork_rng(devices=devices):
                out = model.generate(
                    idx,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    use_cache=True,
                )
            continuation = get_encoder().decode(out[0, len(prompt_ids) :].tolist())
        except Exception as exc:
            logging.warning("Sample generation failed at step %d: %s", self.step, exc)
            return None
        finally:
            try:
                model.clear_cache()
            except Exception:
                pass
        text = " ".join((prompt + continuation).split())
        self.last_sample_text = text
        message = f"sample @ step {self.step:,} | {text[:400]}"
        if pbar is not None:
            pbar.write(message)
        else:
            logging.info("%s", message)
        return text

    def train(self, batches: Iterable[tuple[np.ndarray, np.ndarray]]) -> None:
        cfg = self.train_cfg
        max_steps = cfg.get("steps")
        max_steps = int(max_steps) if max_steps is not None else None
        grad_accum = max(1, int(cfg.get("grad_accum_steps", 1)))
        log_first_steps = max(0, int(cfg.get("log_first_steps", 10) or 0))
        log_interval = max(1, int(cfg.get("log_interval", 100) or 100))
        sample_interval = max(0, int(cfg.get("sample_interval", 500) or 0))
        grad_clip = float(cfg.get("grad_clip", 1.0))
        show_progress = bool(cfg.get("progress", True))

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        batches = iter(batches)
        pbar = tqdm(
            total=max_steps,
            desc="sft",
            unit="step",
            dynamic_ncols=True,
            disable=not show_progress,
            colour="green",
            bar_format="{desc}: {percentage:3.0f}%|{bar}|{postfix}",
        )
        if show_progress:
            style_progress_bar(pbar)
        failed = False
        last_epoch = self.epoch

        try:
            while max_steps is None or self.step < max_steps:
                loss_sum = 0.0
                step_tokens = 0
                micro = 0
                for batch in batches:
                    if len(batch) == 3:
                        x_np, y_np, batch_epoch = batch
                        last_epoch = int(batch_epoch)
                    else:
                        x_np, y_np = batch
                    x = torch.from_numpy(x_np).long().to(self.device)
                    y = torch.from_numpy(y_np).long().to(self.device)
                    step_tokens += int((y != -100).sum().item())
                    if self.use_amp:
                        ctx = torch.amp.autocast(
                            device_type=self.device.type, dtype=self.amp_dtype
                        )
                    else:
                        ctx = contextlib.nullcontext()
                    with ctx:
                        loss = self.compute_loss(x, y)
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError(f"non-finite SFT loss at step {self.step + 1}")
                    self.scaler.scale(loss / grad_accum).backward()
                    loss_sum += float(loss.detach())
                    micro += 1
                    if micro >= grad_accum:
                        break
                if micro == 0:
                    break

                if grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                self.step += 1
                self.epoch = last_epoch
                self.tokens_seen += step_tokens
                mean_loss = loss_sum / micro
                self._record_metric(mean_loss)
                pbar.update(1)
                ppl = math.exp(min(mean_loss, 50.0))
                pbar.set_postfix_str(
                    f"#{self.step} | loss {mean_loss:.3f} | ppl {ppl:,.1f} | "
                    f"{format_compact(self.tokens_seen)} tok | ep {self.epoch}"
                    f"{self.extra_stats()}"
                )
                if self.step <= log_first_steps or self.step % log_interval == 0:
                    logging.info(
                        "step %d | epoch %d | tokens %s | loss %.4f | ppl %.1f",
                        self.step,
                        self.epoch,
                        f"{self.tokens_seen:,}",
                        mean_loss,
                        ppl,
                    )
                if sample_interval > 0 and self.step % sample_interval == 0:
                    self._log_sample(pbar)

                if (
                    self.interval_seconds > 0
                    and (time.monotonic() - self.last_checkpoint_time)
                    >= self.interval_seconds
                ):
                    self.save_checkpoint()
        except KeyboardInterrupt:
            logging.info("SFT interrupted; saving before exit.")
            raise
        except Exception:
            failed = True
            raise
        finally:
            pbar.close()
            if self.step > 0 and not failed:
                with defer_signals(signal.SIGINT, signal.SIGTERM):
                    self.save_checkpoint()


def format_compact(value: float) -> str:
    value = float(value)
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{value / 1e3:.1f}k"
    return f"{value:.0f}"


# ---------------------------------------------------------------------------
# Run setup
# ---------------------------------------------------------------------------
def run_sft(cfg: dict, new_run: bool = False, full_new: bool = False) -> None:
    load_env_file(ROOT_DIR / ".env")
    pid_file = os.environ.get("MINILLM_SFT_PID_FILE", "").strip()
    if pid_file:
        atexit.register(_remove_pid_file_if_ours, Path(pid_file))

    seed = int(cfg.get("seed", 1337))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device, num_gpus = select_device(cfg)
    logging.info("Student device: %s | visible GPUs: %d", device, num_gpus)
    model_cfg = load_model_cfg(cfg)
    data_cfg = cfg.get("data") or {}
    block_size = int(data_cfg.get("block_size") or model_cfg["context_length"])
    if block_size > int(model_cfg["context_length"]):
        raise SystemExit(
            f"data.block_size={block_size} exceeds context_length="
            f"{model_cfg['context_length']}"
        )

    checkpoint_dir = resolve_path(cfg.get("checkpoint_dir", "sft_checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model, resume_state, source = prepare_student(
        cfg, model_cfg, device, checkpoint_dir, new_run, full_new
    )
    logging.info("Student source: %s.", source)
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
    hf_cfg = get_hf_settings(cfg, disable=not bool(hf_section.get("enabled", False)))
    if hf_section.get("enabled") and hf_cfg is None:
        logging.warning("hf.enabled is true but sync is not configured; local only.")

    trainer = SFTTrainer(model, optimizer, device, model_cfg, cfg, hf_cfg)
    if resume_state is not None:
        trainer.restore_state(resume_state)

    data = SFTData(
        block_size=block_size,
        dataset_name=str(data_cfg.get("dataset_name", "databricks/databricks-dolly-15k")),
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
        "SFT: epochs=%d, steps=%s, block_size=%d, batch=%d, lr=%.2e.",
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
        logging.info("SFT data stats: %s", data.stats)
        try:
            trainer.final_sync()
        finally:
            if previous_sigterm is not None:
                try:
                    signal.signal(signal.SIGTERM, previous_sigterm)
                except (ValueError, OSError):
                    pass
    logging.info("SFT finished.")


# ---------------------------------------------------------------------------
# Background runner (Jupyter-safe)
# ---------------------------------------------------------------------------
def _pid_file_path(cfg: dict) -> Path:
    return resolve_path(cfg.get("checkpoint_dir", "sft_checkpoints")) / "train.pid"


def _read_pid(path: Path) -> Optional[int]:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _pid_is_trainer(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return True
    return b"sft.tuning" in cmdline or b"sft/tuning.py" in cmdline


def _latest_log() -> Optional[Path]:
    log_dir = ROOT_DIR / "logs"
    try:
        logs = sorted(log_dir.glob("sft-*.log"), key=lambda p: p.stat().st_mtime)
    except OSError:
        return None
    return logs[-1] if logs else None


def _tail(path: Path, lines: int = 20) -> str:
    try:
        content = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


def _remove_pid_file_if_ours(path: Path) -> None:
    try:
        if path.read_text().strip() == str(os.getpid()):
            path.unlink()
    except OSError:
        pass


def start_detached(args: argparse.Namespace, raw_argv: list[str], cfg: dict) -> int:
    checkpoint_dir = resolve_path(cfg.get("checkpoint_dir", "sft_checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pid_file = checkpoint_dir / "train.pid"
    existing = _read_pid(pid_file)
    if existing and _pid_is_trainer(existing):
        print(f"SFT is already running (pid {existing}).")
        print("Use --status to inspect it or --stop to stop it.")
        return 1
    if pid_file.exists():
        with contextlib.suppress(OSError):
            pid_file.unlink()

    if args.log_file is not None:
        log_path = Path(args.log_file).expanduser()
        if not log_path.is_absolute():
            log_path = ROOT_DIR / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_dir = ROOT_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / time.strftime("sft-%Y%m%d-%H%M%S.log")

    child_argv = [arg for arg in raw_argv if arg != "--detach"]
    env = os.environ.copy()
    env["MINILLM_SFT_PID_FILE"] = str(pid_file)
    env["PYTHONUNBUFFERED"] = "1"
    with open(log_path, "ab", buffering=0) as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-m", "sft.tuning", *child_argv],
            cwd=str(ROOT_DIR),
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
            close_fds=True,
        )
    try:
        pid_file.write_text(str(proc.pid))
    except OSError as exc:
        print(f"Warning: could not write PID file {pid_file}: {exc}")

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        returncode = proc.poll()
        if returncode is not None:
            with contextlib.suppress(OSError):
                pid_file.unlink()
            print(f"SFT exited immediately with code {returncode}.")
            print(f"Log: {log_path}")
            tail = _tail(log_path, 30)
            if tail:
                print("--- last log lines ---")
                print(tail)
            return 1
        time.sleep(0.1)

    print(f"SFT started in the background: pid {proc.pid}")
    print(f"Log: {log_path}")
    print(f"PID file: {pid_file}")
    print("Check:  python -m sft.tuning --status")
    print("Stop:   python -m sft.tuning --stop")
    return 0


def stop_training(cfg: dict, timeout: float = 10.0) -> int:
    pid_file = _pid_file_path(cfg)
    pid = _read_pid(pid_file)
    if not pid or not _pid_is_trainer(pid):
        print(f"No running SFT found (PID file: {pid_file}).")
        if pid_file.exists():
            print("The PID file is stale; removing it.")
            with contextlib.suppress(OSError):
                pid_file.unlink()
        return 1
    print(f"Sending SIGTERM to SFT pid {pid}.")
    print("It will save locally and upload before exiting.")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print("The process already exited.")
        return 0
    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline:
        if not _pid_is_trainer(pid):
            print("SFT exited.")
            return 0
        time.sleep(0.2)
    print("SFT is still running (final save/upload can take minutes).")
    log_path = _latest_log()
    if log_path is not None:
        print(f"Watch: {log_path}")
    return 0


def show_status(cfg: dict) -> int:
    checkpoint_dir = resolve_path(cfg.get("checkpoint_dir", "sft_checkpoints"))
    pid_file = _pid_file_path(cfg)
    pid = _read_pid(pid_file)
    running = bool(pid and _pid_is_trainer(pid))
    print(f"SFT: {'RUNNING (pid %d)' % pid if running else 'not running'}")
    if pid and not running:
        print(f"Stale PID file: {pid_file}")
    log_path = _latest_log()
    if log_path is not None:
        print(f"Log: {log_path}")
        tail = _tail(log_path, 20)
        if tail:
            print("--- last log lines ---")
            print(tail)
    filename = str((cfg.get("hf") or {}).get("filename", "sft_model.pt"))
    checkpoint = checkpoint_dir / filename
    if checkpoint.exists():
        stat = checkpoint.stat()
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
        print(f"Checkpoint: {checkpoint} ({stat.st_size / 1024**2:.1f} MB, {when})")
    else:
        print(f"Checkpoint: none at {checkpoint}")
    metrics = resolve_path(cfg.get("out_dir", "sft_outs")) / "metrics.json"
    if metrics.exists():
        stat = metrics.stat()
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
        print(f"Metrics: {metrics} ({when})")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SFT the mini GPT on streamed Dolly data.")
    parser.add_argument("--config", type=Path, default=DEFAULT_SFT_CONFIG)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument(
        "--checkpoint-seconds",
        type=float,
        default=None,
        help="Testing override: seconds between saves/uploads.",
    )
    parser.add_argument(
        "--new", action="store_true", help="Ignore previous SFT checkpoint."
    )
    parser.add_argument(
        "--fullnew", action="store_true", help="Random student; ignore all checkpoints."
    )
    parser.add_argument("--no-hf", action="store_true")
    parser.add_argument("--upload", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--detach", action="store_true")
    mode.add_argument("--stop", action="store_true")
    mode.add_argument("--status", action="store_true")
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--stop-timeout", type=float, default=10.0)
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


def sft_main(argv: Optional[list[str]] = None) -> None:
    setup_logging()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_arg_parser().parse_args(raw_argv)
    cfg = load_yaml(Path(args.config))
    apply_cli_overrides(cfg, args)
    load_env_file(ROOT_DIR / ".env")

    if args.status:
        raise SystemExit(show_status(cfg))
    if args.stop:
        raise SystemExit(stop_training(cfg, timeout=args.stop_timeout))
    if args.detach:
        raise SystemExit(start_detached(args, raw_argv, cfg))

    try:
        run_sft(cfg, new_run=args.new, full_new=args.fullnew)
    except KeyboardInterrupt:
        logging.info("SFT interrupted by user.")


if __name__ == "__main__":
    sft_main()
