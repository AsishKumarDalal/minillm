"""Pretrain the mini GPT on streaming OpenWebText.

Features
--------
* Uses 2 GPUs via ``nn.DataParallel`` when available, otherwise 1 GPU,
  otherwise CPU. (DataParallel is used because it works from a notebook
  without ``torchrun``/spawn setup.)
* Streams and tokenizes OpenWebText lazily through ``pretrain.data``.
* Automatic mixed precision (bf16/fp16) on CUDA.
* TQDM progress bar via ``tqdm.auto`` so it works in notebooks and terminals.
* Saves a local checkpoint and (optionally) uploads it to HuggingFace Hub
  every ``checkpoint_interval_minutes`` (default: 60), plus at exit/Ctrl+C.
* Optional wall-clock limit via ``--max-hours 6`` (or ``max_hours`` in the
  config): training stops after that many hours and still saves/uploads.
* Writes ``out/metrics.json`` and token-vs-loss / token-vs-perplexity plots
  at the same interval, then uploads them to the HF repo's ``out/`` folder.
* On startup, resumes from ``checkpoints/minillm_latest.pt`` if it exists,
  otherwise downloads ``minillm_latest.pt`` from the configured HF repo.
  It restores model, optimizer, step, tokens, RNG state, and the metrics
  history used by the plots.
* ``--new`` ignores both local and HF checkpoints and starts from a random
  model, but still saves locally and uploads checkpoints/plots to HF.
* ``--no-hf`` ignores both local and HF checkpoints and disables all HF
  download/upload, so the run never pulls from or pushes to HF.
* If the HF repo or token is missing, logs a warning and disables remote
  sync; it still resumes from a local checkpoint when one is present
  (unless ``--new`` or ``--no-hf`` was passed).

Config
------
``configs/pretrain.yaml`` controls training and HF sync.
``configs/model.yaml`` controls model size.
``.env`` (see ``.env.example``) supplies ``HF_TOKEN``.

Usage
-----
::

    # GPU/CPU, auto-selected
    .venv/bin/python -m pretrain.train

    # Force CPU or limit steps
    .venv/bin/python -m pretrain.train --device cpu --max-steps 10

    # Ignore local and HF checkpoints and start from a random model.
    # Training still saves and uploads to HF.
    .venv/bin/python -m pretrain.train --new

    # Fresh local-only run: ignore local and HF checkpoints, never upload.
    .venv/bin/python -m pretrain.train --no-hf

    # Stop after 12 hours and save/upload the final checkpoint
    .venv/bin/python -m pretrain.train --hours 12
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.model import GPT
from pretrain.data import OpenWebTextTokenStream, get_encoder

DEFAULT_PRETRAIN_CONFIG = ROOT_DIR / "configs" / "pretrain.yaml"
DEFAULT_MODEL_CONFIG = ROOT_DIR / "configs" / "model.yaml"

ARCH_KEYS = (
    "vocab_size",
    "context_length",
    "emb_dim",
    "n_heads",
    "n_layers",
    "qkv_bias",
    "tie_weights",
)


# ---------------------------------------------------------------------------
# Config / environment helpers
# ---------------------------------------------------------------------------
def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT_DIR / p


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE lines from .env without overriding real env vars."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and not os.environ.get(key):
            os.environ[key] = value


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


# ---------------------------------------------------------------------------
# Hardware / model helpers
# ---------------------------------------------------------------------------
def select_device(cfg: dict) -> tuple[torch.device, int]:
    """Pick 2 GPUs, 1 GPU, or CPU. Returns ``(device, num_gpus)``."""
    requested = str(cfg.get("device", "auto")).lower().strip()
    max_gpus = int(cfg.get("num_gpus", 2))

    if requested == "cpu":
        return torch.device("cpu"), 0

    if not torch.cuda.is_available():
        logging.warning("CUDA is not available; using CPU.")
        return torch.device("cpu"), 0

    if requested.startswith("cuda"):
        if ":" in requested:
            index = int(requested.split(":", 1)[1])
            if index >= torch.cuda.device_count():
                logging.warning(
                    "Requested cuda:%d but only %d GPU(s) are visible; using cuda:0.",
                    index,
                    torch.cuda.device_count(),
                )
                index = 0
            return torch.device(f"cuda:{index}"), 1
        n = min(max_gpus, torch.cuda.device_count())
        if n <= 0:
            logging.warning("num_gpus <= 0 with device=cuda; using CPU.")
            return torch.device("cpu"), 0
        return torch.device("cuda:0"), n

    # auto
    n = min(max_gpus, torch.cuda.device_count())
    if n <= 0:
        return torch.device("cpu"), 0
    return torch.device("cuda:0"), n


def model_config_mismatches(saved: dict, current: dict) -> list[str]:
    mismatches = []
    for key in ARCH_KEYS:
        old = saved.get(key, True if key == "tie_weights" else None)
        new = current.get(key, True if key == "tie_weights" else None)
        if old != new:
            mismatches.append(f"{key}: checkpoint={old!r}, current={new!r}")
    return mismatches


def lr_for_step(step: int, cfg: dict) -> float:
    base_lr = float(cfg.get("learning_rate", 3e-4))
    min_lr = float(cfg.get("min_lr", base_lr * 0.1))
    warmup = int(cfg.get("warmup_steps", 0))
    max_steps = cfg.get("max_steps")

    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup

    if max_steps is not None:
        max_steps = int(max_steps)
        decay_steps = max_steps - warmup
        if decay_steps <= 0:
            return min_lr
        progress = min(max((step - warmup) / decay_steps, 0.0), 1.0)
        return min_lr + 0.5 * (base_lr - min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )

    return base_lr


def choose_amp_dtype(device: torch.device, cfg: dict) -> Optional[torch.dtype]:
    if device.type != "cuda" or not bool(cfg.get("amp", True)):
        return None

    requested = str(cfg.get("amp_dtype", "auto")).lower()
    if requested == "fp32":
        return None
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def make_grad_scaler(
    device: torch.device,
    use_amp: bool,
    amp_dtype: Optional[torch.dtype],
) -> torch.amp.GradScaler:
    enabled = bool(use_amp and amp_dtype == torch.float16)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    return torch.amp.GradScaler(device_type, enabled=enabled)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# HuggingFace checkpoint sync
# ---------------------------------------------------------------------------
def get_hf_settings(cfg: dict, disable: bool = False) -> Optional[dict]:
    hf = cfg.get("hf") or {}
    repo_id = str(hf.get("repo_id", "")).strip()
    token_env = str(hf.get("token_env", "HF_TOKEN")).strip()
    token = os.environ.get(token_env, "").strip()

    if disable:
        logging.info("HuggingFace checkpoint sync disabled by --no-hf.")
        return None
    if not repo_id:
        logging.warning(
            "HF repo_id is empty in configs/pretrain.yaml; "
            "remote checkpoint sync is disabled. Training will continue."
        )
        return None
    if not token:
        logging.warning(
            "HF token environment variable %r is not set (checked .env); "
            "remote checkpoint sync is disabled. Training will continue.",
            token_env,
        )
        return None

    out_dir = str(hf.get("out_dir", "out")).strip().strip("/")
    if not out_dir:
        out_dir = "out"

    return {
        "repo_id": repo_id,
        "token": token,
        "token_env": token_env,
        "filename": str(hf.get("filename", "minillm_latest.pt")),
        "out_dir": out_dir,
        "private": bool(hf.get("private", True)),
    }


def download_checkpoint(hf: dict, checkpoint_dir: Path) -> Optional[Path]:
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import (
            EntryNotFoundError,
            RepositoryNotFoundError,
        )
    except ImportError as exc:
        logging.warning(
            "huggingface_hub is not installed (%s); remote resume disabled.",
            exc,
        )
        return None

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = hf_hub_download(
            repo_id=hf["repo_id"],
            filename=hf["filename"],
            repo_type="model",
            token=hf["token"],
            local_dir=str(checkpoint_dir),
        )
        logging.info("Found remote checkpoint: %s", path)
        return Path(path)
    except EntryNotFoundError:
        logging.info(
            "No %r in HF repo %s; starting from scratch.",
            hf["filename"],
            hf["repo_id"],
        )
    except RepositoryNotFoundError:
        logging.warning(
            "HF repo %s was not found or is not accessible; starting from scratch.",
            hf["repo_id"],
        )
    except Exception as exc:
        logging.warning(
            "Could not download %r from %s: %s; starting from scratch.",
            hf["filename"],
            hf["repo_id"],
            exc,
        )
    return None


def upload_checkpoint(path: Path, hf: dict, step: int) -> bool:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        logging.warning(
            "huggingface_hub is not installed (%s); HF upload disabled.",
            exc,
        )
        return False

    try:
        api = HfApi(token=hf["token"])
        api.create_repo(
            repo_id=hf["repo_id"],
            repo_type="model",
            private=hf["private"],
            exist_ok=True,
        )
        size_mb = path.stat().st_size / (1024 * 1024)
        logging.info(
            "Uploading %.1f MB to %s/%s ...",
            size_mb,
            hf["repo_id"],
            hf["filename"],
        )
        # Uploading to the same path replaces the previous file in one commit.
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=hf["filename"],
            repo_id=hf["repo_id"],
            repo_type="model",
            commit_message=f"Update {hf['filename']} at step {step}",
        )
        logging.info(
            "Uploaded %s to %s (replaced previous file if it existed).",
            hf["filename"],
            hf["repo_id"],
        )
        return True
    except Exception as exc:
        logging.warning("HF upload failed: %s", exc)
        return False


def upload_outputs(output_dir: Path, hf: dict, step: int) -> bool:
    """Upload metrics.json and plots to the configured HF ``out/`` folder."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        logging.warning(
            "huggingface_hub is not installed (%s); HF output upload disabled.",
            exc,
        )
        return False

    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    if not files:
        logging.info("No output files to upload.")
        return True

    out_dir = str(hf.get("out_dir", "out")).strip().strip("/") or "out"
    try:
        api = HfApi(token=hf["token"])
        api.create_repo(
            repo_id=hf["repo_id"],
            repo_type="model",
            private=hf["private"],
            exist_ok=True,
        )
        logging.info(
            "Uploading %d output file(s) to %s/%s ...",
            len(files),
            hf["repo_id"],
            out_dir,
        )
        api.upload_folder(
            folder_path=str(output_dir),
            path_in_repo=out_dir,
            repo_id=hf["repo_id"],
            repo_type="model",
            commit_message=f"Update training metrics and plots at step {step}",
        )
        logging.info("Uploaded output files to %s/%s.", hf["repo_id"], out_dir)
        return True
    except Exception as exc:
        logging.warning("HF output upload failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    """Training loop with periodic local + HF checkpoints and resume."""

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

        checkpoint_dir = resolve_path(train_cfg.get("checkpoint_dir", "checkpoints"))
        filename = (hf_cfg or {}).get("filename", "minillm_latest.pt")
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / filename

        out_dir_name = str(train_cfg.get("out_dir", "out")).strip() or "out"
        self.out_dir = resolve_path(out_dir_name)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.out_dir / "metrics.json"

        self.interval_seconds = max(
            0.0,
            float(train_cfg.get("checkpoint_interval_minutes", 60)) * 60.0,
        )
        self.last_checkpoint_time = time.monotonic()

        self.step = 0
        self.tokens_seen = 0
        self.history: list[dict] = []
        self.last_loss: Optional[float] = None
        self.last_lr: Optional[float] = None
        self.last_metric_step = 0
        self.train_start_time: Optional[float] = None
        self.max_train_seconds: Optional[float] = None
        self.last_sample_text: Optional[str] = None
        self.nonfinite_streak = 0

        self.amp_dtype = choose_amp_dtype(device, train_cfg)
        self.use_amp = self.amp_dtype is not None
        self.scaler = make_grad_scaler(device, self.use_amp, self.amp_dtype)

    # -- checkpoint loading -------------------------------------------------
    def resume(self) -> bool:
        # Prefer the local checkpoint. Downloading the HF copy can overwrite a
        # good local checkpoint with a diverged (NaN) one.
        if self.checkpoint_path.exists():
            logging.info("Using local checkpoint: %s", self.checkpoint_path)
            if self._load_checkpoint(self.checkpoint_path):
                return True
            logging.warning(
                "Local checkpoint %s is unusable; trying remote.", self.checkpoint_path
            )

        if self.hf_cfg:
            path = download_checkpoint(self.hf_cfg, self.checkpoint_dir)
            if path is not None and self._load_checkpoint(path):
                return True

        logging.warning("No usable checkpoint found; starting training from scratch.")
        return False

    @staticmethod
    def _nonfinite_tensors(state: dict, limit: int = 5) -> list[str]:
        bad = []
        for name, value in state.items():
            if torch.is_tensor(value) and value.is_floating_point():
                if not bool(torch.isfinite(value).all().item()):
                    bad.append(name)
                    if len(bad) >= limit:
                        break
        return bad

    def _load_checkpoint(self, path: Path) -> bool:
        try:
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                # Older PyTorch versions do not support weights_only.
                ckpt = torch.load(path, map_location="cpu")
        except Exception as exc:
            logging.warning("Could not read checkpoint %s (%s).", path, exc)
            return False

        if not isinstance(ckpt, dict) or "model" not in ckpt:
            logging.warning("Checkpoint %s has an unexpected format.", path)
            return False

        saved_cfg = ckpt.get("model_config")
        if not isinstance(saved_cfg, dict):
            logging.warning("Checkpoint %s has no model_config.", path)
            return False

        mismatches = model_config_mismatches(saved_cfg, self.model_cfg)
        if mismatches:
            logging.warning(
                "Checkpoint model config does not match configs/model.yaml. "
                "Differences: %s",
                "; ".join(mismatches),
            )
            return False

        bad_tensors = self._nonfinite_tensors(ckpt["model"])
        if bad_tensors:
            logging.warning(
                "Checkpoint %s contains non-finite (NaN/Inf) weights in %s; "
                "refusing to resume from it. Delete the local and remote "
                "checkpoint, or resume an earlier finite one.",
                path,
                ", ".join(bad_tensors),
            )
            return False

        try:
            unwrap_model(self.model).load_state_dict(ckpt["model"], strict=True)
        except Exception as exc:
            logging.warning("Could not load model weights: %s", exc)
            return False

        optimizer_state = ckpt.get("optimizer")
        if optimizer_state and bool(self.train_cfg.get("save_optimizer", True)):
            try:
                self.optimizer.load_state_dict(optimizer_state)
                bad_optimizer = [
                    name
                    for state in self.optimizer.state.values()
                    for name, value in state.items()
                    if torch.is_tensor(value)
                    and value.is_floating_point()
                    and not bool(torch.isfinite(value).all().item())
                ]
                if bad_optimizer:
                    logging.warning(
                        "Optimizer state contains non-finite values; "
                        "starting it fresh."
                    )
                    self.optimizer.state.clear()
                for state in self.optimizer.state.values():
                    for key, value in state.items():
                        if torch.is_tensor(value):
                            state[key] = value.to(self.device)
            except Exception as exc:
                logging.warning("Could not restore optimizer state: %s", exc)

        if "scaler" in ckpt:
            try:
                self.scaler.load_state_dict(ckpt["scaler"])
            except Exception as exc:
                logging.warning("Could not restore grad scaler state: %s", exc)

        if "rng_state" in ckpt:
            try:
                torch.set_rng_state(ckpt["rng_state"])
            except Exception as exc:
                logging.warning("Could not restore CPU RNG state: %s", exc)

        cuda_rng = ckpt.get("cuda_rng_state_all")
        if cuda_rng and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(cuda_rng)
            except Exception as exc:
                logging.warning("Could not restore CUDA RNG state: %s", exc)

        self.step = int(ckpt.get("step", 0))
        self.tokens_seen = int(ckpt.get("tokens_seen", 0))
        self.last_checkpoint_time = time.monotonic()

        history = ckpt.get("metrics_history")
        if isinstance(history, list) and history:
            self.history = [dict(entry) for entry in history if isinstance(entry, dict)]
            self._sync_last_metric()
            logging.info("Restored %d metric point(s).", len(self.history))
        else:
            self._load_metrics_file()

        logging.info(
            "Resumed from %s at step %d (%d tokens).",
            path,
            self.step,
            self.tokens_seen,
        )
        return True

    # -- metrics / plots ----------------------------------------------------
    def _sync_last_metric(self) -> None:
        if not self.history:
            return
        last = self.history[-1]
        try:
            self.last_loss = float(last.get("loss"))
        except (TypeError, ValueError):
            self.last_loss = None
        try:
            self.last_lr = float(last.get("lr"))
        except (TypeError, ValueError):
            self.last_lr = None
        try:
            self.last_metric_step = int(last.get("step", 0))
        except (TypeError, ValueError):
            self.last_metric_step = 0

    def _load_metrics_file(self) -> None:
        if not self.metrics_path.exists():
            return
        try:
            entries = json.loads(self.metrics_path.read_text())
        except Exception as exc:
            logging.warning("Could not read local metrics %s: %s", self.metrics_path, exc)
            return
        if not isinstance(entries, list):
            return
        self.history = [dict(entry) for entry in entries if isinstance(entry, dict)]
        self._sync_last_metric()
        logging.info(
            "Loaded %d metric point(s) from %s.",
            len(self.history),
            self.metrics_path,
        )

    def _record_metric(self, loss_value: float, lr: float) -> None:
        """Append one loss/perplexity point, avoiding duplicates per step."""
        if self.history:
            try:
                if int(self.history[-1].get("step", -1)) == self.step:
                    return
            except (TypeError, ValueError):
                pass

        loss = float(loss_value)
        if math.isfinite(loss):
            perplexity = math.exp(min(loss, 50.0))
        else:
            perplexity = float("nan")

        self.history.append(
            {
                "step": int(self.step),
                "tokens_seen": int(self.tokens_seen),
                "loss": loss,
                "perplexity": perplexity,
                "lr": float(lr),
                "timestamp": time.time(),
            }
        )
        self.last_loss = loss
        self.last_lr = float(lr)
        self.last_metric_step = int(self.step)

    def _log_sample(self) -> Optional[str]:
        """Generate a short continuation of a fixed prompt for eyeballing."""
        prompt = str(self.train_cfg.get("sample_prompt", "Einstein was a")).strip()
        if not prompt:
            return None

        model = unwrap_model(self.model)
        vocab_size = int(
            getattr(model, "vocab_size", self.model_cfg.get("vocab_size", 0))
        )
        max_new_tokens = max(
            1, int(self.train_cfg.get("sample_max_new_tokens", 60))
        )
        temperature = float(self.train_cfg.get("sample_temperature", 0.8))
        top_k = self.train_cfg.get("sample_top_k", 50)
        top_k = None if top_k in (None, 0) else int(top_k)

        try:
            encoder = get_encoder()
            prompt_ids = encoder.encode(prompt)
        except Exception as exc:
            logging.warning("Could not tokenize sample prompt: %s", exc)
            return None
        if not prompt_ids:
            return None
        if max(prompt_ids) >= vocab_size:
            logging.warning(
                "Sample prompt %r uses token ids outside model vocab_size=%d; "
                "skipping sample.",
                prompt,
                vocab_size,
            )
            return None

        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        try:
            # fork_rng keeps sampling from changing dropout/training RNG.
            devices = (
                [self.device.index]
                if self.device.type == "cuda" and self.device.index is not None
                else []
            )
            with torch.random.fork_rng(devices=devices):
                out = model.generate(
                    idx,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    use_cache=True,
                )
            continuation = encoder.decode(out[0, len(prompt_ids) :].tolist())
        except Exception as exc:
            logging.warning("Sample generation failed at step %d: %s", self.step, exc)
            return None
        finally:
            model.clear_cache()

        text = " ".join((prompt + continuation).split())
        self.last_sample_text = text
        logging.info("sample @ step %d | %s", self.step, text[:400])
        return text

    def _plot_metrics(self) -> list[Path]:
        if len(self.history) < 2:
            logging.info("Not enough metric points for plots; skipping.")
            return []

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            logging.warning("matplotlib is not installed; skipping plots: %s", exc)
            return []

        points: list[tuple[float, float, float]] = []
        for entry in self.history:
            try:
                tokens = float(entry.get("tokens_seen", 0))
                loss = float(entry.get("loss"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(loss):
                continue
            try:
                perplexity = float(entry.get("perplexity"))
            except (TypeError, ValueError):
                perplexity = math.exp(min(loss, 50.0))
            points.append((tokens, loss, perplexity))

        if len(points) < 2:
            logging.info("Not enough finite metric points for plots; skipping.")
            return []

        points.sort(key=lambda item: item[0])
        if len(points) > 5000:
            stride = max(1, len(points) // 5000)
            sampled = points[::stride]
            if sampled[-1] != points[-1]:
                sampled.append(points[-1])
            points = sampled

        tokens = [item[0] for item in points]
        losses = [item[1] for item in points]
        perplexities = [
            (item[0], item[2])
            for item in points
            if math.isfinite(item[2]) and item[2] > 0
        ]

        def ema(values, alpha=0.2):
            smoothed = []
            last = None
            for value in values:
                last = value if last is None else alpha * value + (1 - alpha) * last
                smoothed.append(last)
            return smoothed

        loss_ema = ema(losses)

        files: list[Path] = []

        def save_plot(
            path: Path,
            title: str,
            ylabel: str,
            y_values: list[float],
            x_values: list[float] = tokens,
            smooth_values: Optional[list[float]] = None,
            log_scale: bool = False,
        ) -> None:
            fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=120)
            marker = "o" if len(x_values) <= 80 else None
            if smooth_values is not None:
                ax.plot(
                    x_values,
                    y_values,
                    color="#93c5fd",
                    linewidth=1.0,
                    alpha=0.7,
                    label="raw",
                )
                ax.plot(
                    x_values,
                    smooth_values,
                    color="#2563eb",
                    linewidth=2.0,
                    marker=marker,
                    markersize=3,
                    label="smoothed",
                )
                ax.legend(frameon=False)
            else:
                ax.plot(
                    x_values,
                    y_values,
                    color="#2563eb",
                    linewidth=1.6,
                    marker=marker,
                    markersize=3,
                )
            ax.set_xlabel("Tokens seen")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            if log_scale:
                ax.set_yscale("log")
            fig.tight_layout()
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            files.append(path)

        loss_path = self.out_dir / "token_seen_vs_loss.png"
        save_plot(
            loss_path,
            "Training loss vs. tokens seen",
            "Cross-entropy loss",
            losses,
            smooth_values=loss_ema,
        )

        if perplexities:
            ppl_x = []
            ppl_y = []
            ppl_smooth = []
            for index, (token, perplexity) in enumerate(
                (item[0], item[2]) for item in points
            ):
                if math.isfinite(perplexity) and perplexity > 0:
                    ppl_x.append(token)
                    ppl_y.append(perplexity)
                    ppl_smooth.append(math.exp(min(loss_ema[index], 50.0)))
            ppl_path = self.out_dir / "token_seen_vs_perplexity.png"
            save_plot(
                ppl_path,
                "Training perplexity vs. tokens seen",
                "Perplexity (log scale)",
                ppl_y,
                x_values=ppl_x,
                smooth_values=ppl_smooth,
                log_scale=True,
            )

            curves_path = self.out_dir / "training_curves.png"
            fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), dpi=120)
            marker = "o" if len(tokens) <= 80 else None
            axes[0].plot(
                tokens,
                losses,
                color="#93c5fd",
                linewidth=1.0,
                alpha=0.7,
                label="raw",
            )
            axes[0].plot(
                tokens,
                loss_ema,
                color="#2563eb",
                linewidth=2.0,
                marker=marker,
                markersize=3,
                label="smoothed",
            )
            axes[0].set_xlabel("Tokens seen")
            axes[0].set_ylabel("Cross-entropy loss")
            axes[0].set_title("Loss")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(frameon=False)

            axes[1].plot(
                ppl_x,
                ppl_y,
                color="#fca5a5",
                linewidth=1.0,
                alpha=0.7,
                label="raw",
            )
            axes[1].plot(
                ppl_x,
                ppl_smooth,
                color="#dc2626",
                linewidth=2.0,
                marker=marker,
                markersize=3,
                label="smoothed",
            )
            axes[1].set_xlabel("Tokens seen")
            axes[1].set_ylabel("Perplexity (log scale)")
            axes[1].set_title("Perplexity")
            axes[1].set_yscale("log")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(frameon=False)

            fig.suptitle("Training curves")
            fig.tight_layout()
            fig.savefig(curves_path, bbox_inches="tight")
            plt.close(fig)
            files.append(curves_path)

        return files

    def write_outputs(self) -> list[Path]:
        """Write metrics.json and plots under ``out_dir`` (never raises)."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        files: list[Path] = []

        try:
            self.metrics_path.write_text(json.dumps(self.history, indent=2))
            files.append(self.metrics_path)
        except Exception as exc:
            logging.warning("Could not write metrics %s: %s", self.metrics_path, exc)

        try:
            files.extend(self._plot_metrics())
        except Exception as exc:
            logging.warning("Could not write training plots: %s", exc)

        return files

    # -- checkpoint saving --------------------------------------------------
    def save_checkpoint(self) -> bool:
        model_state = unwrap_model(self.model).state_dict()
        bad_tensors = self._nonfinite_tensors(model_state)
        if bad_tensors:
            logging.warning(
                "Refusing to save checkpoint at step %d: model weights are "
                "non-finite in %s. Training diverged; restart from a finite "
                "checkpoint or lower learning_rate.",
                self.step,
                ", ".join(bad_tensors),
            )
            return False

        # Make sure the checkpoint includes a metric point for the latest step,
        # even if that step was not a multiple of log_interval.
        if self.last_loss is not None and self.last_metric_step != self.step:
            self._record_metric(self.last_loss, self.last_lr or 0.0)

        state = {
            "format_version": 1,
            "model": model_state,
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "model_config": self.model_cfg,
            "train_config": self.train_cfg,
            "metrics_history": self.history,
            "rng_state": torch.get_rng_state(),
        }
        if bool(self.train_cfg.get("save_optimizer", True)):
            state["optimizer"] = self.optimizer.state_dict()
        if self.use_amp:
            state["scaler"] = self.scaler.state_dict()
        if torch.cuda.is_available():
            state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

        tmp_path = self.checkpoint_path.with_name(self.checkpoint_path.name + ".tmp")
        try:
            torch.save(state, tmp_path)
            os.replace(tmp_path, self.checkpoint_path)
        except Exception as exc:
            logging.warning("Could not save local checkpoint: %s", exc)
            return False

        output_files = self.write_outputs()
        size_mb = self.checkpoint_path.stat().st_size / (1024 * 1024)
        logging.info(
            "Saved checkpoint %s (%.1f MB) at step %d (%s tokens); "
            "wrote %d output file(s).",
            self.checkpoint_path,
            size_mb,
            self.step,
            f"{self.tokens_seen:,}",
            len(output_files),
        )

        if self.hf_cfg:
            upload_checkpoint(self.checkpoint_path, self.hf_cfg, self.step)
            upload_outputs(self.out_dir, self.hf_cfg, self.step)

        self.last_checkpoint_time = time.monotonic()
        return True

    # -- main loop ----------------------------------------------------------
    def train(self, batches: Iterable[tuple[np.ndarray, np.ndarray]]) -> None:
        cfg = self.train_cfg
        max_steps = cfg.get("max_steps")
        if max_steps is not None:
            max_steps = int(max_steps)
        grad_accum = max(1, int(cfg.get("grad_accum_steps", 1)))
        log_first_steps = max(0, int(cfg.get("log_first_steps", 10) or 0))
        log_interval = max(1, int(cfg.get("log_interval", 100) or 100))
        sample_interval = max(0, int(cfg.get("sample_interval", 1000) or 0))
        grad_clip = float(cfg.get("grad_clip", 1.0))
        show_progress = bool(cfg.get("progress", True))

        max_hours = cfg.get("max_hours")
        if max_hours is not None:
            max_hours = float(max_hours)
            if max_hours <= 0:
                raise ValueError("max_hours must be > 0")
            self.max_train_seconds = max_hours * 3600.0
        else:
            self.max_train_seconds = None
        self.train_start_time = time.monotonic()
        if self.max_train_seconds is not None:
            logging.info(
                "Time limit: %.2f h; training will stop and save automatically.",
                max_hours,
            )

        batches = iter(batches)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            total=max_steps,
            initial=self.step,
            desc="pretrain",
            unit="step",
            dynamic_ncols=True,
            disable=not show_progress,
        )
        last_log_time = time.monotonic()
        last_log_tokens = self.tokens_seen
        self.nonfinite_streak = 0
        failed = False

        try:
            while max_steps is None or self.step < max_steps:
                micro_losses = []
                step_tokens = 0
                data_exhausted = False

                for _ in range(grad_accum):
                    try:
                        x_np, y_np = next(batches)
                    except StopIteration:
                        data_exhausted = True
                        break

                    x = torch.from_numpy(x_np).long().to(
                        self.device, non_blocking=True
                    )
                    y = torch.from_numpy(y_np).long().to(
                        self.device, non_blocking=True
                    )
                    step_tokens += int(x.numel())

                    if self.use_amp:
                        autocast_ctx = torch.amp.autocast(
                            device_type=self.device.type,
                            dtype=self.amp_dtype,
                        )
                    else:
                        autocast_ctx = contextlib.nullcontext()

                    with autocast_ctx:
                        logits = self.model(x)
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            y.reshape(-1),
                        )

                    micro_losses.append(loss.detach())
                    self.scaler.scale(loss / grad_accum).backward()

                if not micro_losses:
                    break

                loss_value = float(torch.stack(micro_losses).mean().item())
                loss_finite = math.isfinite(loss_value)

                self.scaler.unscale_(self.optimizer)
                grad_norm_value: Optional[float] = None
                grads_finite = True
                if grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), grad_clip
                    )
                    grad_norm_value = float(grad_norm)
                    grads_finite = math.isfinite(grad_norm_value)

                lr = lr_for_step(self.step, cfg)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr

                if loss_finite and (grads_finite or self.scaler.is_enabled()):
                    # An enabled GradScaler skips the optimizer step itself
                    # when unscale_ observed inf/nan gradients.
                    self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                self.step += 1
                self.tokens_seen += step_tokens
                updated = loss_finite and grads_finite

                if updated:
                    self.nonfinite_streak = 0
                    self.last_loss = loss_value
                    self.last_lr = float(lr)
                else:
                    self.nonfinite_streak += 1
                    logging.warning(
                        "step %d: skipped optimizer update because the loss or "
                        "gradients were non-finite (loss=%s, grad_norm=%s); "
                        "%d consecutive skip(s).",
                        self.step,
                        loss_value,
                        "n/a" if grad_norm_value is None else grad_norm_value,
                        self.nonfinite_streak,
                    )
                pbar.update(1)

                now = time.monotonic()
                elapsed = max(now - last_log_time, 1e-9)
                tok_s = (self.tokens_seen - last_log_tokens) / elapsed
                ppl = (
                    math.exp(min(loss_value, 50.0))
                    if loss_finite
                    else float("nan")
                )
                if not math.isfinite(ppl):
                    ppl_text = "nan"
                elif ppl < 1e6:
                    ppl_text = f"{ppl:,.1f}"
                else:
                    ppl_text = f"{ppl:.2e}"
                # Update the bar every step so tokens_seen and loss visibly
                # move; only record/log metrics at the configured intervals.
                postfix = {
                    "loss": f"{loss_value:.4f}",
                    "ppl": ppl_text,
                    "tokens": f"{self.tokens_seen:,}",
                    "lr": f"{lr:.2e}",
                    "tok_s": f"{tok_s:,.1f}",
                }
                if self.max_train_seconds is not None:
                    remaining = max(
                        0.0,
                        self.max_train_seconds
                        - (now - self.train_start_time),
                    )
                    postfix["remaining_h"] = f"{remaining / 3600:.2f}"
                pbar.set_postfix(postfix)
                # Log every step only during the first few steps, then every
                # log_interval steps, so long runs do not flood the output.
                if (
                    self.step <= log_first_steps
                    or self.step % log_interval == 0
                ) and loss_finite:
                    self._record_metric(loss_value, lr)
                    logging.info(
                        "step %d | tokens_seen %s | loss %.4f | ppl %s | "
                        "lr %.2e | tok/s %.1f",
                        self.step,
                        f"{self.tokens_seen:,}",
                        loss_value,
                        ppl_text,
                        lr,
                        tok_s,
                    )
                    last_log_time = now
                    last_log_tokens = self.tokens_seen

                if sample_interval > 0 and self.step % sample_interval == 0:
                    self._log_sample()

                if self.nonfinite_streak >= 5:
                    raise RuntimeError(
                        "Training produced non-finite loss/gradients for 5 "
                        "consecutive steps; stopping without overwriting the "
                        "checkpoint. Restart from a finite checkpoint and/or "
                        "lower learning_rate in configs/pretrain.yaml."
                    )

                time_up = (
                    self.max_train_seconds is not None
                    and (now - self.train_start_time) >= self.max_train_seconds
                )

                if (
                    not time_up
                    and self.interval_seconds > 0
                    and (now - self.last_checkpoint_time) >= self.interval_seconds
                ):
                    self.save_checkpoint()

                if data_exhausted:
                    logging.info("Data stream exhausted at step %d.", self.step)
                    break

                if time_up:
                    logging.info(
                        "Reached --max-hours time limit at step %d; "
                        "stopping and saving checkpoint.",
                        self.step,
                    )
                    break

                if max_steps is not None and self.step >= max_steps:
                    break

        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt: saving checkpoint before exit.")
            raise
        except Exception:
            # Do not save a checkpoint after a crash (e.g. persistent NaN).
            failed = True
            raise
        finally:
            pbar.close()
            if self.step > 0 and not failed:
                self.save_checkpoint()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_training(
    pretrain_cfg: dict,
    model_cfg: dict,
    no_hf: bool = False,
    new_run: bool = False,
) -> None:
    load_env_file(ROOT_DIR / ".env")

    seed = int(pretrain_cfg.get("seed", 1337))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device, num_gpus = select_device(pretrain_cfg)
    logging.info("Device: %s | GPUs used: %d", device, num_gpus)

    model = GPT(model_cfg)
    n_params = count_parameters(model)
    model.to(device)

    if num_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
        logging.info("Using nn.DataParallel across %d GPUs.", num_gpus)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(pretrain_cfg.get("learning_rate", 3e-4)),
        betas=tuple(pretrain_cfg.get("betas", [0.9, 0.95])),
        weight_decay=float(pretrain_cfg.get("weight_decay", 0.1)),
    )

    hf_cfg = get_hf_settings(pretrain_cfg, disable=no_hf)

    data_cfg = pretrain_cfg.get("data") or {}
    block_size = data_cfg.get("block_size") or model_cfg["context_length"]
    block_size = int(block_size)
    if block_size > int(model_cfg["context_length"]):
        raise ValueError(
            f"data.block_size={block_size} exceeds model context_length="
            f"{model_cfg['context_length']}"
        )

    batch_size = int(pretrain_cfg.get("batch_size", 8))
    grad_accum = max(1, int(pretrain_cfg.get("grad_accum_steps", 1)))

    logging.info("Model parameters: %s", f"{n_params:,}")
    logging.info(
        "batch_size=%d block_size=%d grad_accum=%d -> %s tokens per optimizer step",
        batch_size,
        block_size,
        grad_accum,
        f"{batch_size * block_size * grad_accum:,}",
    )
    logging.info(
        "Checkpoint: every %.1f min to %s",
        float(pretrain_cfg.get("checkpoint_interval_minutes", 60)),
        resolve_path(pretrain_cfg.get("checkpoint_dir", "checkpoints")),
    )
    logging.info(
        "HF sync: %s",
        "enabled" if hf_cfg else "disabled",
    )
    logging.info(
        "Local metrics/plots: %s",
        resolve_path(pretrain_cfg.get("out_dir", "out")),
    )
    logging.info(
        "Logging: first %d steps every step, then every %d steps; "
        "sample generation every %d steps.",
        int(pretrain_cfg.get("log_first_steps", 10) or 0),
        int(pretrain_cfg.get("log_interval", 100) or 100),
        int(pretrain_cfg.get("sample_interval", 1000) or 0),
    )
    if hf_cfg:
        logging.info(
            "HF metrics/plots: %s/%s",
            hf_cfg["repo_id"],
            hf_cfg["out_dir"],
        )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        model_cfg=model_cfg,
        train_cfg=pretrain_cfg,
        hf_cfg=hf_cfg,
    )
    if new_run or no_hf:
        reason = "--new" if new_run and not no_hf else "--no-hf"
        logging.warning(
            "Starting from a random model (%s): local and HF checkpoints "
            "are ignored.",
            reason,
        )
    else:
        trainer.resume()

    logging.info(
        "Streaming %s/%s with shuffle_buffer=%d.",
        data_cfg.get("dataset_name", "Skylion007/openwebtext"),
        data_cfg.get("split", "train"),
        int(data_cfg.get("shuffle_buffer", 1000)),
    )
    stream = OpenWebTextTokenStream(
        block_size=block_size,
        dataset_name=str(
            data_cfg.get("dataset_name", "Skylion007/openwebtext")
        ),
        split=str(data_cfg.get("split", "train")),
        shuffle_buffer=int(data_cfg.get("shuffle_buffer", 1000)),
        seed=int(pretrain_cfg.get("seed", 1337)),
        max_docs=data_cfg.get("max_docs"),
        min_chars=int(data_cfg.get("min_chars", 1)),
    )
    batches = stream.batches(batch_size=batch_size, drop_last=True)

    trainer.train(batches)
    logging.info("Training finished.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretrain the mini GPT on streaming OpenWebText."
    )
    parser.add_argument(
        "--pretrain-config",
        type=Path,
        default=DEFAULT_PRETRAIN_CONFIG,
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--max-hours",
        "--hours",
        type=float,
        default=None,
        dest="max_hours",
        help="Stop after this many wall-clock hours (e.g. 6 or 12). "
             "The final checkpoint is still saved/uploaded.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument(
        "--new",
        action="store_true",
        help="Ignore local and HF checkpoints and start from a random model. "
             "Training still saves locally and uploads to HF.",
    )
    parser.add_argument(
        "--no-hf",
        action="store_true",
        help="Start from a random model (ignore local and HF checkpoints) "
             "and disable all HF download/upload.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    setup_logging()
    args = build_arg_parser().parse_args(argv)

    pretrain_cfg = load_yaml(args.pretrain_config)
    model_cfg = load_yaml(args.model_config)

    if args.max_steps is not None:
        pretrain_cfg["max_steps"] = args.max_steps
    if args.max_hours is not None:
        pretrain_cfg["max_hours"] = args.max_hours
    if args.batch_size is not None:
        pretrain_cfg["batch_size"] = args.batch_size
    if args.device is not None:
        pretrain_cfg["device"] = args.device
    if args.num_gpus is not None:
        pretrain_cfg["num_gpus"] = args.num_gpus
    if args.checkpoint_dir is not None:
        pretrain_cfg["checkpoint_dir"] = args.checkpoint_dir

    try:
        run_training(
            pretrain_cfg,
            model_cfg,
            no_hf=args.no_hf,
            new_run=args.new,
        )
    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")


if __name__ == "__main__":
    main()
