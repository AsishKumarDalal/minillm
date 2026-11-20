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
* Optional token limit via ``--budget 15000000``: training stops at that many
  tokens and the bar shows progress toward the budget. Without a limit the
  bar shows tokens seen / the estimated token count of one OpenWebText pass
  (``estimated_total_tokens`` in the config).
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

    # Stop after 15M tokens; the bar counts toward that budget
    .venv/bin/python -m pretrain.train --budget 15000000

    # Testing only: save/upload every 10 seconds instead of 60 minutes
    .venv/bin/python -m pretrain.train --checkpoint-seconds 10

    # Notebook-safe run: start in a separate session, then stop cleanly.
    # A plain ``!python -m pretrain.train`` is force-killed by Jupyter
    # 0.2s after Ctrl+C, before the final upload can finish.
    .venv/bin/python -m pretrain.train --check-hf
    .venv/bin/python -m pretrain.train --detach --new
    .venv/bin/python -m pretrain.train --status
    .venv/bin/python -m pretrain.train --stop
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import io
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


class _TqdmLoggingHandler(logging.StreamHandler):
    """Print log lines above the active tqdm bar instead of through it."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            tqdm.write(message, file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging() -> None:
    handler = _TqdmLoggingHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    # Keep the log readable: these libraries log every HTTP range request or
    # backend probe at INFO level.
    for name in (
        "httpx",
        "httpcore",
        "urllib3",
        "datasets",
        "huggingface_hub",
        "numexpr",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


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


def model_config_mismatches(
    saved: dict,
    current: dict,
    keys: tuple[str, ...] = ARCH_KEYS,
) -> list[str]:
    mismatches = []
    for key in keys:
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


@contextlib.contextmanager
def defer_signals(*signums: int, raise_pending: bool = True):
    """Defer SIGINT/SIGTERM instead of raising while artifacts sync.

    A 100M-parameter checkpoint plus optimizer is ~1.2 GB, so the final save
    and upload can take minutes. A second Ctrl+C (or a notebook stop request)
    during that window must not abort the upload and leave the Hub with only
    the model or only the ``out/`` files. Signals received inside this block
    are remembered; after the block they either raise ``KeyboardInterrupt``
    (so a signal during a periodic save still stops training) or are logged
    and swallowed (``raise_pending=False``, used by the final exit sync).
    """
    installed: dict[int, object] = {}
    pending: list[int] = []

    def _remember(signum, _frame):
        pending.append(signum)
        logging.warning(
            "Received signal %d while finishing the final save/upload; "
            "deferring it until the sync completes.",
            signum,
        )

    for signum in signums:
        try:
            installed[signum] = signal.getsignal(signum)
            signal.signal(signum, _remember)
        except (ValueError, OSError):
            # Signal handlers can only be installed in the main thread.
            pass

    try:
        yield
    finally:
        for signum, previous in installed.items():
            try:
                signal.signal(signum, previous)
            except (ValueError, OSError):
                pass
        if pending:
            logging.info(
                "Final sync finished; handled deferred signal(s): %s.",
                ", ".join(str(signum) for signum in pending),
            )
            if raise_pending:
                raise KeyboardInterrupt(
                    "deferred signal(s): "
                    + ", ".join(str(signum) for signum in pending)
                )


def atomic_torch_save(state: dict, path: Path) -> None:
    """Write a checkpoint atomically, removing the temp file on any failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp_path, "wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()


def atomic_write_text(path: Path, text: str) -> None:
    """Write a small file atomically so an upload never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp_path, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()


def atomic_savefig(fig, path: Path, **kwargs) -> None:
    """Save a matplotlib figure atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        fig.savefig(tmp_path, format=path.suffix.lstrip(".") or "png", **kwargs)
        os.replace(tmp_path, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def format_compact(value: float) -> str:
    """Format a large number as 1.2k / 3.4M / 5.6B for a narrow tqdm bar."""
    value = float(value)
    if not math.isfinite(value):
        return "nan"
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{value / 1e3:.1f}k"
    return f"{value:.1f}"


def style_progress_bar(pbar) -> None:
    """Make the notebook bar green and keep all stats visible beside it."""
    try:
        desc_widget, bar_widget, stats_widget = pbar.container.children
        # tqdm.notebook gives the bar flex=2, which squeezes the right-hand
        # stats until "tok"/"ppl"/"loss" are clipped. Use the full row, keep
        # the bar itself narrow, and let the stats take all remaining width
        # without shrinking below their content.
        pbar.container.layout.width = "100%"
        desc_widget.layout.flex = "0 1 auto"
        desc_widget.layout.overflow = "visible"
        bar_widget.layout.flex = "0 0 15%"
        bar_widget.layout.width = "15%"
        bar_widget.layout.min_width = "60px"
        stats_widget.layout.flex = "1 1 auto"
        stats_widget.layout.min_width = "max-content"
        stats_widget.layout.overflow = "visible"
        bar_widget.bar_style = "success"
    except Exception:
        # Terminal tqdm has no widget container; colour="green" handles it.
        pass


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
            "remote checkpoint sync is disabled."
        )
        return None
    if not token:
        logging.warning(
            "HF token environment variable %r is not set (checked .env); "
            "remote checkpoint sync is disabled.",
            token_env,
        )
        return None

    out_dir = str(hf.get("out_dir", "out")).strip().strip("/")
    if not out_dir:
        out_dir = "out"

    filename = str(hf.get("filename", "minillm_latest.pt")).strip().lstrip("/")
    path_in_repo = str(hf.get("path_in_repo", filename)).strip().lstrip("/")
    if not path_in_repo:
        path_in_repo = filename

    return {
        "repo_id": repo_id,
        "token": token,
        "token_env": token_env,
        "filename": filename,
        "path_in_repo": path_in_repo,
        "out_dir": out_dir,
        "private": bool(hf.get("private", True)),
        "upload_retries": max(1, int(hf.get("upload_retries", 3))),
        "upload_retry_backoff": max(0.0, float(hf.get("upload_retry_backoff", 2.0))),
    }


def _is_permanent_upload_error(exc: Exception) -> bool:
    """True for auth/not-found errors that retrying cannot fix."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in (400, 401, 403, 404)


def _retry_upload(description: str, action, hf: dict) -> bool:
    """Run one Hub upload with bounded retries for transient failures."""
    retries = max(1, int(hf.get("upload_retries", 3)))
    backoff = max(0.0, float(hf.get("upload_retry_backoff", 2.0)))

    for attempt in range(1, retries + 1):
        try:
            action()
            return True
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if _is_permanent_upload_error(exc) or attempt >= retries:
                logging.warning(
                    "%s failed after %d attempt(s): %s", description, attempt, exc
                )
                return False
            delay = backoff * (2 ** (attempt - 1))
            logging.warning(
                "%s failed (attempt %d/%d): %s; retrying in %.1fs.",
                description,
                attempt,
                retries,
                exc,
                delay,
            )
            time.sleep(delay)

    return False


def download_checkpoint(hf: dict, checkpoint_dir: Path) -> Optional[Path]:
    try:
        import shutil

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
    local_name = str(hf.get("filename", "minillm_latest.pt")).strip().lstrip("/")
    remote_name = str(hf.get("path_in_repo", local_name)).strip().lstrip("/")
    target = checkpoint_dir / local_name
    try:
        path = Path(
            hf_hub_download(
                repo_id=hf["repo_id"],
                filename=remote_name,
                repo_type="model",
                token=hf.get("token") or None,
                local_dir=str(checkpoint_dir / "_hf_downloads"),
            )
        )
        if path.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target.unlink()
            shutil.move(str(path), str(target))
        logging.info("Found remote checkpoint: %s -> %s", remote_name, target)
        return target
    except EntryNotFoundError:
        logging.info(
            "No %r in HF repo %s; starting from scratch.",
            remote_name,
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
            remote_name,
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

    if not path.exists():
        logging.warning("Checkpoint %s does not exist; upload skipped.", path)
        return False

    path_in_repo = str(hf.get("path_in_repo", hf.get("filename", "minillm_latest.pt")))
    path_in_repo = path_in_repo.strip().lstrip("/")
    if not path_in_repo:
        logging.warning("hf.path_in_repo is empty; checkpoint upload skipped.")
        return False

    size_mb = path.stat().st_size / (1024 * 1024)

    def _upload() -> None:
        api = HfApi(token=hf["token"])
        api.create_repo(
            repo_id=hf["repo_id"],
            repo_type="model",
            private=hf["private"],
            exist_ok=True,
        )
        logging.info(
            "Uploading %.1f MB to %s/%s ...",
            size_mb,
            hf["repo_id"],
            path_in_repo,
        )
        # Uploading to the same path replaces the previous file in one commit.
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path_in_repo,
            repo_id=hf["repo_id"],
            repo_type="model",
            commit_message=f"Update {path_in_repo} at step {step}",
        )

    if not _retry_upload(f"HF checkpoint upload {path_in_repo}", _upload, hf):
        return False

    logging.info(
        "Uploaded %s to %s (replaced previous file if it existed).",
        path_in_repo,
        hf["repo_id"],
    )
    return True


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

    out_dir = str(hf.get("out_dir", "out")).strip().strip("/").lstrip("/") or "out"

    def _upload() -> None:
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

    if not _retry_upload(f"HF output upload to {out_dir}", _upload, hf):
        return False

    logging.info("Uploaded output files to %s/%s.", hf["repo_id"], out_dir)
    return True


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
        self.arch_keys = ARCH_KEYS

        checkpoint_dir = resolve_path(train_cfg.get("checkpoint_dir", "checkpoints"))
        filename = (hf_cfg or train_cfg.get("hf") or {}).get(
            "filename", "minillm_latest.pt"
        )
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
        # Used by the final sync so every exit path (Ctrl+C, SIGTERM, crash,
        # normal stop) ends with the newest finite checkpoint on HF. The model
        # and the out/ files are tracked separately: losing one upload must
        # never mark the other one as already uploaded.
        self.last_saved_step = 0
        self.last_uploaded_step = 0
        self.last_outputs_uploaded_step = 0
        self._remove_stale_checkpoint_temps()

        self.amp_dtype = choose_amp_dtype(device, train_cfg)
        self.use_amp = self.amp_dtype is not None
        self.scaler = make_grad_scaler(device, self.use_amp, self.amp_dtype)

    # -- model/loss hooks ---------------------------------------------------
    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Default dense next-token loss. MoE trainers override this."""
        logits = self.model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.reshape(-1)
        )

    def extra_stats(self) -> str:
        """Optional text appended to the tqdm postfix and step log."""
        return ""

    def extra_checkpoint_state(self) -> dict:
        """Optional fields merged into the checkpoint payload."""
        return {}

    def _remove_stale_checkpoint_temps(self) -> None:
        """Remove temp checkpoints left by an interrupted save."""
        for stale in self.checkpoint_dir.glob(f"{self.checkpoint_path.name}.tmp*"):
            try:
                stale.unlink()
                logging.info("Removed stale checkpoint temp file: %s", stale)
            except OSError as exc:
                logging.warning(
                    "Could not remove stale checkpoint temp %s: %s", stale, exc
                )

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

        mismatches = model_config_mismatches(
            saved_cfg, self.model_cfg, self.arch_keys
        )
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

    def _log_sample(self, pbar=None) -> Optional[str]:
        """Generate a short continuation of a fixed prompt for eyeballing."""
        prompt = str(
            self.train_cfg.get("sample_prompt", "Albert Einstein is a")
        ).strip()
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
            with torch.no_grad(), torch.random.fork_rng(devices=devices):
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
        message = f"sample @ step {self.step:,} | {text[:400]}"
        if pbar is not None:
            # Write above the live bar instead of logging a new line, which
            # would make Jupyter look like it restarted the progress bar.
            pbar.write(message)
        else:
            logging.info("%s", message)
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
            atomic_savefig(fig, path, bbox_inches="tight")
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
            atomic_savefig(fig, curves_path, bbox_inches="tight")
            plt.close(fig)
            files.append(curves_path)

        return files

    def write_outputs(self) -> list[Path]:
        """Write metrics.json and plots under ``out_dir`` (never raises)."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        files: list[Path] = []

        try:
            atomic_write_text(self.metrics_path, json.dumps(self.history, indent=2))
            files.append(self.metrics_path)
        except Exception as exc:
            logging.warning("Could not write metrics %s: %s", self.metrics_path, exc)

        try:
            files.extend(self._plot_metrics())
        except Exception as exc:
            logging.warning("Could not write training plots: %s", exc)

        return files

    # -- checkpoint saving --------------------------------------------------
    def sync_to_hf(
        self, *, checkpoint: bool = True, outputs: bool = True
    ) -> None:
        """Upload artifacts independently and record each result.

        The ``out/`` files are small, so they go first. If the large model
        upload is aborted, the plots and metrics still land. A failed model
        upload never marks the outputs as uploaded (and vice versa), which is
        what allows :meth:`final_sync` to retry only the missing half.
        """
        if not self.hf_cfg:
            return
        if outputs and self.last_outputs_uploaded_step != self.step:
            if upload_outputs(self.out_dir, self.hf_cfg, self.step):
                self.last_outputs_uploaded_step = self.step
        if checkpoint and self.last_uploaded_step != self.step:
            if upload_checkpoint(self.checkpoint_path, self.hf_cfg, self.step):
                self.last_uploaded_step = self.step

    def save_checkpoint(self, upload: bool = True) -> bool:
        self._remove_stale_checkpoint_temps()
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
        state.update(self.extra_checkpoint_state())

        try:
            atomic_torch_save(state, self.checkpoint_path)
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

        self.last_saved_step = self.step
        if upload and self.hf_cfg:
            self.sync_to_hf()

        self.last_checkpoint_time = time.monotonic()
        return True

    def final_sync(self) -> None:
        """Save/upload the newest finite checkpoint on every exit path.

        ``train()`` already saves on a clean stop, so this usually only
        re-uploads if the periodic upload failed. On Ctrl+C, SIGTERM, or a
        crash it performs the save itself. ``save_checkpoint()`` refuses
        non-finite weights, so a diverged model is never uploaded.

        SIGINT/SIGTERM are deferred for the whole sync: pressing Ctrl+C again
        while the 1.2 GB checkpoint uploads must not abort it or skip the
        ``out/`` upload. The checkpoint and ``out/`` uploads are retried
        independently, so a failure of one never prevents the other.
        """
        if self.step <= 0:
            return

        with defer_signals(
            signal.SIGINT, signal.SIGTERM, raise_pending=False
        ):
            saved = self.last_saved_step == self.step
            if not saved:
                saved = self.save_checkpoint(upload=False)

            if not saved:
                # The model diverged (non-finite weights), but the metrics
                # and plots for the latest steps are still worth uploading.
                self.write_outputs()

            if self.hf_cfg:
                self.sync_to_hf(checkpoint=saved, outputs=True)
                logging.info(
                    "Final sync at step %d: local checkpoint %s, model %s, "
                    "out/ %s.",
                    self.step,
                    "saved" if saved else "skipped (non-finite weights)",
                    "uploaded"
                    if self.last_uploaded_step == self.step
                    else "not uploaded",
                    "uploaded"
                    if self.last_outputs_uploaded_step == self.step
                    else "not uploaded",
                )
            else:
                logging.info(
                    "Final sync at step %d: local checkpoint %s (HF disabled).",
                    self.step,
                    "saved" if saved else "skipped (non-finite weights)",
                )

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
        token_budget = cfg.get("token_budget")
        if token_budget is not None:
            token_budget = int(token_budget)
            if token_budget <= 0:
                raise ValueError("token_budget must be > 0")
        estimated_total = cfg.get("estimated_total_tokens")
        if estimated_total is not None:
            estimated_total = int(estimated_total)
            if estimated_total <= 0:
                estimated_total = None

        self.train_start_time = time.monotonic()
        if self.max_train_seconds is not None:
            logging.info(
                "Time limit: %.2f h; training will stop and save automatically.",
                max_hours,
            )
        if token_budget is not None:
            logging.info(
                "Token budget: %s tokens; training stops at that point.",
                f"{token_budget:,}",
            )

        batches = iter(batches)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Choose what the tqdm percentage measures (first match wins):
        #   * --hours / max_hours       -> percent of the wall-clock budget
        #   * --budget / token_budget   -> percent of that many tokens
        #   * max_steps                 -> percent of requested optimizer steps
        #   * estimated_total_tokens    -> tokens seen / one OpenWebText pass
        #   * otherwise                 -> time until the next checkpoint save
        now = time.monotonic()
        progress_time_based = False
        progress_token_based = False
        if self.max_train_seconds is not None:
            progress_time_based = True
            progress_total: Optional[float] = self.max_train_seconds
            progress_initial: float = min(
                progress_total,
                max(0.0, now - self.train_start_time),
            )
            progress_unit = "s"
            progress_desc = "pretrain (time budget)"
        elif token_budget is not None:
            progress_token_based = True
            progress_total = float(token_budget)
            progress_initial = min(progress_total, float(self.tokens_seen))
            progress_unit = "token"
            progress_desc = "pretrain (token budget)"
        elif max_steps is not None:
            progress_total = float(max_steps)
            progress_initial = min(progress_total, float(self.step))
            progress_unit = "step"
            progress_desc = "pretrain"
        elif estimated_total is not None:
            progress_token_based = True
            progress_total = float(estimated_total)
            progress_initial = min(progress_total, float(self.tokens_seen))
            progress_unit = "token"
            # Keep the widget label short so the four live stats on the right
            # are not clipped. The meaning is logged once at startup.
            progress_desc = "pretrain"
            logging.info(
                "Progress bar: tokens seen / estimated %s tokens "
                "(one full OpenWebText pass).",
                f"{estimated_total:,}",
            )
        elif self.interval_seconds > 0:
            progress_time_based = True
            progress_total = self.interval_seconds
            progress_initial = min(
                progress_total,
                max(0.0, now - self.last_checkpoint_time),
            )
            progress_unit = "s"
            progress_desc = "pretrain (to next save)"
        else:
            progress_total = None
            progress_initial = float(self.step)
            progress_unit = "step"
            progress_desc = "pretrain"

        pbar = tqdm(
            total=progress_total,
            initial=progress_initial,
            desc=progress_desc,
            unit=progress_unit,
            dynamic_ncols=True,
            disable=not show_progress,
            postfix="",
            colour="green",
            # A clean percentage bar; the default time-based bar would print
            # awkward values like "5.386/3600.0 s" and "1.00 s/s".
            # Use {bar} (not {bar:20}): tqdm.notebook only replaces the exact
            # {bar} token with its widget. A width-qualified {bar:20} renders
            # an extra text bar inside the stats text, pushing t/s, tok, ppl
            # and loss out of view in Jupyter.
            bar_format="{desc}: {percentage:3.0f}%|{bar}|{postfix}",
        )
        if show_progress:
            style_progress_bar(pbar)
        pbar_last_time = now
        last_rate_time = time.monotonic()
        last_rate_tokens = self.tokens_seen
        self.nonfinite_streak = 0
        failed = False

        try:
            while (
                (max_steps is None or self.step < max_steps)
                and (token_budget is None or self.tokens_seen < token_budget)
            ):
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
                        loss = self.compute_loss(x, y)

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
                now = time.monotonic()
                if progress_time_based:
                    pbar.update(max(0.0, now - pbar_last_time))
                    pbar_last_time = now
                elif progress_token_based:
                    pbar.update(step_tokens)
                else:
                    pbar.update(1)

                elapsed = max(now - last_rate_time, 1e-9)
                tok_s = (self.tokens_seen - last_rate_tokens) / elapsed
                last_rate_time = now
                last_rate_tokens = self.tokens_seen
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
                # Keep the four requested metrics visible on one continuous
                # bar. Use compact numbers so the notebook widget does not
                # clip "ppl"/"loss" on the right side.
                postfix = (
                    f"#{self.step:,} | {format_compact(tok_s)} t/s | "
                    f"{format_compact(self.tokens_seen)} tok | "
                    f"ppl {format_compact(ppl)} | loss {loss_value:.3f}"
                    f"{self.extra_stats()}"
                )
                pbar.set_postfix_str(postfix)
                # Record metrics for the plots every log_interval steps, but
                # only print full INFO lines for the first log_first_steps.
                # After that the live tqdm suffix is the status display.
                if loss_finite and (
                    self.step <= log_first_steps
                    or self.step % log_interval == 0
                ):
                    self._record_metric(loss_value, lr)
                    if self.step <= log_first_steps:
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

                if sample_interval > 0 and self.step % sample_interval == 0:
                    self._log_sample(pbar)
                    # Do not let sample generation depress the next t/s.
                    last_rate_time = time.monotonic()
                    last_rate_tokens = self.tokens_seen

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
                budget_reached = (
                    token_budget is not None
                    and self.tokens_seen >= token_budget
                )

                if (
                    not time_up
                    and self.interval_seconds > 0
                    and (now - self.last_checkpoint_time) >= self.interval_seconds
                ):
                    if self.save_checkpoint():
                        last_rate_time = time.monotonic()
                        last_rate_tokens = self.tokens_seen
                        if progress_time_based and (
                            self.max_train_seconds is None
                        ):
                            # start a fresh "time until next save" percentage
                            pbar.reset(total=progress_total)
                            pbar.set_description(progress_desc)
                            pbar_last_time = time.monotonic()

                if data_exhausted:
                    logging.info("Data stream exhausted at step %d.", self.step)
                    break

                if budget_reached:
                    logging.info(
                        "Reached --budget (%s tokens) at step %d (%s tokens); "
                        "stopping and saving checkpoint.",
                        f"{token_budget:,}",
                        self.step,
                        f"{self.tokens_seen:,}",
                    )
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
                # This is the Ctrl+C/exit path. Defer further interrupts
                # until the local save and both HF uploads have finished.
                with defer_signals(signal.SIGINT, signal.SIGTERM):
                    self.save_checkpoint()


# ---------------------------------------------------------------------------
# Background runner (Jupyter-safe Ctrl+C)
# ---------------------------------------------------------------------------
def _pid_file_path(pretrain_cfg: dict) -> Path:
    checkpoint_dir = resolve_path(pretrain_cfg.get("checkpoint_dir", "checkpoints"))
    return checkpoint_dir / "train.pid"


def _read_pid(path: Path) -> Optional[int]:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _pid_is_trainer(pid: int) -> bool:
    """True when ``pid`` exists and looks like this training process."""
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
        # /proc is Linux-only; if it is unavailable assume the PID is live.
        return True
    return b"pretrain.train" in cmdline or b"pretrain/train.py" in cmdline


def _latest_log() -> Optional[Path]:
    log_dir = ROOT_DIR / "logs"
    try:
        logs = sorted(
            log_dir.glob("pretrain-*.log"),
            key=lambda path: path.stat().st_mtime,
        )
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


def check_hf(pretrain_cfg: dict, no_hf: bool = False) -> int:
    """Check token, repo, and write access without starting training."""
    load_env_file(ROOT_DIR / ".env")
    hf = pretrain_cfg.get("hf") or {}
    repo_id = str(hf.get("repo_id", "")).strip()
    token_env = str(hf.get("token_env", "HF_TOKEN")).strip()
    token = os.environ.get(token_env, "").strip()

    print(f"HF repo: {repo_id or '(not set)'}")
    print(f"Token: {token_env} is {'set' if token else 'NOT set'}")
    if no_hf:
        print("Result: DISABLED (--no-hf was passed).")
        return 1
    if not repo_id:
        print("Result: DISABLED (hf.repo_id is empty in configs/pretrain.yaml).")
        return 1
    if not token:
        print(f"Result: DISABLED ({token_env} is not set).")
        print("Set it in the notebook, or put it in .env, then re-run --check-hf.")
        return 1

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        who = api.whoami()
        user = who.get("name") or who.get("user") or "unknown"
        print(f"Authenticated as: {user}")

        try:
            info = api.repo_info(repo_id=repo_id, repo_type="model")
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 404:
                print(
                    f"Repo: {repo_id} does not exist yet; "
                    "the first upload will create it."
                )
            else:
                raise
        else:
            print(
                f"Repo: {repo_id} exists "
                f"(last modified {info.last_modified})."
            )

        try:
            api.auth_check(repo_id=repo_id, repo_type="model", write=True)
            print("Write access: OK")
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 404:
                print(
                    "Write access: repo does not exist yet; cannot verify "
                    "until the first upload."
                )
            else:
                raise

        print("Result: HF upload is configured. Training uploads will run.")
        return 0
    except Exception as exc:
        print(f"Result: FAILED: {exc}")
        print("Fix the token or repo permissions; training would not upload.")
        return 1


def show_status(pretrain_cfg: dict) -> int:
    checkpoint_dir = resolve_path(pretrain_cfg.get("checkpoint_dir", "checkpoints"))
    pid_file = _pid_file_path(pretrain_cfg)
    pid = _read_pid(pid_file)
    running = bool(pid and _pid_is_trainer(pid))

    print(f"Training: {'RUNNING (pid %d)' % pid if running else 'not running'}")
    if pid and not running:
        print(f"Stale PID file: {pid_file}")
    if running:
        print(f"PID file: {pid_file}")

    log_path = _latest_log()
    if log_path is not None:
        print(f"Log: {log_path}")
        tail = _tail(log_path, 20)
        if tail:
            print("--- last log lines ---")
            print(tail)

    filename = str(
        (pretrain_cfg.get("hf") or {}).get("filename", "minillm_latest.pt")
    )
    checkpoint = checkpoint_dir / filename
    if checkpoint.exists():
        stat = checkpoint.stat()
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
        print(f"Checkpoint: {checkpoint} ({stat.st_size / 1024**2:.1f} MB, {when})")
    else:
        print(f"Checkpoint: none at {checkpoint}")

    metrics = resolve_path(pretrain_cfg.get("out_dir", "out")) / "metrics.json"
    if metrics.exists():
        stat = metrics.stat()
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
        print(f"Metrics: {metrics} ({when})")
    return 0


def start_detached_training(
    args: argparse.Namespace,
    raw_argv: list[str],
    pretrain_cfg: dict,
) -> int:
    """Start training in its own session so notebook Ctrl+C cannot kill it."""
    checkpoint_dir = resolve_path(pretrain_cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pid_file = checkpoint_dir / "train.pid"

    existing = _read_pid(pid_file)
    if existing and _pid_is_trainer(existing):
        print(f"Training is already running (pid {existing}).")
        print("Use --status to inspect it or --stop to stop it.")
        return 1
    if pid_file.exists():
        try:
            pid_file.unlink()
        except OSError:
            pass

    if args.log_file is not None:
        log_path = Path(args.log_file).expanduser()
        if not log_path.is_absolute():
            log_path = ROOT_DIR / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_dir = ROOT_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / time.strftime("pretrain-%Y%m%d-%H%M%S.log")

    child_argv = [arg for arg in raw_argv if arg != "--detach"]
    env = os.environ.copy()
    env["MINILLM_PID_FILE"] = str(pid_file)
    env["PYTHONUNBUFFERED"] = "1"

    with open(log_path, "ab", buffering=0) as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-m", "pretrain.train", *child_argv],
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

    # Catch immediate failures (bad config/import) before returning. Imports
    # can take a few seconds, so give the child a short grace period.
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        returncode = proc.poll()
        if returncode is not None:
            try:
                pid_file.unlink()
            except OSError:
                pass
            print(f"Training exited immediately with code {returncode}.")
            print(f"Log: {log_path}")
            tail = _tail(log_path, 30)
            if tail:
                print("--- last log lines ---")
                print(tail)
            return 1
        time.sleep(0.1)

    print(f"Training started in the background: pid {proc.pid}")
    print(f"Log: {log_path}")
    print(f"PID file: {pid_file}")
    print("Check:  python -m pretrain.train --status")
    print("Stop:   python -m pretrain.train --stop")
    print(
        "The trainer runs in its own session, so Ctrl+C in this notebook "
        "will not kill it; --stop asks it to save and upload before exit."
    )
    return 0


def stop_training(pretrain_cfg: dict, timeout: float = 10.0) -> int:
    """Ask a detached trainer to stop; it saves/uploads in its signal handler."""
    pid_file = _pid_file_path(pretrain_cfg)
    pid = _read_pid(pid_file)
    if not pid or not _pid_is_trainer(pid):
        print(f"No running training found (PID file: {pid_file}).")
        if pid_file.exists():
            print("The PID file is stale; removing it.")
            try:
                pid_file.unlink()
            except OSError:
                pass
        return 1

    print(f"Sending SIGTERM to training pid {pid}.")
    print(
        "It will stop after the current step, save locally, and upload "
        "the model and out/ files before exiting."
    )
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print("The process already exited.")
        return 0

    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline:
        if not _pid_is_trainer(pid):
            print("Training exited.")
            return 0
        time.sleep(0.2)

    print("Training is still running (the final save/upload can take minutes).")
    print("It is detached, so you can close this cell; it will finish on its own.")
    log_path = _latest_log()
    if log_path is not None:
        print(f"Watch: {log_path}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_training(
    pretrain_cfg: dict,
    model_cfg: dict,
    no_hf: bool = False,
    new_run: bool = False,
    require_hf: bool = False,
) -> None:
    load_env_file(ROOT_DIR / ".env")

    pid_file = os.environ.get("MINILLM_PID_FILE", "").strip()
    if pid_file:
        atexit.register(_remove_pid_file_if_ours, Path(pid_file))
    elif os.environ.get("JPY_PARENT_PID"):
        logging.warning(
            "This looks like a notebook `!python ...` run. Jupyter "
            "force-kills that child 0.2s after Ctrl+C, before the final "
            "upload can finish. Use `python -m pretrain.train --detach ...` "
            "and stop it with `python -m pretrain.train --stop`."
        )

    seed = int(pretrain_cfg.get("seed", 1337))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device, num_gpus = select_device(pretrain_cfg)
    logging.info("Device: %s | GPUs used: %d", device, num_gpus)

    hf_cfg = get_hf_settings(pretrain_cfg, disable=no_hf)
    if require_hf and hf_cfg is None:
        raise SystemExit(
            "HF sync was required (--require-hf), but it is not configured. "
            "Set HF_TOKEN and hf.repo_id, then run --check-hf. No training "
            "was started."
        )

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
    interval_minutes = float(pretrain_cfg.get("checkpoint_interval_minutes", 60))
    if interval_minutes < 1.0:
        interval_text = f"{interval_minutes * 60.0:.0f} s"
    else:
        interval_text = f"{interval_minutes:.1f} min"
    logging.info(
        "Checkpoint: every %s to %s",
        interval_text,
        resolve_path(pretrain_cfg.get("checkpoint_dir", "checkpoints")),
    )
    if hf_cfg:
        logging.info(
            "HF sync: ENABLED | repo=%s | model=%s | out=%s | token_env=%s",
            hf_cfg["repo_id"],
            hf_cfg["filename"],
            hf_cfg["out_dir"],
            hf_cfg["token_env"],
        )
    else:
        logging.warning(
            "HF sync: DISABLED | nothing will be uploaded to HuggingFace. "
            "Set HF_TOKEN and hf.repo_id, then test with --check-hf."
        )
    logging.info(
        "Local metrics/plots: %s",
        resolve_path(pretrain_cfg.get("out_dir", "out")),
    )
    logging.info(
        "Logging: step lines for the first %d steps only; metrics/plot "
        "points every %d steps; sample generation every %d steps.",
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

    def _terminate(signum, _frame):
        # Turn "kill"/SIGTERM into the same clean-exit path as Ctrl+C so the
        # final checkpoint is still saved and uploaded.
        raise KeyboardInterrupt(f"signal {signum}")

    previous_sigterm = None
    try:
        previous_sigterm = signal.signal(signal.SIGTERM, _terminate)
    except (ValueError, OSError):
        pass  # signal handlers can only be installed in the main thread

    try:
        trainer.train(batches)
    except KeyboardInterrupt:
        logging.info("Interrupted; saving the final checkpoint before exit.")
        raise
    except Exception:
        logging.exception(
            "Training crashed; attempting a final finite checkpoint save."
        )
        raise
    finally:
        # Always end with the newest finite checkpoint and out/ files on HF,
        # unless --no-hf. final_sync() defers SIGINT/SIGTERM, so restore the
        # SIGTERM handler only after the sync has completed.
        try:
            trainer.final_sync()
        finally:
            if previous_sigterm is not None:
                try:
                    signal.signal(signal.SIGTERM, previous_sigterm)
                except (ValueError, OSError):
                    pass
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
        "--budget",
        "--token-budget",
        type=int,
        default=None,
        dest="token_budget",
        help="Train until this many tokens have been seen (e.g. 15000000). "
             "The bar shows tokens seen / budget. Without this and --hours, "
             "the bar uses the estimated token count of one OpenWebText pass.",
    )
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
        "--checkpoint-seconds",
        type=float,
        default=None,
        help="Testing override: seconds between checkpoint saves/uploads. "
             "Sets checkpoint_interval_minutes = seconds/60 (e.g. 10 for a "
             "10-second test). Production default is 60 minutes.",
    )
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--detach",
        action="store_true",
        help="Start training in its own background session and return "
             "immediately. Use this from a notebook: a plain `!python ...` "
             "run is force-killed by Jupyter 0.2s after Ctrl+C, before the "
             "final upload can finish.",
    )
    mode.add_argument(
        "--stop",
        action="store_true",
        help="Ask a detached training run to stop cleanly (SIGTERM). It "
             "still saves and uploads the model and out/ files.",
    )
    mode.add_argument(
        "--status",
        action="store_true",
        help="Show whether a detached run is active, plus the newest log, "
             "checkpoint, and metrics timestamps.",
    )
    mode.add_argument(
        "--check-hf",
        action="store_true",
        help="Check HF_TOKEN, repo access, and write permission, then exit "
             "without starting training.",
    )
    parser.add_argument(
        "--require-hf",
        action="store_true",
        help="Refuse to start training when HF sync is not configured, "
             "instead of silently training without uploads.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Log path for --detach (default: logs/pretrain-<timestamp>.log).",
    )
    parser.add_argument(
        "--stop-timeout",
        type=float,
        default=10.0,
        help="Seconds --stop waits for the trainer to exit before returning "
             "(default: 10; the upload may continue in the background).",
    )
    return parser


def apply_cli_overrides(pretrain_cfg: dict, args: argparse.Namespace) -> None:
    if args.max_steps is not None:
        pretrain_cfg["max_steps"] = args.max_steps
    if args.token_budget is not None:
        pretrain_cfg["token_budget"] = args.token_budget
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
    if args.checkpoint_seconds is not None:
        if args.checkpoint_seconds <= 0:
            raise ValueError("--checkpoint-seconds must be greater than zero")
        pretrain_cfg["checkpoint_interval_minutes"] = args.checkpoint_seconds / 60.0


def main(argv: Optional[list[str]] = None) -> None:
    setup_logging()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_arg_parser().parse_args(raw_argv)

    pretrain_cfg = load_yaml(args.pretrain_config)
    model_cfg = load_yaml(args.model_config)
    apply_cli_overrides(pretrain_cfg, args)
    load_env_file(ROOT_DIR / ".env")

    if args.check_hf:
        raise SystemExit(check_hf(pretrain_cfg, no_hf=args.no_hf))
    if args.status:
        raise SystemExit(show_status(pretrain_cfg))
    if args.stop:
        raise SystemExit(stop_training(pretrain_cfg, timeout=args.stop_timeout))
    if args.detach:
        raise SystemExit(start_detached_training(args, raw_argv, pretrain_cfg))

    try:
        run_training(
            pretrain_cfg,
            model_cfg,
            no_hf=args.no_hf,
            new_run=args.new,
            require_hf=args.require_hf,
        )
    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")


if __name__ == "__main__":
    main()
