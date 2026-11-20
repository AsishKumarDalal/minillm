"""Logit distillation of our mini GPT from a pretrained HuggingFace teacher.

The student is the GPT defined by ``configs/model.yaml``. It is loaded from
``sddfhy546y/minillm`` (or a local checkpoint) and trained on the usual
streaming OpenWebText batches. At every step both models see the same token
ids; the teacher is frozen and only the student is optimized.

Loss
----
For labels ``y`` and logits ``s`` (student) / ``t`` (teacher)::

    ce  = cross_entropy(s, y)
    kl  = KL(softmax(t / T) || log_softmax(s / T)) * T**2
    loss = (1 - alpha) * ce + alpha * kl

``alpha`` and ``temperature`` live in ``configs/distill.yaml``. Both models
use the GPT-2 BPE vocabulary, so their logits line up token-for-token.

Usage
-----
::

    python -m distill.distill
    python -m distill.distill --teacher hf-internal-testing/tiny-random-gpt2 --alpha 0.4
    python -m distill.distill --detach --upload
    python -m distill.distill --status
    python -m distill.distill --stop
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
import yaml
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.model import GPT  # noqa: E402
from pretrain.data import OpenWebTextTokenStream, get_encoder  # noqa: E402
from pretrain.train import (  # noqa: E402
    atomic_savefig,
    atomic_torch_save,
    atomic_write_text,
    choose_amp_dtype,
    count_parameters,
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

DEFAULT_DISTILL_CONFIG = ROOT_DIR / "configs" / "distill.yaml"
DEFAULT_MODEL_CONFIG = ROOT_DIR / "configs" / "model.yaml"


# ---------------------------------------------------------------------------
# Config / model loading helpers
# ---------------------------------------------------------------------------
def _token_from_env(token_env: str) -> str:
    return os.environ.get(token_env, "").strip()


def _dtype_from_name(name: str, device: torch.device) -> Optional[torch.dtype]:
    """Map a config dtype string to a torch dtype used for the teacher."""
    requested = str(name or "auto").lower().strip()
    if requested == "fp32":
        return torch.float32
    if requested == "fp16":
        return torch.float16
    if requested == "bf16":
        return torch.bfloat16
    if requested != "auto":
        raise ValueError(f"unknown dtype {name!r}; use auto|bf16|fp16|fp32")
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _parse_device(value, fallback: torch.device) -> torch.device:
    if value is None or str(value).strip() == "":
        return fallback
    text = str(value).strip()
    if text.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA requested for teacher but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(text)


def load_student_model_config(cfg: dict) -> dict:
    path = resolve_path(cfg.get("model_config", "configs/model.yaml"))
    model_cfg = load_yaml(Path(path))
    if not isinstance(model_cfg, dict) or "vocab_size" not in model_cfg:
        raise SystemExit(f"Invalid student model config: {path}")
    return model_cfg


def load_teacher(
    cfg: dict,
    student_cfg: dict,
    student_device: torch.device,
    block_size: int,
):
    """Download a pretrained causal LM and freeze it for logit distillation."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is required for distillation. Install it with "
            "`pip install transformers` (or install requirements.txt)."
        ) from exc

    teacher_cfg = cfg.get("teacher") or {}
    name = str(teacher_cfg.get("name", "openai-community/gpt2-xl")).strip()
    if not name:
        raise SystemExit("teacher.name is empty in configs/distill.yaml")
    token = _token_from_env(str(teacher_cfg.get("token_env", "HF_TOKEN")))

    logging.info("Loading teacher %s from HuggingFace ...", name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(name, token=token or None)
        model = AutoModelForCausalLM.from_pretrained(name, token=token or None)
    except Exception as exc:
        raise SystemExit(
            f"Could not download teacher {name!r} from HuggingFace: {exc}\n"
            "Check the model id and internet access. On Kaggle, turn "
            "Internet ON."
        ) from exc

    teacher_vocab = int(getattr(model.config, "vocab_size", 0) or 0)
    student_vocab = int(student_cfg["vocab_size"])
    if teacher_vocab != student_vocab:
        raise SystemExit(
            f"Teacher vocab ({teacher_vocab}) does not match student vocab "
            f"({student_vocab}). Logit distillation needs identical token "
            "ids, so use a GPT-2-family model (gpt2, gpt2-medium, gpt2-large, "
            "gpt2-xl) or another checkpoint with the GPT-2 tokenizer. "
            "Qwen/Llama/Mistral-style models are not compatible with this "
            "script. Change teacher.name or configs/model.yaml."
        )

    teacher_context = int(
        teacher_cfg.get("context_length")
        or getattr(model.config, "n_positions", 0)
        or 0
    )
    if teacher_context and block_size > teacher_context:
        raise SystemExit(
            f"data.block_size={block_size} exceeds teacher context "
            f"{teacher_context}. Lower block_size or choose a longer teacher."
        )

    device = _parse_device(teacher_cfg.get("device"), student_device)
    dtype = _dtype_from_name(teacher_cfg.get("dtype", "auto"), device)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    model.requires_grad_(False)

    logging.info(
        "Teacher ready: %s | vocab=%d | context=%s | device=%s | dtype=%s",
        name,
        teacher_vocab,
        teacher_context or "unknown",
        device,
        dtype,
    )
    return model, tokenizer, device, dtype


def load_student_model(
    cfg: dict,
    student_cfg: dict,
    device: torch.device,
    checkpoint_dir: Path,
):
    """Load our GPT from a local checkpoint or from the configured HF repo."""
    student_section = cfg.get("student") or {}
    source = str(student_section.get("source", "hf")).strip().lower()
    local_path = resolve_path(
        student_section.get("local_path", "checkpoints/minillm_latest.pt")
    )
    token_env = str(student_section.get("token_env", "HF_TOKEN"))
    prefer_local = bool(student_section.get("prefer_local", True))

    path: Optional[Path] = None
    if source == "local":
        path = local_path
        if not path.exists():
            raise SystemExit(f"student.source=local but {path} does not exist.")
    elif source == "hf":
        if prefer_local and local_path.exists():
            path = local_path
        else:
            hf = {
                "repo_id": str(student_section.get("repo_id", "")).strip(),
                "filename": str(
                    student_section.get("filename", "minillm_latest.pt")
                ).strip(),
                "token": _token_from_env(token_env),
            }
            if not hf["repo_id"]:
                raise SystemExit("student.repo_id is empty in configs/distill.yaml")
            path = download_checkpoint(hf, checkpoint_dir)
        if path is None:
            raise SystemExit(
                "No student checkpoint found. Put it at "
                f"{local_path}, or set HF_TOKEN so it can be downloaded "
                f"from {student_section.get('repo_id')}."
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
    mismatches = model_config_mismatches(saved_cfg, student_cfg)
    if mismatches:
        raise SystemExit(
            "Student checkpoint architecture does not match "
            f"configs/model.yaml: {'; '.join(mismatches)}"
        )

    model = GPT(student_cfg).to(device)
    try:
        model.load_state_dict(state["model"], strict=True)
    except Exception as exc:
        raise SystemExit(f"Could not load student weights from {path}: {exc}") from exc
    model.train()
    logging.info("Student loaded from %s (%s params).", path, f"{count_parameters(model):,}")
    return model, path


def prepare_student(
    cfg: dict,
    student_cfg: dict,
    device: torch.device,
    checkpoint_dir: Path,
    new_run: bool = False,
    full_new: bool = False,
) -> tuple[nn.Module, Optional[dict], str]:
    """Choose the student weights and return ``(model, resume_state, source)``.

    - ``full_new``: random initialization, ignore every checkpoint.
    - ``new_run``: ignore previous distilled checkpoint, load the original
      student (``minillm_latest.pt``).
    - default: resume from local/HF ``distilled_model.pt`` when present,
      otherwise load the original student.
    """
    if full_new:
        student = GPT(student_cfg).to(device)
        student.train()
        logging.info(
            "Full new run: student randomly initialized; no checkpoint loaded."
        )
        return student, None, "random init"

    if not new_run:
        resumed = load_resume_state(cfg, student_cfg, device, checkpoint_dir)
        if resumed is not None:
            student, state = resumed
            return student, state, "distilled checkpoint"

    student, _ = load_student_model(cfg, student_cfg, device, checkpoint_dir)
    return student, None, "original student checkpoint"


def load_resume_state(
    cfg: dict,
    student_cfg: dict,
    device: torch.device,
    checkpoint_dir: Path,
) -> Optional[tuple[nn.Module, dict]]:
    """Load a previous distilled checkpoint, or return None if none exists.

    With no ``--new`` flag, training continues from the newest
    ``distilled_model.pt``: first the local copy, then the HF copy. ``--new``
    skips this entirely and starts from the original student checkpoint.
    """
    hf_section = cfg.get("hf") or {}
    filename = str(hf_section.get("filename", "distilled_model.pt")).strip()
    local_path = checkpoint_dir / filename
    token_env = str(hf_section.get("token_env", "HF_TOKEN"))
    repo_id = str(
        hf_section.get("repo_id")
        or (cfg.get("student") or {}).get("repo_id", "")
    ).strip()

    path: Optional[Path] = local_path if local_path.exists() else None
    if path is None and repo_id:
        hf = {
            "repo_id": repo_id,
            "filename": filename,
            "token": _token_from_env(token_env),
        }
        path = download_checkpoint(hf, checkpoint_dir)
    if path is None:
        return None

    try:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
    except Exception as exc:
        logging.warning("Could not read distilled checkpoint %s: %s", path, exc)
        return None

    if not isinstance(state, dict) or "model" not in state:
        logging.warning("Distilled checkpoint %s has no model state dict.", path)
        return None
    saved_cfg = state.get("model_config")
    if not isinstance(saved_cfg, dict):
        logging.warning("Distilled checkpoint %s has no model_config.", path)
        return None
    mismatches = model_config_mismatches(saved_cfg, student_cfg)
    if mismatches:
        logging.warning(
            "Distilled checkpoint %s architecture mismatch: %s; starting "
            "from the original student instead.",
            path,
            "; ".join(mismatches),
        )
        return None

    model = GPT(student_cfg).to(device)
    try:
        model.load_state_dict(state["model"], strict=True)
    except Exception as exc:
        logging.warning("Could not load distilled weights from %s: %s", path, exc)
        return None
    model.train()
    logging.info(
        "Resuming distillation from %s at step %d (%s tokens).",
        path,
        int(state.get("step", 0)),
        f"{int(state.get('tokens_seen', 0)):,}",
    )
    return model, state


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    temperature: float = 1.0,
    ignore_index: int = -100,
):
    """Return ``(total, ce, kl)`` for one logit-distillation batch.

    ``alpha`` is the weight of the KL term; ``(1-alpha)`` weights the normal
    cross-entropy term. Temperature softens both logits before the KL; the
    KL is multiplied by ``T**2`` so gradients stay comparable to CE.
    """
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student/teacher logit shapes differ: {tuple(student_logits.shape)} "
            f"vs {tuple(teacher_logits.shape)}"
        )

    vocab = student_logits.size(-1)
    student = student_logits.reshape(-1, vocab).float()
    teacher = teacher_logits.reshape(-1, vocab).float()
    targets = labels.reshape(-1)
    mask = targets != ignore_index
    if not bool(mask.any()):
        zero = student.sum() * 0.0
        return zero, zero, zero

    student = student[mask]
    teacher = teacher[mask]
    targets = targets[mask]

    ce = F.cross_entropy(student, targets)
    log_student = F.log_softmax(student / temperature, dim=-1)
    teacher_probs = F.softmax(teacher / temperature, dim=-1)
    kl = F.kl_div(log_student, teacher_probs, reduction="batchmean")
    kl = kl * (temperature**2)

    total = (1.0 - float(alpha)) * ce + float(alpha) * kl
    return total, ce, kl


def _teacher_logits(
    teacher: nn.Module,
    x: torch.Tensor,
    teacher_device: torch.device,
    teacher_dtype: Optional[torch.dtype],
    autocast_ctx,
) -> torch.Tensor:
    """Run the frozen teacher and return logits on the student's device."""
    teacher_x = x.to(teacher_device)
    del autocast_ctx  # the teacher uses its own dtype, not the student's AMP
    with torch.no_grad():
        if teacher_device.type == "cuda" and teacher_dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            ctx = torch.amp.autocast(device_type="cuda", dtype=teacher_dtype)
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            output = teacher(input_ids=teacher_x)
            logits = output.logits if hasattr(output, "logits") else output[0]
    return logits.to(x.device)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _plot_series(out_dir: Path, tokens: list[float], values: list[float],
                 filename: str, title: str, ylabel: str) -> Optional[Path]:
    """Write one line plot atomically and return its path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=120)
    ax.plot(tokens, values, color="#2563eb", linewidth=1.8, marker="o", markersize=2.5)
    ax.set_xlabel("Tokens seen")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / filename
    atomic_savefig(fig, path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_distillation_plots(history: list[dict], out_dir: Path) -> list[Path]:
    """Write loss/perplexity/CE/KL plots into ``distilled_outs``.

    Perplexity is derived from the cross-entropy term only; the total loss
    also contains the KL term, so exp(total) is not a perplexity.
    """
    if len(history) < 2:
        return []
    tokens = [float(entry.get("tokens_seen", 0)) for entry in history]
    losses = [float(entry.get("loss", float("nan"))) for entry in history]
    ces = [float(entry.get("ce", float("nan"))) for entry in history]
    kls = [float(entry.get("kl", float("nan"))) for entry in history]
    ppls = [
        math.exp(min(ce, 50.0)) if math.isfinite(ce) and ce > 0 else float("nan")
        for ce in ces
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    specs = (
        ("token_seen_vs_loss.png", "Total distillation loss", "Loss", losses),
        ("token_seen_vs_perplexity.png", "Perplexity (from CE)", "Perplexity", ppls),
        ("token_seen_vs_ce.png", "Cross-entropy loss", "CE loss", ces),
        ("token_seen_vs_kl.png", "KL divergence to teacher", "KL loss", kls),
    )
    if any("val_loss" in entry for entry in history):
        val_points = [
            (float(entry.get("tokens_seen", 0)), float(entry["val_loss"]))
            for entry in history
            if entry.get("val_loss") is not None
        ]
        if len(val_points) >= 2:
            specs = specs + (
                (
                    "token_seen_vs_val_loss.png",
                    "Validation distillation loss",
                    "Val loss",
                    [point[1] for point in val_points],
                    [point[0] for point in val_points],
                ),
            )
    if any("aux" in entry for entry in history):
        aux_points = [
            (float(entry.get("tokens_seen", 0)), float(entry["aux"]))
            for entry in history
            if entry.get("aux") is not None
        ]
        if len(aux_points) >= 2:
            specs = specs + (
                (
                    "token_seen_vs_router_aux.png",
                    "Router auxiliary loss",
                    "Aux loss",
                    [point[1] for point in aux_points],
                    [point[0] for point in aux_points],
                ),
            )
    for filename, title, ylabel, values in specs:
        try:
            path = _plot_series(out_dir, tokens, values, filename, title, ylabel)
            if path is not None:
                files.append(path)
        except Exception as exc:
            logging.warning("Could not write %s: %s", filename, exc)
    return files


# ---------------------------------------------------------------------------
# Distiller
# ---------------------------------------------------------------------------
class Distiller:
    """Train the student with CE +KL logit distillation."""

    def __init__(
        self,
        model: nn.Module,
        teacher: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        teacher_device: torch.device,
        teacher_dtype: Optional[torch.dtype],
        student_cfg: dict,
        train_cfg: dict,
        hf_cfg: Optional[dict] = None,
    ):
        self.model = model
        self.teacher = teacher
        self.optimizer = optimizer
        self.device = device
        self.teacher_device = teacher_device
        self.teacher_dtype = teacher_dtype
        self.student_cfg = student_cfg
        self.train_cfg = train_cfg
        self.hf_cfg = hf_cfg

        self.checkpoint_dir = resolve_path(
            train_cfg.get("checkpoint_dir", "distill_checkpoints")
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        filename = (hf_cfg or train_cfg.get("hf") or {}).get(
            "filename", "distilled_model.pt"
        )
        self.checkpoint_path = self.checkpoint_dir / filename

        self.out_dir = resolve_path(train_cfg.get("out_dir", "distilled_outs"))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.out_dir / "metrics.json"

        self.alpha = float(train_cfg.get("alpha", 0.3))
        self.temperature = float(train_cfg.get("temperature", 1.0))
        self.interval_seconds = max(
            0.0,
            float(train_cfg.get("checkpoint_interval_minutes", 60)) * 60.0,
        )
        self.last_checkpoint_time = time.monotonic()

        self.step = 0
        self.tokens_seen = 0
        self.history: list[dict] = []
        self.last_loss: Optional[float] = None
        self.last_ce: Optional[float] = None
        self.last_kl: Optional[float] = None
        self.last_aux: Optional[float] = None
        self.last_sample_text: Optional[str] = None
        self.last_saved_step = 0
        self.last_uploaded_step = 0
        self.last_outputs_uploaded_step = 0
        self.train_start_time: Optional[float] = None
        self.max_train_seconds: Optional[float] = None

        self.amp_dtype = choose_amp_dtype(device, train_cfg)
        self.use_amp = self.amp_dtype is not None
        self.scaler = make_grad_scaler(device, self.use_amp, self.amp_dtype)

    # -- model/loss hooks ---------------------------------------------------
    def compute_student(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return student logits and an optional auxiliary loss.

        The dense student returns a tensor; the MoE student returns
        ``(logits, aux_loss)`` and overrides this method.
        """
        out = self.model(x)
        if isinstance(out, tuple):
            return out[0], out[1] if len(out) > 1 else None
        return out, None

    def extra_stats(self) -> str:
        return ""

    def extra_checkpoint_state(self) -> dict:
        return {}

    def restore_state(self, state: dict) -> None:
        """Continue from a saved distilled checkpoint (step, tokens, history)."""
        self.step = int(state.get("step", 0))
        self.tokens_seen = int(state.get("tokens_seen", 0))
        history = state.get("metrics_history")
        if isinstance(history, list):
            self.history = [dict(entry) for entry in history if isinstance(entry, dict)]
        saved_alpha = state.get("distill_alpha")
        saved_temp = state.get("distill_temperature")
        if saved_alpha is not None:
            self.alpha = float(saved_alpha)
        if saved_temp is not None:
            self.temperature = float(saved_temp)
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

    # -- metrics / output ---------------------------------------------------
    def _record_metric(self, loss: float, ce: float, kl: float) -> None:
        self.history.append(
            {
                "step": self.step,
                "tokens_seen": self.tokens_seen,
                "loss": round(float(loss), 6),
                "ce": round(float(ce), 6),
                "kl": round(float(kl), 6),
                "alpha": self.alpha,
                "temperature": self.temperature,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )

    def write_outputs(self) -> list[Path]:
        atomic_write_text(self.metrics_path, json.dumps(self.history, indent=2))
        files: list[Path] = [self.metrics_path]
        try:
            files.extend(save_distillation_plots(self.history, self.out_dir))
        except Exception as exc:
            logging.warning("Could not write distillation plots: %s", exc)
        return files

    def _log_sample(self, pbar=None) -> Optional[str]:
        """Generate a short continuation of a fixed prompt for eyeballing."""
        prompt = str(
            self.train_cfg.get("sample_prompt", "Albert Einstein is a")
        ).strip()
        if not prompt:
            return None

        model = unwrap_model(self.model)
        vocab_size = int(self.student_cfg.get("vocab_size", 0))
        max_new_tokens = max(
            1, int(self.train_cfg.get("sample_max_new_tokens", 60))
        )
        temperature = float(self.train_cfg.get("sample_temperature", 0.8))
        top_k = self.train_cfg.get("sample_top_k", 50)
        top_k = None if top_k in (None, 0) else int(top_k)

        try:
            prompt_ids = get_encoder().encode(prompt)
        except Exception as exc:
            logging.warning("Could not tokenize sample prompt: %s", exc)
            return None
        if not prompt_ids:
            return None
        if max(prompt_ids) >= vocab_size:
            logging.warning(
                "Sample prompt %r uses token ids outside vocab_size=%d; "
                "skipping sample.",
                prompt,
                vocab_size,
            )
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

    # -- checkpoint / HF ----------------------------------------------------
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
            logging.warning(
                "Refusing to save non-finite distilled weights in %s.", ", ".join(bad)
            )
            return False

        if self.last_loss is not None and (
            not self.history or self.history[-1].get("step") != self.step
        ):
            self._record_metric(self.last_loss, self.last_ce or 0.0, self.last_kl or 0.0)

        state = {
            "format_version": 1,
            "model": model_state,
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "model_config": self.student_cfg,
            "train_config": self.train_cfg,
            "distill_alpha": self.alpha,
            "distill_temperature": self.temperature,
            "metrics_history": self.history,
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
            logging.warning("Could not save distilled checkpoint: %s", exc)
            return False

        self.write_outputs()
        logging.info(
            "Saved distilled checkpoint %s (%.1f MB) at step %d (%s tokens).",
            self.checkpoint_path,
            self.checkpoint_path.stat().st_size / (1024 * 1024),
            self.step,
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
                "Final distillation sync at step %d: model %s, out/ %s.",
                self.step,
                "uploaded" if self.last_uploaded_step == self.step else "not uploaded",
                "uploaded"
                if self.last_outputs_uploaded_step == self.step
                else "not uploaded",
            )

    # -- training loop ------------------------------------------------------
    def train(self, batches: Iterable[tuple[np.ndarray, np.ndarray]]) -> None:
        cfg = self.train_cfg
        max_steps = cfg.get("max_steps")
        max_steps = int(max_steps) if max_steps is not None else None
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
        token_budget = cfg.get("token_budget")
        if token_budget is not None:
            token_budget = int(token_budget)
            if token_budget <= 0:
                raise ValueError("token_budget must be > 0")

        self.train_start_time = time.monotonic()
        batches = iter(batches)
        self.model.train()
        self.teacher.eval()
        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            total=max_steps,
            desc="distill",
            unit="step",
            dynamic_ncols=True,
            disable=not show_progress,
            colour="green",
            bar_format="{desc}: {percentage:3.0f}%|{bar}|{postfix}",
        )
        if show_progress:
            style_progress_bar(pbar)
        failed = False

        try:
            while (max_steps is None or self.step < max_steps) and (
                token_budget is None or self.tokens_seen < token_budget
            ):
                total_sum = 0.0
                ce_sum = 0.0
                kl_sum = 0.0
                aux_sum = 0.0
                step_tokens = 0
                data_exhausted = False

                for _ in range(grad_accum):
                    try:
                        x_np, y_np = next(batches)
                    except StopIteration:
                        data_exhausted = True
                        break

                    x = torch.from_numpy(x_np).long().to(self.device)
                    y = torch.from_numpy(y_np).long().to(self.device)
                    step_tokens += int(x.numel())

                    if self.use_amp:
                        student_ctx = torch.amp.autocast(
                            device_type=self.device.type, dtype=self.amp_dtype
                        )
                    else:
                        student_ctx = contextlib.nullcontext()

                    with student_ctx:
                        student_logits, aux_loss = self.compute_student(x)
                    teacher_logits = _teacher_logits(
                        self.teacher,
                        x,
                        self.teacher_device,
                        self.teacher_dtype,
                        student_ctx,
                    )
                    loss, ce, kl = distillation_loss(
                        student_logits,
                        teacher_logits[:, : x.size(1), :],
                        y,
                        self.alpha,
                        self.temperature,
                    )
                    if aux_loss is not None:
                        loss = loss + aux_loss.to(loss.dtype)
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError(
                            f"non-finite distillation loss at step {self.step + 1}"
                        )

                    self.scaler.scale(loss / grad_accum).backward()
                    total_sum += float(loss.detach())
                    ce_sum += float(ce.detach())
                    kl_sum += float(kl.detach())
                    if aux_loss is not None:
                        aux_sum += float(aux_loss.detach())

                if data_exhausted and total_sum == 0.0:
                    break

                if grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                self.step += 1
                self.tokens_seen += step_tokens
                self.last_loss = total_sum / max(1, grad_accum)
                self.last_ce = ce_sum / max(1, grad_accum)
                self.last_kl = kl_sum / max(1, grad_accum)
                self.last_aux = aux_sum / max(1, grad_accum)
                pbar.update(1)
                pbar.set_postfix_str(
                    f"#{self.step} | loss {self.last_loss:.3f} | "
                    f"ce {self.last_ce:.3f} | kl {self.last_kl:.3f} | "
                    f"{format_compact(self.tokens_seen)} tok"
                    f"{self.extra_stats()}"
                )

                if self.step <= log_first_steps or self.step % log_interval == 0:
                    self._record_metric(self.last_loss, self.last_ce, self.last_kl)
                if self.step <= log_first_steps:
                    logging.info(
                        "step %d | tokens %s | loss %.4f | ce %.4f | kl %.4f "
                        "| alpha %.2f",
                        self.step,
                        f"{self.tokens_seen:,}",
                        self.last_loss,
                        self.last_ce,
                        self.last_kl,
                        self.alpha,
                    )

                if sample_interval > 0 and self.step % sample_interval == 0:
                    self._log_sample(pbar)

                now = time.monotonic()
                if (
                    self.interval_seconds > 0
                    and (now - self.last_checkpoint_time) >= self.interval_seconds
                ):
                    self.save_checkpoint()

                if data_exhausted:
                    logging.info("Data stream exhausted at step %d.", self.step)
                    break
                if token_budget is not None and self.tokens_seen >= token_budget:
                    logging.info("Reached token budget at step %d.", self.step)
                    break
                if (
                    self.max_train_seconds is not None
                    and (now - self.train_start_time) >= self.max_train_seconds
                ):
                    logging.info("Reached time limit at step %d.", self.step)
                    break
        except KeyboardInterrupt:
            logging.info("Distillation interrupted; saving before exit.")
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
def run_distillation(
    cfg: dict, new_run: bool = False, full_new: bool = False
) -> None:
    load_env_file(ROOT_DIR / ".env")
    pid_file = os.environ.get("MINILLM_DISTILL_PID_FILE", "").strip()
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

    # By default use every visible GPU for the student and put the teacher on
    # the second GPU when one exists. This keeps both GPUs busy and avoids
    # replicating the large teacher with DataParallel.
    teacher_cfg = cfg.setdefault("teacher", {})
    if (
        not str(teacher_cfg.get("device") or "").strip()
        and device.type == "cuda"
        and num_gpus >= 2
    ):
        teacher_cfg["device"] = f"cuda:{min(1, num_gpus - 1)}"
        logging.info(
            "Auto-placing teacher on %s (student uses cuda:0 with "
            "DataParallel across %d GPUs).",
            teacher_cfg["device"],
            num_gpus,
        )

    student_cfg = load_student_model_config(cfg)
    data_cfg = cfg.get("data") or {}
    block_size = data_cfg.get("block_size") or student_cfg["context_length"]
    block_size = int(block_size)
    if block_size > int(student_cfg["context_length"]):
        raise SystemExit(
            f"data.block_size={block_size} exceeds student context_length="
            f"{student_cfg['context_length']}"
        )

    checkpoint_dir = resolve_path(cfg.get("checkpoint_dir", "distill_checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    student, resume_state, student_source = prepare_student(
        cfg, student_cfg, device, checkpoint_dir, new_run, full_new
    )
    logging.info("Student source: %s.", student_source)

    if num_gpus > 1:
        student = nn.DataParallel(student, device_ids=list(range(num_gpus)))
        logging.info("Student wrapped in DataParallel across %d GPUs.", num_gpus)

    teacher, _tokenizer, teacher_device, teacher_dtype = load_teacher(
        cfg, student_cfg, device, block_size
    )

    alpha = float(cfg.get("alpha", 0.3))
    temperature = float(cfg.get("temperature", 1.0))
    if not 0.0 <= alpha <= 1.0:
        raise SystemExit("alpha must be between 0 and 1")
    if temperature <= 0:
        raise SystemExit("temperature must be > 0")

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(cfg.get("learning_rate", 1e-4)),
        betas=tuple(cfg.get("betas", [0.9, 0.95])),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )
    hf_section = cfg.get("hf") or {}
    hf_cfg = get_hf_settings(cfg, disable=not bool(hf_section.get("enabled", False)))
    if hf_section.get("enabled") and hf_cfg is None:
        logging.warning(
            "hf.enabled is true but HF sync is not configured; "
            "the distilled model will only be saved locally."
        )

    distiller = Distiller(
        model=student,
        teacher=teacher,
        optimizer=optimizer,
        device=device,
        teacher_device=teacher_device,
        teacher_dtype=teacher_dtype,
        student_cfg=student_cfg,
        train_cfg=cfg,
        hf_cfg=hf_cfg,
    )
    if resume_state is not None:
        distiller.restore_state(resume_state)
    logging.info(
        "Distillation: alpha=%.2f (KL weight), CE weight=%.2f, T=%.1f, "
        "block_size=%d, batch=%d.",
        alpha,
        1.0 - alpha,
        temperature,
        block_size,
        int(cfg.get("batch_size", 4)),
    )
    logging.info(
        "Checkpoint: every %s to %s",
        (
            f"{float(cfg.get('checkpoint_interval_minutes', 60)) * 60:.0f} s"
            if float(cfg.get("checkpoint_interval_minutes", 60)) < 1
            else f"{float(cfg.get('checkpoint_interval_minutes', 60)):.1f} min"
        ),
        checkpoint_dir,
    )

    stream = OpenWebTextTokenStream(
        block_size=block_size,
        dataset_name=str(data_cfg.get("dataset_name", "Skylion007/openwebtext")),
        split=str(data_cfg.get("split", "train")),
        shuffle_buffer=int(data_cfg.get("shuffle_buffer", 1000)),
        seed=seed,
        max_docs=data_cfg.get("max_docs"),
        min_chars=int(data_cfg.get("min_chars", 1)),
    )
    batches = stream.batches(batch_size=int(cfg.get("batch_size", 4)), drop_last=True)

    def _terminate(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")

    previous_sigterm = None
    try:
        previous_sigterm = signal.signal(signal.SIGTERM, _terminate)
    except (ValueError, OSError):
        pass

    try:
        distiller.train(batches)
    finally:
        try:
            distiller.final_sync()
        finally:
            if previous_sigterm is not None:
                try:
                    signal.signal(signal.SIGTERM, previous_sigterm)
                except (ValueError, OSError):
                    pass
    logging.info("Distillation finished.")


# ---------------------------------------------------------------------------
# Background runner (same reason as pretraining: Jupyter kills !python children)
# ---------------------------------------------------------------------------
def _pid_file_path(cfg: dict) -> Path:
    return resolve_path(cfg.get("checkpoint_dir", "distill_checkpoints")) / "train.pid"


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
    return b"distill.distill" in cmdline or b"distill/train.py" in cmdline


def _latest_log() -> Optional[Path]:
    log_dir = ROOT_DIR / "logs"
    try:
        logs = sorted(log_dir.glob("distill-*.log"), key=lambda p: p.stat().st_mtime)
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
    checkpoint_dir = resolve_path(cfg.get("checkpoint_dir", "distill_checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pid_file = checkpoint_dir / "train.pid"
    existing = _read_pid(pid_file)
    if existing and _pid_is_trainer(existing):
        print(f"Distillation is already running (pid {existing}).")
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
        log_path = log_dir / time.strftime("distill-%Y%m%d-%H%M%S.log")

    child_argv = [arg for arg in raw_argv if arg != "--detach"]
    env = os.environ.copy()
    env["MINILLM_DISTILL_PID_FILE"] = str(pid_file)
    env["PYTHONUNBUFFERED"] = "1"
    with open(log_path, "ab", buffering=0) as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-m", "distill.distill", *child_argv],
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
            print(f"Distillation exited immediately with code {returncode}.")
            print(f"Log: {log_path}")
            tail = _tail(log_path, 30)
            if tail:
                print("--- last log lines ---")
                print(tail)
            return 1
        time.sleep(0.1)

    print(f"Distillation started in the background: pid {proc.pid}")
    print(f"Log: {log_path}")
    print(f"PID file: {pid_file}")
    print("Check:  python -m distill.distill --status")
    print("Stop:   python -m distill.distill --stop")
    return 0


def stop_training(cfg: dict, timeout: float = 10.0) -> int:
    pid_file = _pid_file_path(cfg)
    pid = _read_pid(pid_file)
    if not pid or not _pid_is_trainer(pid):
        print(f"No running distillation found (PID file: {pid_file}).")
        if pid_file.exists():
            print("The PID file is stale; removing it.")
            with contextlib.suppress(OSError):
                pid_file.unlink()
        return 1
    print(f"Sending SIGTERM to distillation pid {pid}.")
    print("It will save and upload before exiting.")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print("The process already exited.")
        return 0
    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline:
        if not _pid_is_trainer(pid):
            print("Distillation exited.")
            return 0
        time.sleep(0.2)
    print("Distillation is still running (final save/upload can take minutes).")
    log_path = _latest_log()
    if log_path is not None:
        print(f"Watch: {log_path}")
    return 0


def show_status(cfg: dict) -> int:
    checkpoint_dir = resolve_path(cfg.get("checkpoint_dir", "distill_checkpoints"))
    pid_file = _pid_file_path(cfg)
    pid = _read_pid(pid_file)
    running = bool(pid and _pid_is_trainer(pid))
    print(f"Distillation: {'RUNNING (pid %d)' % pid if running else 'not running'}")
    if pid and not running:
        print(f"Stale PID file: {pid_file}")
    log_path = _latest_log()
    if log_path is not None:
        print(f"Log: {log_path}")
        tail = _tail(log_path, 20)
        if tail:
            print("--- last log lines ---")
            print(tail)
    filename = str((cfg.get("hf") or {}).get("filename", "distilled_model.pt"))
    checkpoint = checkpoint_dir / filename
    if checkpoint.exists():
        stat = checkpoint.stat()
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
        print(f"Checkpoint: {checkpoint} ({stat.st_size / 1024**2:.1f} MB, {when})")
    else:
        print(f"Checkpoint: none at {checkpoint}")
    metrics = resolve_path(cfg.get("out_dir", "distilled_outs")) / "metrics.json"
    if metrics.exists():
        stat = metrics.stat()
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
        print(f"Metrics: {metrics} ({when})")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Logit-distill the mini GPT.")
    parser.add_argument("--config", type=Path, default=DEFAULT_DISTILL_CONFIG)
    parser.add_argument("--teacher", type=str, default=None, help="Override teacher.name")
    parser.add_argument("--alpha", type=float, default=None, help="KL loss weight (0-1)")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-hours", "--hours", type=float, default=None, dest="max_hours")
    parser.add_argument(
        "--budget", "--token-budget", type=int, default=None, dest="token_budget"
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument(
        "--checkpoint-seconds",
        type=float,
        default=None,
        help="Testing override: seconds between saves/uploads.",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Ignore any previous distilled_model.pt and start from the "
             "original student checkpoint (minillm_latest.pt).",
    )
    parser.add_argument(
        "--fullnew",
        action="store_true",
        help="Randomly initialize the student. Ignore both the previous "
             "distilled_model.pt and the original minillm_latest.pt.",
    )
    parser.add_argument("--no-hf", action="store_true", help="Disable HF upload")
    parser.add_argument("--upload", action="store_true", help="Enable HF upload")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--detach", action="store_true")
    mode.add_argument("--stop", action="store_true")
    mode.add_argument("--status", action="store_true")
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--stop-timeout", type=float, default=10.0)
    return parser


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> None:
    if args.teacher is not None:
        cfg.setdefault("teacher", {})["name"] = args.teacher
    if args.alpha is not None:
        cfg["alpha"] = args.alpha
    if args.temperature is not None:
        cfg["temperature"] = args.temperature
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.max_hours is not None:
        cfg["max_hours"] = args.max_hours
    if args.token_budget is not None:
        cfg["token_budget"] = args.token_budget
    if args.device is not None:
        cfg["device"] = args.device
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
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


def distill_main(argv: Optional[list[str]] = None) -> None:
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
        run_distillation(cfg, new_run=args.new, full_new=args.fullnew)
    except KeyboardInterrupt:
        logging.info("Distillation interrupted by user.")


if __name__ == "__main__":
    distill_main()
