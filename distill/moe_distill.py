"""Logit distillation of the MoE student from a pretrained HF teacher.

This mirrors ``distill/distill.py``:

    loss = (1 - alpha) * CE(student, labels)
         + alpha * KL(teacher || student) * T^2
         + student MoE auxiliary routing loss

The student uses the GPT-2 BPE vocabulary (50257), so GPT-2-family teachers
(gpt2, gpt2-medium, gpt2-large, gpt2-xl) line up token-for-token.

Local outputs
-------------
``moe_distill_checkpoints/moe_model_distill.pt``
``moe_distill_outs/metrics.json`` plus loss/CE/KL/perplexity/validation plots

HF outputs
----------
``moe/moe_model_distill.pt``
``moe/distill_outs/...``

Usage
-----
::

    python -m distill.moe_distill
    python -m distill.moe_distill --teacher openai-community/gpt2 --alpha 0.4
    python -m distill.moe_distill --new
    python -m distill.moe_distill --fullnew --no-hf
    python -m distill.moe_distill --check-hf
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

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from distill.distill import (  # noqa: E402
    Distiller,
    _teacher_logits,
    distillation_loss,
    load_teacher,
)
from model.moe_model import MOE_ARCH_KEYS, MoELanguageModel  # noqa: E402
from pretrain.data import OpenWebTextTokenStream  # noqa: E402
from pretrain.moe_train import load_moe_model_config  # noqa: E402
from pretrain.train import (  # noqa: E402
    check_hf,
    download_checkpoint,
    get_hf_settings,
    load_env_file,
    load_yaml,
    model_config_mismatches,
    resolve_path,
    select_device,
    setup_logging,
)

DEFAULT_CONFIG = ROOT_DIR / "configs" / "moe_distill.yaml"
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
    strict: bool = True,
) -> MoELanguageModel:
    mismatches = model_config_mismatches(
        state["model_config"], model_cfg, MOE_ARCH_KEYS
    )
    if mismatches:
        if strict:
            raise SystemExit(
                f"Checkpoint {path} architecture mismatch: "
                f"{'; '.join(mismatches)}"
            )
        logging.warning(
            "Checkpoint %s architecture mismatch: %s",
            path,
            "; ".join(mismatches),
        )
        return None  # type: ignore[return-value]
    model = MoELanguageModel(model_cfg).to(device)
    try:
        model.load_state_dict(state["model"], strict=True)
    except Exception as exc:
        if strict:
            raise SystemExit(
                f"Could not load MoE weights from {path}: {exc}"
            ) from exc
        logging.warning("Could not load MoE weights from %s: %s", path, exc)
        return None  # type: ignore[return-value]
    return model


def load_moe_student(
    cfg: dict,
    model_cfg: dict,
    device: torch.device,
) -> tuple[MoELanguageModel, Path]:
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
                raise SystemExit("student.repo_id is empty in moe_distill.yaml")
            path = download_checkpoint(
                {
                    "repo_id": repo_id,
                    "filename": local_path.name,
                    "path_in_repo": str(
                        section.get("filename", "moe/moe_model_latest.pt")
                    ),
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
    return model, path


def load_moe_resume_state(
    cfg: dict,
    model_cfg: dict,
    device: torch.device,
    checkpoint_dir: Path,
) -> Optional[tuple[MoELanguageModel, dict]]:
    hf = cfg.get("hf") or {}
    local_name = str(hf.get("filename", "moe_model_distill.pt")).strip()
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
        model = _load_moe_from_state(
            state, model_cfg, device, path, strict=False
        )
    except SystemExit as exc:
        logging.warning("Ignoring distilled checkpoint %s: %s", path, exc)
        return None
    if model is None:
        return None
    model.train()
    logging.info(
        "Resuming MoE distillation from %s at step %d (%s tokens).",
        path,
        int(state.get("step", 0)),
        f"{int(state.get('tokens_seen', 0)):,}",
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
        resumed = load_moe_resume_state(cfg, model_cfg, device, checkpoint_dir)
        if resumed is not None:
            model, state = resumed
            return model, state, "previous MoE distilled checkpoint"
    model, _ = load_moe_student(cfg, model_cfg, device)
    return model, None, "pretrained MoE checkpoint"


# ---------------------------------------------------------------------------
# Distiller
# ---------------------------------------------------------------------------
class MoEDistiller(Distiller):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.val_batches: list[tuple[np.ndarray, np.ndarray]] = []
        self.last_val: Optional[dict] = None

    def compute_student(self, x: torch.Tensor):
        logits, aux_loss = self.model(x)
        # DataParallel gathers tuple outputs into a 1-D aux tensor; the base
        # Distiller adds aux directly to the loss, so return a scalar here.
        return logits, aux_loss.mean()

    def extra_stats(self) -> str:
        if self.last_aux is None:
            return ""
        return f" | aux {self.last_aux:.3f}"

    def _autocast(self):
        import contextlib

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
        self.teacher.eval()
        totals = {"loss": 0.0, "ce": 0.0, "kl": 0.0, "aux": 0.0}
        try:
            for x_np, y_np in self.val_batches:
                x = torch.from_numpy(x_np).long().to(self.device)
                y = torch.from_numpy(y_np).long().to(self.device)
                with self._autocast():
                    student_logits, aux_loss = self.compute_student(x)
                teacher_logits = _teacher_logits(
                    self.teacher,
                    x,
                    self.teacher_device,
                    self.teacher_dtype,
                    None,
                )
                loss, ce, kl = distillation_loss(
                    student_logits,
                    teacher_logits[:, : x.size(1), :],
                    y,
                    self.alpha,
                    self.temperature,
                )
                aux_value = 0.0 if aux_loss is None else float(aux_loss)
                totals["loss"] += float(loss) + aux_value
                totals["ce"] += float(ce)
                totals["kl"] += float(kl)
                totals["aux"] += aux_value
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

    def save_checkpoint(self, upload: bool = True) -> bool:
        if self.step > 0 and self.val_batches:
            stats = self.validate()
            if stats is not None:
                if self.history and self.history[-1].get("step") == self.step:
                    self.history[-1]["val_loss"] = round(
                        float(stats["loss"]), 6
                    )
                    self.history[-1]["val_ce"] = round(float(stats["ce"]), 6)
                    self.history[-1]["val_kl"] = round(float(stats["kl"]), 6)
                    self.history[-1]["val_aux"] = round(float(stats["aux"]), 6)
                    self.history[-1]["val_perplexity"] = round(
                        float(stats["perplexity"]), 3
                    )
                logging.info(
                    "validation @ step %d | val_loss %.4f | val_ce %.4f | "
                    "val_kl %.4f | val_aux %.4f | val_ppl %.1f",
                    self.step,
                    stats["loss"],
                    stats["ce"],
                    stats["kl"],
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
    stream = OpenWebTextTokenStream(
        block_size=block_size,
        dataset_name=str(
            data_cfg.get("dataset_name", "Skylion007/openwebtext")
        ),
        split=str(data_cfg.get("split", "train")),
        shuffle_buffer=0,
        seed=seed + 7919,
        max_docs=val_docs,
        min_chars=int(data_cfg.get("min_chars", 1)),
    )
    batches: list[tuple[np.ndarray, np.ndarray]] = []
    for x, y in stream.batches(batch_size=val_batch_size, drop_last=False):
        batches.append((x, y))
        if len(batches) >= val_batches:
            break
    logging.info(
        "Distillation validation: %d cached batch(es) from %d docs.",
        len(batches),
        val_docs,
    )
    return batches


def run_distillation(
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

    data_cfg = cfg.get("data") or {}
    block_size = int(data_cfg.get("block_size") or model_cfg["context_length"])
    if block_size > int(model_cfg["context_length"]):
        raise SystemExit(
            f"data.block_size={block_size} exceeds student context_length="
            f"{model_cfg['context_length']}"
        )

    checkpoint_dir = resolve_path(
        cfg.get("checkpoint_dir", "moe_distill_checkpoints")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    student, resume_state, student_source = prepare_moe_student(
        cfg, model_cfg, device, checkpoint_dir, new_run, full_new
    )
    logging.info("Student source: %s.", student_source)
    breakdown = student.parameter_breakdown()
    logging.info(
        "MoE parameters: total %s | active/token %s | experts %s",
        f"{breakdown['total']:,}",
        f"{breakdown['active']:,}",
        f"{breakdown['expert_total']:,}",
    )

    if num_gpus > 1:
        student = nn.DataParallel(student, device_ids=list(range(num_gpus)))
        logging.info("Student wrapped in DataParallel across %d GPUs.", num_gpus)

    teacher, _tokenizer, teacher_device, teacher_dtype = load_teacher(
        cfg, model_cfg, device, block_size
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

    distiller = MoEDistiller(
        model=student,
        teacher=teacher,
        optimizer=optimizer,
        device=device,
        teacher_device=teacher_device,
        teacher_dtype=teacher_dtype,
        student_cfg=model_cfg,
        train_cfg=cfg,
        hf_cfg=hf_cfg,
    )
    if resume_state is not None:
        distiller.restore_state(resume_state)
    distiller.val_batches = build_validation_batches(cfg, block_size, seed)

    logging.info(
        "MoE distillation: alpha=%.2f, CE weight=%.2f, T=%.1f, block=%d, "
        "batch=%d.",
        alpha,
        1.0 - alpha,
        temperature,
        block_size,
        int(cfg.get("batch_size", 4)),
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
    batches = stream.batches(
        batch_size=int(cfg.get("batch_size", 4)), drop_last=True
    )

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
    logging.info("MoE distillation finished.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Logit-distill the MoE student from a HF teacher."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--model-config", type=Path, default=DEFAULT_MODEL_CONFIG
    )
    parser.add_argument("--teacher", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--max-hours", "--hours", type=float, default=None, dest="max_hours"
    )
    parser.add_argument(
        "--budget", "--token-budget", type=int, default=None,
        dest="token_budget",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
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


def main(argv: Optional[list[str]] = None) -> None:
    setup_logging()
    args = build_arg_parser().parse_args(argv)
    cfg = load_yaml(Path(args.config))
    apply_cli_overrides(cfg, args)
    model_cfg = load_moe_model_config(args.model_config)
    load_env_file(ROOT_DIR / ".env")

    if args.check_hf:
        raise SystemExit(check_hf(cfg, no_hf=args.no_hf))

    try:
        run_distillation(
            cfg,
            model_cfg,
            no_hf=args.no_hf,
            new_run=args.new,
            full_new=args.fullnew,
            require_hf=args.require_hf,
        )
    except KeyboardInterrupt:
        logging.info("MoE distillation interrupted by user.")


if __name__ == "__main__":
    main()
