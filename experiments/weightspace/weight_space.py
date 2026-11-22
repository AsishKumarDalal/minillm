"""Pure weight-space programming: project LLaMA-2 weights into the student.

No training. The source checkpoint is streamed one tensor at a time from disk.
Two projection matrices (P: hidden, Q: MLP intermediate) are built from source
weights, then every student parameter is written as a linear function of a
source tensor.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

import torch

try:  # package import
    from .model import LlamaConfig, LlamaForCausalLM
except ImportError:  # script import: python experiments/weightspace/build_student.py
    from model import LlamaConfig, LlamaForCausalLM

try:
    import numpy as np
except ImportError:
    np = None

try:
    import yaml
except ImportError:
    yaml = None


LAYER_RE = re.compile(r"model\.layers\.(\d+)\.(.+)\.weight$")
QKV_RE = re.compile(r"model\.layers\.(\d+)\.self_attn\.[qkvo]_proj\.weight$")
DOWN_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.down_proj\.weight$")


# ---------------------------------------------------------------------------
# Streaming source provider
# ---------------------------------------------------------------------------
class SourceWeights:
    """Yields ``(name, tensor)`` pairs from safetensors or torch .bin shards.

    Only one shard / tensor is materialised at a time, so a 13B model never
    occupies more than a single matrix of process RAM.
    """

    def __init__(self, path: str, strip_prefix: str = ""):
        self.path = Path(path).expanduser()
        self.strip_prefix = strip_prefix
        if self.path.is_dir():
            files = sorted(glob.glob(str(self.path / "*.safetensors")))
            if not files:
                files = sorted(glob.glob(str(self.path / "*.bin")))
        else:
            files = [str(self.path)]
        if not files:
            raise FileNotFoundError(f"no weight shards found in {self.path}")
        self.files = files

    def __len__(self):
        return len(self.files)

    def iter_tensors(self) -> Iterator[Tuple[str, torch.Tensor]]:
        for shard in self.files:
            if shard.endswith(".safetensors"):
                from safetensors import safe_open

                with safe_open(shard, framework="pt", device="cpu") as fp:
                    for key in fp.keys():
                        yield self._clean(key), fp.get_tensor(key)
            else:
                try:
                    state = torch.load(
                        shard, map_location="cpu", weights_only=True
                    )
                except TypeError:
                    state = torch.load(shard, map_location="cpu")
                except Exception:
                    state = torch.load(shard, map_location="cpu", weights_only=False)
                for key, value in state.items():
                    if torch.is_tensor(value):
                        yield self._clean(key), value
                del state

    def _clean(self, key: str) -> str:
        if self.strip_prefix and key.startswith(self.strip_prefix):
            return key[len(self.strip_prefix) :]
        return key


class DictProvider:
    """In-memory provider used by the tiny CPU test."""

    def __init__(self, state: Dict[str, torch.Tensor]):
        self.state = state

    def __len__(self):
        return 1

    def iter_tensors(self):
        for key, value in self.state.items():
            yield key, value


# ---------------------------------------------------------------------------
# Layer selection
# ---------------------------------------------------------------------------
def select_layers(num_source_layers: int, num_student_layers: int) -> list:
    if num_student_layers > num_source_layers:
        raise ValueError("student cannot have more layers than source")
    if np is not None:
        picked = np.linspace(0, num_source_layers - 1, num_student_layers)
        return sorted(set(int(v) for v in picked.round()))
    step = (num_source_layers - 1) / max(1, num_student_layers - 1)
    return sorted(set(int(round(i * step)) for i in range(num_student_layers)))


# ---------------------------------------------------------------------------
# Projection construction (pass 1)
# ---------------------------------------------------------------------------
def _orthonormal_from_cov(cov: torch.Tensor, rank: int, niter: int = 2) -> torch.Tensor:
    """Top ``rank`` right singular vectors as a (d_in x rank) matrix."""
    cov = cov.float()
    _, _, v = torch.svd_lowrank(
        cov, q=min(rank, min(cov.shape)), niter=int(niter)
    )
    return v.contiguous()


def build_projections(
    provider,
    source_dim: int,
    source_intermediate: int,
    student_dim: int,
    student_intermediate: int,
    method: str = "svd",
    profile_layers: int = 8,
    svd_niter: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return P (D_s x D_t) and Q (I_s x I_t) with orthonormal columns."""
    method = method.lower()
    if method == "slice":
        P = torch.eye(source_dim)[:, :student_dim]
        Q = torch.eye(source_intermediate)[:, :student_intermediate]
        return P, Q
    if method == "random":
        P = torch.linalg.qr(torch.randn(source_dim, student_dim))[0]
        Q = torch.linalg.qr(torch.randn(source_intermediate, student_intermediate))[0]
        return P, Q
    if method != "svd":
        raise ValueError(f"unknown projection method: {method}")

    cov_h = torch.zeros(source_dim, source_dim, dtype=torch.float32)
    cov_i = torch.zeros(source_intermediate, source_intermediate, dtype=torch.float32)
    seen_h = set()
    seen_i = set()

    for name, tensor in provider.iter_tensors():
        m_qkv = QKV_RE.match(name)
        m_down = DOWN_RE.match(name)
        if m_qkv:
            layer = int(m_qkv.group(1))
            if layer in seen_h or len(seen_h) >= profile_layers:
                continue
            w = tensor.float()
            if w.shape != (source_dim, source_dim):
                continue
            cov_h.addmm_(w.T, w)
            seen_h.add(layer)
        elif m_down:
            layer = int(m_down.group(1))
            if layer in seen_i or len(seen_i) >= profile_layers:
                continue
            w = tensor.float()
            if w.shape != (source_dim, source_intermediate):
                continue
            cov_i.addmm_(w.T, w)
            seen_i.add(layer)

    if not seen_h:
        raise RuntimeError("no q/k/v/o matrices found for hidden projection")
    if not seen_i:
        raise RuntimeError("no down_proj matrices found for intermediate projection")

    P = _orthonormal_from_cov(cov_h, student_dim, niter=svd_niter)
    Q = _orthonormal_from_cov(cov_i, student_intermediate, niter=svd_niter)
    return P, Q


# ---------------------------------------------------------------------------
# Weight mapping (pass 2)
# ---------------------------------------------------------------------------
def _project(name: str, tensor: torch.Tensor, P, Q, student_layer: Optional[int]):
    """Return (student_key, projected_tensor) or None if tensor is unused."""
    if name == "model.embed_tokens.weight" or name == "lm_head.weight":
        return name, (tensor.float() @ P).contiguous()
    if name == "model.norm.weight":
        return name, (P.T @ tensor.float()).contiguous()

    m = LAYER_RE.match(name)
    if m is None or student_layer is None:
        return None
    sub = m.group(2)
    w = tensor.float()

    if sub in ("input_layernorm", "post_attention_layernorm"):
        value = P.T @ w
    elif sub in ("self_attn.q_proj", "self_attn.k_proj",
                 "self_attn.v_proj", "self_attn.o_proj"):
        value = P.T @ w @ P
    elif sub in ("mlp.gate_proj", "mlp.up_proj"):
        value = Q.T @ w @ P
    elif sub == "mlp.down_proj":
        value = P.T @ w @ Q
    else:
        return None

    key = f"model.layers.{student_layer}.{sub}.weight"
    return key, value.contiguous()


def build_student(
    cfg_path: str,
    provider: Optional[SourceWeights] = None,
    output_dir: str = "checkpoints",
    filename: str = "weightspace_student.pt",
    verbose: bool = True,
):
    """Run both passes and write the student checkpoint. Returns its path."""
    if yaml is None:
        raise RuntimeError("PyYAML is required for build_student()")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    src = cfg["source"]
    stu = cfg["student"]
    provider = provider or SourceWeights(
        src["path"], strip_prefix=src.get("strip_prefix", "")
    )

    d_s = int(src["hidden_size"])
    i_s = int(src["intermediate_size"])
    n_src_layers = int(src["num_hidden_layers"])
    d_t = int(stu["emb_dim"])
    i_t = int(stu["intermediate_size"])
    n_t = int(stu["n_layers"])

    if verbose:
        print(f"[1/2] building projections from {len(provider)} shard(s) ...")
    P, Q = build_projections(
        provider,
        d_s,
        i_s,
        d_t,
        i_t,
        method=cfg.get("projection", {}).get("method", "svd"),
        profile_layers=int(cfg.get("projection", {}).get("profile_layers", 2)),
        svd_niter=int(cfg.get("projection", {}).get("niter", 2)),
    )

    layer_map_list = select_layers(n_src_layers, n_t)
    layer_map = {src_idx: dst_idx for dst_idx, src_idx in enumerate(layer_map_list)}
    if verbose:
        print(f"      source layers -> student layers: {layer_map_list}")

    student_cfg = LlamaConfig(
        {
            "vocab_size": int(stu.get("vocab_size", src.get("vocab_size", 32000))),
            "context_length": int(stu["context_length"]),
            "emb_dim": d_t,
            "intermediate_size": i_t,
            "n_heads": int(stu["n_heads"]),
            "n_layers": n_t,
            "rope_theta": float(src.get("rope_theta", 10000.0)),
            "eos_token_id": int(src.get("eos_token_id", 2)),
            "bos_token_id": int(src.get("bos_token_id", 1)),
            "pad_token_id": int(src.get("pad_token_id", 0)),
            "dtype": stu.get("dtype", "float32"),
            "tie_weights": False,
        }
    )

    model = LlamaForCausalLM(student_cfg)
    state = model.state_dict()
    mapped = set()

    if verbose:
        print("[2/2] streaming source tensors and writing student weights ...")
    for name, tensor in provider.iter_tensors():
        m = LAYER_RE.match(name)
        dst_layer = None
        if m:
            dst_layer = layer_map.get(int(m.group(1)))
        result = _project(name, tensor, P, Q, dst_layer)
        if result is None:
            continue
        key, value = result
        if key not in state:
            raise RuntimeError(f"unexpected student key: {key}")
        if tuple(state[key].shape) != tuple(value.shape):
            raise RuntimeError(
                f"shape mismatch for {key}: student {tuple(state[key].shape)} "
                f"vs projected {tuple(value.shape)}"
            )
        state[key] = value.to(state[key].dtype)
        mapped.add(key)
        del value, tensor

    expected = {n for n, _ in model.named_parameters()}
    missing = sorted(expected - mapped)
    if missing:
        raise RuntimeError(f"student parameters not filled by source: {missing[:10]}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    torch.save(
        {
            "model": state,
            "model_config": dict(student_cfg),
            "weight_space": {
                "source": str(src.get("path", "")),
                "projection_method": cfg.get("projection", {}).get("method", "svd"),
                "source_layers": layer_map_list,
                "P": P,
                "Q": Q,
            },
        },
        out_path,
    )
    if verbose:
        print(f"      wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    tok_src = src.get("tokenizer_dir")
    if tok_src:
        tok_dst = out_dir / "tokenizer"
        tok_dst.mkdir(parents=True, exist_ok=True)
        for pattern in ("tokenizer.json", "tokenizer.model", "special_tokens_map.json",
                        "tokenizer_config.json", "added_tokens.json"):
            for f in glob.glob(str(Path(tok_src).expanduser() / pattern)):
                shutil.copy2(f, tok_dst / Path(f).name)
        if verbose:
            print(f"      copied tokenizer files to {tok_dst}")

    return out_path


__all__ = [
    "SourceWeights",
    "DictProvider",
    "select_layers",
    "build_projections",
    "build_student",
]
