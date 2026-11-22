"""Scaled-down Llama-2 model used by the weight-space experiment.

State-dict keys intentionally match a HuggingFace LlamaForCausalLM so that
source tensors can be mapped one-for-one (after dimension projection):

    model.embed_tokens.weight
    model.layers.{i}.input_layernorm.weight
    model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    model.layers.{i}.post_attention_layernorm.weight
    model.layers.{i}.mlp.{gate,up,down}_proj.weight
    model.norm.weight
    lm_head.weight
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import yaml
except ImportError:  # yaml is optional for pure-python use
    yaml = None

try:
    from .model_parts import (
        Attention,
        DecoderLayer,
        LayerKVCache,
        RMSNorm,
        RotaryEmbedding,
    )
except ImportError:  # script import: python experiments/weightspace/model.py
    from model_parts import (
        Attention,
        DecoderLayer,
        LayerKVCache,
        RMSNorm,
        RotaryEmbedding,
    )


class LlamaConfig(dict):
    """Small dict-like config with attribute access and validation."""

    REQUIRED = (
        "vocab_size",
        "context_length",
        "emb_dim",
        "intermediate_size",
        "n_heads",
        "n_layers",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key in self.REQUIRED:
            if key not in self:
                raise KeyError(f"missing required config key: {key}")
        self.setdefault("rope_theta", 10000.0)
        self.setdefault("eos_token_id", 2)
        self.setdefault("bos_token_id", 1)
        self.setdefault("pad_token_id", 0)
        self.setdefault("dtype", "float32")
        self.setdefault("tie_weights", False)

    @classmethod
    def from_yaml(cls, path) -> "LlamaConfig":
        if yaml is None:
            raise RuntimeError("PyYAML is required to read YAML configs")
        with open(path, encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh))

    def torch_dtype(self) -> torch.dtype:
        name = str(self["dtype"]).lower()
        return {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }[name]


class LlamaModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = int(cfg["emb_dim"])
        v = int(cfg["vocab_size"])
        self.embed_tokens = nn.Embedding(v, d)
        self.layers = nn.ModuleList(
            [DecoderLayer(cfg) for _ in range(int(cfg["n_layers"]))]
        )
        self.norm = RMSNorm(d)
        self.rope = RotaryEmbedding(
            head_dim=d // int(cfg["n_heads"]),
            max_seq_len=int(cfg["context_length"]),
            theta=float(cfg.get("rope_theta", 10000.0)),
        )
        self.kv_caches: Optional[list] = None

    def build_cache(self, batch_size: int, device, dtype, max_seq_len: int):
        self.kv_caches = [
            LayerKVCache(
                batch_size=batch_size,
                num_heads=layer.self_attn.n_heads,
                max_seq_len=max_seq_len,
                head_dim=layer.self_attn.head_dim,
                device=device,
                dtype=dtype,
            )
            for layer in self.layers
        ]

    def clear_cache(self):
        self.kv_caches = None

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: Optional[list] = None,
        offset: int = 0,
    ) -> torch.Tensor:
        _, seq_len = input_ids.shape
        if seq_len + offset > int(self.rope.max_seq_len):
            raise ValueError("sequence exceeds context_length")
        x = self.embed_tokens(input_ids)
        if cache is None:
            for layer in self.layers:
                x = layer(x, self.rope, cache=None, offset=0)
        else:
            for layer, layer_cache in zip(self.layers, cache):
                x = layer(x, self.rope, cache=layer_cache, offset=offset)
        return self.norm(x)


class LlamaForCausalLM(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.config = LlamaConfig(cfg)
        d = int(self.config["emb_dim"])
        v = int(self.config["vocab_size"])
        self.model = LlamaModel(self.config)
        self.lm_head = nn.Linear(d, v, bias=False)
        if bool(self.config.get("tie_weights", False)):
            self.lm_head.weight = self.model.embed_tokens.weight
        self.apply(self._init_weights)
        self.to(self.config.torch_dtype())

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    # -- checkpoint I/O -----------------------------------------------------
    def save_checkpoint(self, path, extra: Optional[dict] = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.state_dict(),
            "model_config": dict(self.config),
        }
        if extra:
            state.update(extra)
        torch.save(state, path)
        return path

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu"):
        state = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(state["model_config"])
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        return model

    # -- forward ------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        hidden = self.model(input_ids, cache=None, offset=0)
        logits = self.lm_head(hidden)
        if labels is None:
            return logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            labels.reshape(-1),
            ignore_index=-100,
        )
        return logits, loss

    # -- generation ---------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        b, t0 = idx.shape
        eos = (
            int(eos_token_id)
            if eos_token_id is not None
            else int(self.config["eos_token_id"])
        )
        ctx = int(self.config["context_length"])
        if t0 + max_new_tokens > ctx:
            raise ValueError("prompt + max_new_tokens exceeds context_length")

        self.model.build_cache(b, idx.device, self.config.torch_dtype(), ctx)
        cache = self.model.kv_caches
        try:
            out = self.model(idx, cache=cache, offset=0)
            logits = self.lm_head(out[:, -1, :])
            for pos in range(t0, t0 + max_new_tokens):
                nxt = self._sample(logits, temperature, top_k)
                idx = torch.cat((idx, nxt), dim=1)
                if bool((nxt.squeeze(-1) == eos).all()):
                    break
                if pos + 1 == t0 + max_new_tokens:
                    break
                offset = cache[0].seq_len
                out = self.model(nxt, cache=cache, offset=offset)
                logits = self.lm_head(out[:, -1, :])
            return idx
        finally:
            self.model.clear_cache()

    def _sample(self, logits, temperature, top_k):
        if top_k is not None and top_k > 0:
            k = min(int(top_k), logits.size(-1))
            vals, _ = torch.topk(logits, k)
            logits = logits.masked_fill(logits < vals[:, -1:], float("-inf"))
        if temperature > 0:
            probs = torch.softmax(logits.float() / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return torch.argmax(logits, dim=-1, keepdim=True)

    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        tokenizer_dir: str,
        max_new_tokens: int = 80,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
    ) -> str:
        """Generate text using tokenizer files copied next to the checkpoint."""
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for generate_text()") from exc
        tok = AutoTokenizer.from_pretrained(tokenizer_dir)
        ids = tok(prompt, return_tensors="pt").input_ids.to(
            next(self.parameters()).device
        )
        out = self.generate(
            ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        return tok.decode(out[0, ids.size(1) :], skip_special_tokens=True)


__all__ = ["LlamaConfig", "LlamaModel", "LlamaForCausalLM"]
