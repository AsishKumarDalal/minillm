import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_parts import *
import yaml
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
config_path = ROOT_DIR / "configs" / "model.yaml"
EOT_TOKEN_ID=50256
with open(config_path) as f:
    cfg = yaml.safe_load(f)

class GPT(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.vocab_size = cfg["vocab_size"]
        self.context_length = cfg["context_length"]
        self.n_heads = cfg["n_heads"]
        self.head_dim = cfg["emb_dim"] // cfg["n_heads"]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
        # GPT-2 style initialization. PyTorch's default N(0, 1) embedding and
        # linear init makes the initial logits huge (loss ~400+) and the run
        # diverges to NaN within a few hundred steps. std=0.02 keeps the
        # initial loss near ln(vocab_size) and training stable.
        self.apply(self._init_weights)

        # GPT-2 style weight tying: the output head shares the token
        # embedding matrix. Saves ~19.3M parameters with this config.
        if cfg.get("tie_weights", True):
            self.out_head.weight = self.tok_emb.weight

        # Scale residual output projections by 1/sqrt(2 * n_layers) so the
        # residual stream variance stays bounded as depth grows.
        residual_std = 0.02 / math.sqrt(2 * cfg["n_layers"])
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith(
                "ff.layers.2.weight"
            ):
                nn.init.normal_(param, mean=0.0, std=residual_std)

        # Plain attribute (not a module/buffer), so cache tensors never enter
        # state_dict() or change checkpoint compatibility.
        self.kv_caches = None

    @staticmethod
    def _init_weights(module):
        """GPT-2 style init: N(0, 0.02) weights, zero biases."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # -- KV cache lifecycle -------------------------------------------------
    def build_cache(
        self,
        batch_size,
        device=None,
        dtype=None,
        max_seq_len=None,
    ):
        """Allocate one KV cache per transformer block."""
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if device is None:
            device = self.tok_emb.weight.device
        if dtype is None:
            dtype = self.tok_emb.weight.dtype
        if max_seq_len is None:
            max_seq_len = self.context_length
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")

        self.kv_caches = [
            LayerKVCache(
                batch_size=batch_size,
                num_heads=self.n_heads,
                max_seq_len=max_seq_len,
                head_dim=self.head_dim,
                device=device,
                dtype=dtype,
            )
            for _ in range(len(self.trf_blocks))
        ]

    def reset_cache(self):
        """Mark every layer cache empty between independent requests."""
        if self.kv_caches is not None:
            for cache in self.kv_caches:
                cache.reset()

    def clear_cache(self):
        """Release cache memory entirely."""
        self.kv_caches = None

    def _ensure_cache(self, batch_size, device, dtype):
        if not self.kv_caches:
            self.build_cache(batch_size, device=device, dtype=dtype)
            return
        if len(self.kv_caches) != len(self.trf_blocks):
            self.build_cache(batch_size, device=device, dtype=dtype)
            return
        first = self.kv_caches[0]
        if (
            first.batch_size != batch_size
            or first.k_cache.device != device
            or first.k_cache.dtype != dtype
        ):
            self.build_cache(batch_size, device=device, dtype=dtype)

    def forward(self, in_idx, y=None, ignore_index=-100, use_cache=False):
        _, seq_len = in_idx.shape

        if use_cache:
            if y is not None:
                raise ValueError(
                    "use_cache=True is inference-only; training targets are "
                    "not supported"
                )
            self._ensure_cache(
                in_idx.size(0), in_idx.device, self.tok_emb.weight.dtype
            )
            past_len = self.kv_caches[0].seq_len
            if any(cache.seq_len != past_len for cache in self.kv_caches):
                raise RuntimeError(
                    "Layer KV caches are out of sync; call reset_cache() "
                    "before retrying"
                )
            total_len = past_len + seq_len
            if total_len > self.context_length:
                raise ValueError(
                    f"KV-cached forward needs positions [0, {total_len}) but "
                    f"context_length={self.context_length}"
                )
            positions = torch.arange(
                past_len, total_len, device=in_idx.device
            )
            caches = self.kv_caches
        else:
            assert seq_len <= self.context_length, (
                f"sequence length {seq_len} exceeds context_length "
                f"{self.context_length}"
            )
            positions = torch.arange(seq_len, device=in_idx.device)
            caches = [None] * len(self.trf_blocks)

        token_embedding = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(positions)
        x = token_embedding + pos_embeds
        x = self.drop_emb(x)
        for block, cache in zip(self.trf_blocks, caches):
            x = block(x, kv_cache=cache)
        x = self.final_norm(x)
        logits = self.out_head(x)

        if y is None:
            return logits

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=ignore_index,
        )
        return logits, loss

    def _sample_next(self, logits, temperature, top_k):
        """Top-k filter, then temperature sample or greedy argmax."""
        if top_k is not None:
            k = min(int(top_k), logits.size(-1))
            if k <= 0:
                raise ValueError("top_k must be > 0")
            top_logits, _ = torch.topk(logits, k)
            min_val = top_logits[:, -1]
            logits = logits.masked_fill(
                logits < min_val.unsqueeze(-1), float("-inf")
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return torch.argmax(logits, dim=-1, keepdim=True)

    def _apply_eos(self, idx_next, finished, eos_id):
        """Update finished rows and pad them with EOS."""
        if eos_id is None:
            return idx_next, finished, False
        finished = finished | (idx_next.squeeze(-1) == eos_id)
        if finished.any():
            idx_next = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(idx_next, eos_id),
                idx_next,
            )
        return idx_next, finished, bool(finished.all())

    def _generate_uncached_tokens(
        self,
        generated,
        num_tokens,
        context_size,
        temperature,
        top_k,
        eos_id,
        finished,
    ):
        """Full-recompute generation loop used by use_cache=False and the
        sliding-window fallback."""
        for _ in range(num_tokens):
            idx_cond = generated[:, -context_size:]
            logits = self(idx_cond)[:, -1, :]
            idx_next = self._sample_next(logits, temperature, top_k)
            idx_next, finished, all_done = self._apply_eos(
                idx_next, finished, eos_id
            )
            generated = torch.cat((generated, idx_next), dim=1)
            if all_done:
                break
        return generated, finished

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=250, context_size=None,
                 temperature=0.0, top_k=None, eos_id=EOT_TOKEN_ID,
                 use_cache=True):
        """
        Generate text with early stopping at EOT token.

        With ``use_cache=True`` (default), the prompt is prefilled once and
        each new token reuses the cached K/V values. Once the cached window
        is full, generation falls back to the old full-recompute sliding
        window so learned absolute positions are never reused incorrectly.
        With ``use_cache=False`` the full-recompute path is used throughout.

        Both training and eval mode are handled: the model is put in eval
        mode (dropout off) for generation, then restored afterwards.
        Works for batch size 1 and for batched generation.
        """
        was_training = self.training
        self.eval()
        try:
            if context_size is None:
                context_size = self.context_length
            context_size = min(int(context_size), self.context_length)
            if context_size <= 0:
                raise ValueError("context_size must be > 0")

            if temperature < 0.0:
                raise ValueError("temperature must be >= 0.0")

            if max_new_tokens <= 0:
                return idx

            generated = idx
            finished = torch.zeros(
                idx.size(0), dtype=torch.bool, device=idx.device
            )

            if not use_cache:
                generated, _ = self._generate_uncached_tokens(
                    generated,
                    max_new_tokens,
                    context_size,
                    temperature,
                    top_k,
                    eos_id,
                    finished,
                )
                return generated

            cond = idx[:, -context_size:]
            self._ensure_cache(
                cond.size(0), cond.device, self.tok_emb.weight.dtype
            )
            self.reset_cache()

            next_logits = self(cond, use_cache=True)[:, -1, :]

            for step in range(max_new_tokens):
                idx_next = self._sample_next(next_logits, temperature, top_k)
                idx_next, finished, all_done = self._apply_eos(
                    idx_next, finished, eos_id
                )
                generated = torch.cat((generated, idx_next), dim=1)

                if all_done or step == max_new_tokens - 1:
                    break

                # Learned absolute positions mean the cache cannot slide.
                # Rebuild the window with full recomputation for the rest of
                # the request; this matches the old generate() behavior.
                if self.kv_caches[0].seq_len + 1 > context_size:
                    self.reset_cache()
                    generated, _ = self._generate_uncached_tokens(
                        generated,
                        max_new_tokens - (step + 1),
                        context_size,
                        temperature,
                        top_k,
                        eos_id,
                        finished,
                    )
                    break

                next_logits = self(idx_next, use_cache=True)[:, -1, :]

            return generated
        finally:
            self.train(was_training)