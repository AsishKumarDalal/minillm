# Image-conditioned decoder — implementation plan

Goal: let the text decoder (dense `GPT` and `MoELanguageModel`) consume image
features as a **soft visual prefix** prepended to the text token embeddings.
No change to the GPT-2 vocabulary or the output head, so existing checkpoints,
SFT and logit-distillation pipelines remain compatible.

## Constraints (from the current code)

| Constraint | Value | Source |
| --- | --- | --- |
| Embedding dim | 768 | `configs/model.yaml`, `configs/moe_config.yaml` |
| Context length | 128 | same configs |
| Vocabulary | GPT-2 BPE, 50,257 | `pretrain/data.py` |
| Loss masking | `ignore_index=-100` | `model/model.py:174`, `pretrain/moe_train.py:129` |
| KV cache positions | absolute `past_len + seq_len` | `model/model.py:145`, `model/moe_model.py:371` |
| MoE aux loss | router balance + z-loss added to CE | `model/moe_parts.py:484` |
| Validation rule | tiny CPU model (~0.1M), not 100M | `agent.md` |

**Decisions:**
- Image encoder: **frozen CLIP ViT-B/32** (`openai/clip-vit-base-patch32`) for
  phase 1. Output 49 patch tokens of 768-d (drop CLS). Alternative for a later
  ablation: DINOv2 ViT-B/14 (`facebook/dinov2-base`, 256 x 768 patches at
  224px) or SigLIP base. DINOv2's 256 tokens **cannot** be prepended directly —
  they must be compressed first.
- Visual prefix length **K = 16** (configurable 8–32). Learned-query
  cross-attention resampler (Q-Former / Perceiver style), not raw patch tokens.
  This leaves 128 - 16 = 112 positions for text.
- Adapter projects into **768-d** directly. A 256-d intermediate is optional
  and only useful if memory is tight; it is not an architectural requirement.
- Keep the vision encoder frozen throughout; train only the resampler first.

## Phase 0 — baseline measurement (no code change)

Use the existing benchmark suite in `experiments/` so the throughput cost of
the visual prefix can be quantified later:

```bash
bash experiments/run_all.sh
# specific numbers for dense vs MoE under trainer settings:
python experiments/bench_e2e.py --iters 10
```

Record tokens/s for dense GPT and MoE. This is the reference for the
"5000 t/s dense vs ~1900 t/s MoE" observation.

## Phase 1 — new module: `model/vision_adapter.py`

```python
class ImageEncoderAdapter(nn.Module):
    """Frozen vision encoder + learned-query resampler -> [B, K, 768]."""

    def __init__(self, cfg):  # cfg from configs/vision.yaml
        super().__init__()
        self.vision = CLIPModel.from_pretrained(cfg["encoder"]).vision_model
        self.vision.eval()
        for p in self.vision.parameters():
            p.requires_grad_(False)
        self.queries = nn.Parameter(torch.randn(cfg["num_visual_tokens"], 768) * 0.02)
        self.cross_attn = nn.MultiheadAttention(768, 12, batch_first=True)
        self.norm = nn.LayerNorm(768)
        self.proj = nn.Linear(768, 768)

    @torch.no_grad()
    def encode(self, pixel_values):
        return self.vision(pixel_values).last_hidden_state[:, 1:]  # drop CLS

    def forward(self, pixel_values):
        feats = self.encode(pixel_values)
        q = self.queries.expand(pixel_values.size(0), -1, -1)
        img = self.cross_attn(q, feats, feats)[0]
        return self.proj(self.norm(img))  # [B, K, 768]
```

Config file `configs/vision.yaml`:
```yaml
encoder: openai/clip-vit-base-patch32
num_visual_tokens: 16
image_size: 224
freeze_encoder: true
projector_lr: 1.0e-4
```

## Phase 2 — language-model changes

Files: `model/model.py` (dense) and `model/moe_model.py` (MoE).

1. `forward(..., image_embeds=None, pixel_values=None)`:
   - build `tok = self.tok_emb(input_ids)` then
     `x = torch.cat([image_embeds, tok], dim=1)`;
   - total length `K + T` must satisfy the existing
     `assert seq_len <= self.context_length` (update the check to use the
     concatenated length);
   - positions: `arange(total_len)` for dense, RoPE offset starting at 0 for
     the prefix in MoE (`model/moe_model.py:389`);
   - KV-cache `total_len = past_len + seq_len` must include the prefix; in
     cached generation the prefix is passed only in the prefill call.
2. Labels: prepend `K` columns of `-100` (and keep the final `-100` on padded
   text positions). `cross_entropy(..., ignore_index=-100)` already handles it.
3. `generate()`: accept `pixel_values`, prime the cache once with
   `[visual prefix | prompt]`, then feed only generated token IDs. The
   full-recompute/sliding-window fallback needs the prefix re-injected each
   step or should be disabled for image-conditioned generation.
4. `state_dict` gains `cross_attn.*`, `queries`, `proj.*`. Load old checkpoints
   with `strict=False` and log the newly initialized keys.

## Phase 3 — paired image/text data

New `pretrain/image_data.py` (or `sft/image_sft_data.py`):

- Stream `COCO Captions` or `Flickr30k` via `datasets` (already installed).
- GPT-2 tokenizer (`pretrain.data.get_encoder`), EOT = 50256.
- Format: `Image prefix (K vectors) + <caption tokens> + EOT`.
- Labels: `-100` for all prompt/padding positions, caption + EOT supervised.
- Cache tokenized captions; decode images once per batch with the frozen
  encoder (under `torch.no_grad()`), so the vision encoder is not in the
  backward graph.

## Phase 4 — training script

New `pretrain/image_train.py` (and later `sft/image_sft.py`):

- Stage 1: freeze the language decoder entirely; optimize only adapter params
  with LR 1e-4. Goal: learn the decoder's embedding space.
- Stage 2: unfreeze top 2–3 decoder blocks with LR ~1e-5 for a few hundred
  steps (optional; validate before enabling).
- Reuse `Trainer` helpers from `pretrain/train.py` (atomic saves, plots, HF
  sync, signal handling). For MoE: subclass `MoETrainer` so the router aux loss
  stays in the objective (`ce + aux`).
- HF upload OFF by default (`--no-hf`); checkpoints under
  `image_checkpoints/` and metrics under `image_outs/`.

## Phase 5 — tests and validation

Per `agent.md`, validate only with a tiny CPU model:

```bash
python -m py_compile model/*.py pretrain/*.py sft/*.py code_test/*.py
python code_test/test_train_tiny.py
python code_test/test_kv_cache.py
python code_test/test_data_pipeline.py
python code_test/test_distill_tiny.py
python code_test/test_sft_tiny.py
python code_test/test_train_curves_real.py
```

New test `code_test/test_image_adapter_tiny.py` must check:
1. adapter output shape `[B, 16, 768]`;
2. concatenated prefix + tokens never exceeds `context_length`;
3. prefix positions are ignored by the loss (`y == -100` there);
4. KV-cached generation with a prefix matches the uncached logits;
5. old checkpoint loads with `strict=False`;
6. train step does not change frozen encoder weights.

## Phase 6 — benchmarks and ablations

Add `experiments/bench_vision.py` (uses the style of `bench_e2e.py`):
- prefix K = 0 (text only), 8, 16, 32;
- encoder on GPU vs CPU-offload;
- dense vs MoE;
- measure tokens/s and peak VRAM.

Add `experiments/vision_ablation.py` to compare CLIP vs DINOv2 vs SigLIP with
the same resampler and training budget.

## Risks / gotchas

- `context_length=128`: never prepend all 256 DINOv2 patches; compress to 16.
- Absolute/RoPE positions shift for text by K — fine after training the
  adapter, but text-only perplexity can regress if no text-only data is mixed
  in. Mix ~10–20% text-only batches in stage 2.
- `nn.DataParallel` with tuple outputs `(logits, aux)`: keep `aux.mean()`
  handling as in `pretrain/moe_train.py:135`.
- The existing SFT/distill datasets are text-only; image-conditioned SFT and
  distillation need new data loaders (distillation also needs teacher logits
  for the same token ids, which is fine since the vocab is unchanged).
- Always keep `.env` unread/uncommitted; `HF_TOKEN` stays in env/Kaggle secrets.

## Acceptance criteria

- `test_image_adapter_tiny.py` and all existing tiny tests pass.
- Tiny CPU run: loss decreases on a synthetic image-caption batch; prefix
  positions have zero gradient contribution to the loss.
- Dense and MoE models both forward/backward with `K=16` prefix on 2x T4.
- `bench_vision.py` reports the throughput delta vs the Phase 0 baseline.
- READMEs/runbooks updated before commit; commit and push only after tests pass.
