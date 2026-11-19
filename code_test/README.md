# Lightweight checks

These tests are deliberately small. The synthetic training test uses a
**4,416-parameter** model on CPU; the real-data smoke test uses a
**~50k-parameter** model. Neither loads the 100M model. The synthetic test
does not touch OpenWebText or HuggingFace Hub; the real-data smoke test only
streams three documents and never uploads anything.

## Setup

```bash
python3 -m pip install --user --break-system-packages virtualenv
python3 -m virtualenv .venv
```

On a CPU-only Linux machine, install the CPU build of torch first to avoid
CUDA wheels:

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r pretrain/requirements.txt
```

On a CUDA machine or Kaggle, torch is usually already installed:

```bash
.venv/bin/pip install -r pretrain/requirements.txt
```

## Tests

Run the data-pipeline unit tests (no network, no torch needed):

```bash
.venv/bin/python code_test/test_data_pipeline.py
```

Run the tiny end-to-end training test (CPU, synthetic data, no network):

```bash
.venv/bin/python code_test/test_train_tiny.py
```

This checks:

- tiny model forward/backward and optimizer steps
- GPT-2 initialization keeps the initial loss near `ln(vocab_size)`
- gradient accumulation
- sparse logging (full INFO lines first N steps; plot points every `log_interval`)
- tqdm percentage modes (`--hours`, `--budget`, `max_steps`, estimated OpenWebText tokens, next checkpoint)
- one continuous green bar with `t/s`, tokens, ppl, loss; samples written above it
- periodic sample generation from `sample_prompt`
- `--new` starts fresh but keeps HF sync; `--no-hf` starts fresh with no HF
- linear warmup + cosine LR schedule
- local checkpoint save/resume (model, optimizer, step, tokens)
- rejection of NaN checkpoints and refusal to save NaN weights
- wall-clock time limit and token budget stop training and save/upload the final checkpoint
- final sync re-uploads the newest finite checkpoint on interrupt/crash paths
- `out/` and model uploads are tracked/retried independently
- repeated Ctrl+C/SIGTERM during the final upload is deferred until both uploads finish
- atomic checkpoint writes and stale `.tmp` cleanup
- transient HF upload errors retried without retrying auth failures
- detached `--stop` sends SIGTERM and cleans stale PID files
- periodic checkpoint/model/`out/` uploads fire at the configured interval
- metrics history restore and loss/perplexity decreasing as tokens increase
- local `out/metrics.json` plus loss, perplexity, and combined plots
- `.env` parsing
- missing HF repo/token warnings
- mock HF checkpoint and `out/` uploads
- model-config mismatch detection

Run the real OpenWebText streaming smoke test (reads only a couple of docs):

```bash
.venv/bin/python code_test/smoke_openwebtext.py
```

Run a real OpenWebText -> tiny GPT training smoke test on CPU. This uses the
real data pipeline and `run_training`, a ~50k-parameter model, three
documents, one optimizer step, and local checkpointing only. A one-step run
cannot show a loss trend; use the curve test below for that.

```bash
.venv/bin/python code_test/smoke_train_openwebtext_tiny.py
```

Run a real OpenWebText training-curve test on CPU. It repeats a few real
batches for 60 optimizer steps with a small (~1.6M-parameter) model and
asserts that `tokens_seen` increases while loss and perplexity decrease. It
also checks all plots and `metrics.json`:

```bash
.venv/bin/python code_test/test_train_curves_real.py
```

Run the tiny distillation tests (CPU, local 64-vocab teacher, no network):

```bash
.venv/bin/python code_test/test_distill_tiny.py
```

This checks CE/KL weighting with `alpha`, temperature scaling, a full
distillation step loop, `distilled_model.pt` / `distilled_outs/metrics.json`
save/load, resume from a previous distilled checkpoint, the periodic fixed
prompt sample, and the four `distilled_outs/` plots (loss, perplexity, CE, KL).
A smoke test with
a small real HuggingFace teacher (two steps on local text, no uploads) is also
available:

```bash
.venv/bin/python code_test/smoke_distill_real_teacher.py
```

Run the tiny SFT tests (CPU, local Dolly-style examples, no download):

```bash
.venv/bin/python code_test/test_sft_tiny.py
```

This checks runtime triage (full / context-dropped / dropped), answer-only
loss masking, epochs/steps, checkpoint save/resume, the three `sft_outs/`
plots, and CLI overrides. A real streamed-Dolly smoke test (two steps, tiny
random student, no upload) is also available:

```bash
.venv/bin/python code_test/smoke_sft_real_data.py
```

Run the KV-cache correctness tests (no network):

```bash
.venv/bin/python code_test/test_kv_cache.py
```

These check cached-vs-uncached logits, chunked and token-by-token decode,
learned position offsets, cache reset/batch resize, generation equivalence
(including sliding-window fallback), EOS padding, and sampling validation.

Check that the cache is also a *speed* win, on a random ~92k-parameter model
(CPU, no network). It generates the same tokens with and without the cache,
asserts the outputs are token-for-token identical, then times both paths:

```bash
.venv/bin/python code_test/bench_kv_cache.py
```

Expected shape of the result (single CPU thread, ~92k params, context 256):
128 new tokens from an 8-token prompt are roughly 3.2-3.7x faster with the
cache (measured 3.2x, 3.4x, 3.4x, 3.7x over repeat runs). The
long-prompt line near `context_length` is much closer to 1x on purpose: with
learned absolute positions the cache cannot slide, so generation falls back
to full recompute once the window is full.

Inspect exactly what the training loop receives:

```bash
.venv/bin/python -m pretrain.data \
  --block-size 32 --batch-size 2 --num-batches 1 \
  --max-docs 2 --shuffle-buffer 0
```

The stream is synchronous and lazy: each `next()` fetches only enough
OpenWebText documents to build the next batch. HuggingFace caches touched
shards under `~/.cache/huggingface`, but the full dataset is never
downloaded.

Optional: set `HF_TOKEN` to avoid anonymous Hub rate limits during a long
pretraining run:

```bash
export HF_TOKEN=hf_...
```

For the actual pretraining script and HF checkpoint sync, see
[`pretrain/README.md`](../pretrain/README.md).
