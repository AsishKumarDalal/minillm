# Pretraining

`pretrain/train.py` streams and tokenizes OpenWebText lazily, trains the GPT,
and periodically saves/resumes checkpoints through HuggingFace Hub.

For a detailed Kaggle walkthrough and config guide, see
[`runbook/pretrain.md`](../runbook/pretrain.md).

## Files

| File | Purpose |
| --- | --- |
| `pretrain/train.py` | Training loop, device selection, AMP, checkpoints, HF sync |
| `pretrain/data.py` | Streaming OpenWebText tokenizer and batch writer |
| `configs/model.yaml` | Model architecture (~102.5M parameters) |
| `configs/pretrain.yaml` | Batch size, LR, checkpoint/HF settings |
| `.env` | `HF_TOKEN` (copy from `.env.example`) |

## Setup

```bash
python3 -m virtualenv .venv
```

CPU-only machine (install torch from the CPU index first so the requirements
step does not pull CUDA wheels):

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r pretrain/requirements.txt
```

CUDA machine / Kaggle (torch is usually preinstalled; pip will skip it if the
version satisfies the requirement):

```bash
.venv/bin/pip install -r pretrain/requirements.txt
```

## HuggingFace checkpoint repo

1. Create a model repo on HuggingFace (the script can also create it on the
   first upload if the token has permission).
2. Copy `.env.example` to `.env` and add a token with **write** access:

   ```bash
   cp .env.example .env
   # edit .env
   HF_TOKEN=hf_...
   ```

3. Put the repo id in `configs/pretrain.yaml`:

   ```yaml
   hf:
     repo_id: "your-username/minillm"
     token_env: HF_TOKEN
     filename: minillm_latest.pt
     out_dir: out
     private: true
   ```

The checkpoint is uploaded at the repo root as `minillm_latest.pt`. Metrics
and plots are uploaded inside the repo's `out/` folder (configurable with
`hf.out_dir`).

The token is read from `.env` or the process environment. It is never saved
inside a checkpoint.

If `hf.repo_id` is empty or the token is missing, the script logs a warning
and trains normally without remote upload/resume. Local checkpoints still
work.

## Run

Auto-select hardware (2 GPUs, else 1 GPU, else CPU):

```bash
.venv/bin/python -m pretrain.train
```

Force CPU and a short run:

```bash
.venv/bin/python -m pretrain.train --device cpu --max-steps 10
```

Stop after 6 or 12 hours of wall-clock training. The final checkpoint and
plots are still saved/uploaded:

```bash
.venv/bin/python -m pretrain.train --hours 6
.venv/bin/python -m pretrain.train --max-hours 12
```

`--hours` is an alias for `--max-hours`. You can also set `max_hours: 6` in
`configs/pretrain.yaml`.

Start a brand-new run from a random model, ignoring both the local and the
HuggingFace checkpoint. Checkpoints and plots are still saved locally and
uploaded to HF:

```bash
.venv/bin/python -m pretrain.train --new
```

Run completely without HuggingFace: ignore local and HF checkpoints, start
from a random model, and never download from or upload to HF:

```bash
.venv/bin/python -m pretrain.train --no-hf
```

| Flags | Uses local checkpoint | Uses HF checkpoint | Uploads to HF |
| --- | --- | --- | --- |
| *(none)* | yes | yes (if local is missing) | yes |
| `--new` | no | no | yes |
| `--no-hf` | no | no | no |
| `--new --no-hf` | no | no | no |

### Logging and sample generations

By default the run logs every step for the first 10 steps, then only every
100 steps, so a multi-hour log stays readable. Every 1000 steps it also
generates a short continuation of `sample_prompt` (default
`"Einstein was a"`) and writes it to the log, for example:

```text
sample @ step 1000 | Einstein was a German-born theoretical physicist ...
```

All four values live in `configs/pretrain.yaml`:

```yaml
log_first_steps: 10
log_interval: 100
sample_interval: 1000
sample_prompt: "Einstein was a"
sample_max_new_tokens: 60
sample_temperature: 0.8
sample_top_k: 50
```

Set `sample_interval: 0` to disable sampling. tqdm still updates every step;
only the INFO lines (and metric points) are sparse.

Change the checkpoint directory or batch size from the CLI:

```bash
.venv/bin/python -m pretrain.train \
  --checkpoint-dir checkpoints \
  --batch-size 8 \
  --max-steps 1000
```

### GPU selection

- `device: auto` uses up to `num_gpus` visible GPUs.
- 2+ GPUs use `nn.DataParallel` in one process, which works from a notebook
  without `torchrun`/`spawn` setup. It is simpler but slower than DDP for very
  long runs; switch to a `torchrun`/DDP launcher if multi-GPU throughput
  becomes the bottleneck.
- 1 GPU uses that GPU directly.
- No GPU falls back to CPU.
- `device: cuda:1` forces one specific GPU.

### Notebook progress

The script imports `tqdm.auto`, so calling `main()` in a Kaggle/Jupyter
notebook renders a notebook progress bar. It falls back to a terminal bar in
a normal shell. Set `progress: false` in `configs/pretrain.yaml` to disable it.

The bar always shows a percentage when there is a target:

- `max_steps` set → percentage of requested optimizer steps.
- `max_hours` set → percentage of the wall-clock budget.
- neither set → percentage of the time until the next checkpoint save,
  resetting after every save. This is the normal open-ended pretraining mode.

The suffix shows `step`, `tokens`, `loss`, `ppl`, `lr`, and `tok/s` every
step. When time-based, it also shows `remaining_h` (with `max_hours`) or
`save_in` (minutes until the next save).

In a notebook, pass an explicit argument list so argparse does not try to
parse kernel arguments such as `-f`:

```python
from pretrain.train import main

main(["--device", "cuda", "--max-steps", "100"])
```

## Checkpoints and resume

Every `checkpoint_interval_minutes` (default **60**), plus at normal exit,
on `Ctrl+C`, and when the `--hours` time limit is reached, the script:

1. Saves `checkpoints/minillm_latest.pt` atomically.
2. Writes the local `out/` folder:
   - `metrics.json` — step, tokens seen, loss, perplexity, LR, timestamp
   - `token_seen_vs_loss.png`
   - `token_seen_vs_perplexity.png`
   - `training_curves.png`
   The loss/perplexity plots show raw points plus a smoothed trend line.
3. Uploads the checkpoint to `hf.repo_id` as `minillm_latest.pt`, replacing
   the previous file.
4. Uploads the local `out/` folder to `hf.out_dir` (default `out/`) in the
   same repo, replacing the previous plots and metrics.
5. Optionally includes optimizer state (`save_optimizer: true`) so training
   resumes with Adam moments as well as weights. A 100M-parameter FP32
   optimizer checkpoint is large; set `save_optimizer: false` for smaller
   uploads if a fresh optimizer on resume is acceptable.

Non-finite (NaN/Inf) checkpoints are never saved or resumed. If the loss
starts printing `nan`, the trainer skips the bad update, aborts after five
consecutive failures without overwriting the checkpoint, and asks you to
restart from a finite checkpoint with a lower `learning_rate`.

On startup it:

1. Tries the local `checkpoints/minillm_latest.pt` first, so a remote upload
   can never overwrite a good local checkpoint with a bad one.
2. Falls back to downloading `minillm_latest.pt` from the configured HF repo
   when the local checkpoint is missing or unusable.
3. Checks the saved model architecture against `configs/model.yaml` and
   rejects checkpoints containing NaN/Inf weights.
4. Restores model, optimizer, step, token count, scaler, RNG state, and the
   metrics history used for the plots.
5. Starts from scratch with a warning if no usable checkpoint exists.

Pass `--new` to skip those steps and start from a random model while still
saving and uploading to HF. Pass `--no-hf` to start from a random model and
disable all HF download/upload.

The data stream itself restarts from the beginning of OpenWebText on resume;
token position is not persisted. With a shuffled stream this is usually
acceptable for continued pretraining.

## KV-cached generation

`GPT.generate()` uses a per-layer KV cache by default. It prefills the prompt
once, then decodes one token at a time:

```python
model.eval()
out = model.generate(
    idx,
    max_new_tokens=32,
    temperature=0.8,
    top_k=50,
    use_cache=True,
)
```

The cache stores K/V after the learned absolute position embedding, so the
position offset and causal mask stay aligned. Training is unchanged:
`forward(in_idx, y=...)` never allocates or reads a cache. `use_cache=True` is
inference-only and raises if training targets are passed.

Because positions are learned absolute positions, a cached request cannot
silently slide its window. Once the cache reaches `context_size`, generation
automatically falls back to full recomputation of the sliding window for the
remaining tokens. This preserves the old `generate()` behavior (and its
results) while still getting the cache speedup for the first window. Pass
`use_cache=False` to use the full-recompute path from the start. Cache tensors
are plain attributes and do not appear in `state_dict()`, so existing
checkpoints still load with `strict=True`.

See [`code_test/test_kv_cache.py`](../code_test/test_kv_cache.py) for cached
vs. uncached equivalence tests.

## Tiny CPU test

Do not run the 100M model on CPU just to test the training loop. Use:

```bash
.venv/bin/python code_test/test_train_tiny.py
```

It trains a 4,416-parameter model for a few synthetic steps, tests gradient
accumulation, saves a local checkpoint, resumes it, checks the warning path
for missing HF settings, and verifies that the loss/perplexity decrease while
`tokens_seen` increases. It also checks that the metrics JSON and all three
plots are written to the local `out/` folder. See
[`code_test/README.md`](../code_test/README.md).

For a real OpenWebText -> tiny training smoke test on CPU (about 50k
parameters, three documents, one optimizer step, no HF upload):

```bash
.venv/bin/python code_test/smoke_train_openwebtext_tiny.py
```

That is only a one-step smoke test, so it cannot show a loss trend. For a
multi-step real-data curve check (tokens increasing, loss/perplexity
decreasing, plots written), run:

```bash
.venv/bin/python code_test/test_train_curves_real.py
```

The training progress bar shows a percentage and updates `step`, `tokens`,
`loss`, and `ppl` every step; checkpoint plots are written every
`checkpoint_interval_minutes`.
