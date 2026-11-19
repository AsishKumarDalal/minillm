# Pretraining runbook (Kaggle)

This runbook explains how to train `minillm` from a Kaggle notebook, what the
two config files control, and how to make common changes safely.

The model is a dense GPT (~102.5M parameters) trained on streaming
OpenWebText. Training uses 2 GPUs via `nn.DataParallel` when available,
otherwise 1 GPU, otherwise CPU. Checkpoints and training plots sync to the
HuggingFace repo `sddfhy546y/minillm`.

For logit distillation from a pretrained HuggingFace teacher, see the
[distillation runbook](distill.md).

---

## 1. Before you start

In the Kaggle notebook:

1. **Settings → Accelerator:** GPU T4 x2 is ideal. One GPU or CPU also works.
2. **Settings → Internet:** ON. OpenWebText is streamed from HuggingFace.
3. **Settings → Secrets → Add secret:** name it `HF_TOKEN`. The value must be
   a HuggingFace token with write access to `sddfhy546y/minillm`.

Never paste the token into a notebook cell or commit it to GitHub. The repo
ignores `.env`; use Kaggle Secrets instead.

---

## 2. Install dependencies

Kaggle images already include torch, datasets, numpy, and tqdm. Install or
upgrade the remaining packages from the top-level `requirements.txt`:

```python
%pip install -q -r requirements.txt
```

If you want to avoid touching the preinstalled torch/datasets versions:

```python
%pip install -q -U huggingface_hub pyyaml matplotlib tiktoken
```

---

## 3. Put the code in `/kaggle/working`

If you uploaded the GitHub repo as a zip dataset:

```python
!unzip -o /kaggle/input/YOUR-DATASET-NAME/minilllm-main.zip -d /kaggle/working/
%cd /kaggle/working/minilllm-main
```

If you copied the folder in another way, just `cd` into the directory that
contains `pretrain/`, `configs/`, and `model/`:

```python
%cd /kaggle/working/minilllm
```

---

## 4. Load the token from Kaggle Secrets

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

The script reads `HF_TOKEN` from the environment. It also supports a local
`.env` file, but do not create or upload one in Kaggle — Secrets is safer.
If the token is missing, training still runs; only HF upload/resume is
disabled with a warning.

Verify the token, repo, and write permission before starting a long run:

```python
!python -m pretrain.train --check-hf
```

A correct setup prints `Write access: OK` and
`Result: HF upload is configured.` If it says `HF_TOKEN is NOT set`, run the
Secrets cell above again in the same kernel; detached children inherit the
kernel environment.

---

## 5. Verify hardware

```python
import torch
print("cuda:", torch.cuda.is_available())
print("gpus:", torch.cuda.device_count())
```

---

## 6. Optional smoke test (2 steps, no HF)

Use a separate checkpoint directory so the smoke run does not become the
resume point for the real run:

```python
from pretrain.train import main

main([
    "--device", "cpu",
    "--max-steps", "2",
    "--batch-size", "2",
    "--checkpoint-dir", "/kaggle/working/smoke_ckpt",
    "--no-hf",
])
```

If you then want the real run to start fresh, remove any stale local
checkpoint:

```python
!rm -f checkpoints/minillm_latest.pt
```

---

## 7. Start the real training

### Notebook-safe start and stop (recommended)

Do **not** start a long run with `!python -m pretrain.train`: Jupyter sends
Ctrl+C to the child and then force-kills it about 0.2 seconds later. SIGKILL
cannot be caught, so the final checkpoint save and upload never run.
`main([])` runs inside the kernel and blocks the notebook; a detached run is
safer and lets you keep using the notebook. Start the trainer in its own
background session instead:

```python
# 1) Verify HF_TOKEN, repo access, and write permission. No training starts.
!python -m pretrain.train --check-hf

# 2) Start in the background. This cell returns immediately.
!python -m pretrain.train --detach --new --require-hf

# 3) Inspect progress, newest log, checkpoint, and metrics.
!python -m pretrain.train --status

# 4) Stop cleanly. SIGTERM triggers the final save and both uploads.
!python -m pretrain.train --stop
```

`--detach` starts a separate session, writes
`logs/pretrain-<timestamp>.log`, and stores its PID in
`checkpoints/train.pid`. If you started with a custom `--checkpoint-dir`,
pass the same flag to `--status` and `--stop` so they find `train.pid`. It is
safe to close the notebook cell; a notebook Ctrl+C cannot kill the trainer. `--stop` sends SIGTERM, then the trainer
finishes the current step, writes `checkpoints/minillm_latest.pt`, and
uploads the model and all `out/` files. If `--stop` says it is still running,
that is normal for a 1.2 GB upload; the detached process keeps going and
finishes on its own. Watch the log for:

```text
Final sync at step ...: local checkpoint saved, model uploaded, out/ uploaded.
```

`--require-hf` refuses to start when `HF_TOKEN` or `hf.repo_id` is missing,
so a misconfigured run cannot quietly train without uploading. Omit it only
when you deliberately want a local-only run.

### Common options

Start from a random model even if a checkpoint exists, but keep saving and
uploading to HF:

```python
!python -m pretrain.train --detach --new
```

Start from a random model with no HuggingFace access at all:

```python
!python -m pretrain.train --detach --no-hf
```

One GPU instead of two:

```python
!python -m pretrain.train --detach --device cuda --num-gpus 1
```

Force CPU:

```python
!python -m pretrain.train --detach --device cpu
```

Test the hourly upload timer in 10 seconds instead of 60 minutes (with HF
configured, it really uploads the checkpoint and `out/` every 10 seconds):

```python
!python -m pretrain.train --detach --checkpoint-seconds 10 --device cpu
```

Watch `logs/pretrain-*.log` for repeated `Saved checkpoint ...` and
`Uploaded ...` lines about 10 seconds apart. Remove `--checkpoint-seconds`
for the real run so the interval goes back to 60 minutes.

Short run with HF sync enabled:

```python
!python -m pretrain.train --detach --device cuda --max-steps 100 --batch-size 8
```

Stop after 6 hours, then save/upload:

```python
!python -m pretrain.train --detach --hours 6
```

`--hours` is an alias for `--max-hours`. You can also set this in
`configs/pretrain.yaml`:

```yaml
max_hours: 6
```

The stop is checked after each optimizer step. When reached, training exits
cleanly and the final checkpoint + `out/` files are saved and uploaded.

Stop after a fixed number of tokens instead; the bar counts toward that
budget (e.g. 15M tokens):

```python
!python -m pretrain.train --detach --budget 15000000
```

```yaml
token_budget: 15000000
```

With neither `--hours` nor `--budget`, the bar shows
`tokens_seen / estimated_total_tokens` — roughly `tokens seen / 8.97B` for
one full pass of OpenWebText. The estimate is stored in
`configs/pretrain.yaml` as `estimated_total_tokens` and was derived from the
HF dataset metadata (39,769,491,688 text bytes, 8,013,769 documents) and a
500-document sample tokenized with `tiktoken`'s `gpt2` encoder (4.4325
bytes/token). It is only a progress denominator; it does not stop training.

For a 2-step smoke test, `main([...])` in-process is fine; use `--detach` for
anything long enough to matter.

```python
from pretrain.train import main

main(["--device", "cpu", "--max-steps", "2", "--no-hf"])
```

### What you should see

```text
Device: cuda:0 | GPUs used: 2
Model parameters: 102,467,328
batch_size=8 block_size=128 grad_accum=1 -> 1,024 tokens per optimizer step
Checkpoint: every 60.0 min to /kaggle/working/minillm/checkpoints
HF sync: ENABLED | repo=sddfhy546y/minillm | model=minillm_latest.pt | out=out | token_env=HF_TOKEN
Local metrics/plots: /kaggle/working/minillm/out
HF metrics/plots: sddfhy546y/minillm/out
Logging: step lines for the first 10 steps only; metrics/plot points every 100 steps; sample generation every 1000 steps.
step 1 | tokens_seen 1,024 | loss 10.9xxx | ppl 48,xxx | lr 1.50e-06 | tok/s ...
...
step 10 | tokens_seen 10,240 | loss 9.7xxx | ppl 17,xxx | lr ...
sample @ step 1000 | Albert Einstein is a ...
```

The `tqdm.auto` progress bar always shows a percentage. In order of
precedence: `--hours` → percent of the wall-clock budget; `--budget` →
percent of that many tokens; `max_steps` → percent of steps; otherwise
`tokens_seen / estimated_total_tokens` (one estimated OpenWebText pass,
~8.97B tokens, from `configs/pretrain.yaml`). Only the bare time-to-next-save
fallback resets after every save. The
suffix shows the compact stats `#step`, `t/s`, `tok`, `ppl`, and `loss` every
step, e.g. `#81 | 5.5k t/s | 82.9k tok | ppl 13.1k | loss 9.480`. The
notebook bar uses the full row, keeps the bar narrow (15%), and gives the
remaining width to the stats so `t/s`, `tok`, `ppl`, and `loss` stay visible.
The bar is created once and keeps running
for the whole run; the step-1000 sample is written above it, so it does not
restart or create a second bar. Full INFO step lines are printed only for the
first `log_first_steps` steps (default 10); after that the live bar is the
status display, so a long run does not flood the output. A metrics point for
the plots is still recorded every `log_interval` steps (default 100). HTTP
range-request and dataset backend INFO chatter is suppressed; warnings still
appear. Every `sample_interval` steps (default 1000) the script generates a
short continuation of `sample_prompt` and logs it, which is the easiest way
to eyeball whether the model is improving. Loss is noisy step-to-step; the
trend is what matters, and the plots include a smoothed line.

---

## 8. Resume after a Kaggle restart

Repeat sections 2–4, then run the same command:

```python
main([])
```

On startup the script uses `checkpoints/minillm_latest.pt` if it is present
and valid, and only then falls back to downloading `minillm_latest.pt` from
the HF repo. This prevents a bad remote upload from overwriting a good local
checkpoint. Checkpoints containing NaN/Inf weights are rejected. It restores
model weights, optimizer state, step, token count, RNG state, and the metrics
history used by the plots.

To intentionally ignore every checkpoint and start a new run, use `--new`
(keep HF uploads) or `--no-hf` (no HF access at all).

The data stream restarts from the beginning of OpenWebText; exact dataset
position is not persisted. With a shuffled stream this is acceptable for
continued pretraining.

---

## 9. Outputs

Local, under the project root:

```text
checkpoints/minillm_latest.pt
out/metrics.json
out/token_seen_vs_loss.png
out/token_seen_vs_perplexity.png
out/training_curves.png
```

Remote, in `sddfhy546y/minillm`:

```text
minillm_latest.pt
out/metrics.json
out/token_seen_vs_loss.png
out/token_seen_vs_perplexity.png
out/training_curves.png
```

Checkpoints and plots are written/uploaded every 60 minutes, at normal exit,
when you run `--stop` from a notebook, on Ctrl+C/SIGTERM in a terminal, when
the `max_hours` time limit or `--budget` token limit is reached, and on any
caught crash. Unless `--no-hf` is passed, the final sync uploads the newest
finite checkpoint and `out/` files on every one of these paths. Use
`--detach` in a notebook: Jupyter force-kills a plain `!python` child 0.2s
after Ctrl+C, so that child cannot finish its upload.

The `out/` files are uploaded first because they are small. The model and
`out/` uploads are tracked separately and retried independently, so a failed
model upload never marks the plots as uploaded (or the reverse). Transient
Hub/network errors are retried three times by default
(`hf.upload_retries` / `hf.upload_retry_backoff`). If you press Ctrl+C again
while the final 1.2 GB upload is running, the signal is deferred until both
the model and `out/` files finish; the same applies to SIGTERM. Checkpoint,
metrics, and plots are written atomically, and stale `.tmp` files from an
interrupted save are removed automatically on the next save. If your Kaggle
session often dies before one hour, change `checkpoint_interval_minutes` to
10.

---

## 10. `configs/model.yaml` — model architecture

| Key | Current value | Meaning | Change carefully |
| --- | --- | --- | --- |
| `vocab_size` | `50257` | GPT-2 vocabulary size. Token ids from the tokenizer go up to `50256`. | Keep at `50257` for OpenWebText unless you also change tokenization. Lowering it breaks token ids. |
| `context_length` | `128` | Maximum sequence length and KV-cache window. | Larger means more memory per attention step. `block_size` in the data config must be `<=` this. |
| `emb_dim` | `768` | Hidden width. | Must be divisible by `n_heads`. Reducing it makes the model much smaller/faster. |
| `n_heads` | `12` | Attention heads. | Must divide `emb_dim`. |
| `n_layers` | `9` | Transformer blocks. | Reducing it makes the model smaller/faster. |
| `drop_rate` | `0.1` | Dropout probability. | `0.0`–`0.1` is normal for pretraining. |
| `qkv_bias` | `false` | Bias in Q/K/V projections. | Keep unless matching a specific checkpoint. |
| `tie_weights` | `true` | Share token embedding and output projection. | Keep `true`; saves ~19M parameters. |

**Important:** if you change any architecture field, an existing HF
checkpoint will fail the config-mismatch check and training will start from
scratch. To really start fresh, also delete or rename the remote
`minillm_latest.pt` (or change `hf.filename` in `configs/pretrain.yaml`).

### Simple smaller-model recipe

Edit `configs/model.yaml`:

```yaml
context_length: 128
emb_dim: 256
n_heads: 4        # 256 / 4 = 64, valid
n_layers: 4
```

Leave `vocab_size: 50257` and `tie_weights: true`.

---

## 11. `configs/pretrain.yaml` — training and HF sync

### Hardware / speed

| Key | Current | Meaning / simple change |
| --- | --- | --- |
| `seed` | `1337` | Reproducibility seed. |
| `device` | `auto` | `auto`, `cpu`, `cuda`, or `cuda:1`. `auto` picks GPU if present. |
| `num_gpus` | `2` | Max GPUs used by `DataParallel`. Set `1` for one GPU. |
| `amp` | `true` | Mixed precision on CUDA. Disable only if you hit AMP issues. |
| `amp_dtype` | `auto` | `auto`, `bf16`, `fp16`, `fp32`. Kaggle T4 usually uses `fp16`. |
| `batch_size` | `8` | Sequences per micro-batch. Lower it if you get CUDA OOM. |
| `grad_accum_steps` | `1` | Micro-batches per optimizer step. Raise it to simulate a larger batch. |
| `max_steps` | `null` | `null` trains until interrupted, time limit, or data exhausted. Set an integer for a fixed step count. |
| `max_hours` | `null` | `null` = no time limit. Set `6` or `12` to stop after that many wall-clock hours and save/upload the final checkpoint. CLI: `--hours 6`. |
| `token_budget` | `null` | `null` = no token limit. Set e.g. `15000000` to stop at that many tokens. CLI: `--budget 15000000`. |
| `estimated_total_tokens` | `8972315576` | Only the default bar denominator (tokens seen / one estimated OpenWebText pass). Does not stop training. |
| `grad_clip` | `1.0` | Gradient clipping. Leave as is. |
| `progress` | `true` | Uses `tqdm.auto` for notebook-friendly bars. |

### Optimizer / schedule

| Key | Current | Meaning / simple change |
| --- | --- | --- |
| `learning_rate` | `3.0e-4` | Peak LR. Lower for more stable long runs. |
| `min_lr` | `3.0e-5` | Final cosine LR. |
| `warmup_steps` | `200` | Linear warmup steps. |
| `weight_decay` | `0.1` | AdamW weight decay. |
| `betas` | `[0.9, 0.95]` | AdamW momentum betas. |
| `log_first_steps` | `10` | Print a full INFO line every step for the first N steps. |
| `log_interval` | `100` | After the first N steps, record a metrics/plot point every N steps (the live bar already shows the metrics). Lower for denser plots. |

### Sample generations

| Key | Current | Meaning / simple change |
| --- | --- | --- |
| `sample_interval` | `1000` | Generate a sample every N steps. `0` disables it. |
| `sample_prompt` | `"Albert Einstein is a"` | Fixed prompt used for the sample. |
| `sample_max_new_tokens` | `60` | Continuation length. Longer is slower at each sample. |
| `sample_temperature` | `0.8` | `0.0` = greedy. |
| `sample_top_k` | `50` | `null`/`0` disables top-k filtering. |

### Checkpoints / outputs

| Key | Current | Meaning / simple change |
| --- | --- | --- |
| `checkpoint_interval_minutes` | `60` | Save/upload interval in minutes. Set `10` on short Kaggle sessions. |
| CLI: `--checkpoint-seconds` | unset | Testing override in seconds, e.g. `10`. Sets `checkpoint_interval_minutes = seconds/60`. Do not use for long runs. |
| `checkpoint_dir` | `checkpoints` | Local checkpoint folder. |
| `out_dir` | `out` | Local metrics/plots folder. Same name is used in HF by default. |
| `save_optimizer` | `true` | Exact resume. Set `false` for much smaller checkpoints if a fresh optimizer is acceptable. |

### Data

| Key | Current | Meaning / simple change |
| --- | --- | --- |
| `data.dataset_name` | `Skylion007/openwebtext` | Streaming HF dataset. |
| `data.split` | `train` | Dataset split. |
| `data.block_size` | `null` | `null` uses `model.context_length`. Must be `<= context_length`. |
| `data.shuffle_buffer` | `1000` | Local shuffle buffer. `0` disables shuffling (useful for tests). |
| `data.max_docs` | `null` | `null` = unlimited. Set `3` for a tiny smoke test. |
| `data.min_chars` | `1` | Skip very short docs. |

### HuggingFace

| Key | Current | Meaning / simple change |
| --- | --- | --- |
| `hf.repo_id` | `sddfhy546y/minillm` | Target HF repo. Do not include the full URL. |
| `hf.token_env` | `HF_TOKEN` | Environment variable name. Must match the Kaggle secret. |
| `hf.filename` | `minillm_latest.pt` | Checkpoint filename in the repo root. |
| `hf.out_dir` | `out` | Folder inside the HF repo for metrics/plots. |
| `hf.private` | `true` | Used when the script creates a repo. Existing repo visibility is unchanged. |
| `hf.upload_retries` | `3` | Retries per upload on a transient Hub/network error. Auth/not-found errors are not retried. |
| `hf.upload_retry_backoff` | `2.0` | Seconds before the first retry; the delay doubles each attempt. |

To disable HF entirely for a run, pass `--no-hf` rather than editing the file.
`--no-hf` also ignores local and HF checkpoints, so the run always starts
from a random model. Use `--new` instead when you want a fresh model but
still want the checkpoint and plots uploaded to HF.

---

## 12. Simple change recipes

| Goal | Change |
| --- | --- |
| Notebook-safe long run | `!python -m pretrain.train --detach --new --require-hf` |
| Stop now and upload | `!python -m pretrain.train --stop` |
| Check upload configuration | `!python -m pretrain.train --check-hf` |
| Fast smoke run | `max_steps: 20`, `batch_size: 2`, `data.max_docs: 5` |
| Stop after 6 hours | `max_hours: 6` or CLI `main(["--hours", "6"])` |
| Stop after 12 hours | `max_hours: 12` or CLI `main(["--hours", "12"])` |
| Stop after 15M tokens | `token_budget: 15000000` or CLI `main(["--budget", "15000000"])` |
| Smaller model | `emb_dim: 256`, `n_heads: 4`, `n_layers: 4` (keep `vocab_size: 50257`) |
| Save more often | `checkpoint_interval_minutes: 10` |
| Denser plots | `log_interval: 1` or `2` |
| Sample more often | `sample_interval: 500` |
| Change the sample prompt | `sample_prompt: "The meaning of life is"` |
| Disable samples | `sample_interval: 0` |
| One GPU | CLI: `main(["--device", "cuda", "--num-gpus", "1"])` |
| No HF upload | CLI: add `--no-hf` |
| Fresh model, keep HF upload | CLI: add `--new` |
| Fresh run from scratch | `--new`, or remove local `checkpoints/minillm_latest.pt` and delete/rename the HF `minillm_latest.pt` |
| Avoid CUDA OOM | Lower `batch_size` to `4`/`2`, or raise `grad_accum_steps` |

After editing YAML, rerun `main([])`; no code changes are needed.

---

## 13. How the data pipeline works

`pretrain/data.py` streams `Skylion007/openwebtext` lazily with
`datasets.load_dataset(..., streaming=True)`. It:

1. tokenizes documents with the GPT-2 BPE tokenizer (`tiktoken`),
2. appends token id `50256` (EOT) between documents,
3. keeps a rolling buffer of token ids,
4. emits `(x, y)` batches of length `block_size`, where `y` is `x` shifted by
   one token for next-token prediction.

The full dataset is never downloaded to disk. Only a few shards are cached
by HuggingFace under `~/.cache/huggingface`. The first batch can take 15–30
seconds; later batches are fast.

---

## 14. KV-cached generation

`model/model.py` provides `GPT.generate()`. By default (`use_cache=True`) it
prefills the prompt once and decodes one token at a time using cached K/V.
It keeps learned absolute positions correct. Once the cache reaches
`context_size`, it falls back to full recomputation of the sliding window so
long generations still work. For the original all-recompute path, call:

```python
out = model.generate(idx, max_new_tokens=250, use_cache=False)
```

Training `forward(in_idx, y=...)` never uses the cache, so training behavior
is unchanged.

---

## 15. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `HF token environment variable 'HF_TOKEN' is not set` | Add the Kaggle Secret named `HF_TOKEN` and re-run the token cell. |
| `HF repo_id is empty` | Use `configs/pretrain.yaml` from this repo; it already contains `sddfhy546y/minillm`. |
| `HF repo ... not found` | Check the repo id and token permissions. |
| `CUDA out of memory` | Lower `batch_size`, lower `context_length`, or raise `grad_accum_steps`. |
| `loss nan` / `ppl nan` | Model diverged. Pull the latest code (GPT-2 init + bad-step skipping), delete the NaN `minillm_latest.pt` locally and on HF, then restart. If it repeats, lower `learning_rate`. |
| `Refusing to save checkpoint ... non-finite` | Same as above: the trainer will not overwrite the good checkpoint with NaN weights. |
| `sequence length ... exceeds context_length` | Ensure `data.block_size <= model.context_length`. |
| Checkpoint config mismatch | You changed `configs/model.yaml`; start fresh or restore the old architecture. |
| First batch is slow | Normal; OpenWebText streaming fetches a shard. |
| `IterableDataset.shuffle() got an unexpected keyword 'max_buffer_input_shards'` | Pull the latest code; the data loader now falls back for older `datasets` versions used by Kaggle. |
| Only local files, no HF files | Run `!python -m pretrain.train --check-hf`; fix a missing/invalid token or repo permission. Also check `--no-hf` was not used. |
| Pressed Ctrl+C, but the model never appeared on HF | Jupyter force-kills a plain `!python` child 0.2s after Ctrl+C. Use `!python -m pretrain.train --detach ...`, then `!python -m pretrain.train --stop`; the trainer runs in its own session and finishes the upload. |
| `--check-hf` says `HF_TOKEN is NOT set` | Run the Kaggle Secrets cell in the same kernel before starting. A detached child inherits the kernel environment. |
| `--stop` says training is still running | Normal while the 1.2 GB checkpoint uploads. The process is detached; watch `logs/pretrain-*.log` for the `Final sync ...` line. |

---

## 16. Tiny tests before a long run

```python
!python -m compileall -q model pretrain code_test
!python code_test/test_train_tiny.py
!python code_test/test_kv_cache.py
!python code_test/test_data_pipeline.py
!python code_test/test_train_curves_real.py
```

These use tiny CPU models and do not upload anything. `test_train_curves_real.py`
reads a few real OpenWebText batches and verifies that `tokens_seen`
increases while loss/perplexity decrease.
