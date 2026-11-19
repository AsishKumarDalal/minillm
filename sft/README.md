# Supervised fine-tuning (SFT)

SFT fine-tunes our GPT on `databricks/databricks-dolly-15k`. The dataset is
**streamed** from HuggingFace — it is never downloaded in full.

The student starts from `sddfhy546y/minillm` by default (or a local copy), and
the whole pipeline mirrors pretraining/distillation: periodic 60-minute saves,
a clean-stop final sync, optional HF upload, detached notebook runner, and loss
plots.

## Runtime triage (`SFTData`)

Every example is triaged when it is read, using GPT-2 BPE token ids and
`block_size` (default 128):

1. `instruction + context + response + EOT` fits → keep everything.
2. Otherwise `instruction + response + EOT` fits → drop the context.
3. Otherwise drop the example.

The prompt is formatted as:

```text
Instruction:
<instruction>

Context:
<context>

Response:
```

(when the context is dropped, the `Context:` block is omitted). The response
and the GPT-2 EOT token (`50256`) are the targets. Prompt tokens are labelled
`-100`, so cross-entropy is computed **only on the answer**.

Batches are padded on the right with EOT tokens (labels `-100`); causal
attention means the padding cannot affect real tokens. `SFTData.stats` reports
`seen`, `full`, `answer_only`, and `dropped` counts.

## Config (`configs/sft_config.yaml`)

| Key | Default | Meaning |
| --- | --- | --- |
| `data.dataset_name` | `databricks/databricks-dolly-15k` | Streamed HF dataset. |
| `data.block_size` | `null` | `null` uses `context_length` (128). |
| `data.shuffle_buffer` | `100` | `0` disables shuffling. |
| `data.max_docs` | `null` | Limit documents (for smoke tests). |
| `epochs` | `1` | Number of passes over the stream. `--epochs` overrides. |
| `steps` | `null` | Hard cap on optimizer steps. `--steps` overrides. |
| `batch_size` / `grad_accum_steps` | `4` / `1` | Effective batch size. |
| `learning_rate` | `2.0e-5` | Lower than pretraining, as usual for SFT. |
| `sample_interval` | `500` | Print a fixed-prompt sample every N steps; `0` disables. |
| `checkpoint_interval_minutes` | `60` | Save/upload interval. |
| `checkpoint_dir` | `sft_checkpoints` | Local folder for `sft_model.pt`. |
| `out_dir` | `sft_outs` | Local metrics and plots. |
| `hf.enabled` | `false` | Upload off by default; use `--upload` to enable. |
| `hf.filename` | `sft_model.pt` | Filename on HF when uploading. |
| `hf.out_dir` | `sft_outs` | Folder on HF for plots/metrics. |

`epochs` and `steps` work together: training stops when either is exhausted.

## Start behavior

| Command | Student weights |
| --- | --- |
| `python -m sft.tuning` | resume from local/HF `sft_model.pt`, else `minillm_latest.pt` |
| `python -m sft.tuning --new` | original student `minillm_latest.pt` |
| `python -m sft.tuning --fullnew` | random initialization, ignore all checkpoints |

## Running

Terminal:

```bash
python -m sft.tuning --epochs 1 --steps 1000
python -m sft.tuning --epochs 2 --steps 5000 --upload
python -m sft.tuning --fullnew --device cpu --steps 10
```

Notebook (use `--detach`; Jupyter kills plain `!python` children on Interrupt):

```python
!python -m sft.tuning --detach --epochs 1
!python -m sft.tuning --status
!python -m sft.tuning --stop
```

Quick 10-second timer test:

```python
!python -m sft.tuning --detach --checkpoint-seconds 10 --no-hf
```

Logs: `logs/sft-<timestamp>.log` · PID: `sft_checkpoints/train.pid`

## Outputs

```text
sft_checkpoints/sft_model.pt
sft_outs/metrics.json
sft_outs/token_seen_vs_loss.png
sft_outs/token_seen_vs_perplexity.png
sft_outs/step_vs_loss.png
```

With `--upload`, the model is pushed to `sddfhy546y/minillm/sft_model.pt` and
the whole `sft_outs/` folder to `sddfhy546y/minillm/sft_outs/`. This does not
overwrite `minillm_latest.pt`.

## Tests

```bash
python code_test/test_sft_tiny.py
python code_test/smoke_sft_real_data.py
```
