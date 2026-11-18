# minillm

A dense GPT language model trained from scratch on streaming OpenWebText.
The training script auto-selects 2 GPUs, 1 GPU, or CPU, saves checkpoints and
training plots every hour, and syncs them to HuggingFace Hub.

Logging is sparse after the first 10 steps (every 100 steps), and every 1000
steps the model generates a short continuation of a fixed prompt so you can
see whether it is improving.

## Quick start

```bash
python3 -m pip install -r requirements.txt
python -m pretrain.train
# or stop after 12 hours and upload the final checkpoint:
python -m pretrain.train --hours 12
# or stop after 15M tokens; the bar counts toward that budget:
python -m pretrain.train --budget 15000000
```

In Kaggle, follow the detailed [pretraining runbook](runbook/pretrain.md).

## Layout

| Path | Purpose |
| --- | --- |
| `model/model.py` | GPT model, forward pass, KV-cached generation |
| `model/model_parts.py` | Attention, MLP, LayerNorm, KV cache |
| `pretrain/train.py` | Training loop, checkpoints, HF upload/resume, plots |
| `pretrain/data.py` | Streaming OpenWebText tokenizer and batches |
| `configs/model.yaml` | Model architecture (~102.5M parameters) |
| `configs/pretrain.yaml` | Training, data, checkpoint, and HF settings |
| `runbook/pretrain.md` | Full Kaggle runbook and config guide |
| `code_test/` | Tiny CPU tests and smoke tests |

## Outputs

- `checkpoints/minillm_latest.pt` — local checkpoint
- `out/metrics.json` — tokens seen, loss, perplexity, LR, timestamp
- `out/token_seen_vs_loss.png`
- `out/token_seen_vs_perplexity.png`
- `out/training_curves.png`

With HF sync enabled, the checkpoint is uploaded to the repo root as
`minillm_latest.pt` and the `out/` files are uploaded to the repo's `out/`
folder.

## Tests

```bash
python code_test/test_train_tiny.py
python code_test/test_kv_cache.py
python code_test/test_data_pipeline.py
python code_test/test_train_curves_real.py
```

## Secrets

`HF_TOKEN` must be supplied through the environment (for example, a Kaggle
Secret) or a local `.env` file. `.env` is gitignored and is never committed.
