# SFT runbook (Kaggle)

Supervised fine-tuning of the mini GPT on streamed
`databricks/databricks-dolly-15k`. See [`sft/README.md`](../sft/README.md)
for the data-triage details.

## 1. Setup

```python
!pip install -q -r requirements.txt
```

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

## 2. Configure `configs/sft_config.yaml`

```yaml
epochs: 1
steps: null          # set a number to cap optimizer steps

batch_size: 4
learning_rate: 2.0e-5

checkpoint_interval_minutes: 60
checkpoint_dir: sft_checkpoints
out_dir: sft_outs

hf:
  enabled: false     # --upload turns this on
  filename: sft_model.pt
  out_dir: sft_outs
```

Dolly is streamed — no dataset download. Runtime triage keeps the full
instruction/context/answer when it fits in 128 tokens, else drops the context,
else drops the example. Loss is computed only on the answer tokens.

## 3. Start / check / stop

```python
# Continue from a previous SFT model if one exists, else the base student
!python -m sft.tuning --detach --epochs 1

# Fresh SFT from the original HF student
!python -m sft.tuning --detach --new

# Random student (ignore all checkpoints)
!python -m sft.tuning --detach --fullnew

!python -m sft.tuning --status
!python -m sft.tuning --stop
```

Always use `--detach` in a notebook; a plain `!python` child is force-killed by
Jupyter 0.2s after Interrupt, before the final save/upload.

## 4. Outputs and upload

Local:

```text
sft_checkpoints/sft_model.pt
sft_outs/metrics.json
sft_outs/token_seen_vs_loss.png
sft_outs/token_seen_vs_perplexity.png
sft_outs/step_vs_loss.png
```

Every 60 minutes (and on clean stop), the model and `sft_outs/` are saved. Add
`--upload` to push them to `sddfhy546y/minillm/sft_model.pt` and
`sddfhy546y/minillm/sft_outs/`. Upload is off by default.

## 5. Tests

```bash
python code_test/test_sft_tiny.py
python code_test/smoke_sft_real_data.py
```
