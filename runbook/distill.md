# Distillation runbook (Kaggle)

This runbook covers distilling a pretrained HuggingFace teacher into our mini
GPT. For ordinary pretraining, see [`pretrain.md`](pretrain.md).

Distillation loads the **student** from `sddfhy546y/minillm` (or a local
copy) and freezes a pretrained **teacher** from HuggingFace. At every step:

```text
loss = (1 - alpha) * cross_entropy(student, labels)
     + alpha * KL(teacher logits || student logits)
```

Both models use GPT-2 token ids, so their logits line up directly.

## Teacher compatibility — use GPT-2 models only

This script does **logit** distillation: it compares the teacher's probability
distribution with the student's at the same token position. That requires both
models to use the exact same tokenizer and vocabulary. Our student uses GPT-2
BPE (50,257 ids), so the teacher must too.

**Do not use Qwen, Llama, Mistral, Gemma, Phi, etc.** They have different
tokenizers and vocabulary sizes (32k–150k+ ids). Their logits are over
different tokens, so the KL loss would be meaningless and the script will
stop with a vocab-mismatch error. Supporting them would need a different
text/sequence-level distillation pipeline or a replaced student tokenizer and
output head.

Suitable teachers (all GPT-2 tokenizer, 50,257 ids):

| Model | Params | Approx weights fp32 / fp16 | Use |
| --- | --- | --- | --- |
| `sshleifer/tiny-gpt2` | ~0.1M | a few MB | Tests only |
| `gpt2` | 124M | ~0.5 GB / ~0.25 GB | Fast, low VRAM, smallest useful teacher |
| `gpt2-medium` | 355M | ~1.4 GB / ~0.7 GB | Best speed/quality balance |
| `gpt2-large` | 774M | ~3.1 GB / ~1.5 GB | Stronger, needs ~6–8 GB GPU |
| `openai-community/gpt2-xl` (default) | 1.5B | ~6 GB / ~3 GB | Best GPT-2 teacher; fine on T4 |

The student is 102.5M parameters, so a teacher that is several times larger
gives a useful signal. Tiny teachers are only for smoke tests.

Change the teacher with:

```python
!python -m distill.distill --teacher gpt2-medium --detach
```

If GPU memory is tight, set `teacher.device: cpu` in `configs/distill.yaml`.

---

## 1. Setup

In the Kaggle notebook:

```python
!pip install -q -r requirements.txt
```

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

Verify the token before starting:

```python
!python -m pretrain.train --check-hf
```

---

## 2. Configure `configs/distill.yaml`

Key settings:

```yaml
teacher:
  name: openai-community/gpt2-xl   # default teacher; change if needed
  device: null                     # auto -> cuda:1 with two GPUs
  dtype: auto

student:
  repo_id: sddfhy546y/minillm
  filename: minillm_latest.pt

model_config: configs/model.yaml

alpha: 0.3        # KL weight; try 0.3, 0.4, or 0.6
temperature: 1.0

num_gpus: 2       # student uses DataParallel across visible GPUs

checkpoint_dir: distill_checkpoints
out_dir: distilled_outs

hf:
  enabled: false              # uploads OFF by default
  filename: distilled_model.pt
  out_dir: distilled_outs
```

Set `hf.enabled: true` only when you want the distilled model pushed to HF.

---

## 3. Choose how the student starts

| Command | Student weights |
| --- | --- |
| `python -m distill.distill` | Resume from local/HF `distilled_model.pt` if it exists; otherwise load `minillm_latest.pt` |
| `python -m distill.distill --new` | Ignore old distilled checkpoints; load original `minillm_latest.pt` |
| `python -m distill.distill --fullnew` | Random initialization; ignore **all** checkpoints |

`--fullnew` is what you want for a completely fresh distillation run.

---

## 4. Start, check, and stop

Always use `--detach` in a notebook. A plain `!python -m distill.distill`
child is force-killed by Jupyter 0.2s after Interrupt, before the final save.

```python
# Normal run (resumes if distilled_model.pt exists)
!python -m distill.distill --detach

# Completely fresh run
!python -m distill.distill --detach --fullnew

# Check progress
!python -m distill.distill --status

# Stop cleanly: saves distilled_model.pt and distilled_outs/, then exits
!python -m distill.distill --stop
```

For a short upload-timer test:

```python
!python -m distill.distill --detach --checkpoint-seconds 10 --no-hf
```

Remove `--checkpoint-seconds 10` for the real 60-minute interval.

Logs: `logs/distill-<timestamp>.log`  
PID: `distill_checkpoints/train.pid`

---

## 5. Resume after a stop

Just run again with no special flag:

```python
!python -m distill.distill --detach
```

It restores from:

1. local `distill_checkpoints/distilled_model.pt`
2. otherwise `sddfhy546y/minillm/distilled_model.pt` on HF

Restored state includes model weights, `step`, `tokens_seen`, metrics history,
and optimizer state when present.

---

## 6. Uploading

Upload is OFF unless you pass `--upload` or set `hf.enabled: true`.

```python
!python -m distill.distill --detach --upload
```

This uploads to:

```text
sddfhy546y/minillm/distilled_model.pt
sddfhy546y/minillm/distilled_outs/metrics.json
```

It does **not** overwrite `minillm_latest.pt` (the student source).

`--no-hf` forces upload OFF even if the YAML says true.

---

## 7. Outputs

Local:

```text
distill_checkpoints/distilled_model.pt
distilled_outs/metrics.json
distilled_outs/token_seen_vs_loss.png
distilled_outs/token_seen_vs_perplexity.png
distilled_outs/token_seen_vs_ce.png
distilled_outs/token_seen_vs_kl.png
```

These are the four graphs: total loss, perplexity (from CE), cross-entropy,
and KL divergence — all against tokens seen. With `--upload`, the whole
`distilled_outs/` folder is pushed to HF under `distilled_outs/`.

`metrics.json` records `step`, `tokens_seen`, `loss`, `ce`, `kl`, `alpha`,
`temperature`, and `timestamp`.

Every 1000 steps the trainer prints a fixed test prompt and its continuation:

```text
sample @ step 1000 | Albert Einstein is a ...
```

This is controlled by `sample_interval`, `sample_prompt`,
`sample_max_new_tokens`, `sample_temperature`, and `sample_top_k` in
`configs/distill.yaml`. Set `sample_interval: 0` to turn it off.

---

## 8. GPUs

- `num_gpus: 2` wraps the student in `DataParallel` across two GPUs.
- The teacher is auto-placed on `cuda:1` when two GPUs are visible.
- If only one GPU exists, both models share it.
- Set `teacher.device: cpu` in the YAML if the teacher does not fit in GPU
  memory.
- `device: cpu` forces a CPU-only run.

---

## 9. Tests

```bash
python -m py_compile distill/*.py code_test/*.py
python code_test/test_distill_tiny.py
python code_test/smoke_distill_real_teacher.py
```

The smoke test downloads a tiny GPT-2 teacher from HuggingFace and runs two
distillation steps on local text. It does not upload anything.
