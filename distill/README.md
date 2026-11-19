# Logit distillation

Distill a pretrained HuggingFace causal LM (the **teacher**) into our mini GPT
(the **student**). The student starts from `sddfhy546y/minillm` and is trained
on the same streaming OpenWebText batches used for pretraining.

The teacher is frozen. Only the student is optimized.

## Loss

For every batch both models see the same GPT-2 token ids. With student logits
`s`, teacher logits `t`, targets `y`, temperature `T`, and weight `alpha`:

```text
ce   = cross_entropy(s, y)
kl   = KL(softmax(t / T) || log_softmax(s / T)) * T^2
loss = (1 - alpha) * ce + alpha * kl
```

- `alpha` is the **weight of the KL/distillation term**.
  Try `0.3`, `0.4`, or `0.6` (default `0.3`).
- `temperature` softens the distributions before the KL. `1.0` uses plain
  softmax. Higher values expose more of the teacher's dark knowledge.
- `T^2` keeps KL gradients on a similar scale to CE when `T != 1`.

Both models must use the **same vocabulary**. The default teacher
`openai-community/gpt2-xl` and our student both use GPT-2 BPE (50,257 ids),
so their logits line up token-for-token. The script checks this at startup.

## Setup

Teacher inference needs `transformers`:

```bash
python3 -m pip install -r requirements.txt
```

Or, minimally:

```bash
python3 -m pip install transformers
```

The student checkpoint is downloaded from HuggingFace on first run. Set
`HF_TOKEN` (Kaggle Secret or `.env`) so the download is authenticated:

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

## Config (`configs/distill.yaml`)

| Key | Default | Meaning |
| --- | --- | --- |
| `teacher.name` | `openai-community/gpt2-xl` | HF model id for the frozen teacher. Change to any GPT-2-family causal LM with the same vocab. |
| `teacher.device` | `null` | `null` = same device as student; `"cpu"` offloads the teacher to save GPU memory. |
| `teacher.dtype` | `auto` | `auto`, `bf16`, `fp16`, `fp32` for teacher weights. |
| `teacher.context_length` | `null` | Only checked against `data.block_size`. |
| `student.source` | `hf` | `hf` downloads `student.filename`; `local` reads `student.local_path`. |
| `student.repo_id` | `sddfhy546y/minillm` | Repo holding the student checkpoint. |
| `student.filename` | `minillm_latest.pt` | Checkpoint file inside that repo. |
| `student.local_path` | `checkpoints/minillm_latest.pt` | Local copy; used first when `prefer_local: true`. |
| `student.prefer_local` | `true` | Avoid a download when a local checkpoint exists. |
| `model_config` | `configs/model.yaml` | Student architecture. |
| `alpha` | `0.3` | KL weight; `1-alpha` is the CE weight. Try `0.3`, `0.4`, `0.6`. |
| `temperature` | `1.0` | Logit softening for the KL term. |
| `learning_rate` | `1.0e-4` | Student optimizer LR (usually lower than pretraining). |
| `batch_size` / `grad_accum_steps` | `4` / `1` | Effective batch = product of the two. |
| `data.block_size` | `null` | `null` uses the student's `context_length` (128). |
| `checkpoint_interval_minutes` | `60` | Save/upload interval. |
| `checkpoint_dir` | `distill_checkpoints` | Local folder for `distilled_model.pt`. |
| `out_dir` | `distilled_outs` | Local folder for `metrics.json` and curves. |
| `hf.enabled` | `false` | Upload is off by default so a test run cannot overwrite the student. |
| `hf.filename` | `distilled_model.pt` | Filename used on HF when upload is enabled. |
| `hf.out_dir` | `distilled_outs` | Folder used on HF for metrics/plots. |

To upload the distilled model to the same repo under a **new filename**:

```yaml
hf:
  enabled: true
  repo_id: sddfhy546y/minillm
  filename: distilled_model.pt
  out_dir: distilled_outs
```

This does not overwrite `minillm_latest.pt` (the student).

## Running

Foreground (terminal):

```bash
python -m distill.distill
python -m distill.distill --alpha 0.4 --teacher gpt2
python -m distill.distill --device cpu --max-steps 20
```

Notebook-safe background run (use this in Kaggle/Jupyter — a plain
`!python ...` child is force-killed by Jupyter 0.2s after Ctrl+C):

```python
!python -m distill.distill --checkpoint-seconds 10 --detach --no-hf
!python -m distill.distill --status
!python -m distill.distill --stop
```

Remove `--checkpoint-seconds 10` for the real run so saves/uploads go back to
every 60 minutes. Remove `--no-hf` (or pass `--upload`) only when you want the
distilled checkpoint uploaded.

`--stop` sends SIGTERM; the trainer finishes the current step, saves locally,
and uploads before exiting. Logs go to `logs/distill-<timestamp>.log` and the
PID to `distill_checkpoints/train.pid`.

## Outputs

Local:

```text
distill_checkpoints/distilled_model.pt
distilled_outs/metrics.json
```

`metrics.json` is a list of records with `step`, `tokens_seen`, `loss`, `ce`,
`kl`, `alpha`, `temperature`, and `timestamp`.

If `hf.enabled: true`, the same files are uploaded to
`sddfhy546y/minillm/distilled_model.pt` and
`sddfhy546y/minillm/distilled_outs/`. The `distilled_outs/` files are uploaded
first, and the two uploads are tracked/retried independently.

The checkpoint format matches the pretraining checkpoints (`model`,
`model_config`, `step`, `tokens_seen`, `optimizer` if enabled, etc.), so a
distilled checkpoint can later be used as a student/resume checkpoint by
pointing `student.local_path` at it.

## Tests

```bash
python code_test/test_distill_tiny.py
python code_test/smoke_distill_real_teacher.py
```

The tiny test uses a local 64-vocab teacher and student, checks the CE/KL
weighting, and verifies checkpoint save/load. The smoke test downloads a small
real GPT-2 teacher from HuggingFace and runs two distillation steps on local
text; it does not upload anything.
