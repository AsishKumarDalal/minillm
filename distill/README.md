# Logit distillation

For a Kaggle step-by-step, see the [distillation runbook](../runbook/distill.md).

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

## Which teachers work (and why not Qwen / Llama / Mistral)

This is **logit distillation**, not text-generation distillation. We take a
softmax over the teacher's logits and compare it directly with the student's
softmax at the same position. That only makes sense when token id `12345`
means the same string in both models.

Our student uses the GPT-2 BPE tokenizer (`tiktoken` `gpt2`, 50,257 ids).
Therefore the teacher must also use GPT-2 BPE with 50,257 ids. That means the
**GPT-2 family** (or another checkpoint explicitly trained with the GPT-2
tokenizer). Models such as Qwen, Llama, Mistral, Gemma, Phi, etc. use totally
different tokenizers and vocabulary sizes (32k–150k+ ids). Their token ids do
not correspond to ours, so:

- their logit vector has the wrong width, and
- even if you padded/truncated it, the probabilities would be over different
  tokens and the KL loss would be meaningless.

The script refuses to start in that case with a vocab-mismatch error. To use
such a model you would need a different pipeline (sequence/KD-on-text, or
replacing the student tokenizer and output head), which is not what this
script does.

Also, the teacher should be **larger than the student** (102.5M params) so its
soft targets are actually useful. A tiny teacher gives little signal; a huge
one is slower and uses more memory.

| Teacher model | Params | Approx weights (fp32 / fp16) | When to use |
| --- | --- | --- | --- |
| `sshleifer/tiny-gpt2` | ~0.1M (mostly embeddings) | a few MB | Tests only. Real vocab, near-zero teacher quality. |
| `gpt2` | 124M | ~0.5 GB / ~0.25 GB | Quick runs and low-VRAM GPUs. Smallest useful teacher. |
| `gpt2-medium` | 355M | ~1.4 GB / ~0.7 GB | Good speed/quality balance. Works on most 8 GB GPUs. |
| `gpt2-large` | 774M | ~3.1 GB / ~1.5 GB | Stronger teacher; needs ~6–8 GB GPU memory. |
| `openai-community/gpt2-xl` (default) | 1.5B | ~6 GB / ~3 GB | Best GPT-2 teacher. Fine on a T4; can set `teacher.device: cpu` if needed. |

Change it in `configs/distill.yaml` or on the command line:

```bash
python -m distill.distill --teacher gpt2-medium
python -m distill.distill --teacher gpt2 --device cpu --max-steps 20
```

`teacher.dtype: auto` uses `bf16`/`fp16` on CUDA and `fp32` on CPU. Set
`teacher.device: cpu` to offload the teacher if GPU memory is tight.

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
| `device` / `num_gpus` | `auto` / `2` | Student uses `cuda:0` and `DataParallel` across up to `num_gpus` GPUs. Set `device: cpu` for CPU. |
| `teacher.device` | `null` | `null` = auto: `cuda:1` when 2+ GPUs are visible, otherwise the student's device. `"cpu"` offloads the teacher to CPU. |
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

## Resume after `--stop`

By default (no `--new`), distillation first looks for the newest
`distilled_model.pt`:

1. local `distill_checkpoints/distilled_model.pt`;
2. otherwise `hf.repo_id/distilled_model.pt` on HuggingFace.

If found, it restores model weights, `step`, `tokens_seen`, the metrics
history, and optimizer state when present. So this continues a stopped run:

```python
!python -m distill.distill --stop
!python -m distill.distill --detach
```

To deliberately throw away that progress and restart from the original
student (`minillm_latest.pt`), pass `--new`:

```python
!python -m distill.distill --detach --new
```

`--new` ignores both the local and HF `distilled_model.pt`, but it still loads
the original student from `minillm_latest.pt`.

To start from **random weights** and ignore every checkpoint (both the old
distilled model and the original student), pass `--fullnew`:

```python
!python -m distill.distill --detach --fullnew
```

| Flag | Starts from |
| --- | --- |
| *(none)* | local/HF `distilled_model.pt` if present, else `minillm_latest.pt` |
| `--new` | original student `minillm_latest.pt` |
| `--fullnew` | randomly initialized GPT (no checkpoint loaded) |

`--fullnew` takes priority over `--new` if both are passed.

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
distilled_outs/token_seen_vs_loss.png
distilled_outs/token_seen_vs_perplexity.png
distilled_outs/token_seen_vs_ce.png
distilled_outs/token_seen_vs_kl.png
```

The four plots show total distillation loss, perplexity, cross-entropy, and
KL divergence against tokens seen. Perplexity is computed from the CE term
only (`exp(ce)`); `exp(total_loss)` would not be a perplexity because the
total also contains the KL term.

When `hf.enabled: true` / `--upload` is used, the whole `distilled_outs/`
folder — metrics **and all four plots** — is uploaded to
`sddfhy546y/minillm/distilled_outs/` alongside `distilled_model.pt`.

`metrics.json` is a list of records with `step`, `tokens_seen`, `loss`, `ce`,
`kl`, `alpha`, `temperature`, and `timestamp`.

Every `sample_interval` steps (default **1000**), the trainer also generates a
short continuation of the fixed `sample_prompt` and writes it above the
progress bar / log:

```text
sample @ step 1000 | Albert Einstein is a German-born theoretical physicist ...
```

This is the quickest way to eyeball whether the student is improving. Set
`sample_interval: 0` in `configs/distill.yaml` to disable it. The relevant
config keys are `sample_prompt`, `sample_max_new_tokens`,
`sample_temperature`, and `sample_top_k`.

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
