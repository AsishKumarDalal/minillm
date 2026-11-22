# Tool-calling plan for minillm

This document describes the full pipeline, the code that was added, the data
format, and the exact commands to run.

---

## 1. Final model configuration

`configs/model.yaml` was changed back to the original width, with a longer
context so tool prompts fit:

```yaml
vocab_size: 50257
context_length: 1024
emb_dim: 768
n_heads: 12          # head_dim = 64
n_layers: 9
drop_rate: 0.1
qkv_bias: false
tie_weights: true
```

Parameter count: about **103M** (same order as the original 102.5M model).
Because `context_length` changed from 128 to 1024, the position embeddings no
longer match the old checkpoint. **A fresh pretrain is required.**

Training memory: BF16 weights ~0.2 GB; with AdamW states roughly ~0.8 GB.
Both T4s can run this easily. GPU 0 is often busy with the local Inkling
server, so use `device: cuda:1` for tests.

---

## 2. Overall checkpoint flow (no RL)

```text
fresh pretrain
   │  python -m pretrain.train
   ▼
checkpoints/minillm_latest.pt
   │  python -m distill.distill        (normal/logit distillation)
   ▼
distill_checkpoints/distilled_model.pt
   │  python -m sft.tuning             (instruction SFT)
   ▼
sft_checkpoints/sft_model.pt
   │  python -m tool_calling.tool_calling_sft
   ▼
tool_calling_checkpoints/tool_sft_model.pt
   │  python -m tool_calling.negative_sampling
   ▼
tool_calling_checkpoints/negative_model.pt
   │  (optional) tool-calling distillation + another tool SFT
   ▼
final model
   │  python -m bench.eval             (written last, see section 9)
   ▼
metrics
```

There is deliberately **no PPO / GRPO / reward-model phase**. Tool calling is
a supervised format-and-decision skill. The negative phase uses rejection
sampling + DPO, which is much cheaper and more stable at this scale.

---

## 3. New code (`tool_calling/`)

| File | Purpose |
|---|---|
| `tool_calling/__init__.py` | package docstring |
| `tool_calling/data_format.py` | compact protocol, OpenAI-row conversion, parser, tokenisation helper |
| `tool_calling/runtime.py` | tool registry, parser, executor, two demo tools |
| `tool_calling/tool_calling_sft_data.py` | JSONL → `(input_ids, labels)` with assistant-only loss |
| `tool_calling/hf_dataset.py` | xLAM/Glaive row converters + streamed HF dataset |
| `tool_calling/convert_hf_dataset.py` | Optional: convert streamed HF rows to a local JSONL snapshot |
| `tool_calling/tool_calling_sft.py` | SFT trainer reusing `sft.tuning.SFTTrainer` |
| `tool_calling/tool_calling_distill.py` | teacher generates tool traces → JSONL → student SFT |
| `tool_calling/negative_sampling.py` | rejection sampling + DPO |

New configs:

| Config | Purpose |
|---|---|
| `configs/tool_calling_sft.yaml` | tool SFT hyperparameters and HF paths |
| `configs/tool_calling_distill.yaml` | teacher API, tools, prompts, dataset path |
| `configs/negative_sampling.yaml` | sampling and DPO hyperparameters |
| `configs/tool_calling_sft_stream.yaml` | streamed xLAM/Glaive tool SFT config |

New seed data:

| Path | Purpose |
|---|---|
| `data/tool_prompts.jsonl` | prompts for the teacher |
| `data/tool_traces.jsonl` | four hand-written traces to start SFT |

New test:

| Path | Purpose |
|---|---|
| `code_test/test_tool_calling_tiny.py` | tiny CPU/GPU test of format, parser, SFT, DPO |

The code follows the existing conventions:

- configs are YAML with the same keys as `sft_config.yaml` / `distill.yaml`
- student loading uses `load_student_model` / `load_sft_state` from `sft.tuning`
- training loop, checkpoints, plots, HF upload use `SFTTrainer`
- HF upload helpers and logging come from `pretrain.train`
- CLI flags mirror `sft.tuning`: `--steps`, `--epochs`, `--lr`, `--no-hf`,
  `--upload`, `--new`, `--fullnew`

---

## 4. Data format for tool calling

We use a compact text protocol because full JSON schemas waste context:

```text
TOOLS: weather(city),calc(expr)
USER: weather in Paris?
ASSISTANT: CALL weather {"city":"Paris"}
RESULT: {"temp_c":18}
ASSISTANT: 18C in Paris.
```

Rules:

- `TOOLS:` lists the available tools and their arguments.
- `USER:` is the user request.
- `ASSISTANT:` is what the model must generate.
- `RESULT:` is written by the Python tool executor, not by the model.
- Loss is computed **only** on `ASSISTANT:` lines. `TOOLS:`, `USER:` and
  `RESULT:` tokens are labelled `-100`.
- A GPT-2 EOT token (`50256`) is appended after every assistant answer.
- Examples that do not need a tool have `call: null` and a direct answer.

JSONL schema:

```json
{"tools":"weather(city)|calc(expr)",
 "user":"weather in Paris?",
 "call":"CALL weather {\"city\":\"Paris\"}",
 "result":"{\"temp_c\":18}",
 "answer":"18C in Paris.",
 "source":"seed"}
```

### OpenAI-style conversion

`from_openai_row()` converts xlam / glaive / raw OpenAI rows into this schema.
It accepts:

- `{"query": ..., "tools": [...], "answers": [{"name":..., "arguments":...}]}`
- `{"messages": [{"role":"user",...}], "tool_calls":[{"function":{...}}]}`

Tool schemas are flattened to `name(p1,p2)`. Tool calls become
`CALL name {json}`. You should execute the tool and pass the real result via
`tool_result=`; otherwise an empty `{}` is written.

---

## 4b. Local JSONL vs streamed HF data

There are two ways to feed tool SFT:

1. **Local JSONL (`data.source: jsonl`)** — reads `data/tool_traces.jsonl`.
   The four committed rows are only seeds for testing. Use
   `tool_calling/convert_hf_dataset.py` to build a larger local snapshot:

   ```bash
   python -m tool_calling.convert_hf_dataset --dataset xlam --max-docs 2000
   python -m tool_calling.convert_hf_dataset --dataset glaive --max-docs 2000 \
       --output data/tool_traces_glaive.jsonl
   ```

2. **Streamed HF (`data.source: hf`)** — recommended. Rows are downloaded one
   at a time and converted on the fly during SFT, exactly like pretraining and
   Dolly SFT. Use `configs/tool_calling_sft_stream.yaml`:

   ```bash
   python -m tool_calling.tool_calling_sft \
       --config configs/tool_calling_sft_stream.yaml
   ```

   Set `data.dataset: xlam` or `data.dataset: glaive`. `data.max_docs` caps the
   stream. Use `epochs: 1` with streaming; the HF stream is consumed once.

   **Note:** `Salesforce/xlam-function-calling-60k` is a gated dataset. You must
   accept its license on HuggingFace and set `HF_TOKEN` before streaming it.
   `glaiveai/glaive-function-calling-v2` worked without authentication in
   testing and is the easiest default.

Converters live in `tool_calling/hf_dataset.py`:

- xLAM: parses `query`, `tools`, `answers` (each may be a JSON string).
- Glaive: parses the function schemas from `system` and the multi-turn `chat`
  field, extracting `<functioncall>` and `FUNCTION RESPONSE` turns.

Both emit the same compact `ToolExample` schema as the local JSONL reader.

## 5. Phase-by-phase commands

### 5.1 Fresh pretrain

```bash
python -m pretrain.train --budget 200000000 --no-hf
```

Use `--detach` in a notebook. Aim for at least 200M tokens; more is better.

### 5.2 Normal (logit) distillation

```bash
python -m distill.distill --teacher openai-community/gpt2-xl --alpha 0.3 --no-hf
```

This improves general language quality. It is separate from tool-calling
distillation and can be skipped if compute is tight.

### 5.3 Instruction SFT

```bash
python -m sft.tuning --epochs 1 --steps 1000 --no-hf
```

Use Dolly plus optional filtered OpenHermes / Tulu data. Output:
`sft_checkpoints/sft_model.pt`.

### 5.4 Tool-calling SFT

```bash
python -m tool_calling.tool_calling_sft --config configs/tool_calling_sft.yaml
python -m tool_calling.tool_calling_sft --steps 500 --no-hf
```

Student defaults to `sft_checkpoints/sft_model.pt`. Data defaults to
`data/tool_traces.jsonl`. Output:
`tool_calling_checkpoints/tool_sft_model.pt`.

HF paths (when `hf.enabled: true`):

- `tool_calling/tool_sft_model.pt`
- `tool_calling/sft_outs/`

### 5.5 Negative sampling + DPO

```bash
python -m tool_calling.negative_sampling --config configs/negative_sampling.yaml
python -m tool_calling.negative_sampling --collect-only --no-hf
python -m tool_calling.negative_sampling --train-only --steps 100
```

Stage 1 samples 8 candidates per prompt. Valid calls become *chosen*;
malformed calls, unknown tools, or wrong arguments become *rejected*.
Stage 2 trains DPO with a frozen copy of the SFT model as reference. No reward
model, no PPO.

HF paths:

- `tool_calling/negative_model.pt`
- `tool_calling/negative_outs/`

### 5.6 Optional tool-calling distillation

Use this only if tool accuracy is still poor after 5.4/5.5.

```bash
python -m tool_calling.tool_calling_distill \
  --config configs/tool_calling_distill.yaml --limit 500
```

The default teacher is the local Inkling-2-Mini OpenAI-compatible server at
`http://127.0.0.1:8080/v1`. It generates tool calls, the code validates them by
executing the configured stub tools, writes `data/tool_traces.jsonl`, and then
runs tool SFT again.

HF paths:

- `tool_calling/distill_data/tool_traces.jsonl`
- plus the student checkpoint paths from 5.4.

---

## 6. HF upload layout

All new artifacts go under `tool_calling/` in the configured HF repo:

```text
tool_calling/
  tool_sft_model.pt
  sft_outs/            metrics + plots
  negative_model.pt
  negative_outs/
  distill_data/
    tool_traces.jsonl
```

`HF_TOKEN` must be set in the environment or `.env`. Use `--no-hf` for local
runs and `--upload` to force uploads. Uploads use the existing retrying
helpers in `pretrain.train`.

---

## 7. Testing

After every code change run the full tiny suite:

```bash
python -m py_compile model/*.py pretrain/*.py sft/*.py tool_calling/*.py code_test/*.py
python code_test/test_train_tiny.py
python code_test/test_kv_cache.py
python code_test/test_data_pipeline.py
python code_test/test_distill_tiny.py
python code_test/test_sft_tiny.py
python code_test/test_train_curves_real.py
python code_test/test_tool_calling_tiny.py
```

`test_tool_calling_tiny.py` verifies:

1. `parse_call` handles valid and invalid lines.
2. OpenAI-style rows convert to the compact format.
3. Prompt/result tokens are masked; assistant tokens are trained.
4. The tool registry executes calls and rejects unknown tools.
5. A tiny tool SFT run completes on GPU (`cuda:1` when available) and saves a checkpoint.
6. Negative sampling and one DPO step complete and save a model.

All of these passed at the time this document was written.

---

## 8. Known limitations

- The model is small. Expect it to learn the *format* and a handful of tools,
  not general tool use.
- Prompt/result markers are plain text, not special tokens. This keeps the
  checkpoint compatible with the GPT-2 vocabulary.
- Segment-wise tokenisation can differ slightly from whole-string
  tokenisation at boundaries. For a prototype it is acceptable; for a
  production run, tokenize the full string and use offsets.
- The DPO reference model doubles memory during that phase. At 103M params
  this is fine on a T4.
- Tool execution must be sandboxed. Never run untrusted tool calls.

---

## 9. Benchmark code (write last)

`bench/eval.py` should be written only after the training code above is final.
It should evaluate on 50–100 held-out prompts and report:

| Metric | Meaning |
|---|---|
| parse rate | output matches `CALL name {json}` |
| tool accuracy | correct tool chosen |
| JSON validity | arguments parse |
| argument exact match | correct values |
| final answer correctness | answer is right |
| false-positive rate | called a tool when it should not |

Compare: prompt-only baseline → tool SFT → SFT + negative/DPO → optional
tool distillation.

---

## 10. Summary

- Final config: 768 dim / 12 heads / 1024 context.
- Fresh pretrain, then normal distillation, then instruction SFT.
- Tool calling: compact protocol, assistant-only loss, `tool_calling/` package.
- Negative phase: rejection sampling + DPO, not RL.
- Optional behavioural tool distillation from a teacher if needed.
- Benchmark code last.
- All artifacts upload under `tool_calling/` on HuggingFace.
