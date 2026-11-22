# Weight-space programming — direct weight transplant into a ~500M student

**Status:** plan only. No changes to existing `model/`, `pretrain/`, `distill/`, `sft/`, or
`configs/` files. All new code lives in `experiments/weightspace/`.

**Goal:** no fine-tuning, no training loop, no optimizer, no loss. Read a free
medical LLaMA-2 checkpoint from disk one tensor at a time, project its weights
into a smaller Llama-style student, and write the student checkpoint directly.

---

## 1. Why the student architecture must change

The source models are LLaMA-2 decoders. The current minillm is a GPT. Their
operators do not match, so weights cannot be copied faithfully.

| Current minillm GPT | LLaMA-2 source | Direct copy possible? |
| --- | --- | --- |
| LayerNorm (scale + shift) | RMSNorm (scale only) | No |
| Learned positional embedding | RoPE (no weights) | No |
| GELU 2-matrix FF | SwiGLU 3-matrix FF (`gate`, `up`, `down`) | No |
| Tied input/output embedding | Untied `embed_tokens` + `lm_head` | No |

Therefore the student is rebuilt as a **Llama-2-style decoder**. It is the same
kind of model as the source, only smaller. Weight editing then becomes
semantically exact: same operators, only different dimensions.

---

## 2. Source model — free weights, no HF token required

| Choice | Model | Architecture | Size | License |
| --- | --- | --- | --- | --- |
| Recommended | `nlpie/Llama2-MedTuned-13b` | hidden 5120, MLP 13824, 40 layers, 40 heads, vocab 32000 | ~26GB, three `.bin` shards | Apache-2.0 |
| Lighter | `malhajar/meditron-7b-chat` | hidden 4096, MLP 11008, 32 layers, 32 heads, vocab 32017 | ~13.5GB, safetensors | llama2 |

Both are medical-tuned LLaMA-2 models. `nlpie` is architecturally identical to
Me-LLaMA-13B, the original target. `malhajar` is smaller and uses safetensors.

Models that are *not* usable here:

- Me-LLaMA itself: HF repo has no weights; real files are on PhysioNet behind
  credentialing + DUA.
- Meditron 70B / Llama-3.1-405B / DeepSeek-R1: gated and/or far too large
  (70B ≈ 140GB, 405B ≈ 810GB, R1 ≈ 1.3TB).
- MedGemma 4B: gated and Gemma-3 architecture; usable for distillation, not for
  this weight transplant.
- GPT/Claude/Gemini/Grok/Med-PaLM: closed.

---

## 3. Target student — architecture + dimension changes

| Property | Source 13B | Student | Change |
| --- | --- | --- | --- |
| Hidden dim `D` | 5120 | **1280** | 4x smaller |
| MLP intermediate `I` | 13824 | **3456** | same ~2.7x ratio |
| Layers `L` | 40 | **21** | select 21 of 40 |
| Attention heads | 40 (head_dim 128) | **20** (head_dim 64) | 2x fewer |
| Vocab `V` | 32000 | **32000** | unchanged; same tokenizer |
| Embedding | untied | **untied** | copy both `embed_tokens` and `lm_head` |
| Positional | RoPE | **RoPE** | no learned weights; copy `rope_theta` |
| Context | 2048/4096 | **1024** | RoPE handles any length up to max |

Parameter count:

| Part | Params |
| --- | --- |
| `embed_tokens` | 32000 x 1280 = 41.0M |
| `lm_head` (untied) | 32000 x 1280 = 41.0M |
| 21 blocks x 19.83M | 416.3M |
| final RMSNorm | ~0.001M |
| **Total** | **498.3M** |

---

## 4. Dimension adjustment — the projection math

Two projection matrices, built once from source weights:

- `P`: `D_s x D_t` = `5120 x 1280`. Source hidden -> student hidden.
- `Q`: `I_s x I_t` = `13824 x 3456`. Source MLP intermediate -> student intermediate.

`P` and `Q` always have **orthonormal columns**, so `P.T P = I` and `Q.T Q = I`.
That is what makes the formulas below valid.

How to build them (pure linear algebra, no training):

| Method | Description | Quality |
| --- | --- | --- |
| slice | Take the first `D_t` / `I_t` identity columns | Fastest, arbitrary |
| random orthogonal | QR of a Gaussian `D_s x D_t` matrix | Decent mixing |
| **SVD (recommended)** | `torch.svd_lowrank` on stacked `q/k/v/o` matrices for `P`; on stacked `down_proj` matrices for `Q`; keep the top right-singular vectors | Best, since no fine-tuning will repair errors |

Layer selection: choose 21 of 40 layers evenly, preserving early/middle/late depth:

```text
[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
 21, 23, 25, 27, 29, 31, 33, 35, 37, 39]
```

For the 7B source (32 layers) the map is computed automatically with
`numpy.linspace(0, 31, 21).round()`.

---

## 5. Exact weight mapping

`P` is `D_s x D_t`, `Q` is `I_s x I_t`. For a source matrix `W_s`, the student
matrix is obtained by projecting input and output bases:

| Source tensor | Shape | Student tensor | Shape | Direct edit |
| --- | --- | --- | --- | --- |
| `embed_tokens.weight` | V x D_s | `embed_tokens.weight` | V x D_t | `E_s @ P` |
| `lm_head.weight` | V x D_s | `lm_head.weight` | V x D_t | `H_s @ P` |
| `input_layernorm.weight` | D_s | `input_layernorm.weight` | D_t | `P.T @ w_s` |
| `self_attn.q_proj.weight` | D_s x D_s | `q_proj.weight` | D_t x D_t | `P.T @ W_s @ P` |
| `self_attn.k_proj.weight` | D_s x D_s | `k_proj.weight` | D_t x D_t | `P.T @ W_s @ P` |
| `self_attn.v_proj.weight` | D_s x D_s | `v_proj.weight` | D_t x D_t | `P.T @ W_s @ P` |
| `self_attn.o_proj.weight` | D_s x D_s | `o_proj.weight` | D_t x D_t | `P.T @ W_s @ P` |
| `post_attention_layernorm.weight` | D_s | `post_attention_layernorm.weight` | D_t | `P.T @ w_s` |
| `mlp.gate_proj.weight` | I_s x D_s | `gate_proj.weight` | I_t x D_t | `Q.T @ W_s @ P` |
| `mlp.up_proj.weight` | I_s x D_s | `up_proj.weight` | I_t x D_t | `Q.T @ W_s @ P` |
| `mlp.down_proj.weight` | D_s x I_s | `down_proj.weight` | D_t x I_t | `P.T @ W_s @ Q` |
| `model.norm.weight` | D_s | `model.norm.weight` | D_t | `P.T @ w_s` |

Neither model has bias terms in these linears. RoPE contributes no weights;
only `rope_theta` is copied into the student config. Tokenizer files are copied
unchanged so token IDs match exactly.

---

## 6. Streaming from disk — never load the source into RAM

Two passes, both streaming one tensor at a time:

**Pass 1 — build `P` and `Q`.**
Open shards with `safe_open(..., device="cpu")` for safetensors or
`torch.load(..., weights_only=True, map_location="cpu")` for `.bin` files.
Read only a few attention/MLP matrices, run low-rank SVD, save `P` and `Q`.

**Pass 2 — write student weights.**
Re-open the shards. For each needed tensor: read one tensor -> project ->
assign into the student state dict -> delete. Peak process RAM is one source
matrix (about 140MB), not 26GB.

Output:

```text
checkpoints/weightspace_student.pt
```

with keys `model`, `model_config`, `projection`, `source_layers`. Load it with
`LlamaForCausalLM.from_checkpoint()` in `experiments/weightspace/model.py`.

---

## 7. New files only

```text
experiments/plan/weightspaceprograaming.md   (this file)
experiments/weightspace/
    __init__.py
    model_parts.py          RMSNorm, RoPE, SwiGLU, attention, decoder layer, KV cache
    model.py                LlamaModel / LlamaForCausalLM, config, generate, checkpoint I/O
    weight_space.py         streaming reader, P/Q builders, layer selection, mapping, save
    config.yaml             student architecture config
    weight_space_config.yaml source/repo, format, projection method, layer count
    build_student.py        CLI: pass 1 + pass 2 -> student checkpoint
    test_weight_space_tiny.py tiny CPU end-to-end test with a random source model
    README.md               usage and notes
```

No existing project file is modified.

---

## 8. What to expect

- A ~498M Llama-2-architecture student with context 1024.
- 100% of weights produced by linear projection of a medical source model.
- Zero gradient updates, zero training data, zero optimizer state.
- Can generate immediately after the conversion finishes.

Honest caveat: this is lossy compression. A 13B -> 500M projection cannot keep
all capability, and without any fine-tuning the result depends heavily on
projection quality. SVD projection retains the highest-energy weight
directions, which is why it is the recommended default.

---

## 9. Usage outline

```bash
# 1. tiny CPU sanity test (no downloads)
python experiments/weightspace/test_weight_space_tiny.py

# 2. edit experiments/weightspace/weight_space_config.yaml to point at the source

# 3. build the student (downloads source shards to disk, streams them)
python experiments/weightspace/build_student.py --config experiments/weightspace/weight_space_config.yaml

# 4. load and generate
python -c "
from experiments.weightspace.model import LlamaForCausalLM
m = LlamaForCausalLM.from_checkpoint('checkpoints/weightspace_student.pt')
print(m.generate_text('The symptoms of pneumonia include',
                      tokenizer_dir='checkpoints/tokenizer'))
"
```
