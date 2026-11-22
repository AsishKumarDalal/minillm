# Weight-space programming experiment

Pure weight editing. **No training, no fine-tuning, no optimizer, no dataset.**

This folder is isolated. It does not import or modify the project's `model/`,
`pretrain/`, `distill/`, `sft/`, or `configs/` directories.

Full plan: `experiments/plan/weightspaceprograaming.md`.

## Idea

Take a free medical LLaMA-2 checkpoint, stream its weights from disk one tensor
at a time, and project them into a smaller Llama-2-style student. Because the
student uses the same operators (RMSNorm, RoPE, SwiGLU, untied embeddings),
the weight mapping is exact apart from the dimension reduction.

## Files

| File | Purpose |
| --- | --- |
| `model_parts.py` | RMSNorm, RoPE, SwiGLU, attention, decoder layer, KV cache |
| `model.py` | `LlamaModel` / `LlamaForCausalLM`, config, checkpoint I/O, generation |
| `weight_space.py` | streaming reader, P/Q projection builders, layer selection, weight mapping |
| `config.yaml` | student architecture (~498M, ctx 1024) |
| `weight_space_config.yaml` | source model, projection method, layer count, output path |
| `build_student.py` | CLI entry point: source shards -> student checkpoint |
| `test_weight_space_tiny.py` | tiny CPU end-to-end test, no downloads |

## Quick test (no download)

```bash
python experiments/weightspace/test_weight_space_tiny.py
```

## Real conversion

1. Download a free source model, e.g.:

   ```bash
   huggingface-cli download nlpie/Llama2-MedTuned-13b \
       --local-dir /root/.cache/nlpie-13b
   ```

2. Edit `weight_space_config.yaml`:

   - `source.path` → local directory with the shards
   - `source.tokenizer_dir` → same directory (tokenizer files are copied)
   - `source.hidden_size` / `intermediate_size` / `num_hidden_layers` / `vocab_size`
   - `student.vocab_size` must equal `source.vocab_size`

3. Run:

   ```bash
   python experiments/weightspace/build_student.py \
       --config experiments/weightspace/weight_space_config.yaml
   ```

4. Load:

   ```python
   from experiments.weightspace.model import LlamaForCausalLM

   m = LlamaForCausalLM.from_checkpoint("checkpoints/weightspace_student.pt")
   text = m.generate_text(
       "The symptoms of pneumonia include",
       tokenizer_dir="checkpoints/tokenizer",
       max_new_tokens=80,
   )
   print(text)
   ```

## Projection methods

| Method | Command flag | Notes |
| --- | --- | --- |
| SVD | `--method svd` | Recommended. Keeps highest-energy weight directions. |
| Random orthogonal | `--method random` | Cheap; mixes all dimensions. |
| Slice | `--method slice` | Fastest; keeps first N dimensions only. |

## Memory note

The source 13B model is never loaded as a whole. `SourceWeights.iter_tensors()`
opens one shard, yields one tensor, then discards it. Peak process RAM is about
the size of the largest source matrix (~140MB), not 26GB.

## Caveat

This is lossy compression, not a behaviour clone. There is no training step to
repair projection errors, so result quality depends heavily on the projection
method. SVD is the default for that reason.
