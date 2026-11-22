"""Tool-calling pipeline for minillm.

Modules
-------
data_format
    Compact text protocol, OpenAI-tool-row conversion, and parsing.
tool_calling_sft_data
    JSONL -> (input_ids, labels) dataset with assistant-only loss masking.
hf_dataset
    xLAM/Glaive converters and a streamed HuggingFace dataset.
convert_hf_dataset
    Optional CLI to write a local JSONL snapshot from a streamed HF dataset.
tool_calling_sft
    SFT trainer that reuses ``sft.tuning.SFTTrainer``.
tool_calling_distill
    Teacher-generated tool traces -> JSONL -> student SFT.
negative_sampling
    Rejection sampling (chosen/rejected) plus DPO fine-tuning.
runtime
    Tool registry, parser, and executor used by the other modules.

HF artifacts are written under ``tool_calling/`` in the configured repo,
mirroring the existing ``dense/`` and ``moe/`` layouts.
"""
