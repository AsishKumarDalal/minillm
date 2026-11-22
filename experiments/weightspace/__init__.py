"""Self-contained weight-space programming experiment for minillm.

This package is isolated from the rest of the repository. It rebuilds the
student as a scaled-down Llama-2 decoder so that weights from a medical
LLaMA-2 checkpoint can be projected directly into it, with no training.
"""

from .model import LlamaConfig, LlamaForCausalLM, LlamaModel
from .weight_space import (
    SourceWeights,
    build_projections,
    build_student,
    select_layers,
)

__all__ = [
    "LlamaConfig",
    "LlamaModel",
    "LlamaForCausalLM",
    "SourceWeights",
    "build_projections",
    "build_student",
    "select_layers",
]
