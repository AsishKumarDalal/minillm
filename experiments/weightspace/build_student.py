#!/usr/bin/env python3
"""CLI: stream a medical LLaMA-2 checkpoint into the weight-space student.

Usage:
    python experiments/weightspace/build_student.py \
        --config experiments/weightspace/weight_space_config.yaml

No training is performed. The script only reads source tensors, projects them,
and writes a student checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .weight_space import build_student
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from weight_space import build_student


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(here / "weight_space_config.yaml"),
        help="weight-space conversion config",
    )
    parser.add_argument("--source", default=None, help="override source path")
    parser.add_argument("--output-dir", default=None, help="override output dir")
    parser.add_argument("--filename", default=None, help="override checkpoint filename")
    parser.add_argument(
        "--method", default=None, choices=["svd", "random", "slice"],
        help="override projection method",
    )
    args = parser.parse_args()

    import yaml as _yaml

    with open(args.config, encoding="utf-8") as fh:
        cfg = _yaml.safe_load(fh)

    if args.source:
        cfg["source"]["path"] = args.source
        cfg["source"].setdefault("tokenizer_dir", args.source)
    if args.method:
        cfg.setdefault("projection", {})["method"] = args.method

    # Write overrides to a temporary config so build_student stays config-driven.
    tmp_cfg = here / "_resolved_config.yaml"
    with open(tmp_cfg, "w", encoding="utf-8") as fh:
        _yaml.safe_dump(cfg, fh, sort_keys=False)

    out_dir = args.output_dir or cfg.get("output", {}).get("dir", "checkpoints")
    filename = args.filename or cfg.get("output", {}).get(
        "filename", "weightspace_student.pt"
    )

    path = build_student(str(tmp_cfg), output_dir=out_dir, filename=filename)
    print(f"Done: {path}")
    print("Load with: LlamaForCausalLM.from_checkpoint('" + str(path) + "')")


if __name__ == "__main__":
    main()
