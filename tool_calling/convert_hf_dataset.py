"""Convert a streamed HF tool-calling dataset to compact JSONL.

This is a convenience script: ``tool_calling_sft.py`` can stream HF directly.
Use this when you want a fixed local snapshot of the converted data.

Usage
-----
::

    python -m tool_calling.convert_hf_dataset --dataset xlam --max-docs 2000
    python -m tool_calling.convert_hf_dataset --dataset glaive --max-docs 2000 \
        --output data/tool_traces_glaive.jsonl

xLAM is gated on HuggingFace; set HF_TOKEN and accept its license first.
Glaive works without authentication.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pretrain.train import setup_logging  # noqa: E402
from .data_format import write_jsonl  # noqa: E402
from .hf_dataset import HF_DATASET_NAMES, StreamingToolSFTData, convert_row  # noqa: E402


def main(argv=None) -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Convert HF tool data to compact JSONL.")
    parser.add_argument("--dataset", choices=["xlam", "glaive"], required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--shuffle-buffer", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    output = Path(args.output or f"data/tool_traces_{args.dataset}.jsonl")
    if not output.is_absolute():
        output = ROOT_DIR / output

    data = StreamingToolSFTData.from_hf(
        dataset_name=args.dataset_name or HF_DATASET_NAMES[args.dataset],
        dataset=args.dataset,
        split=args.split,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
        max_docs=args.max_docs,
    )

    examples = []
    for row in data.rows:
        if args.max_docs is not None and len(examples) >= args.max_docs:
            break
        examples.extend(convert_row(row, args.dataset))

    count = write_jsonl(examples, output)
    logging.info("Wrote %d compact traces to %s", count, output)


if __name__ == "__main__":
    main()
