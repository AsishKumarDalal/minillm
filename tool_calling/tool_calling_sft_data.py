"""Dataset for tool-calling SFT.

Reads the compact JSONL produced by ``data_format.write_jsonl`` (or by
``tool_calling_distill``), tokenises it with the GPT-2 encoder and produces
``(input_ids, labels)`` batches where **only assistant lines are labelled**.

This mirrors ``sft/sft_data.py``: prompt/result tokens are -100, EOT is the
GPT-2 end-of-text token, and right padding uses EOT with label -100.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

from pretrain.data import get_encoder
from .data_format import ToolExample, encode_example, read_jsonl


class ToolCallingSFTData:
    def __init__(
        self,
        examples: Iterable[ToolExample],
        block_size: int = 1024,
        eot_token_id: Optional[int] = None,
        tokenizer_name: str = "gpt2",
    ):
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.block_size = int(block_size)
        self.encoder = get_encoder(tokenizer_name)
        self.eot_token_id = (
            self.encoder.eot_token if eot_token_id is None else int(eot_token_id)
        )
        self.examples = list(examples)
        self.stats = {
            "seen": 0,
            "kept": 0,
            "call": 0,
            "no_tool": 0,
            "dropped": 0,
        }

    # -- constructors --------------------------------------------------------
    @classmethod
    def from_jsonl(cls, path: str | Path, block_size: int = 1024,
                   tokenizer_name: str = "gpt2", max_docs: Optional[int] = None):
        examples = read_jsonl(path)
        if max_docs is not None:
            examples = (ex for i, ex in enumerate(examples) if i < int(max_docs))
        return cls(examples, block_size=block_size, tokenizer_name=tokenizer_name)

    # -- examples ------------------------------------------------------------
    def examples_iter(self) -> Iterator[dict]:
        for ex in self.examples:
            self.stats["seen"] += 1
            encoded = encode_example(ex, self.encoder, self.block_size,
                                     self.eot_token_id)
            if encoded is None:
                self.stats["dropped"] += 1
                continue
            ids, labels = encoded
            key = "call" if ex.call else "no_tool"
            self.stats[key] += 1
            self.stats["kept"] += 1
            yield {"input_ids": ids, "labels": labels, "kind": key,
                   "source": ex.source}

    # -- collate / batches ---------------------------------------------------
    def _collate(self, xs: list[list[int]], ys: list[list[int]]):
        max_len = max(len(row) for row in xs)
        input_ids = np.full((len(xs), max_len), self.eot_token_id, dtype=np.int64)
        labels = np.full((len(ys), max_len), -100, dtype=np.int64)
        for i, (x, y) in enumerate(zip(xs, ys)):
            input_ids[i, : len(x)] = x
            labels[i, : len(y)] = y
        return input_ids, labels

    def batches(self, batch_size: int = 4, epochs: int = 1,
                drop_last: bool = False, return_epoch: bool = False,
                return_kind: bool = False) -> Iterator[tuple]:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        xs: list[list[int]] = []
        ys: list[list[int]] = []
        kinds: list[str] = []
        for epoch in range(max(0, epochs)):
            for example in self.examples_iter():
                xs.append(example["input_ids"])
                ys.append(example["labels"])
                kinds.append(example["kind"])
                if len(xs) == batch_size:
                    batch = self._collate(xs, ys)
                    out = [batch[0], batch[1]]
                    if return_epoch:
                        out.append(epoch)
                    if return_kind:
                        out.append(list(kinds))
                    yield tuple(out)
                    xs, ys, kinds = [], [], []
        if xs and not drop_last:
            batch = self._collate(xs, ys)
            out = [batch[0], batch[1]]
            if return_epoch:
                out.append(0)
            if return_kind:
                out.append(list(kinds))
            yield tuple(out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect tool-calling SFT examples.")
    parser.add_argument("--path", type=str, default="data/tool_traces.jsonl")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--num-batches", type=int, default=1)
    args = parser.parse_args()
    data = ToolCallingSFTData.from_jsonl(args.path, block_size=args.block_size)
    for i, (x, y) in enumerate(data.batches(batch_size=2, epochs=1)):
        print(f"batch {i + 1}: {x.shape}")
        print(data.encoder.decode(x[0].tolist()))
        if i + 1 >= args.num_batches:
            break
    print("stats:", data.stats)
