"""Streaming SFT data for databricks/databricks-dolly-15k.

The dataset is streamed with ``datasets.load_dataset(..., streaming=True)``;
the full corpus is never downloaded. Each example is triaged at runtime:

1. ``instruction + context + response + EOT`` fits in ``block_size`` -> keep all
2. otherwise ``instruction + response + EOT`` fits -> drop the context
3. otherwise drop the example

Loss is computed only on response tokens. Prompt tokens are labelled ``-100``
so ``cross_entropy(..., ignore_index=-100)`` ignores them.
"""

from __future__ import annotations

import logging
from typing import Iterable, Iterator, Optional

import numpy as np

from pretrain.data import _shuffle_streaming_dataset, get_encoder

DEFAULT_DATASET = "databricks/databricks-dolly-15k"


class SFTData:
    """Stream and tokenize Dolly-style instruction/response examples."""

    def __init__(
        self,
        block_size: int = 128,
        dataset_name: str = DEFAULT_DATASET,
        split: str = "train",
        shuffle_buffer: int = 100,
        seed: int = 42,
        max_docs: Optional[int] = None,
        min_chars: int = 1,
        tokenizer_name: str = "gpt2",
        eot_token_id: Optional[int] = None,
        documents: Optional[Iterable[dict]] = None,
    ):
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.block_size = int(block_size)
        self.dataset_name = dataset_name
        self.split = split
        self.shuffle_buffer = int(shuffle_buffer)
        self.seed = int(seed)
        self.max_docs = max_docs
        self.min_chars = int(min_chars)
        self.encoder = get_encoder(tokenizer_name)
        self.eot_token_id = (
            self.encoder.eot_token if eot_token_id is None else int(eot_token_id)
        )
        self._documents = documents
        self.stats = {"seen": 0, "full": 0, "answer_only": 0, "dropped": 0}

    # -- document source ----------------------------------------------------
    def _documents_once(self, epoch: int = 0) -> Iterator[dict]:
        if self._documents is not None:
            yield from self._documents
            return

        from datasets import load_dataset

        ds = load_dataset(self.dataset_name, split=self.split, streaming=True)
        ds = _shuffle_streaming_dataset(
            ds, self.shuffle_buffer, self.seed + epoch
        )
        yielded = 0
        for example in ds:
            if not isinstance(example, dict):
                continue
            text = " ".join(
                str(example.get(key) or "")
                for key in ("instruction", "context", "response", "answer")
            )
            if len(text.strip()) < self.min_chars:
                continue
            yield example
            yielded += 1
            if self.max_docs is not None and yielded >= self.max_docs:
                break

    # -- formatting ---------------------------------------------------------
    @staticmethod
    def _prompt(instruction: str, context: str, with_context: bool) -> str:
        if with_context and context:
            return (
                f"Instruction:\n{instruction}\n\n"
                f"Context:\n{context}\n\n"
                "Response:\n"
            )
        return f"Instruction:\n{instruction}\n\nResponse:\n"

    def _encode(self, instruction: str, context: str, response: str):
        """Return ``(input_ids, labels, triage)`` or ``None`` if unusable."""
        instruction = " ".join(instruction.split())
        context = " ".join(context.split())
        response = " ".join(response.split())
        if not instruction or not response:
            return None

        def build(with_context: bool):
            prompt = self._prompt(instruction, context, with_context)
            prompt_ids = self.encoder.encode(prompt)
            response_ids = self.encoder.encode(response) + [self.eot_token_id]
            ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + response_ids
            return ids, labels

        ids, labels = build(True)
        triage = "full"
        if len(ids) > self.block_size:
            ids, labels = build(False)
            triage = "answer_only"
        if len(ids) > self.block_size:
            return None
        return ids, labels, triage

    # -- public generators --------------------------------------------------
    def examples(self, epochs: int = 1) -> Iterator[dict]:
        for epoch in range(max(0, epochs)):
            for raw in self._documents_once(epoch):
                self.stats["seen"] += 1
                instruction = str(raw.get("instruction") or "")
                context = str(raw.get("context") or "")
                response = str(raw.get("response") or raw.get("answer") or "")
                encoded = self._encode(instruction, context, response)
                if encoded is None:
                    self.stats["dropped"] += 1
                    continue
                ids, labels, triage = encoded
                self.stats[triage] += 1
                yield {
                    "input_ids": ids,
                    "labels": labels,
                    "triage": triage,
                    "epoch": epoch,
                }

    def _collate(self, xs: list[list[int]], ys: list[list[int]]):
        max_len = max(len(row) for row in xs)
        input_ids = np.full((len(xs), max_len), self.eot_token_id, dtype=np.int64)
        labels = np.full((len(ys), max_len), -100, dtype=np.int64)
        for i, (x, y) in enumerate(zip(xs, ys)):
            input_ids[i, : len(x)] = x
            labels[i, : len(y)] = y
        return input_ids, labels

    def batches(
        self,
        batch_size: int = 4,
        epochs: int = 1,
        drop_last: bool = False,
        return_epoch: bool = False,
    ) -> Iterator[tuple]:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        xs: list[list[int]] = []
        ys: list[list[int]] = []
        epoch = 0
        for example in self.examples(epochs):
            xs.append(example["input_ids"])
            ys.append(example["labels"])
            epoch = int(example["epoch"])
            if len(xs) == batch_size:
                batch = self._collate(xs, ys)
                yield (batch[0], batch[1], epoch) if return_epoch else batch
                xs, ys = [], []
        if xs and not drop_last:
            batch = self._collate(xs, ys)
            yield (batch[0], batch[1], epoch) if return_epoch else batch


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect streamed SFT examples.")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-docs", type=int, default=5)
    args = parser.parse_args()

    data = SFTData(block_size=args.block_size, max_docs=args.max_docs)
    for batch_no, (x, y) in enumerate(
        data.batches(batch_size=args.batch_size, epochs=1)
    ):
        print(f"batch {batch_no + 1}: shape={x.shape}")
        print(data.encoder.decode(x[0].tolist()))
        if batch_no + 1 >= args.num_batches:
            break
    print("stats:", data.stats)
