"""Streaming OpenWebText data for GPT pretraining.

The dataset is read through HuggingFace ``datasets`` in **streaming** mode:
the full ~38 GB corpus is never downloaded. Documents are pulled one at a
time, tokenized with tiktoken's GPT-2 BPE, and turned into fixed-size
next-token blocks.

Typical training loop
---------------------
::

    from pretrain.data import OpenWebTextTokenStream
    import torch

    stream = OpenWebTextTokenStream(block_size=128)
    for step, (x, y) in enumerate(stream.batches(batch_size=8)):
        x = torch.from_numpy(x).long()
        y = torch.from_numpy(y).long()
        logits, loss = model(x, y)
        ...

The generator is lazy: it fetches and tokenizes only enough documents to
produce the next batch. While the model trains on that batch, nothing is
prefetched; when the loop asks for the next batch, the next part of
OpenWebText is fetched and encoded.

Notes
-----
* HuggingFace caches downloaded shards under ``~/.cache/huggingface``, but
  only the shards/rows touched by the stream are fetched.
* Blocks are NumPy ``int32`` arrays. Convert them to ``torch.long`` in the
  training loop.
* A GPT-2 end-of-text token (50256) is appended after every document, so
  blocks may contain document boundaries.
"""

from __future__ import annotations

import argparse
import time
from functools import lru_cache
from typing import Iterable, Iterator, Optional

import numpy as np
import tiktoken

DEFAULT_DATASET = "Skylion007/openwebtext"
DEFAULT_TOKENIZER = "gpt2"


@lru_cache(maxsize=8)
def get_encoder(name: str = DEFAULT_TOKENIZER) -> tiktoken.Encoding:
    """Return (and cache) a tiktoken encoding by name."""
    return tiktoken.get_encoding(name)


def iter_openwebtext_documents(
    dataset_name: str = DEFAULT_DATASET,
    split: str = "train",
    shuffle_buffer: int = 1000,
    seed: int = 42,
    max_docs: Optional[int] = None,
    min_chars: int = 1,
) -> Iterator[str]:
    """Yield raw OpenWebText documents from a streaming HF dataset.

    Parameters
    ----------
    dataset_name:
        HuggingFace dataset id. The canonical OpenWebText mirror is
        ``Skylion007/openwebtext``.
    split:
        Dataset split to stream.
    shuffle_buffer:
        Document-level shuffle buffer. ``0`` disables shuffling and starts
        yielding immediately. Larger values give better mixing but require
        that many documents to be read before the first yield.
    seed:
        Shuffle seed.
    max_docs:
        Stop after this many documents. ``None`` streams the whole split.
        Useful for smoke tests.
    min_chars:
        Skip documents whose stripped text is shorter than this.
    """
    # Imported lazily so the module can be used for local tests without
    # paying the datasets/pyarrow import cost.
    from datasets import load_dataset

    if min_chars < 0:
        raise ValueError("min_chars must be >= 0")
    if max_docs is not None and max_docs <= 0:
        return

    ds = load_dataset(dataset_name, split=split, streaming=True)
    if shuffle_buffer > 1:
        # Keep the shuffle buffer within one input shard. The datasets
        # default (10) can wait on many remote shards before yielding the
        # first example; one shard is enough to mix documents while keeping
        # startup fast for a streaming pipeline.
        ds = ds.shuffle(
            buffer_size=shuffle_buffer,
            seed=seed,
            max_buffer_input_shards=1,
        )

    yielded = 0
    for example in ds:
        text = example.get("text")
        if not isinstance(text, str) or len(text.strip()) < min_chars:
            continue
        yield text
        yielded += 1
        if max_docs is not None and yielded >= max_docs:
            break


def iter_token_blocks(
    documents: Iterable[str],
    encoder: tiktoken.Encoding,
    block_size: int,
    eot_token_id: Optional[int] = None,
    add_eot: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Tokenize documents and yield ``(input_ids, target_ids)`` blocks.

    Each yielded pair has shape ``(block_size,)`` and dtype ``int32``.
    ``target_ids`` is the input shifted left by one token, so the training
    loss is ordinary next-token cross-entropy.

    A rolling token buffer is used. Documents can be shorter or longer than
    ``block_size``; the stream is continuous across document boundaries.
    A final partial block is discarded so every batch has a fixed shape.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")

    if eot_token_id is None:
        eot_token_id = encoder.eot_token
    if not 0 <= eot_token_id <= encoder.max_token_value:
        raise ValueError(
            f"eot_token_id {eot_token_id} is outside the tokenizer range "
            f"0..{encoder.max_token_value}"
        )

    need = block_size + 1
    buffer: list[int] = []

    for text in documents:
        ids = encoder.encode_ordinary(text)
        if add_eot:
            ids.append(eot_token_id)
        if not ids:
            continue

        buffer.extend(ids)

        while len(buffer) >= need:
            block = np.asarray(buffer[:need], dtype=np.int32)
            yield block[:-1], block[1:]
            # Remove block_size tokens, not block_size + 1, so the last
            # target token becomes the first input token of the next block.
            del buffer[:block_size]


class OpenWebTextTokenStream:
    """Lazy token stream over OpenWebText.

    Parameters
    ----------
    block_size:
        Context length of one training example.
    dataset_name / split:
        HuggingFace dataset to stream.
    tokenizer_name:
        tiktoken encoding name. The model's vocab is GPT-2 (50257).
    eot_token_id:
        Document separator. Defaults to the tokenizer's EOT token.
    shuffle_buffer:
        Document shuffle buffer. ``0`` keeps the dataset order.
    seed:
        Shuffle seed.
    max_docs:
        Optional document limit, mainly for smoke tests.
    min_chars:
        Skip very short documents.
    documents:
        Optional local iterable of strings. When provided, HuggingFace is
        not contacted. This is useful for tests and local corpora.
    """

    def __init__(
        self,
        block_size: int = 128,
        dataset_name: str = DEFAULT_DATASET,
        split: str = "train",
        tokenizer_name: str = DEFAULT_TOKENIZER,
        eot_token_id: Optional[int] = None,
        shuffle_buffer: int = 1000,
        seed: int = 42,
        max_docs: Optional[int] = None,
        min_chars: int = 1,
        documents: Optional[Iterable[str]] = None,
    ):
        if block_size < 1:
            raise ValueError("block_size must be >= 1")

        self.block_size = block_size
        self.dataset_name = dataset_name
        self.split = split
        self.encoder = get_encoder(tokenizer_name)
        self.eot_token_id = (
            self.encoder.eot_token if eot_token_id is None else int(eot_token_id)
        )
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.max_docs = max_docs
        self.min_chars = min_chars
        self._provided_documents = documents

    def documents(self) -> Iterator[str]:
        """Yield raw documents, either local or streamed from HuggingFace."""
        if self._provided_documents is not None:
            yield from self._provided_documents
            return

        yield from iter_openwebtext_documents(
            dataset_name=self.dataset_name,
            split=self.split,
            shuffle_buffer=self.shuffle_buffer,
            seed=self.seed,
            max_docs=self.max_docs,
            min_chars=self.min_chars,
        )

    def blocks(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(input_ids, target_ids)`` blocks of ``block_size`` tokens."""
        return iter_token_blocks(
            documents=self.documents(),
            encoder=self.encoder,
            block_size=self.block_size,
            eot_token_id=self.eot_token_id,
        )

    def batches(
        self,
        batch_size: int = 8,
        drop_last: bool = True,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(x, y)`` NumPy batches of shape ``(batch_size, block_size)``.

        The stream is infinite unless ``max_docs`` or a local ``documents``
        iterator is exhausted. ``drop_last`` only matters in that finite case.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []

        for x, y in self.blocks():
            xs.append(x)
            ys.append(y)

            if len(xs) == batch_size:
                yield np.stack(xs, axis=0), np.stack(ys, axis=0)
                xs.clear()
                ys.clear()

        if xs and not drop_last:
            yield np.stack(xs, axis=0), np.stack(ys, axis=0)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream and tokenize OpenWebText without downloading it fully."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--num-batches",
        type=int,
        default=2,
        help="How many batches to print before exiting (default: 2).",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional document limit for a quick smoke test.",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=1000,
        help="Document shuffle buffer; 0 disables shuffling.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=1)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    if args.num_batches <= 0:
        raise SystemExit("--num-batches must be >= 1")

    stream = OpenWebTextTokenStream(
        block_size=args.block_size,
        dataset_name=args.dataset,
        split=args.split,
        tokenizer_name=args.tokenizer,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
        max_docs=args.max_docs,
        min_chars=args.min_chars,
    )

    start = time.time()
    batches_seen = 0
    tokens_seen = 0

    for x, y in stream.batches(batch_size=args.batch_size, drop_last=False):
        if batches_seen == 0:
            print(
                f"first batch: x={x.shape} {x.dtype} "
                f"y={y.shape} {y.dtype} "
                f"token_range=[{x.min()}, {x.max()}]"
            )
            sample = stream.encoder.decode(x[0].tolist())
            print("decoded sample:", repr(sample[:300]))

        batches_seen += 1
        tokens_seen += int(x.size)
        if batches_seen >= args.num_batches:
            break

    elapsed = time.time() - start
    tokens_per_sec = tokens_seen / elapsed if elapsed > 0 else 0.0
    print(
        f"streamed {batches_seen} batch(es), {tokens_seen} tokens in "
        f"{elapsed:.2f}s ({tokens_per_sec:,.0f} tokens/s)"
    )


if __name__ == "__main__":
    main()
