"""Lightweight tests for pretrain/data.py.

No torch, no model loading, no HuggingFace network access. Uses a few
local strings and the real GPT-2 tokenizer.

Run from the project root:
    .venv/bin/python code_test/test_data_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pretrain.data import (  # noqa: E402
    OpenWebTextTokenStream,
    _shuffle_streaming_dataset,
    get_encoder,
    iter_token_blocks,
)

DOCS = [
    "The quick brown fox jumps over the lazy dog. " * 20,
    "OpenWebText is a large corpus of web pages. " * 20,
    "Tiny test documents for the streaming tokenizer. " * 20,
]


def test_blocks() -> None:
    encoder = get_encoder()
    blocks = list(iter_token_blocks(DOCS, encoder, block_size=16))
    assert blocks, "expected at least one block"

    eot = encoder.eot_token
    saw_eot = False

    for x, y in blocks:
        assert x.shape == (16,), x.shape
        assert y.shape == (16,), y.shape
        assert x.dtype == np.int32, x.dtype
        assert y.dtype == np.int32, y.dtype
        assert np.array_equal(x[1:], y[:-1]), "targets are not shifted inputs"
        assert x.min() >= 0
        assert y.min() >= 0
        assert x.max() <= encoder.max_token_value
        assert y.max() <= encoder.max_token_value
        saw_eot = saw_eot or bool((x == eot).any() or (y == eot).any())

    assert saw_eot, "document separator token was not found"
    print(f"test_blocks: {len(blocks)} blocks, EOT present")


def test_batches() -> None:
    stream = OpenWebTextTokenStream(
        documents=DOCS,
        block_size=16,
        shuffle_buffer=0,
    )
    batches = list(stream.batches(batch_size=4, drop_last=False))
    assert batches, "expected at least one batch"

    for x, y in batches:
        assert x.ndim == 2 and y.ndim == 2
        assert x.shape[1] == 16 and y.shape[1] == 16
        assert x.shape[0] <= 4 and y.shape[0] == x.shape[0]
        assert x.dtype == np.int32 and y.dtype == np.int32
        assert np.array_equal(x[:, 1:], y[:, :-1]), "batch targets are misaligned"

    print(f"test_batches: {len(batches)} batches, shapes aligned")


def test_drop_last() -> None:
    stream = OpenWebTextTokenStream(
        documents=DOCS,
        block_size=16,
        shuffle_buffer=0,
    )
    batches = list(stream.batches(batch_size=7, drop_last=True))
    assert all(x.shape[0] == 7 for x, _ in batches)
    print(f"test_drop_last: {len(batches)} full batches")


def test_bad_block_size() -> None:
    encoder = get_encoder()
    try:
        list(iter_token_blocks(DOCS, encoder, block_size=0))
    except ValueError:
        print("test_bad_block_size: rejected block_size=0")
        return
    raise AssertionError("block_size=0 should have raised ValueError")


def test_bad_eot() -> None:
    encoder = get_encoder()
    try:
        list(iter_token_blocks(DOCS, encoder, block_size=16, eot_token_id=10**9))
    except ValueError:
        print("test_bad_eot: rejected out-of-range EOT token")
        return
    raise AssertionError("out-of-range EOT token should have raised ValueError")


def test_shuffle_compatibility() -> None:
    class NewDataset:
        def shuffle(self, **kwargs):
            self.kwargs = kwargs
            return self

    new_ds = NewDataset()
    assert _shuffle_streaming_dataset(new_ds, 10, seed=3) is new_ds
    assert new_ds.kwargs["buffer_size"] == 10
    assert new_ds.kwargs["seed"] == 3
    assert new_ds.kwargs["max_buffer_input_shards"] == 1

    class OldDataset:
        def __init__(self):
            self.calls = []

        def shuffle(self, **kwargs):
            self.calls.append(kwargs)
            if "max_buffer_input_shards" in kwargs:
                raise TypeError(
                    "IterableDataset.shuffle() got an unexpected keyword "
                    "argument 'max_buffer_input_shards'"
                )
            return self

    old_ds = OldDataset()
    assert _shuffle_streaming_dataset(old_ds, 10, seed=3) is old_ds
    assert len(old_ds.calls) == 2, old_ds.calls
    assert "max_buffer_input_shards" not in old_ds.calls[-1]
    assert old_ds.calls[-1]["buffer_size"] == 10
    assert old_ds.calls[-1]["seed"] == 3
    print("shuffle compatibility: new + old datasets OK")


def main() -> None:
    test_blocks()
    test_batches()
    test_drop_last()
    test_bad_block_size()
    test_bad_eot()
    test_shuffle_compatibility()
    print("all data pipeline tests passed")


if __name__ == "__main__":
    main()
