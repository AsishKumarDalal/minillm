"""Tiny real-network smoke test for the OpenWebText stream.

This reads only a couple of documents (a few KB), tokenizes a 32-token
block, and exits. It does not download the full dataset and does not load
the model.

Run from the project root:
    .venv/bin/python code_test/smoke_openwebtext.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pretrain.data import OpenWebTextTokenStream  # noqa: E402


def main() -> None:
    stream = OpenWebTextTokenStream(
        block_size=32,
        shuffle_buffer=0,  # start yielding immediately
        max_docs=2,        # only touch a couple of documents
    )

    x, y = next(stream.batches(batch_size=1, drop_last=False))

    assert x.shape == (1, 32), x.shape
    assert y.shape == (1, 32), y.shape
    assert (x[0, 1:] == y[0, :-1]).all()

    print(f"x: {x.shape} {x.dtype}, y: {y.shape} {y.dtype}")
    print("decoded:", repr(stream.encoder.decode(x[0].tolist())[:200]))
    print("OpenWebText streaming smoke test passed")


if __name__ == "__main__":
    main()
