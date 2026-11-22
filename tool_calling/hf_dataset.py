"""HF tool-calling datasets -> compact ToolExample, with streaming support.

Supported datasets
------------------
* ``xlam``  -> Salesforce/xlam-function-calling-60k
* ``glaive`` -> glaiveai/glaive-function-calling-v2

Rows are converted on the fly so SFT never downloads the full corpus, exactly
like ``pretrain/data.py`` and ``sft/sft_data.py``.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any, Iterable, Iterator, Optional

import numpy as np

from pretrain.data import _shuffle_streaming_dataset, get_encoder
from .data_format import (
    ToolExample,
    encode_example,
    make_call,
    make_result,
    _tool_header,
)

CALL_TAG_RE = re.compile(r"<functioncall>\s*(\{.*?\})\s*(?:<|$)", re.S)
SPEAKER_RE = re.compile(r"^(USER|ASSISTANT|FUNCTION RESPONSE):\s*(.*)$", re.S)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _loads(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _json_objects(text: str) -> list[dict]:
    """Scan text for concatenated JSON objects (used for Glaive system field)."""
    decoder = json.JSONDecoder()
    out = []
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text) or text[idx] != "{":
            idx += 1
            continue
        try:
            obj, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            idx += 1
            continue
        if isinstance(obj, dict):
            out.append(obj)
        idx += end
    return out


def _parse_function_payload(raw: str) -> Optional[tuple[str, dict]]:
    text = raw.strip()
    try:
        obj, _ = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        try:
            obj = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    args = obj.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if not name or not isinstance(args, dict):
        return None
    return name, args


# ---------------------------------------------------------------------------
# xLAM
# ---------------------------------------------------------------------------
def xlam_to_examples(row: dict) -> list[ToolExample]:
    query = _loads(row.get("query"))
    if isinstance(query, dict):
        query = query.get("query") or query.get("text")
    if not query:
        return []
    tools = _loads(row.get("tools")) or []
    tools_header = _tool_header(tools)
    if not tools_header:
        return []
    answers = _loads(row.get("answers")) or []
    if isinstance(answers, dict):
        answers = [answers]

    examples = []
    for answer in answers:
        answer = _loads(answer)
        if not isinstance(answer, dict):
            continue
        name = answer.get("name")
        args = _loads(answer.get("arguments")) or {}
        if not name or not isinstance(args, dict):
            continue
        result = _loads(answer.get("response") or answer.get("result") or {})
        final = answer.get("answer") or answer.get("response")
        if isinstance(final, (dict, list)):
            final = make_result(final)
        if not final:
            final = f"Called {name}."
        examples.append(ToolExample(
            tools=tools_header,
            user=str(query).strip(),
            call=make_call(str(name), args),
            result=make_result(result),
            answer=str(final).strip(),
            source="xlam",
            meta={"dataset": "xlam"},
        ))
    return examples


# ---------------------------------------------------------------------------
# Glaive
# ---------------------------------------------------------------------------
def glaive_to_examples(row: dict, max_tools: int = 8) -> list[ToolExample]:
    system = str(row.get("system") or "")
    chat = str(row.get("chat") or "")
    schemas = [o for o in _json_objects(system) if o.get("name")]
    if not schemas:
        return []
    tools_header = _tool_header([
        {"function": {
            "name": s.get("name"),
            "parameters": {"properties": (s.get("parameters") or {}).get("properties") or {}},
        }}
        for s in schemas[:max_tools]
    ])

    turns = []
    markers = list(re.finditer(r"(USER|ASSISTANT|FUNCTION RESPONSE):", chat))
    for i, match in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(chat)
        text = chat[match.end():end].replace("<|endoftext|>", " ").strip()
        if text:
            turns.append((match.group(1), text))

    examples = []
    i = 0
    while i < len(turns):
        speaker, text = turns[i]
        if speaker != "ASSISTANT":
            i += 1
            continue
        tag = CALL_TAG_RE.search(text)
        if tag:
            parsed = _parse_function_payload(tag.group(1))
            if parsed and i + 2 < len(turns) and turns[i + 1][0] == "FUNCTION RESPONSE":
                name, args = parsed
                result_text = turns[i + 1][1]
                final_text = turns[i + 2][1] if i + 2 < len(turns) and turns[i + 2][0] == "ASSISTANT" else ""
                user_text = ""
                for j in range(i - 1, -1, -1):
                    if turns[j][0] == "USER":
                        user_text = turns[j][1]
                        break
                if user_text:
                    examples.append(ToolExample(
                        tools=tools_header,
                        user=user_text,
                        call=make_call(name, args),
                        result=result_text,
                        answer=final_text or f"Called {name}.",
                        source="glaive",
                        meta={"dataset": "glaive"},
                    ))
                i += 3
                continue
        else:
            # plain answer -> no-tool example
            user_text = ""
            for j in range(i - 1, -1, -1):
                if turns[j][0] == "USER":
                    user_text = turns[j][1]
                    break
            if user_text and text:
                examples.append(ToolExample(
                    tools=tools_header,
                    user=user_text,
                    call=None,
                    result=None,
                    answer=text,
                    source="glaive",
                    meta={"dataset": "glaive"},
                ))
        i += 1
    return examples


DATASET_CONVERTERS = {"xlam": xlam_to_examples, "glaive": glaive_to_examples}

HF_DATASET_NAMES = {
    "xlam": "Salesforce/xlam-function-calling-60k",
    "glaive": "glaiveai/glaive-function-calling-v2",
}


def convert_row(row: dict, dataset: str) -> list[ToolExample]:
    converter = DATASET_CONVERTERS.get(str(dataset).lower())
    if converter is None:
        raise ValueError(f"unknown tool dataset {dataset!r}; use xlam or glaive")
    return converter(row)


# ---------------------------------------------------------------------------
# Streaming dataset
# ---------------------------------------------------------------------------
class StreamingToolSFTData:
    """Stream HF rows, convert, tokenise, and yield batches on demand."""

    def __init__(self, rows: Iterable[dict], dataset: str, block_size: int,
                 eot_token_id: Optional[int] = None,
                 tokenizer_name: str = "gpt2", max_docs: Optional[int] = None):
        self.rows = rows
        self.dataset = str(dataset).lower()
        self.block_size = int(block_size)
        self.encoder = get_encoder(tokenizer_name)
        self.eot_token_id = (
            self.encoder.eot_token if eot_token_id is None else int(eot_token_id)
        )
        self.max_docs = int(max_docs) if max_docs is not None else None
        self.stats = {"seen": 0, "kept": 0, "call": 0, "no_tool": 0, "dropped": 0}

    @classmethod
    def from_hf(cls, dataset_name: str, dataset: str, split: str = "train",
                shuffle_buffer: int = 100, seed: int = 42,
                block_size: int = 1024, tokenizer_name: str = "gpt2",
                max_docs: Optional[int] = None):
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split=split, streaming=True)
        if shuffle_buffer > 1:
            ds = _shuffle_streaming_dataset(ds, shuffle_buffer, seed)
        return cls(ds, dataset=dataset, block_size=block_size,
                   tokenizer_name=tokenizer_name, max_docs=max_docs)

    def examples_iter(self) -> Iterator[dict]:
        converted = 0
        for row in self.rows:
            if self.max_docs is not None and converted >= self.max_docs:
                return
            converted += 1
            for example in convert_row(row, self.dataset):
                self.stats["seen"] += 1
                encoded = encode_example(example, self.encoder, self.block_size,
                                         self.eot_token_id)
                if encoded is None:
                    self.stats["dropped"] += 1
                    continue
                ids, labels = encoded
                key = "call" if example.call else "no_tool"
                self.stats[key] += 1
                self.stats["kept"] += 1
                yield {"input_ids": ids, "labels": labels, "kind": key}

    def _collate(self, xs, ys):
        max_len = max(len(row) for row in xs)
        input_ids = np.full((len(xs), max_len), self.eot_token_id, dtype=np.int64)
        labels = np.full((len(ys), max_len), -100, dtype=np.int64)
        for i, (x, y) in enumerate(zip(xs, ys)):
            input_ids[i, : len(x)] = x
            labels[i, : len(y)] = y
        return input_ids, labels

    def batches(self, batch_size: int = 4, epochs: int = 1,
                drop_last: bool = False, return_epoch: bool = False):
        xs, ys = [], []
        for epoch in range(max(0, epochs)):
            for example in self.examples_iter():
                xs.append(example["input_ids"])
                ys.append(example["labels"])
                if len(xs) == batch_size:
                    batch = self._collate(xs, ys)
                    yield (batch[0], batch[1], epoch) if return_epoch else batch
                    xs, ys = [], []
            if xs and not drop_last:
                batch = self._collate(xs, ys)
                yield (batch[0], batch[1], epoch) if return_epoch else batch
                xs, ys = [], []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preview streamed tool examples.")
    parser.add_argument("--dataset", choices=["xlam", "glaive"], required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--max-docs", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=1024)
    args = parser.parse_args()

    data = StreamingToolSFTData.from_hf(
        args.dataset_name or HF_DATASET_NAMES[args.dataset], args.dataset,
        block_size=args.block_size, max_docs=args.max_docs, shuffle_buffer=0,
    )
    for i, (x, y) in enumerate(data.batches(batch_size=1, epochs=1)):
        print(f"--- example {i + 1} ---")
        print(data.encoder.decode(x[0].tolist()))
    print("stats:", data.stats)
