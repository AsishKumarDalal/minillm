"""Compact tool-calling text format and OpenAI-style conversion.

The model is small and the context window is limited, so we do **not** put
full JSON tool schemas in the prompt.  A tool turn is represented as::

    TOOLS: weather(city),calc(expr)
    USER: weather in Paris?
    ASSISTANT: CALL weather {"city":"Paris"}
    RESULT: {"temp_c":18}
    ASSISTANT: 18C in Paris.

Loss is computed only on ``ASSISTANT:`` lines.  ``TOOLS:``, ``USER:`` and
``RESULT:`` lines are environment/prompt text and are masked with -100.

This module also converts common OpenAI / xlam / glaive rows into that format.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

CALL_RE = re.compile(r"^\s*CALL\s+(\w+)\s+(\{.*\})\s*$", re.S)


# ---------------------------------------------------------------------------
# Example container
# ---------------------------------------------------------------------------
@dataclass
class ToolExample:
    """One tool-calling training example.

    ``call`` / ``result`` are ``None`` for a "no tool needed" example.
    """

    tools: str
    user: str
    call: Optional[str]
    result: Optional[str]
    answer: str
    source: str = "manual"
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Formatting / parsing
# ---------------------------------------------------------------------------
def make_prompt(tools: str, user: str) -> str:
    """First-turn prompt. Generation starts after the trailing ``ASSISTANT:``."""
    return f"TOOLS: {tools.strip()}\nUSER: {user.strip()}\nASSISTANT: "


def make_call(name: str, arguments: dict) -> str:
    """Canonical assistant tool-call line."""
    return f"CALL {name} {json.dumps(arguments, separators=(',', ':'), ensure_ascii=False)}"


def make_result(result: Any) -> str:
    """Canonical tool-result line payload."""
    return json.dumps(result, separators=(',', ':'), ensure_ascii=False)


def format_full(example: ToolExample) -> str:
    """Full conversation text for display / debugging."""
    if example.call:
        return (
            make_prompt(example.tools, example.user)
            + example.call.strip()
            + "\nRESULT: "
            + (example.result or "{}").strip()
            + "\nASSISTANT: "
            + example.answer.strip()
        )
    return make_prompt(example.tools, example.user) + example.answer.strip()


def parse_call(text: str) -> Optional[tuple[str, dict]]:
    """Parse an assistant line into ``(tool_name, arguments)``.

    Returns ``None`` when the line is a plain answer or is malformed.
    """
    if not text:
        return None
    match = CALL_RE.match(text.strip())
    if not match:
        return None
    name, raw = match.group(1), match.group(2)
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(args, dict):
        return None
    return name, args


# ---------------------------------------------------------------------------
# OpenAI-style -> compact conversion
# ---------------------------------------------------------------------------
def _tool_header(tools: Any) -> str:
    """Build ``name(p1,p2)`` header from an OpenAI tool list."""
    if isinstance(tools, str):
        return tools.strip()
    parts: list[str] = []
    for item in tools or []:
        if not isinstance(item, dict):
            continue
        fn = item.get("function", item)
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters") or {}
        props = params.get("properties") or {}
        parts.append(f"{name}({','.join(props.keys())})")
    return "|".join(parts)


def _call_from_openai(row: dict) -> Optional[tuple[str, dict]]:
    """Extract the first tool call from an OpenAI-like row."""
    calls = row.get("tool_calls") or []
    if not calls:
        for msg in row.get("messages") or []:
            if isinstance(msg, dict) and msg.get("tool_calls"):
                calls = msg["tool_calls"]
                break
    if not calls:
        answers = row.get("answers") or []
        if answers:
            first = answers[0] if isinstance(answers, list) else answers
            if isinstance(first, dict) and first.get("name"):
                return first["name"], dict(first.get("arguments") or {})
        return None

    first = calls[0]
    if not isinstance(first, dict):
        return None
    fn = first.get("function", first)
    name = fn.get("name")
    raw = fn.get("arguments", "{}")
    if isinstance(raw, str):
        try:
            args = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return None
    else:
        args = raw or {}
    if not name or not isinstance(args, dict):
        return None
    return name, args


def _user_from_openai(row: dict) -> str:
    if row.get("query"):
        return str(row["query"]).strip()
    if row.get("user"):
        return str(row["user"]).strip()
    for msg in row.get("messages") or []:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content") or "").strip()
    return ""


def from_openai_row(
    row: dict,
    tool_result: Any = None,
    final_answer: Optional[str] = None,
    source: str = "openai",
) -> Optional[ToolExample]:
    """Convert an OpenAI / xlam / glaive row to :class:`ToolExample`.

    ``tool_result`` and ``final_answer`` can be supplied by the caller; when
    omitted, the original row's text answer is used as the final answer and an
    empty result object is written.  Prefer generating real results with
    ``runtime.ToolRegistry`` before training.
    """
    user = _user_from_openai(row)
    tools = _tool_header(row.get("tools"))
    if not user or not tools:
        return None
    parsed = _call_from_openai(row)
    if parsed is None:
        answer = final_answer or str(row.get("answer") or row.get("response") or "").strip()
        if not answer:
            return None
        return ToolExample(tools=tools, user=user, call=None, result=None,
                           answer=answer, source=source, meta=dict(row))

    name, args = parsed
    answer = final_answer or str(row.get("answer") or row.get("response") or "").strip()
    if not answer:
        return None
    return ToolExample(
        tools=tools,
        user=user,
        call=make_call(name, args),
        result=make_result({} if tool_result is None else tool_result),
        answer=answer,
        source=source,
        meta=dict(row),
    )


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------
def read_jsonl(path: str | Path) -> Iterator[ToolExample]:
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                logging.warning("Skipping invalid JSON at %s:%d: %s", path, line_no, exc)
                continue
            if "call" in row:
                yield ToolExample(
                    tools=str(row.get("tools") or ""),
                    user=str(row.get("user") or ""),
                    call=(str(row["call"]) if row.get("call") else None),
                    result=(str(row["result"]) if row.get("result") else None),
                    answer=str(row.get("answer") or ""),
                    source=str(row.get("source") or "jsonl"),
                    meta=row.get("meta") or {},
                )
            else:
                example = from_openai_row(row, source="jsonl-openai")
                if example is not None:
                    yield example


def write_jsonl(examples: Iterable[ToolExample], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            row = {
                "tools": ex.tools,
                "user": ex.user,
                "call": ex.call,
                "result": ex.result,
                "answer": ex.answer,
                "source": ex.source,
            }
            if ex.meta:
                row["meta"] = ex.meta
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# Tokenisation helper (shared by SFT data)
# ---------------------------------------------------------------------------
def encode_example(example: ToolExample, encoder, block_size: int, eot_token_id: int):
    """Return ``(input_ids, labels)`` or ``None`` when the example is too long.

    Only assistant text is labelled; prompt and tool-result tokens are -100.
    """
    if example.call:
        segments = [
            (make_prompt(example.tools, example.user), False),
            (example.call.strip() + "\n", True),
            (f"RESULT: {(example.result or '{}').strip()}\nASSISTANT: ", False),
            (example.answer.strip() + "\n", True),
        ]
    else:
        segments = [
            (make_prompt(example.tools, example.user), False),
            (example.answer.strip() + "\n", True),
        ]

    ids: list[int] = []
    labels: list[int] = []
    for text, train in segments:
        tokens = encoder.encode(text)
        ids.extend(tokens)
        labels.extend(tokens if train else [-100] * len(tokens))

    ids.append(eot_token_id)
    labels.append(eot_token_id)
    if len(ids) > block_size:
        return None
    return ids, labels
