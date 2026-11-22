"""Tool-calling distillation: teacher traces -> compact JSONL -> student SFT.

This is behavioural distillation, not the logit distillation in
``distill/distill.py``.  A strong OpenAI-compatible teacher (the local Inkling
server works) is asked to use a small tool set.  Its tool calls are converted
to the compact ``CALL name {json}`` protocol, validated by executing the tool,
and written as tool-trace JSONL.  ``ToolCallingSFTData`` then trains the student.

HF artifacts:

* generated dataset: ``tool_calling/distill_data/`` (via ``--upload``)
* student checkpoint/plots: paths from ``configs/tool_calling_sft.yaml``
  (normally ``tool_calling/tool_sft_model.pt`` and ``tool_calling/sft_outs/``)

Usage
-----
::

    python -m tool_calling.tool_calling_distill --config configs/tool_calling_distill.yaml
    python -m tool_calling.tool_calling_distill --limit 50 --skip-sft --no-hf
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pretrain.train import (  # noqa: E402
    get_hf_settings,
    load_env_file,
    load_yaml,
    setup_logging,
    upload_outputs,
)
from .data_format import ToolExample, from_openai_row, parse_call, write_jsonl  # noqa: E402
from .runtime import Tool, ToolRegistry  # noqa: E402

DEFAULT_CONFIG = ROOT_DIR / "configs" / "tool_calling_distill.yaml"


def build_registry(tool_specs: list[dict]) -> ToolRegistry:
    """Build a registry with safe stub tools for validation/results."""
    tools: list[Tool] = []
    for spec in tool_specs or []:
        name = str(spec.get("name") or "").strip()
        params = tuple(str(p) for p in (spec.get("params") or []))
        if not name:
            continue
        stub = dict(spec.get("stub_result") or {})

        def _fn(_stub=stub, _params=params, **kwargs):
            if _stub:
                return _stub
            return {k: kwargs.get(k) for k in _params if k in kwargs}

        tools.append(Tool(name=name, params=params, fn=_fn,
                          description=str(spec.get("description") or "")))
    if not tools:
        raise SystemExit("tool_calling_distill: no tools configured")
    return ToolRegistry(tools)


def _openai_tools(registry: ToolRegistry) -> list[dict]:
    out = []
    for tool in registry.tools.values():
        out.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {p: {"type": "string"} for p in tool.params},
                    "required": list(tool.params),
                },
            },
        })
    return out


def _client(cfg: dict):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "tool_calling_distill needs the 'openai' package "
            "(pip install openai)."
        ) from exc
    teacher = cfg.get("teacher") or {}
    token_env = str(teacher.get("api_key_env", "") or "")
    api_key = os.environ.get(token_env, "") if token_env else str(teacher.get("api_key") or "local")
    return OpenAI(
        base_url=str(teacher.get("base_url", "http://127.0.0.1:8080/v1")),
        api_key=api_key or "local",
    )


def _read_prompts(cfg: dict) -> list[str]:
    source = cfg.get("prompts") or {}
    path = source.get("path")
    if path:
        prompts = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, str):
                    prompts.append(row)
                elif isinstance(row, dict) and row.get("user"):
                    prompts.append(str(row["user"]))
        return prompts
    return [str(p) for p in (source.get("list") or [])]


def generate_traces(cfg: dict) -> list[ToolExample]:
    registry = build_registry(cfg.get("tools") or [])
    prompts = _read_prompts(cfg)
    limit = cfg.get("limit")
    if limit is not None:
        prompts = prompts[: int(limit)]
    if not prompts:
        raise SystemExit("tool_calling_distill: no prompts configured")

    client = _client(cfg)
    teacher = cfg.get("teacher") or {}
    model = str(teacher.get("model", "inkling-2-mini"))
    max_tokens = int(teacher.get("max_tokens", 256))
    temperature = float(teacher.get("temperature", 0.7))
    answer_mode = str(cfg.get("final_answer_mode", "template")).lower()

    examples: list[ToolExample] = []
    for user in prompts:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user}],
                tools=_openai_tools(registry),
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            logging.warning("teacher call failed for %r: %s", user[:60], exc)
            continue

        message = response.choices[0].message
        row = {
            "user": user,
            "tools": [
                {"function": {"name": t.name,
                              "parameters": {"properties": {p: {} for p in t.params}}}}
                for t in registry.tools.values()
            ],
            "tool_calls": [
                {
                    "function": {
                        "name": c.function.name,
                        "arguments": c.function.arguments or "{}",
                    }
                }
                for c in (message.tool_calls or [])
            ],
        }
        result = {}
        parsed = parse_call(
            f"CALL {row['tool_calls'][0]['function']['name']} "
            f"{row['tool_calls'][0]['function']['arguments']}"
        ) if row["tool_calls"] else None
        if parsed is not None:
            verified = registry.try_call(
                f"CALL {parsed[0]} {json.dumps(parsed[1], separators=(',', ':'))}"
            )
            if verified is not None:
                result = json.loads(verified[2])

        final = (message.content or "").strip()
        if answer_mode == "template" or not final:
            final = f"Result: {json.dumps(result, separators=(',', ':'))}"

        example = from_openai_row(row, tool_result=result, final_answer=final,
                                  source="teacher-distill")
        if example is None:
            logging.warning("could not convert teacher output for %r", user[:60])
            continue
        if example.call is None:
            # No tool used: keep it as a no-tool example with the teacher answer.
            example.answer = final
        examples.append(example)

    return examples


def main(argv: Optional[list[str]] = None) -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Distill tool calls from a teacher.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--skip-sft", action="store_true")
    parser.add_argument("--no-hf", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    load_env_file(ROOT_DIR / ".env")
    if args.limit is not None:
        cfg["limit"] = args.limit
    if args.no_hf:
        cfg.setdefault("hf", {})["enabled"] = False
    if args.upload:
        cfg.setdefault("hf", {})["enabled"] = True

    examples = generate_traces(cfg)
    out_path = Path(args.output or (cfg.get("data") or {}).get("path", "data/tool_traces.jsonl"))
    if not out_path.is_absolute():
        out_path = ROOT_DIR / out_path
    count = write_jsonl(examples, out_path)
    logging.info("Wrote %d distilled tool traces to %s", count, out_path)

    hf_cfg = get_hf_settings(cfg, disable=not bool((cfg.get("hf") or {}).get("enabled", False)))
    if hf_cfg:
        upload_outputs(out_path.parent, hf_cfg, step=0)

    if not args.skip_sft:
        from .tool_calling_sft import run_tool_sft

        sft_cfg_path = ROOT_DIR / str(
            (cfg.get("sft") or {}).get("config", "configs/tool_calling_sft.yaml")
        )
        sft_cfg = load_yaml(sft_cfg_path)
        if args.no_hf:
            sft_cfg.setdefault("hf", {})["enabled"] = False
        run_tool_sft(sft_cfg)


if __name__ == "__main__":
    main()
