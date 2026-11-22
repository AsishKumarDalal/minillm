"""Tool registry, parser and executor.

Used by negative sampling (to decide chosen vs rejected) and available as a
harness for inference.  Tool functions stay in Python; the model only emits
the compact ``CALL name {json}`` text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from .data_format import make_result, parse_call


@dataclass
class Tool:
    name: str
    params: tuple[str, ...]
    fn: Callable[..., Any]
    description: str = ""


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        if not tools:
            raise ValueError("ToolRegistry needs at least one tool")
        self.tools: dict[str, Tool] = {t.name: t for t in tools}
        if len(self.tools) != len(tools):
            raise ValueError("duplicate tool names")

    @property
    def header(self) -> str:
        return ",".join(
            f"{t.name}({','.join(t.params)})" for t in self.tools.values()
        )

    def parse(self, text: str):
        return parse_call(text)

    def execute(self, name: str, args: dict):
        if name not in self.tools:
            raise KeyError(f"unknown tool {name!r}")
        tool = self.tools[name]
        unknown = set(args) - set(tool.params)
        if unknown:
            raise ValueError(f"{name} got unexpected args {sorted(unknown)}")
        return tool.fn(**args)

    def try_call(self, text: str):
        """Parse and execute an assistant line.

        Returns ``(name, args, result_string)`` or ``None`` when invalid.
        """
        parsed = self.parse(text)
        if parsed is None:
            return None
        name, args = parsed
        try:
            result = self.execute(name, args)
        except Exception as exc:
            logging.debug("tool execution failed: %s", exc)
            return None
        return name, args, make_result(result)


def demo_registry() -> ToolRegistry:
    """Two tiny deterministic tools used by tests and examples."""

    def weather(city: str):
        readings = {"Paris": 22, "London": 15, "Tokyo": 21, "New York": 18}
        return {"city": city, "temp_c": readings.get(city, 20)}

    def calc(expr: str):
        # Restrictive on purpose: only simple arithmetic for demos.
        allowed = set("0123456789+-*/(). ")
        if not set(expr) <= allowed:
            raise ValueError("unsafe expression")
        return {"result": eval(expr, {"__builtins__": {}}, {})}  # noqa: S307

    return ToolRegistry([
        Tool("weather", ("city",), weather, "Get temperature for a city"),
        Tool("calc", ("expr",), calc, "Evaluate simple arithmetic"),
    ])
