from __future__ import annotations

from typing import Any

from ..schemas import ToolCall, ToolResult
from .base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"no tool registered under {name!r}")
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def call(self, call: ToolCall, *, case_metadata: dict[str, Any]) -> ToolResult:
        tool = self.get(call.tool_name)
        return tool.run(call, case_metadata=case_metadata)


tool_registry = ToolRegistry()
