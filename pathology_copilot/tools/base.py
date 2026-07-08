"""Tool base class.

Every tool obeys the same contract:
    run(args: dict, *, case_metadata: dict) -> ToolResult

`ToolResult` enforces uncertainty + grounding in its own validator, so tools
that forget to attach evidence fail loudly rather than silently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..schemas import ToolCall, ToolResult


class Tool(ABC):
    name: str
    description: str = ""

    @abstractmethod
    def run(
        self,
        call: ToolCall,
        *,
        case_metadata: dict[str, Any],
    ) -> ToolResult:
        raise NotImplementedError
