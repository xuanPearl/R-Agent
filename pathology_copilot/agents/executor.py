"""Executor — walks the plan, calls the tool registry, updates state.

Every step is a checkpoint: after each ToolResult is appended to state,
the caller can dump state to disk and resume later.
"""

from __future__ import annotations

from typing import Callable, Optional

from ..schemas import CriticNote, ToolResult
from ..state import ExternalState
from ..tools import ToolRegistry


StepHook = Callable[[ExternalState, ToolResult], None]


class Executor:
    def __init__(self, registry: ToolRegistry, on_step: Optional[StepHook] = None) -> None:
        self._registry = registry
        self._on_step = on_step

    def run(self, state: ExternalState, *, max_steps: Optional[int] = None) -> ExternalState:
        taken = 0
        while state.pending:
            if max_steps is not None and taken >= max_steps:
                break
            call = state.pending[0]
            try:
                result = self._registry.call(
                    call, case_metadata=state.case_metadata
                )
            except Exception as exc:  # noqa: BLE001 — tool failures are data.
                result = ToolResult(
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    output={},
                    uncertainty=1.0,
                    grounding=[],
                    error=f"{type(exc).__name__}: {exc}",
                )
                state.critic_notes.append(
                    CriticNote(
                        kind="inconsistency",
                        call_id=call.call_id,
                        message=result.error or "tool failed",
                    )
                )
            state.record_result(result)
            taken += 1
            if self._on_step is not None:
                self._on_step(state, result)
        if not state.pending:
            state.finished = True
        return state
