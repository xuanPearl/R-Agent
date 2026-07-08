"""Planner — decomposes the diagnostic workflow into ToolCalls.

Calls MockLLMClient with role="planner". The mock inspects case metadata to
build a typical plan; a real LLM would derive the same plan from case notes.
"""

from __future__ import annotations

from ..llm import MockLLMClient
from ..schemas import PlannerOutput
from ..state import ExternalState


class Planner:
    def __init__(self, llm: MockLLMClient) -> None:
        self._llm = llm

    def plan(self, state: ExternalState) -> PlannerOutput:
        ctx = {
            "case_id": state.case_id,
            "case_metadata": state.case_metadata,
            "executed_tools": state.executed,
            "critic_notes": state.critic_notes,
        }
        return self._llm.complete(
            role="planner", context=ctx, response_schema=PlannerOutput
        )
