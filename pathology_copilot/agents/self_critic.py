"""Self-Critic — evidence / grounding / consistency review.

Uses MockLLMClient with role="critic". Three rules today:
1. Every non-error ToolResult must carry grounding (missing_grounding).
2. High-uncertainty results should have a VLM cross-check (high_uncertainty).
3. If cancer was detected, grading must be present (inconsistency).
"""

from __future__ import annotations

from ..config import DEFAULT_CONFIG, Config
from ..llm import MockLLMClient
from ..schemas import CriticOutput
from ..state import ExternalState


class SelfCritic:
    def __init__(
        self, llm: MockLLMClient, config: Config = DEFAULT_CONFIG
    ) -> None:
        self._llm = llm
        self._config = config

    def review(self, state: ExternalState) -> CriticOutput:
        ctx = {
            "executed_results": state.executed,
            "high_uncertainty_threshold": self._config.high_uncertainty_threshold,
        }
        output = self._llm.complete(
            role="critic", context=ctx, response_schema=CriticOutput
        )
        # Replace, don't append: a review reflects the *current* state of the
        # world. A successful retry should clear a previously-raised flag.
        state.critic_notes = list(output.notes)
        return output
