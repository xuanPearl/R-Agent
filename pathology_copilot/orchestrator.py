"""Orchestrator — Planner → Executor → Self-Critic → Report.

If the Critic finds flags AND we're allowed another pass, we go back to the
Planner (which will only add tools not yet run — see MockLLMClient._planner_response).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from .agents import Executor, Planner, SchemaReportBuilder, SelfCritic
from .config import DEFAULT_CONFIG, Config
from .llm import MockLLMClient
from .schemas import CriticOutput, DiagnosticReport, ToolResult
from .state import ExternalState
from .tools import tool_registry


CriticHook = Callable[[int, CriticOutput], None]


class Orchestrator:
    def __init__(
        self,
        *,
        config: Config = DEFAULT_CONFIG,
        step_hook: Optional[Callable[[ExternalState, ToolResult], None]] = None,
        critic_hook: Optional[CriticHook] = None,
        max_critic_rounds: int = 1,
    ) -> None:
        self._config = config
        self._llm = MockLLMClient()
        self._planner = Planner(self._llm)
        self._executor = Executor(tool_registry, on_step=step_hook)
        self._critic = SelfCritic(self._llm, config)
        self._report = SchemaReportBuilder()
        self._critic_hook = critic_hook
        self._max_critic_rounds = max_critic_rounds

    def run(
        self,
        case_id: str,
        case_metadata: dict[str, Any],
        *,
        state: Optional[ExternalState] = None,
        max_steps: Optional[int] = None,
    ) -> tuple[ExternalState, Optional[DiagnosticReport]]:
        if state is None:
            state = ExternalState.from_case(case_id, case_metadata)

        if not state.plan:
            plan_out = self._planner.plan(state)
            state.register_plan(plan_out.plan)

        self._executor.run(state, max_steps=max_steps)

        # If we stopped early (max_steps), skip critic + report so caller can resume.
        if not state.finished:
            return state, None

        for round_num in range(1, self._max_critic_rounds + 2):
            output = self._critic.review(state)
            if self._critic_hook is not None:
                self._critic_hook(round_num, output)
            if not output.notes:
                break
            # Feed the notes back to the Planner. The Planner is trusted to
            # emit only the *new* steps that address the flags — we don't
            # de-dupe by tool name here, because a retry may re-run the same
            # tool on a different region.
            plan_out = self._planner.plan(state)
            new_calls = list(plan_out.plan)
            if not new_calls:
                break
            state.pending.extend(new_calls)
            state.plan.extend(new_calls)
            state.finished = False
            self._executor.run(state)

        report = self._report.build(state)
        return state, report
