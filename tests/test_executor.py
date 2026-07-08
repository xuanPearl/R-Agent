from __future__ import annotations

from pathlib import Path

from pathology_copilot.agents import Executor, Planner
from pathology_copilot.llm import MockLLMClient
from pathology_copilot.state import ExternalState
from pathology_copilot.tools import tool_registry


CASE = {
    "case_id": "R-EXE-1",
    "hints": {"cancer": True, "subtype": "adenocarcinoma", "grade": "G2"},
}


def _prime_state() -> ExternalState:
    state = ExternalState.from_case("R-EXE-1", CASE)
    plan = Planner(MockLLMClient()).plan(state).plan
    state.register_plan(plan)
    return state


def test_executor_runs_full_plan():
    state = _prime_state()
    executor = Executor(tool_registry)
    executor.run(state)
    assert state.finished
    assert len(state.executed) == len(state.plan)
    assert not state.pending


def test_executor_stops_at_max_steps_and_resumes(tmp_path: Path):
    state = _prime_state()
    executor = Executor(tool_registry)
    executor.run(state, max_steps=2)
    assert not state.finished
    assert len(state.executed) == 2

    snapshot = tmp_path / "state.json"
    state.dump(snapshot)

    resumed = ExternalState.load(snapshot)
    assert len(resumed.executed) == 2
    Executor(tool_registry).run(resumed)
    assert resumed.finished
    assert len(resumed.executed) == len(resumed.plan)


def test_executor_captures_tool_failure_as_note():
    state = _prime_state()
    # Inject an unknown tool at the front of the pending queue.
    from pathology_copilot.schemas import ToolCall

    bad = ToolCall(tool_name="does_not_exist")
    state.pending.insert(0, bad)
    state.plan.insert(0, bad)

    Executor(tool_registry).run(state)
    assert any(n.kind == "inconsistency" for n in state.critic_notes)
    assert state.executed[0].error is not None
