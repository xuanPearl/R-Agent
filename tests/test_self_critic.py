from __future__ import annotations

from pathology_copilot.agents import SelfCritic
from pathology_copilot.llm import MockLLMClient
from pathology_copilot.schemas import Grounding, ToolResult
from pathology_copilot.state import ExternalState


def _mk_result(**kwargs):
    kwargs.setdefault("call_id", "c1")
    kwargs.setdefault("tool_name", "cancer_detection")
    kwargs.setdefault("output", {"has_cancer": True, "probability": 0.9})
    kwargs.setdefault("uncertainty", 0.1)
    kwargs.setdefault(
        "grounding", [Grounding(region_id="roi_center", source="patch")]
    )
    return ToolResult(**kwargs)


def test_flags_cancer_without_grading():
    state = ExternalState.from_case("R-CRT-1", {})
    state.executed.append(_mk_result())
    critic = SelfCritic(MockLLMClient())
    out = critic.review(state)
    assert any(n.kind == "inconsistency" for n in out.notes)


def test_flags_high_uncertainty_without_vlm():
    state = ExternalState.from_case("R-CRT-2", {})
    state.executed.append(_mk_result(uncertainty=0.7))
    state.executed.append(
        _mk_result(
            call_id="c2",
            tool_name="grading",
            output={"grade": "G2", "probability": 0.8},
        )
    )
    critic = SelfCritic(MockLLMClient())
    out = critic.review(state)
    assert any(n.kind == "high_uncertainty" for n in out.notes)


def test_clean_run_produces_no_flags():
    state = ExternalState.from_case("R-CRT-3", {})
    state.executed.append(_mk_result())
    state.executed.append(
        _mk_result(
            call_id="c2",
            tool_name="grading",
            output={"grade": "G2", "probability": 0.85},
            uncertainty=0.15,
        )
    )
    critic = SelfCritic(MockLLMClient())
    out = critic.review(state)
    assert not out.has_flags
