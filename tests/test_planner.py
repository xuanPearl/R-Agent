from __future__ import annotations

from pathology_copilot.agents import Planner
from pathology_copilot.llm import MockLLMClient
from pathology_copilot.state import ExternalState


def test_planner_positive_case_covers_diagnostic_pipeline():
    state = ExternalState.from_case(
        "R-TEST-1", {"case_id": "R-TEST-1", "hints": {"cancer": True, "subtype": "adenocarcinoma"}}
    )
    planner = Planner(MockLLMClient())
    plan = planner.plan(state).plan
    names = [c.tool_name for c in plan]
    assert names[0] == "thumbnail"
    assert "cancer_detection" in names
    assert "subtype_classifier" in names
    assert "grading" in names
    assert "mutation_prediction" in names


def test_planner_negative_case_skips_downstream_experts():
    state = ExternalState.from_case(
        "R-TEST-2", {"case_id": "R-TEST-2", "hints": {"cancer": False}}
    )
    planner = Planner(MockLLMClient())
    plan = planner.plan(state).plan
    names = [c.tool_name for c in plan]
    assert "cancer_detection" in names
    assert "grading" not in names
    assert "mutation_prediction" not in names


def test_planner_replan_skips_already_executed_tools():
    state = ExternalState.from_case(
        "R-TEST-3", {"case_id": "R-TEST-3", "hints": {"cancer": True}}
    )
    planner = Planner(MockLLMClient())
    first_plan = planner.plan(state).plan
    # Simulate executor having run thumbnail + cancer_detection.
    from pathology_copilot.schemas import Grounding, ToolResult

    for c in first_plan[:2]:
        state.executed.append(
            ToolResult(
                call_id=c.call_id,
                tool_name=c.tool_name,
                output={},
                uncertainty=0.1,
                grounding=[Grounding(region_id="thumbnail", source="thumbnail")],
            )
        )
    second_plan = planner.plan(state).plan
    names = [c.tool_name for c in second_plan]
    assert "thumbnail" not in names
    assert "cancer_detection" not in names
