"""Tool registry + tool contract tests."""

from __future__ import annotations

import pytest

from pathology_copilot.schemas import Grounding, ToolCall, ToolResult
from pathology_copilot.tools import tool_registry


CASE_META = {"case_id": "TEST-001", "hints": {"cancer": True, "subtype": "adenocarcinoma", "grade": "G2"}}


def test_expected_tools_registered():
    names = set(tool_registry.names())
    assert {
        "thumbnail",
        "region_view",
        "cancer_detection",
        "subtype_classifier",
        "grading",
        "mutation_prediction",
        "guideline_search",
        "similar_case_retrieval",
        "vlm_ask",
    } <= names


@pytest.mark.parametrize("tool_name", ["thumbnail", "cancer_detection", "grading", "vlm_ask"])
def test_tool_output_has_grounding_and_uncertainty(tool_name):
    call = ToolCall(
        tool_name=tool_name,
        args={"case_id": "TEST-001", "region_id": "roi_center", "question": "features"},
    )
    result = tool_registry.call(call, case_metadata=CASE_META)
    assert isinstance(result, ToolResult)
    assert 0.0 <= result.uncertainty <= 1.0
    assert len(result.grounding) >= 1


def test_tool_result_rejects_missing_grounding():
    with pytest.raises(ValueError):
        ToolResult(
            call_id="x",
            tool_name="fake",
            output={},
            uncertainty=0.1,
            grounding=[],
            error=None,
        )


def test_tool_result_allows_missing_grounding_on_error():
    result = ToolResult(
        call_id="x",
        tool_name="fake",
        output={},
        uncertainty=1.0,
        grounding=[],
        error="boom",
    )
    assert result.error == "boom"


def test_domain_tool_reads_case_hints():
    call = ToolCall(tool_name="cancer_detection", args={"region_id": "roi_center"})
    result = tool_registry.call(call, case_metadata=CASE_META)
    assert result.output["has_cancer"] is True
    assert result.output["probability"] > 0.5
