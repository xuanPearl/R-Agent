from __future__ import annotations

import pytest

from pathology_copilot.agents import SchemaReportBuilder
from pathology_copilot.orchestrator import Orchestrator
from pathology_copilot.schemas import Evidence, Grounding, ToolResult
from pathology_copilot.state import ExternalState


CASE = {
    "case_id": "R-RPT-1",
    "hints": {"cancer": True, "subtype": "adenocarcinoma", "grade": "G2", "mutations": ["HER2"]},
}


def test_report_end_to_end_positive():
    orch = Orchestrator()
    state, report = orch.run(case_id="R-RPT-1", case_metadata=CASE)
    assert report is not None
    assert state.finished
    assert report.primary_diagnosis == "adenocarcinoma"
    assert report.grade == "G2"
    assert "HER2" in report.mutations
    assert report.findings
    assert 0.0 <= report.confidence <= 1.0
    # Every finding's grounding region must exist in state.
    known = state.known_region_ids()
    for ev in report.findings:
        for g in ev.grounding:
            assert g.region_id in known


def test_report_rejects_unknown_grounding_region():
    state = ExternalState.from_case("R-RPT-2", CASE)
    state.executed.append(
        ToolResult(
            call_id="c1",
            tool_name="cancer_detection",
            output={"has_cancer": True, "probability": 0.9},
            uncertainty=0.1,
            grounding=[Grounding(region_id="ghost", source="patch")],
        )
    )
    # Force builder to run — it should refuse: 'ghost' is in state (via executed
    # grounding), so add a Finding referencing a region NOT anywhere in state.
    builder = SchemaReportBuilder()
    report = builder.build(state)
    # Now craft one manually with a fake region.
    with pytest.raises(ValueError):
        fake_ev = Evidence(
            call_id="c1",
            statement="fake",
            confidence=0.5,
            grounding=[Grounding(region_id="not_in_state", source="patch")],
        )
        SchemaReportBuilder._audit(
            report.model_copy(update={"findings": [fake_ev]}),
            known_regions=state.known_region_ids(),
            known_call_ids={r.call_id for r in state.executed},
        )
