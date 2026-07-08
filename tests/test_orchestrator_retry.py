"""Regression: Self-Critic flags ambiguous ROI → Planner replans → Critic clears."""

from __future__ import annotations

from pathology_copilot.orchestrator import Orchestrator
from pathology_copilot.schemas import CriticOutput


AMBIGUOUS_CASE = {
    "case_id": "R-RETRY-1",
    "hints": {
        "cancer": True,
        "ambiguous_regions": ["roi_center"],
        "retry_roi": "roi_periphery",
        "subtype": "adenocarcinoma",
        "grade": "G2",
    },
}


def test_orchestrator_retries_on_ambiguous_roi():
    rounds: list[tuple[int, CriticOutput]] = []
    orch = Orchestrator(critic_hook=lambda n, out: rounds.append((n, out)))
    state, report = orch.run(case_id="R-RETRY-1", case_metadata=AMBIGUOUS_CASE)
    assert report is not None

    # Round 1 should have flagged; a later round should end clean.
    assert len(rounds) >= 2
    assert rounds[0][1].notes, "critic should flag ambiguous ROI on round 1"
    assert any("different ROI" in n.message for n in rounds[0][1].notes)
    assert not rounds[-1][1].notes, "critic should clear after successful retry"

    # Two subtype_classifier calls exist: the ambiguous one and the retry.
    subtype_calls = [r for r in state.executed if r.tool_name == "subtype_classifier"]
    assert len(subtype_calls) == 2
    regions = [c.grounding[0].region_id for c in subtype_calls]
    assert "roi_center" in regions
    assert "roi_periphery" in regions

    # The retry ROI must show up as its own region_view step in the plan.
    assert any(
        r.tool_name == "region_view"
        and r.grounding
        and r.grounding[0].region_id == "roi_periphery"
        for r in state.executed
    )

    # Final report resolves to the confident retry.
    assert report.subtype == "adenocarcinoma"
    assert report.unresolved_flags == []
