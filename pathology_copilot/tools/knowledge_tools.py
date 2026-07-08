"""Knowledge tools: guideline lookup + similar-case retrieval.

Real impl would hit a vector store + curated guideline corpus. Mock uses a
built-in dict keyed by keyword.
"""

from __future__ import annotations

from typing import Any

from ..schemas import Grounding, ToolCall, ToolResult
from .base import Tool
from .registry import tool_registry


_GUIDELINES: dict[str, dict[str, Any]] = {
    "adenocarcinoma": {
        "doc_id": "NCCN-GA-2025",
        "excerpt": (
            "For gastric adenocarcinoma, staging per AJCC 8th; molecular "
            "testing for HER2, MSI, PD-L1 recommended."
        ),
    },
    "squamous_cell_carcinoma": {
        "doc_id": "NCCN-ESO-2025",
        "excerpt": "Esophageal SCC: consider CROSS regimen for locally advanced.",
    },
    "carcinoma": {
        "doc_id": "GEN-CARCINOMA-2024",
        "excerpt": "Generic carcinoma workup: staging, grade, IHC panel.",
    },
}


class GuidelineSearchTool(Tool):
    name = "guideline_search"
    description = "Retrieve a treatment-guideline excerpt matching a query."

    def run(self, call: ToolCall, *, case_metadata: dict[str, Any]) -> ToolResult:
        query = str(call.args.get("query", "")).lower()
        hit = next(
            (payload for key, payload in _GUIDELINES.items() if key in query),
            _GUIDELINES["carcinoma"],
        )
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.name,
            output={"query": query, **hit},
            uncertainty=0.15,
            grounding=[
                Grounding(region_id=hit["doc_id"], source="kb"),
            ],
        )


class SimilarCaseRetrievalTool(Tool):
    name = "similar_case_retrieval"
    description = "Return top-K historical cases similar to the current one."

    def run(self, call: ToolCall, *, case_metadata: dict[str, Any]) -> ToolResult:
        k = int(call.args.get("k", 3))
        cases = [
            {"case_id": f"HIST-{i:04d}", "similarity": round(0.95 - i * 0.05, 2)}
            for i in range(k)
        ]
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.name,
            output={"cases": cases},
            uncertainty=0.2,
            grounding=[
                Grounding(region_id=c["case_id"], source="kb") for c in cases
            ],
        )


tool_registry.register(GuidelineSearchTool())
tool_registry.register(SimilarCaseRetrievalTool())
