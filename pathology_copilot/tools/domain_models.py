"""Wrap each downstream expert as a Tool.

Every expert tool grabs an embedding from the mock ViT for the requested
region, calls the expert, and packages the output with grounding + a heuristic
uncertainty derived from the expert's probability.
"""

from __future__ import annotations

from typing import Any

from ..base_models.pathology_vit import PathologyViT
from ..experts import expert_registry
from ..schemas import Grounding, ToolCall, ToolResult
from .base import Tool
from .registry import tool_registry


_vit = PathologyViT()


def _uncertainty_from_prob(prob: float | None) -> float:
    if prob is None:
        return 0.5
    return max(0.0, min(1.0, 1.0 - float(prob)))


class _ExpertTool(Tool):
    expert_name: str

    def run(self, call: ToolCall, *, case_metadata: dict[str, Any]) -> ToolResult:
        region_id = call.args.get("region_id", "roi_center")
        case_id = call.args.get("case_id", case_metadata.get("case_id", "unknown"))
        vit_out = _vit.embed(f"{case_id}:{region_id}")
        expert = expert_registry.get(self.expert_name)
        output = expert.predict(vit_out, case_metadata)
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.name,
            output=output,
            uncertainty=_uncertainty_from_prob(output.get("probability")),
            grounding=[
                Grounding(region_id=region_id, source="patch"),
            ],
        )


class CancerDetectionTool(_ExpertTool):
    name = "cancer_detection"
    expert_name = "cancer_detection"
    description = "Binary tumor screen on a region."


class SubtypeClassifierTool(_ExpertTool):
    name = "subtype_classifier"
    expert_name = "subtype_classifier"
    description = "Tumor subtype from cellular features."


class GradingTool(_ExpertTool):
    name = "grading"
    expert_name = "grading"
    description = "Histologic grade."


class MutationPredictionTool(_ExpertTool):
    name = "mutation_prediction"
    expert_name = "mutation_prediction"
    description = "Predict actionable driver mutations."


tool_registry.register(CancerDetectionTool())
tool_registry.register(SubtypeClassifierTool())
tool_registry.register(GradingTool())
tool_registry.register(MutationPredictionTool())
