"""VLM Tool — "镜下所见 / 问答"."""

from __future__ import annotations

from typing import Any

from ..base_models.pathology_vlm import PathologyVLM
from ..schemas import Grounding, ToolCall, ToolResult
from .base import Tool
from .registry import tool_registry


_vlm = PathologyVLM()


class VLMAskTool(Tool):
    name = "vlm_ask"
    description = "Ask the pathology VLM about a specific region."

    def run(self, call: ToolCall, *, case_metadata: dict[str, Any]) -> ToolResult:
        region_id = call.args.get("region_id", "roi_center")
        case_id = call.args.get("case_id", case_metadata.get("case_id", "unknown"))
        question = str(call.args.get("question", "Describe the findings."))
        answer = _vlm.ask(f"patch://{case_id}/{region_id}", question)
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.name,
            output={"question": question, "answer": answer.text},
            uncertainty=max(0.0, min(1.0, 1.0 - answer.confidence)),
            grounding=[
                Grounding(region_id=region_id, source="vlm"),
            ],
        )


tool_registry.register(VLMAskTool())
