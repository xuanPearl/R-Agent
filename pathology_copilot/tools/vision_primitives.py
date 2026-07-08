"""Vision primitives: thumbnail + region_view.

These are the entry points for grounding — every downstream tool references
regions defined here. In real deployment they'd read an OpenSlide handle;
here they return synthetic bbox coordinates.
"""

from __future__ import annotations

from typing import Any

from ..schemas import Grounding, ToolCall, ToolResult
from .base import Tool
from .registry import tool_registry


class ThumbnailTool(Tool):
    name = "thumbnail"
    description = "Return a low-magnification overview of the WSI."

    def run(self, call: ToolCall, *, case_metadata: dict[str, Any]) -> ToolResult:
        case_id = call.args.get("case_id", case_metadata.get("case_id", "unknown"))
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.name,
            output={
                "case_id": case_id,
                "image_ref": f"thumbnail://{case_id}",
                "size": [1024, 1024],
            },
            uncertainty=0.0,
            grounding=[
                Grounding(
                    region_id="thumbnail",
                    bbox=(0, 0, 1024, 1024),
                    source="thumbnail",
                )
            ],
        )


class RegionViewTool(Tool):
    name = "region_view"
    description = "Return a high-magnification patch handle for a bbox."

    def run(self, call: ToolCall, *, case_metadata: dict[str, Any]) -> ToolResult:
        case_id = call.args.get("case_id", case_metadata.get("case_id", "unknown"))
        bbox = tuple(call.args.get("bbox", [512, 512, 768, 768]))
        region_id = call.args.get("region_id", f"roi_{bbox[0]}_{bbox[1]}")
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.name,
            output={
                "case_id": case_id,
                "image_ref": f"patch://{case_id}/{region_id}",
                "bbox": list(bbox),
            },
            uncertainty=0.0,
            grounding=[
                Grounding(region_id=region_id, bbox=bbox, source="patch"),
            ],
        )


tool_registry.register(ThumbnailTool())
tool_registry.register(RegionViewTool())
