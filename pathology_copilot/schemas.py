"""Global pydantic contracts shared across tools, agents, and reports.

These are the invariants that keep the system pluggable:
- Every ToolResult carries `uncertainty` and `grounding` (the image's
  "统一接口 + 不确定性 + grounding" rule).
- Every Evidence in the final report references a call_id + grounding so the
  Schema Report layer can audit provenance.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


GroundingSource = Literal["thumbnail", "patch", "region", "kb", "vlm"]


class Grounding(BaseModel):
    region_id: str
    bbox: Optional[tuple[int, int, int, int]] = None
    source: GroundingSource


class ToolCall(BaseModel):
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    call_id: str = Field(default_factory=lambda: f"call_{uuid4().hex[:8]}")
    rationale: str = ""


class ToolResult(BaseModel):
    call_id: str
    tool_name: str
    output: dict[str, Any]
    uncertainty: float = Field(ge=0.0, le=1.0)
    grounding: list[Grounding] = Field(default_factory=list)
    error: Optional[str] = None

    @model_validator(mode="after")
    def _at_least_one_grounding_if_no_error(self) -> "ToolResult":
        if self.error is None and not self.grounding:
            raise ValueError(
                "ToolResult without error must carry at least one Grounding entry"
            )
        return self


class Evidence(BaseModel):
    call_id: str
    statement: str
    confidence: float = Field(ge=0.0, le=1.0)
    grounding: list[Grounding]


class DiagnosticReport(BaseModel):
    case_id: str
    primary_diagnosis: str
    subtype: Optional[str] = None
    grade: Optional[str] = None
    mutations: list[str] = Field(default_factory=list)
    findings: list[Evidence]
    confidence: float = Field(ge=0.0, le=1.0)
    unresolved_flags: list[str] = Field(default_factory=list)


class PlannerOutput(BaseModel):
    """Structured response the (mock) LLM returns to the Planner."""

    plan: list[ToolCall]
    rationale: str = ""


class CriticNote(BaseModel):
    kind: Literal["missing_grounding", "high_uncertainty", "inconsistency"]
    call_id: Optional[str] = None
    message: str


class CriticOutput(BaseModel):
    notes: list[CriticNote] = Field(default_factory=list)

    @property
    def has_flags(self) -> bool:
        return bool(self.notes)
