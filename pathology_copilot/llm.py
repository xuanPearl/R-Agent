"""Mock LLM client.

Signature is intentionally close to a real client (`complete(prompt, schema)`),
so swapping in Anthropic/OpenAI later is a one-file change. The mock inspects
`prompt` for role keywords and dispatches to a hand-written response builder.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from .schemas import CriticNote, CriticOutput, PlannerOutput, ToolCall


T = TypeVar("T", bound=BaseModel)


class MockLLMClient:
    """Deterministic pseudo-LLM used by Planner and Self-Critic."""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[dict[str, Any]], BaseModel]] = {
            "planner": self._planner_response,
            "critic": self._critic_response,
        }

    def complete(
        self,
        *,
        role: str,
        context: dict[str, Any],
        response_schema: type[T],
    ) -> T:
        if role not in self._handlers:
            raise ValueError(f"MockLLMClient has no handler for role={role!r}")
        result = self._handlers[role](context)
        if not isinstance(result, response_schema):
            raise TypeError(
                f"Mock handler returned {type(result).__name__}, "
                f"expected {response_schema.__name__}"
            )
        return result

    # ------------------------------------------------------------------ Planner

    def _planner_response(self, ctx: dict[str, Any]) -> PlannerOutput:
        """Build a typical diagnostic plan.

        The mock uses `case_metadata.hints` to branch — in production the real
        LLM would derive the same plan from clinical text + first-look tools.
        """
        case_id = ctx["case_id"]
        hints = ctx.get("case_metadata", {}).get("hints", {})
        already_run = {c.tool_name for c in ctx.get("executed_tools", [])}

        steps: list[ToolCall] = []

        def add(tool_name: str, args: dict[str, Any], rationale: str) -> None:
            if tool_name in already_run:
                return
            steps.append(ToolCall(tool_name=tool_name, args=args, rationale=rationale))

        add("thumbnail", {"case_id": case_id}, "overview of the slide")
        add(
            "cancer_detection",
            {"case_id": case_id, "region_id": "roi_center"},
            "screen for malignancy on the central ROI",
        )

        if hints.get("cancer", True):
            add(
                "subtype_classifier",
                {"case_id": case_id, "region_id": "roi_center"},
                "subtype the tumor once malignancy is suspected",
            )
            add(
                "grading",
                {"case_id": case_id, "region_id": "roi_center"},
                "histologic grade for treatment planning",
            )
            add(
                "mutation_prediction",
                {"case_id": case_id, "region_id": "roi_center"},
                "predict actionable mutations",
            )
            add(
                "guideline_search",
                {"query": f"{hints.get('subtype', 'carcinoma')} treatment guideline"},
                "align findings with clinical guideline",
            )
            add(
                "vlm_ask",
                {
                    "case_id": case_id,
                    "region_id": "roi_center",
                    "question": "Describe cytologic features supporting the diagnosis.",
                },
                "cross-check with VLM narrative",
            )

        return PlannerOutput(plan=steps, rationale="rule-based mock plan")

    # ------------------------------------------------------------------- Critic

    def _critic_response(self, ctx: dict[str, Any]) -> CriticOutput:
        notes: list[CriticNote] = []
        results = ctx.get("executed_results", [])
        threshold = float(ctx.get("high_uncertainty_threshold", 0.3))

        seen_tool_names = {r.tool_name for r in results}
        for r in results:
            if r.error is not None:
                notes.append(
                    CriticNote(
                        kind="inconsistency",
                        call_id=r.call_id,
                        message=f"{r.tool_name} failed: {r.error}",
                    )
                )
                continue
            if not r.grounding:
                notes.append(
                    CriticNote(
                        kind="missing_grounding",
                        call_id=r.call_id,
                        message=f"{r.tool_name} produced no grounding evidence",
                    )
                )
            if r.uncertainty >= threshold and "vlm_ask" not in seen_tool_names:
                notes.append(
                    CriticNote(
                        kind="high_uncertainty",
                        call_id=r.call_id,
                        message=(
                            f"{r.tool_name} uncertainty={r.uncertainty:.2f} "
                            f">= {threshold}; no VLM confirmation observed"
                        ),
                    )
                )

        # Consistency: cancer detected but grading missing.
        has_cancer = any(
            r.tool_name == "cancer_detection"
            and r.output.get("has_cancer") is True
            for r in results
        )
        graded = any(r.tool_name == "grading" for r in results)
        if has_cancer and not graded:
            notes.append(
                CriticNote(
                    kind="inconsistency",
                    message="cancer detected but no histologic grade was produced",
                )
            )
        return CriticOutput(notes=notes)
