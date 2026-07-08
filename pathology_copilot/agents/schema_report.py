"""Schema Report — build the final DiagnosticReport with grounding audit.

Rules enforced here:
- Every Evidence in the report references a call_id that appeared in state.executed.
- Every Grounding.region_id in the report must appear somewhere in state
  (either from a vision primitive or a domain tool). Anything else is
  hallucinated and rejected.
"""

from __future__ import annotations

from ..schemas import DiagnosticReport, Evidence, Grounding
from ..state import ExternalState


class SchemaReportBuilder:
    def build(self, state: ExternalState) -> DiagnosticReport:
        findings: list[Evidence] = []
        subtype: str | None = None
        grade: str | None = None
        mutations: list[str] = []
        primary_diagnosis = "No malignancy detected"
        confidences: list[float] = []
        has_cancer = False

        known_regions = state.known_region_ids()
        known_call_ids = {r.call_id for r in state.executed}

        for r in state.executed:
            if r.error is not None:
                continue

            grounding = r.grounding

            if r.tool_name == "cancer_detection":
                has_cancer = bool(r.output.get("has_cancer"))
                prob = float(r.output.get("probability", 0.0))
                confidences.append(prob)
                findings.append(
                    Evidence(
                        call_id=r.call_id,
                        statement=(
                            f"Cancer detection: {'positive' if has_cancer else 'negative'} "
                            f"(confidence={prob:.2f})"
                        ),
                        confidence=prob,
                        grounding=grounding,
                    )
                )
            elif r.tool_name == "subtype_classifier":
                subtype = r.output.get("subtype")
                prob = float(
                    max(r.output.get("probabilities", {"_": 0.0}).values())
                )
                confidences.append(prob)
                findings.append(
                    Evidence(
                        call_id=r.call_id,
                        statement=f"Subtype: {subtype} (p={prob:.2f})",
                        confidence=prob,
                        grounding=grounding,
                    )
                )
            elif r.tool_name == "grading":
                grade = r.output.get("grade")
                prob = float(r.output.get("probability", 0.0))
                confidences.append(prob)
                findings.append(
                    Evidence(
                        call_id=r.call_id,
                        statement=f"Histologic grade: {grade} (p={prob:.2f})",
                        confidence=prob,
                        grounding=grounding,
                    )
                )
            elif r.tool_name == "mutation_prediction":
                mutations = list(r.output.get("mutations", []))
                findings.append(
                    Evidence(
                        call_id=r.call_id,
                        statement=(
                            f"Predicted mutations: {', '.join(mutations) or 'none'}"
                        ),
                        confidence=float(r.output.get("probability", 0.0)),
                        grounding=grounding,
                    )
                )
            elif r.tool_name == "vlm_ask":
                findings.append(
                    Evidence(
                        call_id=r.call_id,
                        statement=f"VLM: {r.output.get('answer', '')}",
                        confidence=1.0 - r.uncertainty,
                        grounding=grounding,
                    )
                )
            elif r.tool_name == "guideline_search":
                findings.append(
                    Evidence(
                        call_id=r.call_id,
                        statement=(
                            f"Guideline {r.output.get('doc_id')}: "
                            f"{r.output.get('excerpt', '')}"
                        ),
                        confidence=1.0 - r.uncertainty,
                        grounding=grounding,
                    )
                )
            # thumbnail / region_view / similar_case_retrieval — evidence but not narrated.

        if has_cancer:
            primary_diagnosis = subtype or "Malignancy, subtype pending"

        report = DiagnosticReport(
            case_id=state.case_id,
            primary_diagnosis=primary_diagnosis,
            subtype=subtype,
            grade=grade,
            mutations=mutations,
            findings=findings,
            confidence=(
                sum(confidences) / len(confidences) if confidences else 0.5
            ),
            unresolved_flags=[n.message for n in state.critic_notes],
        )
        self._audit(report, known_regions=known_regions, known_call_ids=known_call_ids)
        return report

    @staticmethod
    def _audit(
        report: DiagnosticReport,
        *,
        known_regions: set[str],
        known_call_ids: set[str],
    ) -> None:
        for ev in report.findings:
            if ev.call_id not in known_call_ids:
                raise ValueError(
                    f"Evidence references unknown call_id={ev.call_id!r}"
                )
            for g in ev.grounding:
                if g.region_id not in known_regions:
                    raise ValueError(
                        f"Grounding region_id={g.region_id!r} not present in state"
                    )
