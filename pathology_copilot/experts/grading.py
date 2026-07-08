from __future__ import annotations

from typing import Any

from ..base_models.pathology_vit import ViTOutput
from .base import ExpertModel


class Grading(ExpertModel):
    name = "grading"

    GRADES = ["G1", "G2", "G3"]

    def predict(
        self, vit_output: ViTOutput, case_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        hint = case_metadata.get("hints", {}).get("grade")
        if hint in self.GRADES:
            return {"grade": hint, "probability": 0.85}
        # Fallback: bucket embedding norm.
        norm = sum(abs(x) for x in vit_output.embedding[:16]) / 16.0
        if norm < 0.4:
            pick = "G1"
        elif norm < 0.7:
            pick = "G2"
        else:
            pick = "G3"
        return {"grade": pick, "probability": 0.65}
