from __future__ import annotations

from typing import Any

from ..base_models.pathology_vit import ViTOutput
from .base import ExpertModel


class CancerDetection(ExpertModel):
    name = "cancer_detection"

    def predict(
        self, vit_output: ViTOutput, case_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        hint = case_metadata.get("hints", {}).get("cancer")
        if hint is True:
            return {"has_cancer": True, "probability": 0.91}
        if hint is False:
            return {"has_cancer": False, "probability": 0.95}
        # Fallback: derive a pseudo-probability from embedding energy.
        energy = sum(x * x for x in vit_output.embedding) / len(vit_output.embedding)
        p_cancer = min(0.99, energy)
        has_cancer = p_cancer > 0.5
        confidence = p_cancer if has_cancer else 1.0 - p_cancer
        return {"has_cancer": has_cancer, "probability": confidence}
