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
            return {"has_cancer": False, "probability": 0.05}
        # Fallback: derive a pseudo-probability from embedding energy.
        energy = sum(x * x for x in vit_output.embedding) / len(vit_output.embedding)
        return {"has_cancer": energy > 0.33, "probability": min(0.99, energy)}
