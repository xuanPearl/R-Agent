from __future__ import annotations

from typing import Any

from ..base_models.pathology_vit import ViTOutput
from .base import ExpertModel


class SubtypeClassifier(ExpertModel):
    name = "subtype_classifier"

    CANDIDATES = ["adenocarcinoma", "squamous_cell_carcinoma", "neuroendocrine"]

    def predict(
        self, vit_output: ViTOutput, case_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        hints = case_metadata.get("hints", {})
        region_id = case_metadata.get("_current_region", "")
        ambiguous_regions = hints.get("ambiguous_regions", [])
        if region_id and region_id in ambiguous_regions:
            # Flat distribution — the model cannot commit to a subtype here.
            probs = {c: round(1.0 / len(self.CANDIDATES), 2) for c in self.CANDIDATES}
            return {"subtype": None, "probabilities": probs}
        hint = hints.get("subtype")
        if hint in self.CANDIDATES:
            probs = {c: 0.05 for c in self.CANDIDATES}
            probs[hint] = 0.9
            return {"subtype": hint, "probabilities": probs}
        # Fallback: pick by first embedding sign.
        idx = 0 if vit_output.embedding[0] >= 0 else 1
        pick = self.CANDIDATES[idx]
        return {
            "subtype": pick,
            "probabilities": {c: (0.6 if c == pick else 0.2) for c in self.CANDIDATES},
        }
