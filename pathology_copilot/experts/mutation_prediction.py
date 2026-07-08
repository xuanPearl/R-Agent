from __future__ import annotations

from typing import Any

from ..base_models.pathology_vit import ViTOutput
from .base import ExpertModel


class MutationPrediction(ExpertModel):
    name = "mutation_prediction"

    PANEL = ["HER2", "KRAS", "EGFR", "TP53"]

    def predict(
        self, vit_output: ViTOutput, case_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        hints = case_metadata.get("hints", {})
        if "mutations" in hints and isinstance(hints["mutations"], list):
            mutations = [m for m in hints["mutations"] if m in self.PANEL]
        else:
            # Fallback: pick mutations by sign of leading embedding dims.
            mutations = [
                gene
                for gene, x in zip(self.PANEL, vit_output.embedding[: len(self.PANEL)])
                if x > 0.4
            ]
        return {
            "mutations": mutations,
            "panel": self.PANEL,
            "probability": 0.6 if mutations else 0.3,
        }
