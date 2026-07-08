from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..base_models.pathology_vit import ViTOutput


class ExpertModel(ABC):
    name: str

    @abstractmethod
    def predict(
        self, vit_output: ViTOutput, case_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Return a JSON-serializable dict with at least a `probability` field."""
        raise NotImplementedError
