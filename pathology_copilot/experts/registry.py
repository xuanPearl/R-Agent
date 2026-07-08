"""ExpertRegistry — plug-in point for the 36+ downstream tasks.

Today we register 4 representative experts. Adding a new task = writing a
class that subclasses ExpertModel and calling `expert_registry.register(...)`.
"""

from __future__ import annotations

from .base import ExpertModel
from .cancer_detection import CancerDetection
from .grading import Grading
from .mutation_prediction import MutationPrediction
from .subtype_classifier import SubtypeClassifier


class ExpertRegistry:
    def __init__(self) -> None:
        self._experts: dict[str, ExpertModel] = {}

    def register(self, expert: ExpertModel) -> None:
        if expert.name in self._experts:
            raise ValueError(f"expert {expert.name!r} already registered")
        self._experts[expert.name] = expert

    def get(self, name: str) -> ExpertModel:
        if name not in self._experts:
            raise KeyError(f"no expert registered under {name!r}")
        return self._experts[name]

    def names(self) -> list[str]:
        return sorted(self._experts)


expert_registry = ExpertRegistry()
expert_registry.register(CancerDetection())
expert_registry.register(SubtypeClassifier())
expert_registry.register(Grading())
expert_registry.register(MutationPrediction())
