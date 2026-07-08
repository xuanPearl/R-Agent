from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    mock_seed: int = 42
    high_uncertainty_threshold: float = 0.3
    verbose: bool = True


DEFAULT_CONFIG = Config()
