"""Mock Pathology ViT.

Real: DINOv3 self-supervised ViT-L (450k WSI / 200M patches) with 2D RoPE.
Mock: deterministic hash-derived pseudo-embedding + fake attention map.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass


EMBED_DIM = 768
ATTENTION_HW = 16


@dataclass
class ViTOutput:
    embedding: list[float]
    attention: list[list[float]]


class PathologyViT:
    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def embed(self, patch_ref: str) -> ViTOutput:
        # Deterministic per patch_ref so repeated calls are stable.
        digest = hashlib.sha256(f"{self._seed}:{patch_ref}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        embedding = [rng.uniform(-1.0, 1.0) for _ in range(EMBED_DIM)]
        attention = [
            [rng.random() for _ in range(ATTENTION_HW)]
            for _ in range(ATTENTION_HW)
        ]
        return ViTOutput(embedding=embedding, attention=attention)
