"""Mock Pathology VLM.

Real: Qwen2.5-VL with dual encoder (semantic + cellular ViT), two-stage SFT.
Mock: rule-based response by matching question keywords to canned findings.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VLMAnswer:
    text: str
    confidence: float


class PathologyVLM:
    def ask(self, image_ref: str, question: str) -> VLMAnswer:
        q = question.lower()
        if "cytologic" in q or "features" in q:
            return VLMAnswer(
                text=(
                    "Enlarged pleomorphic nuclei with prominent nucleoli and "
                    "irregular glandular architecture consistent with adenocarcinoma."
                ),
                confidence=0.82,
            )
        if "necrosis" in q:
            return VLMAnswer(text="Focal necrosis observed.", confidence=0.7)
        return VLMAnswer(text="No additional findings.", confidence=0.5)
