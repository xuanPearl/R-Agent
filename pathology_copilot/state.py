"""External Structured State — checkpointable, resumable, auditable.

Mirrors the image's "token 可控 / 可中断恢复" bullet: state is a plain pydantic
object serializable to JSON, so the Executor can be interrupted after any step
and picked up by another process.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .schemas import CriticNote, ToolCall, ToolResult


class ExternalState(BaseModel):
    case_id: str
    case_metadata: dict[str, Any] = Field(default_factory=dict)
    plan: list[ToolCall] = Field(default_factory=list)
    executed: list[ToolResult] = Field(default_factory=list)
    pending: list[ToolCall] = Field(default_factory=list)
    critic_notes: list[CriticNote] = Field(default_factory=list)
    step: int = 0
    finished: bool = False

    def register_plan(self, plan: list[ToolCall]) -> None:
        self.plan = list(plan)
        self.pending = list(plan)
        self.executed = []
        self.step = 0
        self.finished = False

    def record_result(self, result: ToolResult) -> None:
        self.executed.append(result)
        if self.pending:
            self.pending.pop(0)
        self.step += 1

    def known_region_ids(self) -> set[str]:
        ids: set[str] = set()
        for r in self.executed:
            for g in r.grounding:
                ids.add(g.region_id)
        return ids

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ExternalState":
        return cls.model_validate_json(Path(path).read_text())

    @classmethod
    def from_case(cls, case_id: str, case_metadata: dict[str, Any]) -> "ExternalState":
        return cls(case_id=case_id, case_metadata=case_metadata)
