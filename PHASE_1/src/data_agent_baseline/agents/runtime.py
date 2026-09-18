from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from data_agent_baseline.benchmark.schema import AnswerTable


@dataclass(frozen=True, slots=True)
class StepRecord:
    step_index: int
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str
    observation: dict[str, Any]
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentRuntimeState:
    steps: list[StepRecord] = field(default_factory=list)
    answer: AnswerTable | None = None
    failure_reason: str | None = None
    evidence_memory: dict[str, Any] | None = None
    verification_required: list[str] = field(default_factory=list)
    verification_attempts: int = 0


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    task_id: str
    answer: AnswerTable | None
    steps: list[StepRecord]
    failure_reason: str | None
    evidence_memory: dict[str, Any] | None = None
    verification_required: list[str] = field(default_factory=list)
    verification_attempts: int = 0

    @property
    def succeeded(self) -> bool:
        return self.answer is not None and self.failure_reason is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "answer": self.answer.to_dict() if self.answer is not None else None,
            "steps": [step.to_dict() for step in self.steps],
            "failure_reason": self.failure_reason,
            "evidence_memory": self.evidence_memory,
            "verification_required": list(self.verification_required),
            "verification_attempts": self.verification_attempts,
            "succeeded": self.succeeded,
        }
