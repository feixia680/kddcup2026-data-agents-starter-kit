from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import build_observation_prompt, build_system_prompt, build_task_prompt
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.agents.result_verifier import verify_tool_result
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16
    force_answer_remaining: int = 2
    repeat_warning_threshold: int = 2
    model_retry_limit: int = 1
    verification_retry_limit: int = 2
    tool_call_budgets: dict[str, int] = field(
        default_factory=lambda: {
            "list_context": 2,
            "read_doc": 4,
            "read_csv": 4,
            "read_json": 4,
            "inspect_sqlite_schema": 2,
            "execute_context_sql": 4,
            "execute_python": 6,
            "extract_document_records": 3,
        }
    )


def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    return text


def _load_single_json_object(text: str) -> dict[str, object]:
    payload, end = json.JSONDecoder().raw_decode(text)
    remainder = text[end:].strip()
    if remainder:
        cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
        if cleaned_remainder:
            raise ValueError("Model response must contain only one JSON object.")
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    return payload


def _compact_evidence(action: str, action_input: dict[str, Any], content: dict[str, Any]) -> dict[str, Any]:
    source_input = {}
    for key in ("path", "sql", "code"):
        if key in action_input:
            value = str(action_input[key])
            source_input[key] = value if len(value) <= 1200 else value[:1200] + "...[truncated]"
    serialized = json.dumps(content, ensure_ascii=False, default=str)
    if len(serialized) <= 6000:
        compact_content: object = content
    else:
        compact_content = {"truncated": True, "preview": serialized[:6000] + "...[truncated]"}
    return {"source_action": action, "source_input": source_input, "content": compact_content}


def parse_model_step(raw_response: str) -> ModelStep:
    payload = _load_single_json_object(_strip_json_fence(raw_response))
    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")
    return ModelStep(thought=thought, action=action, action_input=action_input, raw_response=raw_response)


class ReActAgent:
    def __init__(self, *, model: ModelAdapter, tools: ToolRegistry, config: ReActAgentConfig | None = None, system_prompt: str | None = None, checkpoint_callback: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt
        self.checkpoint_callback = checkpoint_callback

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState, step_index: int, *, answer_only: bool = False) -> list[ModelMessage]:
        tool_descriptions = self.tools.describe_for_prompt()
        system_prompt = self.system_prompt
        if answer_only:
            answer_spec = self.tools.specs.get("answer")
            if answer_spec is not None:
                tool_descriptions = f"- {answer_spec.name}: {answer_spec.description}\n  input_schema: {answer_spec.input_schema}"
            system_prompt = (system_prompt or "") + "\nFINAL ANSWER ENFORCEMENT: call answer now. Do not select or describe any exploratory tool."
        system_content = build_system_prompt(tool_descriptions, system_prompt=system_prompt)
        messages = [ModelMessage(role="system", content=system_content)]
        remaining = self.config.max_steps - step_index + 1
        messages.append(ModelMessage(role="user", content=build_task_prompt(task, remaining_steps=remaining)))
        for step in state.steps:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(ModelMessage(role="user", content=build_observation_prompt(step.observation)))
        if state.evidence_memory is not None:
            messages.append(ModelMessage(
                role="user",
                content="CURRENT EVIDENCE MEMORY (preserve provenance; do not treat it as a new tool call):\n"
                + json.dumps(state.evidence_memory, ensure_ascii=False, indent=2),
            ))
        return messages

    def _checkpoint(self, task: PublicTask, state: AgentRuntimeState) -> None:
        if self.checkpoint_callback is None:
            return
        payload = AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
            evidence_memory=state.evidence_memory,
            verification_required=list(state.verification_required),
            verification_attempts=state.verification_attempts,
        ).to_dict()
        payload["verification_required"] = list(state.verification_required)
        payload["verification_attempts"] = state.verification_attempts
        self.checkpoint_callback(payload)

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        previous_signature: tuple[str, str] | None = None
        repeated_actions = 0
        consecutive_model_errors = 0
        tool_call_counts: dict[str, int] = {}
        for step_index in range(1, self.config.max_steps + 1):
            step_started = perf_counter()
            remaining_steps = self.config.max_steps - step_index + 1
            answer_only = remaining_steps <= self.config.force_answer_remaining
            raw_response = ""
            model_started = perf_counter()
            try:
                raw_response = self.model.complete(self._build_messages(task, state, step_index, answer_only=answer_only))
                model_elapsed = perf_counter() - model_started
                consecutive_model_errors = 0
            except Exception as exc:
                model_elapsed = perf_counter() - model_started
                consecutive_model_errors += 1
                observation = {
                    "ok": False,
                    "error": str(exc),
                    "retryable": consecutive_model_errors <= self.config.model_retry_limit,
                    "telemetry": {
                        "step_index": step_index,
                        "model_seconds": model_elapsed,
                        "tool_seconds": 0.0,
                        "step_seconds": perf_counter() - step_started,
                    },
                }
                state.steps.append(StepRecord(step_index=step_index, thought="", action="__model_error__", action_input={}, raw_response=raw_response, observation=observation, ok=False))
                self._checkpoint(task, state)
                if consecutive_model_errors > self.config.model_retry_limit:
                    state.failure_reason = f"Model request failed after {consecutive_model_errors} attempts: {exc}"
                    break
                continue
            try:
                model_step = parse_model_step(raw_response)
                signature = (model_step.action, json.dumps(model_step.action_input, sort_keys=True, ensure_ascii=False))
                repeated_actions = repeated_actions + 1 if signature == previous_signature else 0
                previous_signature = signature
                if answer_only and model_step.action != "answer" and not state.verification_required:
                    observation = {
                        "ok": False,
                        "error": "answer_required",
                        "control": "Exploration is disabled in the final-answer budget. Call answer with the best verified evidence.",
                        "telemetry": {
                            "step_index": step_index,
                            "model_seconds": model_elapsed,
                            "tool_seconds": 0.0,
                            "step_seconds": perf_counter() - step_started,
                        },
                    }
                    state.steps.append(StepRecord(step_index=step_index, thought=model_step.thought, action="__answer_required__", action_input=model_step.action_input, raw_response=raw_response, observation=observation, ok=False))
                    self._checkpoint(task, state)
                    continue
                if model_step.action == "answer" and state.verification_required:
                    if state.verification_attempts < self.config.verification_retry_limit:
                        state.verification_attempts += 1
                        observation = {
                            "ok": False,
                            "error": "verification_required",
                            "verification_required": list(state.verification_required),
                            "control": "Do not submit this answer yet. Re-run the relevant query or computation and remove the critical verification warnings.",
                            "telemetry": {
                                "step_index": step_index,
                                "model_seconds": model_elapsed,
                                "tool_seconds": 0.0,
                                "step_seconds": perf_counter() - step_started,
                            },
                        }
                        state.steps.append(StepRecord(step_index=step_index, thought=model_step.thought, action="__verification_required__", action_input=model_step.action_input, raw_response=raw_response, observation=observation, ok=False))
                        self._checkpoint(task, state)
                        continue
                    state.failure_reason = "Critical verification warnings remained unresolved before answer submission."
                    break
                if model_step.action != "answer":
                    tool_limit = self.config.tool_call_budgets.get(model_step.action)
                    tool_calls = tool_call_counts.get(model_step.action, 0)
                    if tool_limit is not None and tool_calls >= tool_limit:
                        observation = {
                            "ok": False,
                            "error": "tool_budget_exceeded",
                            "tool": model_step.action,
                            "tool_calls": tool_calls,
                            "tool_limit": tool_limit,
                            "control": "This tool has reached its call budget. Switch to a different tool or submit the best supported answer.",
                            "telemetry": {
                                "step_index": step_index,
                                "model_seconds": model_elapsed,
                                "tool_seconds": 0.0,
                                "step_seconds": perf_counter() - step_started,
                            },
                        }
                        state.steps.append(StepRecord(step_index=step_index, thought=model_step.thought, action="__tool_budget_exceeded__", action_input={"tool": model_step.action, "limit": tool_limit}, raw_response=raw_response, observation=observation, ok=False))
                        self._checkpoint(task, state)
                        continue
                    tool_call_counts[model_step.action] = tool_calls + 1
                tool_started = perf_counter()
                tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
                tool_elapsed = perf_counter() - tool_started
                verification = verify_tool_result(task, model_step.action, model_step.action_input, tool_result.content)
                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                    "verification": verification.to_dict(),
                    "telemetry": {
                        "step_index": step_index,
                        "model_seconds": model_elapsed,
                        "tool_seconds": tool_elapsed,
                        "step_seconds": perf_counter() - step_started,
                    },
                }
                if repeated_actions >= self.config.repeat_warning_threshold:
                    observation["control"] = "This action repeats a previous action. Stop exploring and submit the best supported answer."
                step_record = StepRecord(step_index=step_index, thought=model_step.thought, action=model_step.action, action_input=model_step.action_input, raw_response=raw_response, observation=observation, ok=tool_result.ok)
                state.steps.append(step_record)
                if tool_result.ok and model_step.action != "answer":
                    state.evidence_memory = _compact_evidence(model_step.action, model_step.action_input, tool_result.content)
                critical_warnings = [
                    warning
                    for warning in verification.warnings
                    if any(token in warning.lower() for token in ("sum without avg", "average question uses sum", "fixed 12", "divides a total", "avg without sum", "count posts", "per-row ratio"))
                ]
                state.verification_required = critical_warnings
                if not critical_warnings:
                    state.verification_attempts = 0
                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    self._checkpoint(task, state)
                    break
                self._checkpoint(task, state)
            except Exception as exc:
                observation = {
                    "ok": False,
                    "error": str(exc),
                    "telemetry": {
                        "step_index": step_index,
                        "model_seconds": model_elapsed,
                        "tool_seconds": 0.0,
                        "step_seconds": perf_counter() - step_started,
                    },
                }
                state.steps.append(StepRecord(step_index=step_index, thought="", action="__error__", action_input={}, raw_response=raw_response, observation=observation, ok=False))
                self._checkpoint(task, state)

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."
        result = AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
            evidence_memory=state.evidence_memory,
            verification_required=list(state.verification_required),
            verification_attempts=state.verification_attempts,
        )
        self._checkpoint(task, state)
        return result
