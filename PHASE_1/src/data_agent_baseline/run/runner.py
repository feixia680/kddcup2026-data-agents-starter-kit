from __future__ import annotations

import csv
import json
import multiprocessing
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AppConfig
from data_agent_baseline.tools.registry import ToolRegistry, create_default_tool_registry


@dataclass(frozen=True, slots=True)
class TaskRunArtifacts:
    task_id: str
    task_output_dir: Path
    prediction_csv_path: Path | None
    trace_path: Path
    succeeded: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_output_dir": str(self.task_output_dir),
            "prediction_csv_path": str(self.prediction_csv_path) if self.prediction_csv_path else None,
            "trace_path": str(self.trace_path),
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
        }


def create_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_run_id(run_id: str | None = None) -> str:
    if run_id is None:
        return create_run_id()
    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must not be empty.")
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError("run_id must be a single directory name, not a path.")
    return normalized


def create_run_output_dir(output_root: Path, *, run_id: str | None = None) -> tuple[str, Path]:
    effective_run_id = resolve_run_id(run_id)
    run_output_dir = output_root / effective_run_id
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return effective_run_id, run_output_dir


def build_model_adapter(config: AppConfig):
    return OpenAIModelAdapter(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        temperature=config.agent.temperature,
        timeout_seconds=config.agent.request_timeout_seconds,
        max_retries=config.agent.request_max_retries,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    _write_json(temporary_path, payload)
    temporary_path.replace(path)


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)


def _failure_run_result_payload(task_id: str, failure_reason: str) -> dict[str, Any]:
    return {"task_id": task_id, "answer": None, "steps": [], "failure_reason": failure_reason, "succeeded": False}


def _run_single_task_core(*, task_id: str, config: AppConfig, checkpoint_path: Path | None = None, model=None, tools: ToolRegistry | None = None) -> dict[str, Any]:
    public_dataset = DABenchPublicDataset(config.dataset.root_path)
    task = public_dataset.get_task(task_id)

    def checkpoint_callback(payload: dict[str, Any]) -> None:
        if checkpoint_path is not None:
            _write_json_atomic(checkpoint_path, payload)

    agent = ReActAgent(
        model=model or build_model_adapter(config),
        tools=tools or create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=config.agent.max_steps, model_retry_limit=config.agent.model_retry_limit),
        checkpoint_callback=checkpoint_callback if checkpoint_path is not None else None,
    )
    return agent.run(task).to_dict()


def _run_single_task_in_subprocess(task_id: str, config: AppConfig, result_path: str, checkpoint_path: str) -> None:
    try:
        result = _run_single_task_core(task_id=task_id, config=config, checkpoint_path=Path(checkpoint_path))
        _write_json_atomic(Path(result_path), result)
    except BaseException as exc:  # noqa: BLE001
        _write_json_atomic(Path(result_path), _failure_run_result_payload(task_id, f"Task failed with uncaught error: {exc}"))


def _read_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _run_single_task_with_timeout(*, task_id: str, config: AppConfig) -> dict[str, Any]:
    timeout_seconds = config.run.task_timeout_seconds
    if timeout_seconds <= 0:
        return _run_single_task_core(task_id=task_id, config=config)

    # Use atomic files instead of Queue+join. A large trace can fill the Queue
    # pipe while the parent waits for child exit, causing the old 600s deadlock.
    with tempfile.TemporaryDirectory(prefix="dabench-task-") as temp_dir:
        temp_root = Path(temp_dir)
        result_path = temp_root / "result.json"
        checkpoint_path = temp_root / "checkpoint.json"
        process = multiprocessing.Process(target=_run_single_task_in_subprocess, args=(task_id, config, result_path.as_posix(), checkpoint_path.as_posix()))
        process.start()
        deadline = perf_counter() + timeout_seconds
        while process.is_alive():
            remaining = deadline - perf_counter()
            if remaining <= 0:
                break
            process.join(timeout=min(0.2, remaining))

        timed_out = process.is_alive()
        if timed_out:
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join()

        result = _read_result(result_path)
        checkpoint = _read_result(checkpoint_path)
        if timed_out:
            timeout_reason = f"Task timed out after {timeout_seconds} seconds."
            if checkpoint is not None:
                checkpoint["failure_reason"] = timeout_reason
                checkpoint["succeeded"] = False
                return checkpoint
            return _failure_run_result_payload(task_id, timeout_reason)
        if result is not None:
            return result
        return _failure_run_result_payload(task_id, f"Task exited without returning a result (exit code {process.exitcode}).")


def _write_task_outputs(task_id: str, run_output_dir: Path, run_result: dict[str, Any]) -> TaskRunArtifacts:
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = task_output_dir / "trace.json"
    _write_json(trace_path, run_result)
    prediction_csv_path: Path | None = None
    answer = run_result.get("answer")
    if isinstance(answer, dict):
        prediction_csv_path = task_output_dir / "prediction.csv"
        _write_csv(prediction_csv_path, list(answer.get("columns", [])), [list(row) for row in answer.get("rows", [])])
    return TaskRunArtifacts(task_id=task_id, task_output_dir=task_output_dir, prediction_csv_path=prediction_csv_path, trace_path=trace_path, succeeded=bool(run_result.get("succeeded")), failure_reason=run_result.get("failure_reason"))


def run_single_task(*, task_id: str, config: AppConfig, run_output_dir: Path, model=None, tools: ToolRegistry | None = None) -> TaskRunArtifacts:
    started_at = perf_counter()
    if model is None and tools is None:
        run_result = _run_single_task_with_timeout(task_id=task_id, config=config)
    else:
        run_result = _run_single_task_core(task_id=task_id, config=config, model=model, tools=tools)
    run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
    return _write_task_outputs(task_id, run_output_dir, run_result)


def run_benchmark(*, config: AppConfig, model=None, tools: ToolRegistry | None = None, limit: int | None = None, progress_callback: Callable[[TaskRunArtifacts], None] | None = None) -> tuple[Path, list[TaskRunArtifacts]]:
    effective_run_id, run_output_dir = create_run_output_dir(config.run.output_dir, run_id=config.run.run_id)
    dataset = DABenchPublicDataset(config.dataset.root_path)
    tasks = dataset.iter_tasks()
    if limit is not None:
        tasks = tasks[:limit]
    effective_workers = config.run.max_workers
    if effective_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    if model is not None or tools is not None:
        effective_workers = 1
    task_ids = [task.task_id for task in tasks]
    if effective_workers == 1:
        shared_model = model or build_model_adapter(config)
        shared_tools = tools or create_default_tool_registry()
        task_artifacts = []
        for task_id in task_ids:
            artifact = run_single_task(task_id=task_id, config=config, run_output_dir=run_output_dir, model=shared_model, tools=shared_tools)
            task_artifacts.append(artifact)
            if progress_callback is not None:
                progress_callback(artifact)
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_index = {executor.submit(run_single_task, task_id=task_id, config=config, run_output_dir=run_output_dir): index for index, task_id in enumerate(task_ids)}
            indexed_artifacts: list[TaskRunArtifacts | None] = [None] * len(task_ids)
            for future in as_completed(future_to_index):
                artifact = future.result()
                indexed_artifacts[future_to_index[future]] = artifact
                if progress_callback is not None:
                    progress_callback(artifact)
            task_artifacts = [artifact for artifact in indexed_artifacts if artifact is not None]
    _write_json(run_output_dir / "summary.json", {"run_id": effective_run_id, "task_count": len(task_artifacts), "succeeded_task_count": sum(1 for artifact in task_artifacts if artifact.succeeded), "max_workers": effective_workers, "tasks": [artifact.to_dict() for artifact in task_artifacts]})
    return run_output_dir, task_artifacts
