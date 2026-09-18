from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from data_agent_baseline.agents.answer_guard import normalize_answer_for_question
from data_agent_baseline.agents.result_verifier import verify_answer_shape
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.document_extract import extract_document_records
from data_agent_baseline.tools.filesystem import list_context_tree, read_csv_preview, read_doc_preview, read_json_preview, resolve_context_path
from data_agent_baseline.tools.python_exec import execute_python_code
from data_agent_baseline.tools.sqlite import execute_read_only_sql, inspect_sqlite_schema

EXECUTE_PYTHON_TIMEOUT_SECONDS = 30


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    ok: bool
    content: dict[str, Any]
    is_terminal: bool = False
    answer: AnswerTable | None = None


ToolHandler = Callable[[PublicTask, dict[str, Any]], ToolExecutionResult]


def _list_context(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    return ToolExecutionResult(ok=True, content=list_context_tree(task, max_depth=int(action_input.get("max_depth", 4))))


def _read_csv(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    return ToolExecutionResult(ok=True, content=read_csv_preview(task, str(action_input["path"]), max_rows=int(action_input.get("max_rows", 20))))


def _read_json(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    return ToolExecutionResult(ok=True, content=read_json_preview(task, str(action_input["path"]), max_chars=int(action_input.get("max_chars", 4000))))


def _read_doc(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    return ToolExecutionResult(ok=True, content=read_doc_preview(task, str(action_input["path"]), max_chars=int(action_input.get("max_chars", 4000))))


def _extract_document(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    return ToolExecutionResult(ok=True, content=extract_document_records(task, str(action_input["path"])))


def _inspect_sqlite_schema(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    return ToolExecutionResult(ok=True, content=inspect_sqlite_schema(resolve_context_path(task, str(action_input["path"]))))


def _execute_context_sql(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    return ToolExecutionResult(ok=True, content=execute_read_only_sql(resolve_context_path(task, str(action_input["path"])), str(action_input["sql"]), limit=int(action_input.get("limit", 200))))


def _execute_python(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    content = execute_python_code(task.context_dir, str(action_input["code"]), timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS)
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)


def _answer(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    columns = action_input.get("columns")
    rows = action_input.get("rows")
    if not isinstance(columns, list) or not columns or not all(isinstance(item, str) for item in columns):
        raise ValueError("answer.columns must be a non-empty list of strings.")
    if not isinstance(rows, list):
        raise ValueError("answer.rows must be a list.")
    raw_rows: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(columns):
            raise ValueError("Each answer row must match the number of columns.")
        raw_rows.append(list(row))
    answer, warnings = normalize_answer_for_question(task, list(columns), raw_rows)
    verification = verify_answer_shape(task, answer.columns, answer.rows)
    if verification.errors:
        raise ValueError("; ".join(verification.errors))
    warnings.extend(verification.warnings)
    content: dict[str, Any] = {"status": "submitted", "column_count": len(answer.columns), "row_count": len(answer.rows)}
    if warnings:
        content["answer_guard"] = warnings
    return ToolExecutionResult(ok=True, content=content, is_terminal=True, answer=answer)


@dataclass(slots=True)
class ToolRegistry:
    specs: dict[str, ToolSpec]
    handlers: dict[str, ToolHandler]

    def describe_for_prompt(self) -> str:
        lines = []
        for name in sorted(self.specs):
            spec = self.specs[name]
            lines.append(f"- {spec.name}: {spec.description}")
            lines.append(f"  input_schema: {spec.input_schema}")
        return "\n".join(lines)

    def execute(self, task: PublicTask, action: str, action_input: dict[str, Any]) -> ToolExecutionResult:
        if action not in self.handlers:
            raise KeyError(f"Unknown tool: {action}")
        return self.handlers[action](task, action_input)


def create_default_tool_registry() -> ToolRegistry:
    specs = {
        "answer": ToolSpec("answer", "Submit the minimal final answer table.", {"columns": ["column_name"], "rows": [["value_1"]]}),
        "execute_context_sql": ToolSpec("execute_context_sql", "Run read-only SQL inside context.", {"path": "relative/path.sqlite", "sql": "SELECT ...", "limit": 200}),
        "execute_python": ToolSpec("execute_python", f"Execute Python in the task context. Timeout={EXECUTE_PYTHON_TIMEOUT_SECONDS}s.", {"code": "import os\nprint(sorted(os.listdir('.')))"}),
        "inspect_sqlite_schema": ToolSpec("inspect_sqlite_schema", "Inspect a SQLite schema inside context.", {"path": "relative/path.sqlite"}),
        "list_context": ToolSpec("list_context", "List files under context.", {"max_depth": 4}),
        "read_csv": ToolSpec("read_csv", "Read a CSV preview.", {"path": "relative/path.csv", "max_rows": 20}),
        "read_doc": ToolSpec("read_doc", "Read a text document preview.", {"path": "relative/path.md", "max_chars": 4000}),
        "extract_document_records": ToolSpec("extract_document_records", "Extract structured records from a document.", {"path": "relative/path.md"}),
        "read_json": ToolSpec("read_json", "Read a JSON preview.", {"path": "relative/path.json", "max_chars": 4000}),
    }
    handlers = {"answer": _answer, "execute_context_sql": _execute_context_sql, "execute_python": _execute_python, "inspect_sqlite_schema": _inspect_sqlite_schema, "list_context": _list_context, "read_csv": _read_csv, "read_doc": _read_doc, "extract_document_records": _extract_document, "read_json": _read_json}
    return ToolRegistry(specs=specs, handlers=handlers)
