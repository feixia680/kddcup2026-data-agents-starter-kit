from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask


@dataclass(frozen=True, slots=True)
class VerificationReport:
    warnings: list[str]
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "warnings": self.warnings, "errors": self.errors}


def verify_answer_shape(task: PublicTask, columns: list[str], rows: list[list[Any]]) -> VerificationReport:
    warnings: list[str] = []
    errors: list[str] = []
    if len(columns) != len(set(columns)):
        errors.append("answer contains duplicate column names")
    if any(len(row) != len(columns) for row in rows):
        errors.append("answer row width does not match columns")
    if len(rows) > 1 and len({tuple(map(str, row)) for row in rows}) < len(rows):
        warnings.append("answer contains duplicate rows")
    q = task.question.lower()
    if any(token in q for token in ("percentage", "percent")) and len(columns) > 1:
        warnings.append("percentage question should normally return one numeric column")
    if "comment" in q and not any(name.lower() in {"text", "comment"} for name in columns):
        warnings.append("comment question has no obvious Text/Comment output column")
    if "state the post id" in q and len(columns) != 1:
        warnings.append("post-id question should return only the post identifier")
    return VerificationReport(warnings=warnings, errors=errors)


def verify_tool_result(task: PublicTask, action: str, action_input: dict[str, Any], content: dict[str, Any]) -> VerificationReport:
    warnings: list[str] = []
    errors: list[str] = []
    q = task.question.lower()
    sql = str(action_input.get("sql", "")).lower()
    if action == "execute_context_sql":
        if "rank" in q and "position" in sql and "rank" not in sql:
            warnings.append("SQL uses position for a rank question; verify the semantic rank column")
        if ("per unit" in q or "unit price" in q) and "/" not in sql:
            warnings.append("SQL does not visibly compute a per-row ratio")
        if ("lowest" in q or "highest" in q) and "group by" in sql:
            warnings.append("row-level min/max question may have been grouped before comparison")
        if ("percentage" in q or "percent" in q) and "count" in sql and "/" not in sql:
            warnings.append("percentage query has no visible denominator")
        if ("average" in q or "mean" in q) and "sum(" in sql and "avg(" not in sql:
            warnings.append("average question uses SUM without AVG; verify the requested aggregation")
        if ("total" in q or "sum" in q) and "avg(" in sql and "sum(" not in sql:
            warnings.append("total question uses AVG without SUM; verify the requested aggregation")
        if any(token in q for token in ("vote", "votes", "voted")) and "post" in sql and "vote" not in sql:
            warnings.append("vote question appears to count posts; verify the vote entity and table")
    if action == "execute_python":
        code = str(action_input.get("code", ""))
        if "1k" in code.lower() and not any(token in q for token in ("sample", "1k")):
            warnings.append("Python code reads a sample/1k file; verify full-dataset coverage")
        if "position" in code.lower() and "rank" in q and "rank" not in code.lower():
            warnings.append("Python code uses position for a rank question")
        if ("per unit" in q or "unit price" in q or ("price" in q and "amount" in q)) and "/" not in code:
            warnings.append("Python code does not visibly compute the per-row ratio (price per amount)")
        if ("average" in q or "mean" in q) and "sum(" in code and not any(token in code for token in ("mean(", "average(", ".avg(")):
            warnings.append("average question uses SUM without a visible mean/average operation")
        if ("average" in q or "mean" in q) and re.search(r"(?:total|aggregate)[^\n]{0,80}/\s*12", code, flags=re.IGNORECASE):
            warnings.append("average question divides a total by a fixed 12 without computing the requested mean")
        if any(token in q for token in ("vote", "votes", "voted")) and "post" in code and "vote" not in code:
            warnings.append("vote question appears to count posts; verify the vote entity and table")
    if content.get("success") is False or content.get("error"):
        errors.append("tool returned an execution error")
    return VerificationReport(warnings=warnings, errors=errors)
