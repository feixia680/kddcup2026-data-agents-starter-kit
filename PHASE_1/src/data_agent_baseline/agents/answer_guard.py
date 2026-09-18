from __future__ import annotations

import re
from typing import Any

from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask

_NUMBER_WITH_PERCENT = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*%\s*$")
_FINAL_SCORE = re.compile(r"^\s*(-?\d+)\s*[-:]\s*(-?\d+)\s*$")


def _select_column(columns: list[str], candidates: tuple[str, ...]) -> int | None:
    lowered = {column.lower(): index for index, column in enumerate(columns)}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def normalize_answer_for_question(task: PublicTask, columns: list[str], rows: list[list[Any]]) -> tuple[AnswerTable, list[str]]:
    """Conservative fixes for unambiguous presentation errors seen in traces."""
    question = task.question.lower()
    normalized_columns = list(columns)
    normalized_rows = [list(row) for row in rows]
    warnings: list[str] = []

    if ("percentage" in question or "percent" in question) and len(normalized_columns) == 1:
        changed = False
        for row in normalized_rows:
            if row and isinstance(row[0], str):
                match = _NUMBER_WITH_PERCENT.match(row[0])
                if match:
                    row[0] = match.group(1)
                    changed = True
        if changed:
            warnings.append("normalized percentage presentation")

    if "final score" in question and len(normalized_columns) == 1:
        split_rows: list[list[Any]] = []
        for row in normalized_rows:
            if len(row) != 1 or not isinstance(row[0], str):
                break
            match = _FINAL_SCORE.match(row[0])
            if not match:
                break
            split_rows.append([match.group(1), match.group(2)])
        if len(split_rows) == len(normalized_rows) and split_rows:
            normalized_columns = ["home_team_goal", "away_team_goal"]
            normalized_rows = split_rows
            warnings.append("split final score into home/away columns")

    if "comment" in question and len(normalized_columns) > 1:
        index = _select_column(normalized_columns, ("Text", "Comment"))
        if index is not None:
            normalized_rows = [[row[index]] for row in normalized_rows if index < len(row)]
            normalized_columns = [normalized_columns[index]]
            warnings.append("selected comment text")

    if "withdrawal" in question and len(normalized_columns) > 1:
        index = _select_column(normalized_columns, ("trans_id", "transaction_id", "id"))
        if index is not None:
            normalized_rows = [[row[index]] for row in normalized_rows if index < len(row)]
            normalized_columns = [normalized_columns[index]]
            warnings.append("selected transaction identifiers")

    if "state the post id" in question and len(normalized_columns) > 1:
        index = _select_column(normalized_columns, ("PostId", "post_id", "Id"))
        if index is not None:
            normalized_rows = [[row[index]] for row in normalized_rows if index < len(row)]
            normalized_columns = [normalized_columns[index]]
            warnings.append("selected post identifier")

    return AnswerTable(columns=normalized_columns, rows=normalized_rows), warnings
