from __future__ import annotations

import re
from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.filesystem import resolve_context_path


_ID_RE = re.compile(r"(?:identifier|registration number|reference (?:code|ID)|registry number|ID)\s+(\d+)", re.IGNORECASE)
_PATIENT_RE = re.compile(r"\bpatient\s+(\d+)\b", re.IGNORECASE)
_NAME_RE = re.compile(r"(?:known as|designated|codename|operative)\s+['\"]?([^,.'\"]+)", re.IGNORECASE)
_FULL_NAME_RE = re.compile(r"full (?:legal )?name[^.]{0,100}?(?:is|to be|as)\s+([^.;]+)", re.IGNORECASE)
_HEIGHT_RE = re.compile(r"height[^.]{0,160}?(\d+(?:\.\d+)?)\s*centimeters", re.IGNORECASE)
_CREATININE_RE = re.compile(r"creatinine[^.]{0,180}?(\d+(?:\.\d+)?)\s*mg/dL", re.IGNORECASE)
_CREATININE_VALUE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mg/dL", re.IGNORECASE)


def extract_document_records(task: PublicTask, relative_path: str, *, max_records: int = 500) -> dict[str, Any]:
    path = resolve_context_path(task, relative_path)
    text = path.read_text(errors="replace")
    has_patient = bool(_PATIENT_RE.search(text))
    has_height = bool(_HEIGHT_RE.search(text))
    has_creatinine = bool(_CREATININE_RE.search(text))
    rows: list[list[Any]] = []
    for index, paragraph in enumerate((part.strip() for part in text.split("\n\n")), start=1):
        if not paragraph:
            continue
        identifier = _ID_RE.search(paragraph) or _PATIENT_RE.search(paragraph)
        if identifier is None:
            continue
        name_match = _NAME_RE.search(paragraph)
        full_name_match = _FULL_NAME_RE.search(paragraph)
        patient_match = _PATIENT_RE.search(paragraph)
        height_values = _HEIGHT_RE.findall(paragraph)
        creatinine_values = _CREATININE_VALUE_RE.findall(paragraph) if _CREATININE_RE.search(paragraph) else []
        row: list[Any] = [
            index,
            name_match.group(1).strip() if name_match else "",
            int(identifier.group(1)),
            full_name_match.group(1).strip() if full_name_match else "",
        ]
        if has_patient:
            row.append(int(patient_match.group(1)) if patient_match else None)
        if has_height:
            row.append(float(height_values[-1]) if height_values else None)
        if has_creatinine:
            row.append(float(creatinine_values[-1]) if creatinine_values else None)
        row.append(paragraph[:500])
        rows.append(row)
        if len(rows) >= max_records:
            break

    columns = ["record_index", "name", "identifier", "full_name"]
    if has_patient:
        columns.append("patient_id")
    if has_height:
        columns.append("height_cm")
    if has_creatinine:
        columns.append("creatinine_mg_dl")
    columns.append("evidence")
    return {
        "path": relative_path,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "extraction": "paragraph-level regex with task-relevant numeric fields and evidence retained",
    }
