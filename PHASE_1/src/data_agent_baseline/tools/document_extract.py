from __future__ import annotations

import re
from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.filesystem import resolve_context_path


_ID_RE = re.compile(r"(?:identifier|registration number|reference (?:code|ID)|registry number|ID)\s+(\d+)", re.IGNORECASE)
_NAME_RE = re.compile(r"(?:known as|designated|codename|operative)\s+['\"]?([^,.'\"]+)", re.IGNORECASE)
_FULL_NAME_RE = re.compile(r"full (?:legal )?name[^.]{0,100}?(?:is|to be|as)\s+([^.;]+)", re.IGNORECASE)


def extract_document_records(task: PublicTask, relative_path: str, *, max_records: int = 500) -> dict[str, Any]:
    path = resolve_context_path(task, relative_path)
    text = path.read_text(errors="replace")
    rows: list[list[Any]] = []
    for index, paragraph in enumerate((part.strip() for part in text.split("\n\n")), start=1):
        if not paragraph:
            continue
        identifier = _ID_RE.search(paragraph)
        if identifier is None:
            continue
        name_match = _NAME_RE.search(paragraph)
        full_name_match = _FULL_NAME_RE.search(paragraph)
        rows.append(
            [
                index,
                name_match.group(1).strip() if name_match else "",
                int(identifier.group(1)),
                full_name_match.group(1).strip() if full_name_match else "",
                paragraph[:500],
            ]
        )
        if len(rows) >= max_records:
            break
    return {
        "path": relative_path,
        "columns": ["record_index", "name", "identifier", "full_name", "evidence"],
        "rows": rows,
        "row_count": len(rows),
        "extraction": "paragraph-level regex with evidence retained",
    }

