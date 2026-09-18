from __future__ import annotations

import ast
import builtins
import re
from pathlib import Path
from typing import Any


class PythonPolicyError(ValueError):
    """Raised when agent Python attempts an operation outside the task sandbox."""


_SHELL_ESCAPE_RE = re.compile(r"(?:find\s+/|curl\s+|wget\s+|nc\s+|/etc/|/proc/|/sys/)", re.IGNORECASE)
_BLOCKED_MODULES = {"subprocess", "socket", "requests", "httpx", "urllib.request"}
_BLOCKED_CALLS = {"system", "popen", "walk", "rmtree", "remove", "unlink"}


def validate_python_code(code: str) -> None:
    if _SHELL_ESCAPE_RE.search(code):
        raise PythonPolicyError("Python policy rejected shell/network/system path access.")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise PythonPolicyError(f"Python syntax error: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = {alias.name for alias in node.names}
            if names & _BLOCKED_MODULES:
                raise PythonPolicyError("Python policy rejected a network/process module.")
        elif isinstance(node, ast.ImportFrom) and node.module in _BLOCKED_MODULES:
            raise PythonPolicyError("Python policy rejected a network/process module.")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in _BLOCKED_CALLS:
                raise PythonPolicyError(f"Python policy rejected os/shutil call: {node.func.attr}.")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                if Path(node.args[0].value).is_absolute():
                    raise PythonPolicyError("Python policy rejected an absolute open path.")


def make_restricted_builtins(context_root: Path) -> dict[str, Any]:
    builtins_map = dict(vars(builtins))
    original_open = builtins.open
    resolved_root = context_root.resolve()

    def safe_open(file: Any, *args: Any, **kwargs: Any):
        candidate = Path(file)
        resolved = (resolved_root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        if resolved != resolved_root and resolved_root not in resolved.parents:
            raise PythonPolicyError(f"open path escapes task context: {file}")
        return original_open(resolved, *args, **kwargs)

    builtins_map["open"] = safe_open
    return builtins_map

