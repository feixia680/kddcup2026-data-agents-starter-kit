from __future__ import annotations

import json

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.agents.semantic_plan import infer_semantic_plan

REACT_SYSTEM_PROMPT = """
You are a reliable ReAct-style data agent for benchmarked data analysis.

You are solving a task from a public dataset. You may only inspect files inside the task's `context/` directory through the provided tools.

Rules:
1. Inspect the available context before answering and use only observed evidence.
2. Before executing, identify the target entity, requested output fields, filters, joins, aggregation grain, denominator, units, and tie behavior.
3. Prefer the narrowest tool that can answer the question. Do not repeatedly query after the candidate answer is stable.
4. The task is complete only when you call `answer` with the smallest table that answers the question.
5. Do not add explanation columns, IDs, totals, percentage signs, currency signs, or extra rows unless requested.
6. Preserve numeric precision; do not round early. Return numeric values as numeric-looking cells.
7. Distinguish row-level min/max from grouped sum/min/max, rank from position, per-unit values from totals, and counting entities from counting attributes.
8. Always return exactly one JSON object with keys `thought`, `action`, and `action_input`, wrapped in one ```json fenced block.
9. If the remaining budget is small, stop exploration and submit the best evidence-supported answer.

The evaluator ignores column names and row order, but extra or incorrectly shaped columns reduce the score.
""".strip()

RESPONSE_EXAMPLES = """
Example response when you need to inspect the context:
```json
{"thought":"I should inspect the available files first.","action":"list_context","action_input":{"max_depth":4}}
```

Example response when you have the final answer:
```json
{"thought":"I have the minimal final result table.","action":"answer","action_input":{"columns":["average_long_shots"],"rows":[["63.5"]]}}
```
""".strip()


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    return f"{base_prompt}\n\nAvailable tools:\n{tool_descriptions}\n\n{RESPONSE_EXAMPLES}\n\nReturn only one fenced JSON object."


def build_task_prompt(task: PublicTask, remaining_steps: int | None = None) -> str:
    urgency = ""
    if remaining_steps is not None and remaining_steps <= 2:
        urgency = " FINAL-ANSWER MODE: do not call exploratory tools; call answer now using the best verified evidence."
    plan = infer_semantic_plan(task.question)
    return (
        f"Question: {task.question}\n"
        f"Semantic plan (treat as checks, revise only when evidence contradicts it):\n{plan.render()}\n"
        "All tool file paths are relative to the task context directory. "
        f"When you have the final table, call `answer`.{urgency}"
    )


def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}"
