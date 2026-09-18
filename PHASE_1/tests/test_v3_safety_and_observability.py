from pathlib import Path

from data_agent_baseline.agents.model import ScriptedModelAdapter
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.document_extract import extract_document_records
from data_agent_baseline.tools.python_exec import execute_python_code
from data_agent_baseline.tools.python_policy import PythonPolicyError, validate_python_code
from data_agent_baseline.tools.registry import ToolExecutionResult, ToolRegistry, ToolSpec


def make_task(root: Path) -> PublicTask:
    return PublicTask(
        record=TaskRecord(task_id="v3-test", difficulty="easy", question="Return the value"),
        assets=TaskAssets(task_dir=root, context_dir=root),
    )


def test_python_policy_blocks_process_network_and_absolute_path_access(tmp_path):
    for code in ("import subprocess", "import socket", "os.system('find /')", "open('/etc/passwd')"):
        try:
            validate_python_code(code)
        except PythonPolicyError:
            pass
        else:
            raise AssertionError(f"policy did not reject: {code}")

    result = execute_python_code(tmp_path, "open('/etc/passwd').read()")
    assert result["success"] is False
    assert "policy" in result["error"].lower()


def test_document_extractor_retains_structured_evidence(tmp_path):
    doc = tmp_path / "records.md"
    doc.write_text("An entry is known as Alpha, filed under the unique registration number 7. The full name is Alice.")
    result = extract_document_records(make_task(tmp_path), "records.md")
    assert result["row_count"] == 1
    assert result["rows"][0][1:4] == ["Alpha", 7, "Alice"]
    assert result["rows"][0][4]


def test_react_trace_contains_latency_telemetry(tmp_path):
    task = make_task(tmp_path)

    def answer_handler(_task, _input):
        return ToolExecutionResult(True, {"status": "submitted"}, True, AnswerTable(["value"], [[1]]))

    registry = ToolRegistry(
        specs={"answer": ToolSpec("answer", "submit", {})},
        handlers={"answer": answer_handler},
    )
    result = ReActAgent(
        model=ScriptedModelAdapter(['{"thought":"done","action":"answer","action_input":{"columns":["value"],"rows":[[1]]}}']),
        tools=registry,
        config=ReActAgentConfig(max_steps=2),
    ).run(task)
    telemetry = result.steps[0].observation["telemetry"]
    assert telemetry["model_seconds"] >= 0
    assert telemetry["tool_seconds"] >= 0
    assert telemetry["step_seconds"] >= 0




def test_critical_verification_warning_blocks_answer_until_repair():
    from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
    from data_agent_baseline.benchmark.schema import AnswerTable
    from data_agent_baseline.tools.registry import ToolExecutionResult, ToolRegistry, ToolSpec

    task = PublicTask(
        record=TaskRecord(task_id="avg", difficulty="easy", question="What is the average expense?"),
        assets=TaskAssets(task_dir=Path(".").resolve(), context_dir=Path(".").resolve()),
    )
    calls = []

    def execute_handler(_task, action_input):
        calls.append(action_input["code"])
        if len(calls) == 1:
            return ToolExecutionResult(True, {"success": True, "output": "SUM without AVG"})
        return ToolExecutionResult(True, {"success": True, "output": "mean = 4.0"})

    def answer_handler(_task, _input):
        calls.append("answer")
        return ToolExecutionResult(True, {"status": "submitted"}, True, AnswerTable(["value"], [[4.0]]))

    class RepairModel:
        def __init__(self):
            self.calls = 0

        def complete(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return '{"thought":"compute","action":"execute_python","action_input":{"code":"total = df[\'amount\'].sum()"}}'
            if self.calls == 2:
                return '{"thought":"submit","action":"answer","action_input":{"columns":["value"],"rows":[[82027220]]}}'
            if self.calls == 3:
                return '{"thought":"repair with mean","action":"execute_python","action_input":{"code":"mean = df[\'amount\'].mean()"}}'
            return '{"thought":"submit corrected","action":"answer","action_input":{"columns":["value"],"rows":[[4.0]]}}'

    registry = ToolRegistry(
        specs={
            "answer": ToolSpec("answer", "submit", {}),
            "execute_python": ToolSpec("execute_python", "compute", {}),
        },
        handlers={"answer": answer_handler, "execute_python": execute_handler},
    )
    result = ReActAgent(
        model=RepairModel(),
        tools=registry,
        config=ReActAgentConfig(max_steps=6),
    ).run(task)
    assert result.succeeded
    assert result.answer == AnswerTable(["value"], [[4.0]])
    assert "__verification_required__" in [step.action for step in result.steps]
    assert calls[-1] == "answer"
