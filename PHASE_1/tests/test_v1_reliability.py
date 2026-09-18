from pathlib import Path

import data_agent_baseline.agents.model as model_module
from data_agent_baseline.agents.answer_guard import normalize_answer_for_question
from data_agent_baseline.agents.model import OpenAIModelAdapter, ScriptedModelAdapter
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig, parse_model_step
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.registry import ToolExecutionResult, ToolRegistry, ToolSpec


def make_task(question: str) -> PublicTask:
    root = Path(".").resolve()
    return PublicTask(
        record=TaskRecord(task_id="test", difficulty="easy", question=question),
        assets=TaskAssets(task_dir=root, context_dir=root),
    )


def test_answer_guard_normalizes_known_baseline_presentation_failures():
    task = make_task("What percentage is the answer?")
    answer, warnings = normalize_answer_for_question(task, ["value"], [["52.17%"]])
    assert answer == AnswerTable(columns=["value"], rows=[["52.17"]])
    assert warnings

    task = make_task("State the final score")
    answer, _ = normalize_answer_for_question(task, ["score"], [["1-1"]])
    assert answer == AnswerTable(
        columns=["home_team_goal", "away_team_goal"], rows=[["1", "1"]]
    )

    task = make_task("Which comment was posted?")
    answer, _ = normalize_answer_for_question(
        task, ["PostId", "Text"], [[12, "useful comment"]]
    )
    assert answer == AnswerTable(columns=["Text"], rows=[["useful comment"]])


def test_parse_model_step_rejects_trailing_non_json_text():
    parsed = parse_model_step('{"thought":"done","action":"answer","action_input":{}}')
    assert parsed.action == "answer"
    try:
        parse_model_step('{"thought":"done","action":"answer","action_input":{}} trailing')
    except ValueError as exc:
        assert "only one JSON object" in str(exc)
    else:
        raise AssertionError("trailing text must be rejected")


def test_react_checkpoint_and_terminal_answer():
    task = make_task("Return the value")

    def answer_handler(_task, action_input):
        return ToolExecutionResult(
            ok=True,
            content={"status": "submitted"},
            is_terminal=True,
            answer=AnswerTable(columns=["value"], rows=[[42]]),
        )

    registry = ToolRegistry(
        specs={"answer": ToolSpec("answer", "submit", {"columns": [], "rows": []})},
        handlers={"answer": answer_handler},
    )
    checkpoints = []
    agent = ReActAgent(
        model=ScriptedModelAdapter(
            ['{"thought":"submit","action":"answer","action_input":{"columns":["value"],"rows":[[42]]}}']
        ),
        tools=registry,
        config=ReActAgentConfig(max_steps=3),
        checkpoint_callback=checkpoints.append,
    )
    result = agent.run(task)
    assert result.succeeded
    assert result.answer == AnswerTable(columns=["value"], rows=[[42]])
    assert len(checkpoints) >= 2
    assert checkpoints[-1]["succeeded"] is True


def test_react_retries_model_error_and_records_recoverable_step():
    task = make_task("Return the value")

    class FlakyModel:
        def __init__(self):
            self.calls = 0

        def complete(self, _messages):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary request timeout")
            return '{"thought":"submit","action":"answer","action_input":{"columns":["value"],"rows":[[42]]}}'

    def answer_handler(_task, _input):
        return ToolExecutionResult(True, {"status": "submitted"}, True, AnswerTable(["value"], [[42]]))

    registry = ToolRegistry(
        specs={"answer": ToolSpec("answer", "submit", {})},
        handlers={"answer": answer_handler},
    )
    result = ReActAgent(model=FlakyModel(), tools=registry, config=ReActAgentConfig(max_steps=3, model_retry_limit=1)).run(task)
    assert result.succeeded
    assert result.steps[0].action == "__model_error__"
    assert result.steps[0].observation["retryable"] is True


def test_react_enforces_answer_only_budget():
    task = make_task("Return the value")
    calls = []

    def list_handler(_task, _input):
        calls.append("list_context")
        return ToolExecutionResult(True, {"files": []})

    def answer_handler(_task, _input):
        calls.append("answer")
        return ToolExecutionResult(True, {"status": "submitted"}, True, AnswerTable(["value"], [[42]]))

    registry = ToolRegistry(
        specs={
            "answer": ToolSpec("answer", "submit", {}),
            "list_context": ToolSpec("list_context", "inspect", {}),
        },
        handlers={"answer": answer_handler, "list_context": list_handler},
    )
    result = ReActAgent(
        model=ScriptedModelAdapter([
            '{"thought":"I should inspect first","action":"list_context","action_input":{}}',
            '{"thought":"submit","action":"answer","action_input":{"columns":["value"],"rows":[[42]]}}',
        ]),
        tools=registry,
        config=ReActAgentConfig(max_steps=2, force_answer_remaining=2),
    ).run(task)
    assert result.succeeded
    assert calls == ["answer"]
    assert result.steps[0].action == "__answer_required__"



def test_openai_adapter_disables_sdk_retries_and_uses_hard_request_timeout(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["request"] = kwargs
            return type("Response", (), {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]})()

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(model_module, "OpenAI", FakeClient)
    adapter = OpenAIModelAdapter(
        model="test-model",
        api_base="https://example.test/v1",
        api_key="test-key",
        temperature=0.0,
        timeout_seconds=12.5,
        max_retries=0,
    )
    assert adapter.complete([]) == "ok"
    assert captured["client"]["timeout"] == 12.5
    assert captured["client"]["max_retries"] == 0



def test_react_persists_last_successful_evidence_for_final_answer():
    task = make_task("Return the name")
    checkpoints = []
    captured_messages = []

    def read_handler(_task, _input):
        return ToolExecutionResult(True, {"columns": ["name"], "rows": [["Ada"]]})

    def answer_handler(_task, _input):
        return ToolExecutionResult(True, {"status": "submitted"}, True, AnswerTable(["name"], [["Ada"]]))

    class EvidenceAwareModel:
        def __init__(self):
            self.calls = 0

        def complete(self, messages):
            self.calls += 1
            captured_messages.append(messages)
            if self.calls == 1:
                return '{"thought":"inspect","action":"read_csv","action_input":{"path":"people.csv"}}'
            assert "CURRENT EVIDENCE MEMORY" in messages[-1].content
            assert '"source_action": "read_csv"' in messages[-1].content
            return '{"thought":"submit","action":"answer","action_input":{"columns":["name"],"rows":[["Ada"]]}}'

    registry = ToolRegistry(
        specs={
            "answer": ToolSpec("answer", "submit", {}),
            "read_csv": ToolSpec("read_csv", "read", {}),
        },
        handlers={"answer": answer_handler, "read_csv": read_handler},
    )
    result = ReActAgent(
        model=EvidenceAwareModel(),
        tools=registry,
        config=ReActAgentConfig(max_steps=3),
        checkpoint_callback=checkpoints.append,
    ).run(task)
    assert result.succeeded
    assert result.evidence_memory["source_action"] == "read_csv"
    assert any(checkpoint.get("evidence_memory", {}).get("source_action") == "read_csv" for checkpoint in checkpoints)
