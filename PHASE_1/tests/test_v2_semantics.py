from pathlib import Path

from data_agent_baseline.agents.result_verifier import verify_answer_shape, verify_tool_result
from data_agent_baseline.agents.semantic_plan import infer_semantic_plan
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord


def task(question: str) -> PublicTask:
    root = Path(".").resolve()
    return PublicTask(
        record=TaskRecord(task_id="v2-test", difficulty="easy", question=question),
        assets=TaskAssets(task_dir=root, context_dir=root),
    )


def test_semantic_plan_catches_rank_and_per_unit_ambiguity():
    plan = infer_semantic_plan("Which driver has rank 2 and what is the price per unit?")
    assert plan.target_grain == "row-level"
    assert any("semantic rank" in check for check in plan.checks)
    assert any("per-row ratio" in check for check in plan.checks)


def test_semantic_plan_warns_about_sample_coverage_and_percentage_denominator():
    plan = infer_semantic_plan("What percentage of all records is in the sample_1k file?")
    assert "sample" in plan.source_scope
    assert "denominator" in plan.checks[0] or any("denominator" in check for check in plan.checks)


def test_tool_verifier_flags_known_wrong_query_shapes():
    report = verify_tool_result(
        task("Return the driver with rank 2"),
        "execute_context_sql",
        {"sql": "SELECT * FROM results WHERE position = 2"},
        {"rows": []},
    )
    assert not report.errors
    assert any("position" in warning for warning in report.warnings)

    report = verify_tool_result(
        task("Which items have price per unit above 29?"),
        "execute_python",
        {"code": "filtered = df[df['Price'] > 29]"},
        {"success": True},
    )
    assert any("per-row ratio" in warning for warning in report.warnings)


def test_answer_verifier_rejects_duplicate_columns_and_flags_shape():
    report = verify_answer_shape(task("What percentage is the answer?"), ["value", "value"], [[1, 2]])
    assert report.errors




def test_tool_verifier_flags_average_and_vote_entity_mismatches():
    report = verify_tool_result(
        task("What is the average expense per month?"),
        "execute_context_sql",
        {"sql": "SELECT SUM(amount) / 12 FROM expenses"},
        {"rows": [[82027220]]},
    )
    assert any("SUM without AVG" in warning for warning in report.warnings)

    report = verify_tool_result(
        task("How many votes did the user receive?"),
        "execute_python",
        {"code": "answer = posts[posts.user_id == target].shape[0]"},
        {"success": True},
    )
    assert any("count posts" in warning for warning in report.warnings)
