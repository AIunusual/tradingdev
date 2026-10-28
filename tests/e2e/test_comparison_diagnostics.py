"""Comparison failure diagnostics keep identities and failed attempts, not payloads."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import pytest

from tests.e2e.comparison_diagnostics import comparison_failure_diagnostics
from tests.e2e.llm_client import ToolCall, tool_payload


def test_large_results_do_not_expand_diagnostics_or_hide_failed_candidate_inputs() -> (
    None
):
    large = "large-source-and-market-record-" * 30_000
    parameters = [
        {"label": "K=9", "rationale": "固定比較", "parameters": {"k_period": 9}},
        {"label": "K=14", "rationale": "固定比較", "parameters": {"k_period": 14}},
    ]
    calls = [
        ToolCall(
            "get_strategy",
            {"strategy_id": "kd_crossover"},
            {
                "success": True,
                "strategy_id": "kd_crossover",
                "source": large,
                "schema": {"description": large},
            },
        ),
        ToolCall(
            "prepare_comparison",
            {"strategy_id": "kd_crossover", "candidates": parameters},
            {
                "success": False,
                "code": "invalid_candidate",
                "error": "candidate_2 failed",
                "required_parameter_paths": ["/config/k_period"],
                "details": large,
            },
        ),
        ToolCall(
            "prepare_comparison",
            {"strategy_id": "kd_crossover", "candidates": parameters},
            {
                "success": True,
                "plan_id": "reviewed_plan",
                "manifest_hash": "a" * 64,
                "preflight": {"details": large},
            },
        ),
        ToolCall(
            "get_run_trades",
            {"run_id": "parent_candidate_2", "scope": "full"},
            {
                "success": True,
                "run_id": "parent_candidate_2",
                "scope": "full",
                "trades": [large],
            },
        ),
        ToolCall(
            "generate_report",
            {"comparison_job_id": "parent"},
            {
                "success": True,
                "comparison_job_id": "parent",
                "run_ids": ["parent_candidate_1", "parent_candidate_2"],
                "report_id": "report",
                "path": "/report.html",
                "html": large,
            },
        ),
    ]
    original = deepcopy(calls)
    evidence = comparison_failure_diagnostics(calls)
    text = json.dumps(evidence, ensure_ascii=False)
    assert len(text) < 6000
    assert "large-source-and-market-record" not in text
    assert evidence["attempt_count"] == len(calls)
    assert [attempt["tool"] for attempt in evidence["attempts"]] == [
        call.name for call in calls
    ]
    assert [attempt["index"] for attempt in evidence["attempts"]] == list(
        range(1, len(calls) + 1)
    )
    failed = evidence["attempts"][1]
    assert failed["arguments"]["candidates"] == parameters
    assert failed["result"] == {
        "success": False,
        "code": "invalid_candidate",
        "error": "candidate_2 failed",
        "required_parameter_paths": ["/config/k_period"],
    }
    assert failed["result_keys"] == sorted(calls[1].result)
    assert evidence["attempts"][2]["result"]["plan_id"] == "reviewed_plan"
    assert evidence["attempts"][3]["arguments"] == {
        "run_id": "parent_candidate_2",
        "scope": "full",
    }
    assert evidence["attempts"][4]["result"]["run_ids"] == [
        "parent_candidate_1",
        "parent_candidate_2",
    ]
    assert calls == original, "Diagnostic formatting must not mutate raw evidence"


def test_unusually_large_argument_and_error_have_explicit_bounded_evidence() -> None:
    calls = [
        ToolCall(
            "prepare_comparison",
            {"candidates": "x" * 100_000},
            {"success": False, "error": "y" * 100_000},
        )
    ]
    evidence = comparison_failure_diagnostics(calls)
    attempt = evidence["attempts"][0]
    for value in (attempt["arguments"]["candidates"], attempt["result"]["error"]):
        assert value["complete"] is False
        assert value["characters"] == 100_002
        assert len(value["sha256"]) == 64
        assert len(value["preview"]) == 400
    assert len(json.dumps(evidence)) < 2000


@pytest.mark.parametrize("result", [None, "failed " * 20_000, ["failed " * 20_000]])
def test_nonobject_results_preserve_attempts_and_bounded_evidence(result: Any) -> None:
    candidates = [{"label": "K=9", "parameters": {"k_period": 9}}]
    calls = [
        ToolCall("get_strategy", {"strategy_id": "kd_crossover"}, {"success": True}),
        ToolCall(
            "prepare_comparison",
            {"candidates": candidates, "unexpected": "large argument " * 20_000},
            result,
        ),
        ToolCall(
            "get_execution_plan", {"plan_id": "reviewed_plan"}, {"success": False}
        ),
    ]
    original = deepcopy(calls)
    evidence = comparison_failure_diagnostics(calls)
    assert evidence["attempt_count"] == 3
    assert [attempt["tool"] for attempt in evidence["attempts"]] == [
        call.name for call in calls
    ]
    assert [attempt["index"] for attempt in evidence["attempts"]] == [1, 2, 3]
    failed = evidence["attempts"][1]
    assert failed["result_type"] == type(result).__name__
    assert failed["result_keys"] is None
    assert failed["arguments"]["candidates"] == candidates
    assert failed["arguments"]["unexpected"]["complete"] is False
    if result is None:
        assert failed["result"] == {"value": None}
    else:
        value = failed["result"]["value"]
        assert value["complete"] is False
        assert value["characters"] == len(json.dumps(result, ensure_ascii=False))
        assert len(value["preview"]) == 400
        assert len(value["sha256"]) == 64
    assert len(json.dumps(evidence)) < 3000
    assert calls == original


@pytest.mark.parametrize(
    "result", [None, "tool rejected the request", ["tool rejected"]]
)
def test_nonobject_wire_reply_does_not_mask_original_assertion(result: Any) -> None:
    wire = {
        "content": [
            {
                "type": "text",
                "text": "Unstructured tool failure"
                if result is None
                else json.dumps(result),
            }
        ]
    }
    call = ToolCall(
        "prepare_comparison", {"strategy_id": "kd_crossover"}, tool_payload(wire)
    )
    with pytest.raises(AssertionError, match="Original comparison failure") as caught:
        try:
            raise AssertionError("Original comparison failure")
        except AssertionError as exc:
            exc.add_note(json.dumps(comparison_failure_diagnostics([call])))
            raise
    note = json.loads(caught.value.__notes__[0])
    assert note["attempts"][0]["result_type"] == type(result).__name__
    assert note["attempts"][0]["result"] == {"value": result}
