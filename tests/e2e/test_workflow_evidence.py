"""Offline checks that model evidence belongs to the requested repair target."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

from tests.e2e.llm_client import ToolCall
from tests.e2e.strategy_scenarios import SCENARIOS
from tests.e2e.test_llm_workflows import assert_workflow
from tests.integration.execution_fixtures import (
    execution_evidence_draft,
    synthetic_preflight_receipt,
)


def _repair_evidence() -> list[ToolCall]:
    strategy_id = SCENARIOS["repair"].strategy_id
    target = {"strategy_id": strategy_id}
    versioned = {**target, "revision_id": "revision_a"}
    metrics = {"total_return": 0.25}
    # Synthetic tool output, independently recorded from the later prepare input.
    config: dict[str, dict[str, Any]] = {
        "research": {"strategy_outline": "Saved moving-average fixture rules."},
        "backtest": {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01T00:00:00Z",
            "end_date": "2024-01-08T00:00:00Z",
            "init_cash": 10000.0,
            "fees": 0.0,
            "slippage": 0.0,
            "periods_per_year": 365.0,
        },
    }
    return [
        ToolCall("get_strategy_contract", {}, {"lifecycle": "fixture"}),
        ToolCall(
            "get_strategy", target, {"success": True, "revision_id": "broken-revision"}
        ),
        ToolCall(
            "validate_strategy",
            target,
            {"success": False, "diagnostics": [{"code": "invalid_signal_values"}]},
        ),
        ToolCall(
            "save_strategy",
            {
                **target,
                "research": {
                    **SCENARIOS["repair"].research,
                    "strategy_outline": config["research"]["strategy_outline"],
                },
                "expected_revision_id": "broken-revision",
            },
            {"success": True, "status": "draft", "revision_id": "revision_a"},
        ),
        ToolCall(
            "validate_strategy",
            versioned,
            {
                "success": True,
                "status": "validated",
                "revision_id": "revision_a",
                "effective_config": deepcopy(config),
            },
        ),
        ToolCall(
            "dry_run_strategy",
            versioned,
            {
                "success": True,
                "status": "runnable",
                "revision_id": "revision_a",
                "effective_config": deepcopy(config),
            },
        ),
        ToolCall(
            "save_research_evidence",
            {"draft": execution_evidence_draft(SCENARIOS["repair"].parameters)},
            {"success": True, "evidence_id": "e" * 32},
        ),
        ToolCall(
            "prepare_backtest",
            {
                **versioned,
                "research": {"kind": "inherit_selected_revision"},
                "evidence_id": "e" * 32,
            },
            {
                "success": True,
                "status": "ready",
                "plan_id": "plan",
                "manifest_hash": "a" * 64,
                "confirmation_summary": "確認本次參數與歷史模擬期間。",
                "html_path": "/fixture/confirmation.html",
                "artifact_id": "plan:confirmation",
                "preflight": synthetic_preflight_receipt("a" * 64),
            },
        ),
        ToolCall(
            "submit_execution_decision",
            {"plan_id": "plan", "decision": "approve"},
            {
                "success": True,
                "plan_id": "plan",
                "job_id": "job",
                "manifest_hash": "a" * 64,
            },
        ),
        ToolCall(
            "get_job_status",
            {"job_id": "job"},
            {
                "status": "done",
                "run_id": "run",
                "metrics": metrics,
                "revision_id": "revision_a",
                "manifest_hash": "a" * 64,
            },
        ),
        ToolCall(
            "get_run",
            {"run_id": "run"},
            {
                "success": True,
                "run": {
                    "strategy_id": strategy_id,
                    "available_metric_ids": [
                        "daily_pnl_mean",
                        "total_volume",
                        "n_days",
                    ],
                    "metrics": metrics,
                    "revision_id": "revision_a",
                    "manifest_hash": "a" * 64,
                },
            },
        ),
        ToolCall(
            "list_artifacts",
            {"run_id": "run"},
            {
                "items": [{"artifact_id": "artifact"}],
                "offset": 0,
                "limit": 20,
                "total": 1,
                "next_offset": None,
            },
        ),
        ToolCall(
            "get_metric_catalog",
            {"mode": "signal"},
            {
                "success": True,
                "definitions": [
                    {"id": metric_id}
                    for metric_id in ["daily_pnl_mean", "total_volume", "n_days"]
                ],
            },
        ),
        ToolCall(
            "get_run_metrics",
            {"run_id": "run"},
            {
                "success": True,
                "scope": "full",
                "metrics": {
                    "daily_pnl_mean": 1.0,
                    "total_volume": 100.0,
                    "n_days": 8,
                },
            },
        ),
    ]


@pytest.mark.parametrize("event_index", [1, 2, 3], ids=["read", "diagnosis", "save"])
def test_other_strategy_cannot_supply_repair_evidence(event_index: int) -> None:
    calls = _repair_evidence()
    assert_workflow(calls, SCENARIOS["repair"])
    original = calls[event_index]
    calls[event_index] = ToolCall(
        original.name, {"strategy_id": "unrelated"}, original.result
    )
    with pytest.raises(AssertionError, match=SCENARIOS["repair"].strategy_id):
        assert_workflow(calls, SCENARIOS["repair"])


@pytest.mark.parametrize("tool", ["get_metric_catalog", "get_run_metrics"])
def test_missing_metric_discovery_or_detail_is_rejected(tool: str) -> None:
    calls = [call for call in _repair_evidence() if call.name != tool]
    with pytest.raises(AssertionError, match=tool):
        assert_workflow(calls, SCENARIOS["repair"])


@pytest.mark.parametrize("metrics_before_summary", [False, True])
def test_natural_runtime_repair_accepts_report_without_artifact_listing(
    metrics_before_summary: bool,
) -> None:
    calls = [call for call in _repair_evidence() if call.name != "list_artifacts"]
    calls[2].result["diagnostics"] = [{"code": "strategy_check_failed"}]
    if metrics_before_summary:
        summary = next(call for call in calls if call.name == "get_run")
        calls.remove(summary)
        calls.append(summary)
    report = ToolCall(
        "generate_report",
        {"run_ids": ["run"]},
        {"success": True, "run_ids": ["run"], "artifact_id": "report:html"},
    )
    calls.append(report)
    scenario = replace(SCENARIOS["repair"], runtime_exit=True)
    assert_workflow(calls, scenario)
    for result in (
        {"success": False},
        {"success": True, "run_ids": ["unrelated"], "artifact_id": "report:html"},
    ):
        report.result = result
        with pytest.raises(AssertionError, match="generate_report"):
            assert_workflow(calls, scenario)


def test_scripted_repair_still_requires_artifact_listing() -> None:
    calls = [call for call in _repair_evidence() if call.name != "list_artifacts"]
    with pytest.raises(AssertionError, match="list_artifacts"):
        assert_workflow(calls, SCENARIOS["repair"])


def test_detail_from_another_run_is_rejected() -> None:
    calls = _repair_evidence()
    calls[-1].arguments["run_id"] = "unrelated"
    with pytest.raises(AssertionError, match="get_run_metrics"):
        assert_workflow(calls, SCENARIOS["repair"])


def test_summary_does_not_substitute_for_explicit_detail_query() -> None:
    calls = _repair_evidence()
    calls[-1].result["metrics"].pop("daily_pnl_mean")
    with pytest.raises(AssertionError, match="get_run_metrics"):
        assert_workflow(calls, SCENARIOS["repair"])


def test_unrelated_events_do_not_hide_correct_target_diagnosis() -> None:
    target = {"strategy_id": SCENARIOS["repair"].strategy_id}
    unrelated = {"strategy_id": "unrelated"}
    distractors = [
        ToolCall("save_strategy", unrelated, {"success": True, "status": "draft"}),
        ToolCall("get_strategy", unrelated, {"success": False}),
        ToolCall(
            "validate_strategy",
            unrelated,
            {"success": False, "diagnostics": [{"code": "invalid_signal_values"}]},
        ),
        ToolCall(
            "validate_strategy",
            target,
            {"success": False, "diagnostics": [{"code": "ruff_failed"}]},
        ),
    ]
    assert_workflow([*distractors, *_repair_evidence()], SCENARIOS["repair"])


def test_repair_requires_the_seeded_signal_error() -> None:
    calls = _repair_evidence()
    calls[2].result = {"success": False, "diagnostics": [{"code": "ruff_failed"}]}
    with pytest.raises(AssertionError, match="diagnostic code='invalid_signal_values'"):
        assert_workflow(calls, SCENARIOS["repair"])


def test_failed_lookup_does_not_count_as_reading_the_draft() -> None:
    calls = _repair_evidence()
    calls[1].result = {"success": False, "error": "Unknown strategy"}
    with pytest.raises(AssertionError, match="get_strategy.*success=True"):
        assert_workflow(calls, SCENARIOS["repair"])


@pytest.mark.parametrize(
    ("event_index", "expected_details"),
    [
        (0, ("get_strategy_contract", "strategy_id='llm_repair'")),
        (1, ("get_strategy", "strategy_id='llm_repair'", "success=True")),
        (2, ("validate_strategy", "success=False", "invalid_signal_values")),
        (3, ("save_strategy", "strategy_id='llm_repair'", "success=True")),
        (4, ("validate_strategy", "status='validated'", "after save_strategy")),
        (5, ("dry_run_strategy", "status='runnable'", "after validate_strategy")),
        (7, ("prepare_backtest", "ready plan_id", "after successful dry_run_strategy")),
        (
            8,
            ("submit_execution_decision", "plan_id='plan'", "nonempty job_id"),
        ),
        (
            9,
            (
                "get_job_status",
                "job_id='job'",
                "status='done'",
                "after submit_execution_decision",
            ),
        ),
        (10, ("get_run", "run_id='run'", "after get_job_status reported done")),
        (11, ("list_artifacts", "run_id='run'", "nonempty result")),
    ],
    ids=[
        "contract",
        "read",
        "diagnosis",
        "save",
        "validate",
        "dry-run",
        "prepare",
        "confirm",
        "done",
        "query",
        "artifacts",
    ],
)
def test_missing_step_reports_required_tool_target_and_outcome(
    event_index: int, expected_details: tuple[str, ...]
) -> None:
    calls = _repair_evidence()
    del calls[event_index]
    with pytest.raises(AssertionError, match="Missing required MCP step:") as error:
        assert_workflow(calls, SCENARIOS["repair"])
    message = str(error.value)
    for detail in expected_details:
        assert detail in message


@pytest.mark.parametrize(
    ("event_index", "actual_status", "expected_status"),
    [(4, "draft", "validated"), (5, "validated", "runnable"), (9, "failed", "done")],
)
def test_wrong_status_reports_the_required_status(
    event_index: int, actual_status: str, expected_status: str
) -> None:
    calls = _repair_evidence()
    calls[event_index].result["status"] = actual_status
    with pytest.raises(AssertionError, match=f"status='{expected_status}'"):
        assert_workflow(calls, SCENARIOS["repair"])


def test_contract_order_failure_explains_required_order() -> None:
    calls = _repair_evidence()
    calls[0], calls[3] = calls[3], calls[0]
    with pytest.raises(AssertionError, match="get_strategy_contract must precede"):
        assert_workflow(calls, SCENARIOS["repair"])


def test_empty_artifact_page_is_not_completed_result_evidence() -> None:
    calls = _repair_evidence()
    artifacts = next(call for call in calls if call.name == "list_artifacts")
    artifacts.result.update(items=[], total=0)
    with pytest.raises(AssertionError, match="list_artifacts.*nonempty result"):
        assert_workflow(calls, SCENARIOS["repair"])


def test_dry_run_before_validation_does_not_satisfy_required_step() -> None:
    calls = _repair_evidence()
    calls[4], calls[5] = calls[5], calls[4]
    with pytest.raises(
        AssertionError, match="dry_run_strategy.*after validate_strategy"
    ):
        assert_workflow(calls, SCENARIOS["repair"])


def test_repair_order_failure_explains_read_diagnose_save_sequence() -> None:
    calls = _repair_evidence()
    calls[1], calls[2] = calls[2], calls[1]
    with pytest.raises(AssertionError, match="Invalid MCP repair order") as error:
        assert_workflow(calls, SCENARIOS["repair"])
    assert "strategy_id='llm_repair'" in str(error.value)
    assert "get_strategy must precede validate_strategy" in str(error.value)
    assert "precede successful save_strategy" in str(error.value)


@pytest.mark.parametrize("revision", [None, "another-draft"])
def test_repair_cannot_claim_a_save_without_the_selected_draft(
    revision: str | None,
) -> None:
    calls = _repair_evidence()
    calls[3].arguments["expected_revision_id"] = revision
    with pytest.raises(AssertionError, match="bind expected_revision_id"):
        assert_workflow(calls, SCENARIOS["repair"])


@pytest.mark.parametrize("event_index", [4, 5, 9, 10])
def test_mixed_revisions_cannot_supply_workflow_evidence(event_index: int) -> None:
    calls = _repair_evidence()
    if event_index == 10:
        calls[event_index].result["run"]["revision_id"] = "revision_b"
    else:
        calls[event_index].result["revision_id"] = "revision_b"
    with pytest.raises(AssertionError):
        assert_workflow(calls, SCENARIOS["repair"])


@pytest.mark.parametrize("event_index", [7, 8, 9, 10])
def test_different_manifests_cannot_supply_workflow_evidence(event_index: int) -> None:
    calls = _repair_evidence()
    result = calls[event_index].result
    payload = result["run"] if event_index == 10 else result
    payload["manifest_hash"] = "b" * 64
    with pytest.raises(AssertionError):
        assert_workflow(calls, SCENARIOS["repair"])


def test_parameter_experiment_requires_explicit_overrides_and_original_revision() -> (
    None
):
    scenario = replace(SCENARIOS["repair"], experiment=True)
    calls = _repair_evidence()
    with pytest.raises(AssertionError, match="prepare_backtest arguments differ"):
        assert_workflow(calls, scenario)
    calls[7].arguments["parameters"] = scenario.overrides
    assert_workflow(calls, scenario)
    calls[7].arguments["revision_id"] = "another_revision"
    with pytest.raises(AssertionError, match="prepare_backtest arguments differ"):
        assert_workflow(calls, scenario)


@pytest.mark.parametrize(
    "optional_arguments",
    [
        {},
        {"parameters": None},
        {"backtest_overrides": None},
        {"parameters": None, "backtest_overrides": None},
    ],
    ids=["omitted", "null-parameters", "null-backtest", "both-null"],
)
def test_nonexperiment_preparation_accepts_omitted_or_null_overrides(
    optional_arguments: dict[str, Any],
) -> None:
    calls = _repair_evidence()
    calls[7].arguments.update(optional_arguments)
    assert_workflow(calls, SCENARIOS["repair"])


@pytest.mark.parametrize(
    "repeated",
    [
        {"fees": 0, "slippage": 0},
        {
            "start_date": "2024-01-01T08:00:00+08:00",
            "end_date": "2024-01-08",
        },
        {
            "strategy_outline": "Saved moving-average fixture rules.",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01T00:00:00Z",
            "end_date": "2024-01-08T00:00:00+00:00",
            "init_cash": 10000,
            "fees": 0,
            "slippage": 0,
            "annualization": 365,
        },
    ],
    ids=["zero-cost-subset", "equivalent-utc-period", "all-research-fields"],
)
@pytest.mark.parametrize("json_encoded", [False, True], ids=["object", "sdk-json"])
def test_repeated_research_is_compared_to_recorded_revision_values(
    repeated: dict[str, Any], json_encoded: bool
) -> None:
    calls = _repair_evidence()
    research = {"kind": "inherit_selected_revision", **repeated}
    calls[7].arguments["research"] = json.dumps(research) if json_encoded else research
    original = deepcopy(calls)
    assert_workflow(calls, SCENARIOS["repair"])
    assert calls == original, "Verification must preserve raw arguments and results"


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("strategy_outline", "Different entry rules"),
        ("symbol", "ETH/USDT"),
        ("timeframe", "4h"),
        ("start_date", "2024-01-02"),
        ("end_date", "2024-01-09"),
        ("init_cash", 20000),
        ("fees", 0.0006),
        ("slippage", 0.0005),
        ("annualization", 252),
        ("annualization", "unavailable"),
    ],
)
def test_inherited_research_cannot_change_any_recorded_setting(
    field: str, changed: Any
) -> None:
    calls = _repair_evidence()
    calls[7].arguments["research"][field] = changed
    with pytest.raises(AssertionError, match=f"research changes {field}"):
        assert_workflow(calls, SCENARIOS["repair"])


def test_fixed_research_cannot_reselect_market_annualization_policy() -> None:
    calls = _repair_evidence()
    calls[7].arguments["research"]["annualization"] = "market"
    with pytest.raises(AssertionError, match="reselect.*annualization policy"):
        assert_workflow(calls, SCENARIOS["repair"])


def test_explicit_unavailable_annualization_matches_only_recorded_none() -> None:
    calls = _repair_evidence()
    for call in calls[4:6]:
        call.result["effective_config"]["backtest"]["periods_per_year"] = None
    calls[7].arguments["research"]["annualization"] = "unavailable"
    assert_workflow(calls, SCENARIOS["repair"])
    calls[7].arguments["research"]["annualization"] = 365
    with pytest.raises(AssertionError, match="research changes annualization"):
        assert_workflow(calls, SCENARIOS["repair"])


def test_same_new_research_does_not_substitute_for_explicit_inheritance() -> None:
    calls = _repair_evidence()
    calls[7].arguments["research"] = SCENARIOS["repair"].research
    with pytest.raises(AssertionError, match="must inherit the selected revision"):
        assert_workflow(calls, SCENARIOS["repair"])


@pytest.mark.parametrize("event_index", [4, 5], ids=["validate", "dry-run"])
@pytest.mark.parametrize(
    "missing",
    ["effective_config", "strategy_outline", "fees", "periods_per_year"],
)
def test_research_comparison_never_supplies_missing_recorded_evidence(
    event_index: int, missing: str
) -> None:
    calls = _repair_evidence()
    result = calls[event_index].result
    if missing == "effective_config":
        result.pop(missing)
    else:
        section = "research" if missing == "strategy_outline" else "backtest"
        result["effective_config"][section].pop(missing)
    with pytest.raises(AssertionError, match="effective_config"):
        assert_workflow(calls, SCENARIOS["repair"])


def test_validate_and_dry_run_must_record_the_same_research() -> None:
    calls = _repair_evidence()
    calls[5].result["effective_config"]["backtest"]["fees"] = 0.0006
    with pytest.raises(
        AssertionError, match="effective_config research settings differ"
    ):
        assert_workflow(calls, SCENARIOS["repair"])


def test_validation_evidence_accepts_equivalent_utc_dates() -> None:
    calls = _repair_evidence()
    calls[4].result["effective_config"]["backtest"]["start_date"] = "2024-01-01"
    calls[5].result["effective_config"]["backtest"]["start_date"] = (
        "2024-01-01T08:00:00+08:00"
    )
    assert_workflow(calls, SCENARIOS["repair"])


@pytest.mark.parametrize("field", ["fees", "periods_per_year"])
@pytest.mark.parametrize("invalid", [True, "0", float("nan")])
def test_equal_but_invalid_recorded_numbers_are_not_research_evidence(
    field: str, invalid: Any
) -> None:
    calls = _repair_evidence()
    for call in calls[4:6]:
        call.result["effective_config"]["backtest"][field] = invalid
    with pytest.raises(AssertionError, match="must record a finite number"):
        assert_workflow(calls, SCENARIOS["repair"])


@pytest.mark.parametrize(
    "overrides",
    [
        {"parameters": SCENARIOS["repair"].parameters},
        {"parameters": SCENARIOS["repair"].overrides},
        {"parameters": {}},
        {"backtest_overrides": {"fees": 0}},
        {"backtest_overrides": {"fees": 0.01}},
        {"backtest_overrides": {}},
    ],
    ids=[
        "same-parameters",
        "different-parameters",
        "empty-parameters",
        "same-backtest",
        "different-backtest",
        "empty-backtest",
    ],
)
def test_nonexperiment_preparation_rejects_override_mappings(
    overrides: dict[str, Any],
) -> None:
    calls = _repair_evidence()
    calls[7].arguments.update(overrides)
    with pytest.raises(AssertionError, match="prepare_backtest arguments differ"):
        assert_workflow(calls, SCENARIOS["repair"])


@pytest.mark.parametrize(
    "parameters",
    [None, {}, SCENARIOS["repair"].parameters, {"fast_period": 3}],
    ids=["null", "empty", "base-parameters", "partial-override"],
)
def test_parameter_experiment_requires_the_complete_requested_override(
    parameters: dict[str, int] | None,
) -> None:
    scenario = replace(SCENARIOS["repair"], experiment=True)
    calls = _repair_evidence()
    calls[7].arguments["parameters"] = parameters
    with pytest.raises(AssertionError, match="prepare_backtest arguments differ"):
        assert_workflow(calls, scenario)


def test_parameter_experiment_cannot_resave_after_base_is_runnable() -> None:
    scenario = replace(SCENARIOS["repair"], experiment=True)
    calls = _repair_evidence()
    calls[7].arguments["parameters"] = scenario.overrides
    calls.append(
        ToolCall(
            "save_strategy",
            {"strategy_id": scenario.strategy_id},
            {"success": True, "revision_id": "duplicate_source"},
        )
    )
    with pytest.raises(AssertionError, match="reuse the runnable revision"):
        assert_workflow(calls, scenario)


@pytest.mark.parametrize("location", ["arguments", "result"])
def test_confirmation_of_another_plan_cannot_supply_execution_evidence(
    location: str,
) -> None:
    calls = _repair_evidence()
    getattr(calls[8], location)["plan_id"] = "another_plan"
    with pytest.raises(AssertionError, match="submit_execution_decision"):
        assert_workflow(calls, SCENARIOS["repair"])


def test_legacy_approval_field_does_not_match_the_decision_contract() -> None:
    calls = _repair_evidence()
    calls[8].arguments["approved"] = True
    with pytest.raises(AssertionError, match="submit_execution_decision"):
        assert_workflow(calls, SCENARIOS["repair"])


@pytest.mark.parametrize(
    "field", ["confirmation_summary", "html_path", "artifact_id", "preflight"]
)
def test_preparation_requires_reviewable_confirmation_evidence(field: str) -> None:
    calls = _repair_evidence()
    calls[7].result.pop(field)
    with pytest.raises(AssertionError):
        assert_workflow(calls, SCENARIOS["repair"])


@pytest.mark.parametrize(
    "changes",
    [
        {"manifest_hash": "b" * 64},
        {"schema_version": 1},
        {"synthetic_fixture_rows": []},
        {"checked_paths": ["configuration", "engine"]},
        {"not_checked": ["backtest_engine", "training", "performance"]},
        {"trade_count": 3},
    ],
)
def test_workflow_rejects_stale_or_historical_preparation_evidence(
    changes: dict[str, Any],
) -> None:
    calls = _repair_evidence()
    calls[7].result["preflight"].update(changes)
    with pytest.raises(AssertionError):
        assert_workflow(calls, SCENARIOS["repair"])


def test_workflow_accepts_sdk_decoded_metadata_and_parameter_arguments() -> None:
    scenario = replace(SCENARIOS["repair"], experiment=True)
    calls = _repair_evidence()
    arguments = calls[7].arguments
    calls[6].arguments["draft"] = json.dumps(calls[6].arguments["draft"])
    arguments["parameters"] = json.dumps(scenario.overrides)
    arguments["backtest_overrides"] = "null"
    original = dict(arguments)
    assert_workflow(calls, scenario)
    assert calls[7].arguments == original, "Raw model evidence must remain available"


@pytest.mark.parametrize(
    "presentation",
    [
        None,
        "not JSON",
        "{",
        "[]",
        '"string only"',
        "null",
        "{}",
        '{"title":"研究","summary":"說明","parameter_descriptions":{}}',
        {
            "title": "研究",
            "summary": "說明",
            "parameter_descriptions": {"/fast_period": {"label": "快線"}},
        },
    ],
)
def test_workflow_rejects_invalid_or_incomplete_presentation(
    presentation: Any,
) -> None:
    calls = _repair_evidence()
    calls[6].arguments["draft"]["presentation"] = presentation
    with pytest.raises(AssertionError):
        assert_workflow(calls, SCENARIOS["repair"])


def test_workflow_does_not_convert_nested_strategy_parameter_strings() -> None:
    scenario = replace(SCENARIOS["repair"], experiment=True)
    calls = _repair_evidence()
    calls[7].arguments["parameters"] = json.dumps(
        {name: str(value) for name, value in scenario.overrides.items()}
    )
    with pytest.raises(AssertionError, match="prepare_backtest arguments differ"):
        assert_workflow(calls, scenario)


def test_history_workflow_accepts_sdk_encoded_report_and_query_arguments() -> None:
    scenario = replace(SCENARIOS["repair"], history=True)
    calls = _repair_evidence()
    calls.extend(
        [
            ToolCall(
                "find_runs",
                {
                    "strategy_id": scenario.strategy_id,
                    "parameters": json.dumps(scenario.parameters),
                },
                {"success": True, "runs": [{"run_id": "run", "scope": "full"}]},
            ),
            ToolCall(
                "get_run_trades",
                {"run_id": "run", "limit": "1"},
                {"success": True, "next_offset": 1},
            ),
            ToolCall(
                "get_run_trades",
                {"run_id": "run", "offset": "1"},
                {"success": True, "next_offset": None},
            ),
            ToolCall("get_run_equity", {"run_id": "run"}, {"success": True}),
            ToolCall(
                "get_run_executions",
                {"run_id": "run", "limit": "1"},
                {"success": True, "availability": "available"},
            ),
            ToolCall(
                "get_run_account_history",
                {"run_id": "run", "limit": "2"},
                {"success": True, "availability": "available"},
            ),
            ToolCall("get_report_sections", {}, {"success": True}),
            ToolCall(
                "generate_report",
                {
                    "run_ids": '["run"]',
                    "sections": '["overview","metrics","ledger","trades"]',
                    "commentary": json.dumps(
                        [{"title": "研究評語", "text": "本次結果僅為歷史模擬。"}]
                    ),
                },
                {"success": True},
            ),
        ]
    )
    assert_workflow(calls, scenario)
    calls[-1].arguments["run_ids"] = '["another-run"]'
    with pytest.raises(AssertionError, match="generate_report"):
        assert_workflow(calls, scenario)
