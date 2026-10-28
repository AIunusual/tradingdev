"""Offline checks that legacy model recovery requires actual discovery evidence."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from tests.e2e.llm_client import ToolCall
from tests.e2e.strategy_scenarios import SCENARIOS
from tests.e2e.test_llm_workflows import assert_workflow
from tests.integration.execution_fixtures import (
    execution_evidence_draft,
    synthetic_preflight_receipt,
)


@pytest.fixture
def legacy_calls() -> list[ToolCall]:
    strategy_id = SCENARIOS["legacy"].strategy_id
    identity = {"strategy_id": strategy_id, "revision_id": "new-revision"}
    legacy = {
        "strategy_id": SCENARIOS["legacy"].legacy_strategy_id,
        "revision_id": None,
        "kind": "legacy",
        "status": "revision_required",
        "code": "strategy_revision_required",
    }
    config: dict[str, dict[str, Any]] = {
        "research": {"strategy_outline": "Saved legacy moving-average rules."},
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
        ToolCall("get_strategy_contract", {}, {"example_strategy_code": "source"}),
        ToolCall(
            "list_strategies",
            {},
            {
                "items": [legacy],
                "offset": 0,
                "limit": 20,
                "total": 1,
                "next_offset": None,
            },
        ),
        ToolCall(
            "get_strategy",
            {"strategy_id": SCENARIOS["legacy"].legacy_strategy_id},
            {
                **legacy,
                "success": True,
                "source_code": "legacy source",
                "yaml_config": "legacy config",
            },
        ),
        ToolCall(
            "save_strategy",
            {
                "strategy_id": strategy_id,
                "research": {
                    **SCENARIOS["legacy"].research,
                    "strategy_outline": config["research"]["strategy_outline"],
                },
            },
            {**identity, "success": True, "status": "draft"},
        ),
        ToolCall(
            "validate_strategy",
            identity,
            {
                **identity,
                "success": True,
                "status": "validated",
                "effective_config": deepcopy(config),
            },
        ),
        ToolCall(
            "dry_run_strategy",
            identity,
            {
                **identity,
                "success": True,
                "status": "runnable",
                "effective_config": deepcopy(config),
            },
        ),
        ToolCall(
            "save_research_evidence",
            {"draft": execution_evidence_draft(SCENARIOS["legacy"].parameters)},
            {"success": True, "evidence_id": "e" * 32},
        ),
        ToolCall(
            "prepare_backtest",
            {
                **identity,
                "research": {"kind": "inherit_selected_revision"},
                "evidence_id": "e" * 32,
            },
            {
                "success": True,
                "status": "ready",
                "plan_id": "plan-1",
                "manifest_hash": "a" * 64,
                "confirmation_summary": "確認本次歷史研究設定。",
                "html_path": "/fixture/confirmation.html",
                "artifact_id": "plan:confirmation",
                "preflight": synthetic_preflight_receipt("a" * 64),
            },
        ),
        ToolCall(
            "submit_execution_decision",
            {"plan_id": "plan-1", "decision": "approve"},
            {
                "success": True,
                "plan_id": "plan-1",
                "job_id": "job-1",
                "manifest_hash": "a" * 64,
            },
        ),
        ToolCall(
            "get_job_status",
            {"job_id": "job-1"},
            {
                **identity,
                "status": "done",
                "run_id": "run-1",
                "metrics": {},
                "manifest_hash": "a" * 64,
            },
        ),
        ToolCall(
            "get_run",
            {"run_id": "run-1"},
            {
                "success": True,
                "run": {
                    **identity,
                    "metrics": {},
                    "manifest_hash": "a" * 64,
                    "available_metric_ids": [
                        "daily_pnl_mean",
                        "total_volume",
                        "n_days",
                    ],
                },
            },
        ),
        ToolCall(
            "list_artifacts",
            {"run_id": "run-1"},
            {
                "items": [{"path": "result.json"}],
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
            {"run_id": "run-1"},
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


def test_legacy_evidence_accepts_discovery_read_and_fresh_revision(
    legacy_calls: list[ToolCall],
) -> None:
    assert_workflow(legacy_calls, SCENARIOS["legacy"])


@pytest.mark.parametrize("missing", ["list_strategies", "get_strategy"])
def test_legacy_evidence_rejects_skipping_existing_contents(
    legacy_calls: list[ToolCall], missing: str
) -> None:
    calls = [call for call in legacy_calls if call.name != missing]
    with pytest.raises(AssertionError, match="Missing required MCP step"):
        assert_workflow(calls, SCENARIOS["legacy"])


@pytest.mark.parametrize("name", ["list_strategies", "get_strategy"])
def test_legacy_evidence_rejects_nonlegacy_responses(
    legacy_calls: list[ToolCall], name: str
) -> None:
    calls = deepcopy(legacy_calls)
    call = next(call for call in calls if call.name == name)
    payload = call.result["items"][0] if name == "list_strategies" else call.result
    payload["status"] = "runnable"
    with pytest.raises(AssertionError, match="Missing required MCP step"):
        assert_workflow(calls, SCENARIOS["legacy"])


def test_empty_strategy_page_does_not_identify_the_legacy_strategy(
    legacy_calls: list[ToolCall],
) -> None:
    discovery = next(call for call in legacy_calls if call.name == "list_strategies")
    discovery.result.update(items=[], total=0)
    with pytest.raises(AssertionError, match="list_strategies identifying"):
        assert_workflow(legacy_calls, SCENARIOS["legacy"])


def test_legacy_evidence_rejects_reading_only_after_resaving(
    legacy_calls: list[ToolCall],
) -> None:
    legacy_calls[2], legacy_calls[3] = legacy_calls[3], legacy_calls[2]
    with pytest.raises(AssertionError, match="Invalid MCP legacy recovery order"):
        assert_workflow(legacy_calls, SCENARIOS["legacy"])


def test_legacy_evidence_rejects_inheriting_runnable_status(
    legacy_calls: list[ToolCall],
) -> None:
    legacy_calls[3].result["status"] = "runnable"
    with pytest.raises(AssertionError, match="without inherited evidence"):
        assert_workflow(legacy_calls, SCENARIOS["legacy"])


def test_legacy_recovery_cannot_overwrite_the_occupied_source_namespace(
    legacy_calls: list[ToolCall],
) -> None:
    legacy_calls[3].arguments["strategy_id"] = SCENARIOS["legacy"].legacy_strategy_id
    with pytest.raises(AssertionError, match="save_strategy"):
        assert_workflow(legacy_calls, SCENARIOS["legacy"])
