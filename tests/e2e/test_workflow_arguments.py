"""Keep evidence conversion aligned with the actual registered SDK input models."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import pytest

from tests.e2e.llm_client import ToolCall
from tests.e2e.workflow_arguments import effective_workflow_calls
from tests.integration.execution_fixtures import (
    execution_evidence_draft,
    research_input,
)
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.mcp.server import create_server

if TYPE_CHECKING:
    from pathlib import Path


def _samples() -> list[tuple[str, dict[str, Any]]]:
    prepare = {
        "strategy_id": "fixture",
        "research": json.dumps(research_input(fees=0)),
        "parameters": '{"period": "20"}',
        "backtest_overrides": '{"position_size": 100}',
        "evidence_id": "e" * 32,
    }
    return [
        (
            "save_research_evidence",
            {"draft": json.dumps(execution_evidence_draft({"period": 20}))},
        ),
        ("prepare_backtest", prepare),
        (
            "prepare_comparison",
            {name: value for name, value in prepare.items() if name != "parameters"}
            | {
                "candidates": '[{"label":"固定候選","rationale":"事先指定",'
                '"parameters":{"period":20}}]'
            },
        ),
        (
            "prepare_walk_forward",
            prepare
            | {
                "validation_config": (
                    '{"mode": "automatic", "n_splits": 2, "train_ratio": 0.6}'
                )
            },
        ),
        (
            "prepare_backtest",
            prepare
            | {
                "research": '{"kind": "inherit_selected_revision"}',
                "revision_id": "selected-revision",
            },
        ),
        (
            "prepare_optimization",
            prepare
            | {
                "research": json.dumps(
                    research_input(end_date="2024-01-08T23:59:59.999999", fees=0)
                ),
                "param_ranges": '{"period": [10, 20]}',
                "optimization_metric": "total_return",
                "train_start": "2024-01-01",
                "train_end": "2024-01-03",
                "test_start": "2024-01-04",
                "test_end": "2024-01-08",
            },
        ),
        ("find_runs", {"parameters": '{"period": "20"}'}),
        (
            "generate_report",
            {
                "run_ids": '["run"]',
                "sections": '["metrics", "trades"]',
                "commentary": '[{"title":"研究評語","text":"模擬結果"}]',
                "analysis": json.dumps(
                    {
                        "summary": "  summary  ",
                        "observations": "observed",
                        "hypotheses_and_limits": "unknown",
                        "next_steps": "proposed",
                        "references": [{"run_id": "run", "scope": "full"}],
                    }
                ),
            },
        ),
        ("generate_report", {"run_ids": '["run"]', "sections": "null"}),
        ("generate_report", {"comparison_job_id": "parent", "run_ids": "null"}),
        (
            "generate_report",
            {
                "ml_entry_ids": '["candidate-a", "candidate-b"]',
                "ml_ranking_metric": "ic_mean",
                "ml_allow_parameter_axes": "true",
                "sections": '["ml"]',
            },
        ),
        (
            "leaderboard",
            {
                "metric": "ic_mean",
                "allow_parameter_axes": "true",
                "allow_mixed": "false",
                "horizon": "1",
                "offset": "0",
                "limit": "20",
            },
        ),
        *[
            (name, {"run_id": "run", "limit": "1", "offset": "1"})
            for name in (
                "get_run_trades",
                "get_run_executions",
                "get_run_account_history",
            )
        ],
        (
            "cleanup_strategy_drafts",
            {"strategy_id": "fixture", "revision_ids": '["old"]', "apply": "true"},
        ),
        (
            "cleanup_strategy_drafts",
            {"strategy_id": "fixture", "revision_ids": "null", "apply": "false"},
        ),
    ]


@pytest.mark.parametrize(("name", "arguments"), _samples())
def test_comparison_matches_registered_sdk_argument_conversion(
    tmp_path: Path, name: str, arguments: dict[str, Any]
) -> None:
    server = create_server(WorkspacePaths(tmp_path / "workspace"))
    tool = server._tool_manager.get_tool(name)
    assert tool is not None
    metadata = tool.fn_metadata
    expected = metadata.arg_model.model_validate(
        metadata.pre_parse_json(arguments)
    ).model_dump(mode="json", exclude_unset=True)
    call = ToolCall(name, deepcopy(arguments), {"success": True})
    assert effective_workflow_calls([call])[0].arguments == expected
    assert call.arguments == arguments


def test_conversion_preserves_failed_calls_source_text_and_unknown_fields() -> None:
    raw = {
        "presentation": '{"malformed":true}',
        "parameters": "[]",
        "code": "[1,2]",
        "yaml_config": '{"strategy":{}}',
    }
    failed = ToolCall("prepare_backtest", raw, {"success": False})
    assert effective_workflow_calls([failed])[0] is failed
    saved = ToolCall("save_strategy", raw, {"success": True})
    assert effective_workflow_calls([saved])[0].arguments == raw
    confirmed = ToolCall(
        "submit_execution_decision",
        {"plan_id": "plan", "approved": "true"},
        {"success": True},
    )
    assert effective_workflow_calls([confirmed])[0].arguments == confirmed.arguments


@pytest.mark.parametrize("value", ["{}", '"run"', "[", "[1]"])
@pytest.mark.parametrize("field", ["run_ids", "ml_entry_ids"])
def test_conversion_rejects_invalid_report_list_shapes(value: str, field: str) -> None:
    call = ToolCall("generate_report", {field: value}, {"success": True})
    with pytest.raises(AssertionError, match=field):
        effective_workflow_calls([call])
