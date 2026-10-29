"""ML report selection and bounded results through real MCP registration."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from mcp.server import MCPServer
from mcp_types import CallToolResult

from tradingdev.adapters.storage.filesystem import WorkspacePaths, sha256_file
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.contracts.reports import ReportResponse
from tradingdev.app.report_service import ReportService
from tradingdev.app.trade_history_service import TradeHistoryService
from tradingdev.domain.ml.leaderboard import EvaluationContext, LeaderboardEntry
from tradingdev.domain.ml.registry import LeaderboardStore
from tradingdev.mcp.strict_server import StrictMCPServer
from tradingdev.mcp.tools import history


@pytest.fixture(params=[MCPServer, StrictMCPServer], ids=["sdk", "strict"])
def context(
    tmp_path: Path, request: pytest.FixtureRequest
) -> tuple[WorkspacePaths, SQLiteStore, MCPServer, ReportService]:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    reports = ReportService(workspace=workspace, store=store)
    server: MCPServer = request.param("ml-report-test")
    history.register(
        server, TradeHistoryService(workspace=workspace, store=store), reports
    )
    return workspace, store, server, reports


async def _call(server: MCPServer, arguments: dict[str, Any]) -> dict[str, Any]:
    result = await server.call_tool("generate_report", arguments)
    assert isinstance(result, CallToolResult)
    payload = result.structured_content
    assert isinstance(payload, dict)
    tool = next(
        tool for tool in await server.list_tools() if tool.name == "generate_report"
    )
    assert tool.output_schema is not None
    Draft202012Validator(tool.output_schema).validate(payload)
    response = payload.get("result", payload)
    assert isinstance(response, dict)
    return response


def _record(workspace: WorkspacePaths) -> list[str]:
    board = LeaderboardStore(workspace.ml_leaderboard)
    for entry_id, score in (("candidate", 0.2), ("unavailable", None)):
        board.record(
            LeaderboardEntry(
                entry_id=entry_id,
                label=f"候選 {entry_id}",
                context=EvaluationContext(
                    symbol="BTC/USDT",
                    timeframe="1h",
                    start_date="2024-01-01",
                    end_date="2024-02-01",
                    horizon=1,
                    window=20,
                    feature_set="returns",
                    code_fingerprint="recorded-feature-code",
                    data_fingerprint="recorded-data-content",
                ),
                metrics={"ic_mean": score, "beta": 1.2},
                n_observations=200,
                model_id=f"model-{entry_id}",
                dataset_id="recorded-dataset",
                signal_fingerprint=f"signal-{entry_id}",
            )
        )
    return ["candidate", "unavailable"]


def test_ml_report_schema_discovers_selection_and_limits(
    context: tuple[WorkspacePaths, SQLiteStore, MCPServer, ReportService],
) -> None:
    async def check() -> None:
        tool = next(
            tool
            for tool in await context[2].list_tools()
            if tool.name == "generate_report"
        )
        Draft202012Validator.check_schema(tool.input_schema)
        properties = tool.input_schema["properties"]
        assert properties["ml_ranking_metric"]["default"] == "ic_mean"
        assert properties["ml_allow_parameter_axes"]["default"] is False
        assert properties["ml_entry_ids"]["default"] is None
        Draft202012Validator(tool.input_schema).validate(
            {"ml_entry_ids": ["candidate"], "sections": ["ml"]}
        )
        assert tool.description is not None
        assert "leaderboard" in tool.description
        assert "evaluate_signal" in tool.description
        assert "1..50 distinct" in tool.description
        assert "not a backtest" in tool.description
        assert "final answer" in tool.description

    asyncio.run(check())


@pytest.mark.parametrize(
    "selection", [{}, {"run_ids": ["run"]}, {"comparison_job_id": "comparison"}]
)
def test_ml_report_tool_delegates_explicit_selection_and_preserves_summary(
    context: tuple[WorkspacePaths, SQLiteStore, MCPServer, ReportService],
    monkeypatch: pytest.MonkeyPatch,
    selection: dict[str, Any],
) -> None:
    response = {
        "success": True,
        "report_id": "report",
        "artifact_id": "html",
        "manifest_artifact_id": "manifest",
        "path": "報告.html",
        "sha256": "digest",
        "run_ids": selection.get("run_ids", []),
        "ml_entry_ids": ["candidate", "unavailable"],
        "comparison_job_id": selection.get("comparison_job_id"),
        "navigation_instructions": "查看 ML 候選與缺值。",
        "scope_count": 2,
    }

    def generate(run_ids: list[str] | None, **options: Any) -> dict[str, Any]:
        assert run_ids == selection.get("run_ids")
        assert options == {
            "sections": ["ml"],
            "commentary": [{"title": "觀察", "text": "保留缺值。"}],
            "analysis": None,
            "comparison_job_id": selection.get("comparison_job_id"),
            "ml_entry_ids": ["candidate", "unavailable"],
            "ml_ranking_metric": "rank_ic_mean",
            "ml_allow_parameter_axes": True,
        }
        return response

    monkeypatch.setattr(context[3], "generate_report", generate)
    payload = asyncio.run(
        _call(
            context[2],
            {
                **selection,
                "sections": ["ml"],
                "commentary": [{"title": "觀察", "text": "保留缺值。"}],
                "analysis": None,
                "ml_entry_ids": ["candidate", "unavailable"],
                "ml_ranking_metric": "rank_ic_mean",
                "ml_allow_parameter_axes": True,
            },
        )
    )
    assert payload == response
    assert "html" not in payload and "scopes" not in payload


def test_ml_report_tool_preserves_service_failures(
    context: tuple[WorkspacePaths, SQLiteStore, MCPServer, ReportService],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = {
        "success": False,
        "code": "ml_entry_not_found",
        "error": "Saved ML entry is unavailable",
    }

    def generate(_: list[str] | None, **options: Any) -> dict[str, Any]:
        assert options["ml_entry_ids"] == ["missing"]
        return response

    monkeypatch.setattr(context[3], "generate_report", generate)
    assert asyncio.run(_call(context[2], {"ml_entry_ids": ["missing"]})) == response


def test_ml_only_report_is_a_registered_artifact_through_mcp(
    context: tuple[WorkspacePaths, SQLiteStore, MCPServer, ReportService],
) -> None:
    workspace, store, server, _ = context
    entry_ids = _record(workspace)
    original = (workspace.ml_leaderboard / "leaderboard.jsonl").read_bytes()
    arguments = {"ml_entry_ids": entry_ids}
    report = ReportResponse.model_validate(asyncio.run(_call(server, arguments)))
    assert report.run_ids == []
    assert report.ml_entry_ids == entry_ids
    assert report.scope_count == 2
    path = Path(report.path)
    assert path.is_relative_to(workspace.root / "reports")
    assert sha256_file(path) == report.sha256
    content = path.read_text(encoding="utf-8")
    encoded = content.split("<script id='report-data' type='application/json'>", 1)[1]
    payload = json.loads(encoded.split("</script>", 1)[0])
    assert payload["runs"] == []
    assert payload["sections"] == ["ml"]
    scopes = payload["ml"]["scopes"]
    assert [scope["scope_id"] for scope in scopes] == entry_ids
    assert scopes[1]["values"]["ic_mean"] is None
    assert scopes[1]["observations"] is None
    assert "candidate" in content and "unavailable" in content
    assert "ML" in report.navigation_instructions
    assert store.get_artifact(report.artifact_id) is not None
    manifest_artifact = store.get_artifact(report.manifest_artifact_id)
    assert manifest_artifact is not None
    manifest = json.loads(Path(manifest_artifact["path"]).read_text(encoding="utf-8"))
    assert manifest["ml_entry_ids"] == entry_ids
    assert manifest["ml"] == payload["ml"]
    assert (
        ReportResponse.model_validate(asyncio.run(_call(server, arguments))) == report
    )
    assert (workspace.ml_leaderboard / "leaderboard.jsonl").read_bytes() == original


@pytest.mark.parametrize(
    "selection",
    [[], ["candidate", "candidate"], [f"candidate-{index}" for index in range(51)]],
)
def test_invalid_ml_selection_does_not_publish_a_report(
    context: tuple[WorkspacePaths, SQLiteStore, MCPServer, ReportService],
    selection: list[str],
) -> None:
    workspace, _, server, _ = context
    _record(workspace)
    result = asyncio.run(_call(server, {"ml_entry_ids": selection}))
    assert result["success"] is False
    assert result["code"] == "invalid_report_ml_selection"
    assert not list((workspace.root / "reports").glob("**/*.html"))


def test_unknown_ml_entry_does_not_publish_a_partial_report(
    context: tuple[WorkspacePaths, SQLiteStore, MCPServer, ReportService],
) -> None:
    workspace, _, server, _ = context
    _record(workspace)
    result = asyncio.run(_call(server, {"ml_entry_ids": ["candidate", "missing"]}))
    assert result["success"] is False
    assert result["code"] == "ml_entry_not_found"
    assert not list((workspace.root / "reports").glob("**/*.html"))
