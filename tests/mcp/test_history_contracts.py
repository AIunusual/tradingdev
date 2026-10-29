"""History/report contracts through real MCP dispatch and persisted observations."""

from __future__ import annotations

import asyncio
import base64
import csv
import io
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from jsonschema import Draft202012Validator
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError
from mcp_types import CallToolResult
from pydantic import ValidationError

from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths, sha256_file
from tradingdev.adapters.storage.performance import PerformanceStore
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.contracts.common import ErrorResponse
from tradingdev.app.contracts.history import (
    FindRunsResponse,
    HistoryQueryError,
    RunAccountHistoryResponse,
    RunEquityResponse,
    RunExecutionsResponse,
    RunTradesResponse,
)
from tradingdev.app.contracts.reports import ReportResponse, ReportSectionCatalog
from tradingdev.app.report_service import ReportService
from tradingdev.app.trade_history_service import TradeHistoryService
from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.performance.artifacts import (
    PerformanceScope,
    ScopeObservations,
    build_artifacts,
)
from tradingdev.domain.strategies.execution import StrategyExecution
from tradingdev.mcp.strict_server import StrictMCPServer
from tradingdev.mcp.tools import history

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(params=[MCPServer, StrictMCPServer], ids=["sdk", "strict"])
def context(
    tmp_path: Path, request: pytest.FixtureRequest
) -> Iterator[
    tuple[WorkspacePaths, SQLiteStore, MCPServer, TradeHistoryService, ReportService]
]:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    service = TradeHistoryService(workspace=workspace, store=store)
    reports = ReportService(workspace=workspace, store=store)
    server: MCPServer = request.param("history-contract-test")
    history.register(server, service, reports)
    _publish(workspace, store)
    yield workspace, store, server, service, reports


def _publish(
    workspace: WorkspacePaths,
    store: SQLiteStore,
    *,
    run_id: str = "run",
    missing_entry: bool = False,
    timestamps: bool = True,
) -> None:
    params = {"fast": 12, "slow": 26, "nested": {"signal": 9}}
    config = {
        "strategy": {"id": "macd_fixture", "parameters": {"fast": 12}},
        "backtest": {
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "start_date": "2024-01-01",
            "end_date": "2024-01-03",
            "init_cash": 1000.0,
            "fees": 0.0004,
            "slippage": 0.0,
        },
    }
    manifest = ExecutionManifest.create(
        kind="backtest",
        config=config,
        strategy_execution=StrategyExecution(
            kind="generated", constructor_kwargs=params
        ),
    )
    path = ExecutionManifestStore(workspace).publish(run_id, manifest)
    store.create_run(
        run_id=run_id,
        job_id="job",
        strategy_id="macd_fixture",
        artifact_dir=path.parent,
        manifest_hash=manifest.manifest_hash,
        metrics={"total_pnl": 15.0},
        dataset_id="historical-data-fingerprint",
    )
    store.create_artifact(
        artifact_id=f"{run_id}:execution_manifest",
        run_id=run_id,
        artifact_type="execution_manifest",
        path=path,
        sha256=sha256_file(path),
    )
    scope = PerformanceScope.model_validate(
        {
            "mode": "signal",
            "values": {"total_pnl": 15.0},
            "parameters": params,
            "metadata": {"execution_context": config["backtest"]},
        }
    )
    observations = ScopeObservations(
        init_cash=1000.0,
        equity_curve=[1000.0, 1010.0, 1015.0],
        returns=[0.0, 0.01, 0.004950495],
        timestamps=[
            "2024-01-01T00:00:00Z",
            "2024-01-02T00:00:00Z",
            "2024-01-03T00:00:00Z",
        ]
        if timestamps
        else None,
        trades=[
            {
                "entry_idx": 0,
                "exit_idx": 1,
                "direction": 1,
                "status": "closed",
                "size": 0.01,
                "entry_price": 100.0,
                "exit_price": 110.0,
                "entry_fees": 0.1,
                "exit_fees": 0.11,
                "fee": 0.21,
                "net_pnl": 0.79,
            },
            {
                "entry_idx": 1,
                "exit_idx": 2,
                "direction": -1,
                "status": "open",
                "size": 0.02,
                "entry_price": 110.0,
                "exit_price": 105.0,
                "entry_fees": 0.11,
                "exit_fees": 0.0,
                "fee": 0.11,
                "net_pnl": -0.01,
            },
        ],
    )
    if missing_entry:
        observations.trades[0].pop("entry_idx")
    PerformanceStore(workspace, store).publish(
        build_artifacts(
            run_id,
            manifest.manifest_hash,
            "full",
            {"full": scope},
            {"full": observations},
        )
    )


async def _call(
    server: MCPServer, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Validate the actual structured MCP payload against its advertised schema."""
    result = await server.call_tool(name, arguments)
    assert isinstance(result, CallToolResult)
    assert result.structured_content is not None
    payload = result.structured_content
    assert isinstance(payload, dict)
    tool = next(tool for tool in await server.list_tools() if tool.name == name)
    assert tool.output_schema is not None
    Draft202012Validator(tool.output_schema).validate(payload)
    response = payload.get("result", payload)
    assert isinstance(response, dict)
    return response


def test_history_tool_schemas_and_real_paginated_queries(
    context: tuple[
        WorkspacePaths, SQLiteStore, MCPServer, TradeHistoryService, ReportService
    ],
) -> None:
    _, _, server, _, _ = context

    async def check() -> None:
        tools = {tool.name: tool for tool in await server.list_tools()}
        assert set(tools) == {
            "find_runs",
            "get_run_trades",
            "get_run_equity",
            "get_run_executions",
            "get_run_account_history",
            "get_report_sections",
            "get_run_diagnostics",
            "generate_report",
        }
        for name, tool in tools.items():
            Draft202012Validator.check_schema(tool.input_schema)
            assert tool.output_schema is not None
            Draft202012Validator.check_schema(tool.output_schema)
            if isinstance(server, StrictMCPServer):
                assert tool.input_schema["additionalProperties"] is False
            assert tool.annotations is not None
            assert tool.annotations.read_only_hint is (name != "generate_report")
            assert tool.annotations.destructive_hint is False
            assert tool.annotations.open_world_hint is False
        assert "run_id" in tools["get_run_trades"].input_schema["required"]
        assert "scope" not in tools["get_run_trades"].input_schema["required"]
        # Strategy parameter names are dynamic even though top-level names are closed.
        Draft202012Validator(tools["find_runs"].input_schema).validate(
            {"parameters": {"user_parameter": {"leaf": 7}}}
        )
        found = FindRunsResponse.model_validate(
            await _call(
                server,
                "find_runs",
                {
                    "strategy_id": "macd_fixture",
                    "parameters": {"nested": {"signal": 9}},
                    "symbol": "BTC/USDT",
                    "timeframe": "1d",
                    "limit": 1,
                },
            )
        )
        assert found.complete and found.matched == 1
        assert found.runs[0].parameters == {
            "fast": 12,
            "slow": 26,
            "nested": {"signal": 9},
        }
        first = RunTradesResponse.model_validate(
            await _call(server, "get_run_trades", {"run_id": "run", "limit": 1})
        )
        assert first.next_offset == 1 and first.trades[0].trade_id == 0
        second = RunTradesResponse.model_validate(
            await _call(
                server, "get_run_trades", {"run_id": "run", "offset": 1, "limit": 1}
            )
        )
        assert second.trades[0].trade_id == 1 and second.next_offset is None
        selected = RunTradesResponse.model_validate(
            await _call(
                server,
                "get_run_trades",
                {
                    "run_id": "run",
                    "scope": "full",
                    "status": "open",
                    "direction": "short",
                    "entry_start": "2024-01-02",
                    "entry_end": "2024-01-02",
                },
            )
        )
        assert selected.matched == 1 and selected.total == 2
        assert selected.trades[0].exit_timestamp is None
        assert selected.trades[0].exit_price is None
        assert selected.trades[0].mark_timestamp == "2024-01-03T00:00:00+00:00"
        assert selected.trades[0].mark_price == 105.0
        assert selected.trades[0].record["exit_price"] == 105.0
        equity = RunEquityResponse.model_validate(
            await _call(
                server,
                "get_run_equity",
                {
                    "run_id": "run",
                    "start": "2024-01-02",
                    "end": "2024-01-03",
                    "limit": 1,
                },
            )
        )
        assert equity.init_cash == 1000 and equity.equity_basis == "account_equity"
        assert equity.total == 3 and equity.matched == 2 and equity.next_offset == 1
        assert equity.points[0].bar_index == 1 and equity.points[0].equity == 1010.0
        executions = RunExecutionsResponse.model_validate(
            await _call(server, "get_run_executions", {"run_id": "run"})
        )
        assert executions.availability == "not_recorded"
        assert executions.records == [] and executions.total == 0
        accounts = RunAccountHistoryResponse.model_validate(
            await _call(server, "get_run_account_history", {"run_id": "run"})
        )
        assert accounts.availability == "not_recorded"
        assert accounts.states == [] and accounts.total == 0

    asyncio.run(check())


def test_trade_date_filter_excludes_individual_unknown_times_through_mcp(
    context: tuple[
        WorkspacePaths, SQLiteStore, MCPServer, TradeHistoryService, ReportService
    ],
) -> None:
    workspace, store, server, _, _ = context
    _publish(workspace, store, run_id="mixed", missing_entry=True)
    _publish(workspace, store, run_id="missing_series", timestamps=False)

    async def check() -> None:
        unfiltered = RunTradesResponse.model_validate(
            await _call(server, "get_run_trades", {"run_id": "mixed"})
        )
        assert unfiltered.total == unfiltered.matched == 2
        assert [trade.trade_id for trade in unfiltered.trades] == [0, 1]
        assert unfiltered.trades[0].entry_timestamp is None
        selected = RunTradesResponse.model_validate(
            await _call(
                server,
                "get_run_trades",
                {
                    "run_id": "mixed",
                    "entry_start": "2024-01-02",
                    "entry_end": "2024-01-02",
                },
            )
        )
        assert (selected.total, selected.matched, selected.next_offset) == (2, 1, None)
        assert [trade.trade_id for trade in selected.trades] == [1]
        assert selected.trades[0].entry_timestamp == "2024-01-02T00:00:00+00:00"
        missing = HistoryQueryError.model_validate(
            await _call(
                server,
                "get_run_trades",
                {"run_id": "missing_series", "entry_start": "2024-01-01"},
            )
        )
        assert missing.code == "timestamps_unavailable"

    asyncio.run(check())


def test_report_catalogue_and_composition_are_real_reusable_artifacts(
    context: tuple[
        WorkspacePaths, SQLiteStore, MCPServer, TradeHistoryService, ReportService
    ],
) -> None:
    workspace, store, server, _, _ = context
    before = {
        path: path.read_bytes() for path in (workspace.runs / "run").glob("*.json")
    }

    async def check() -> None:
        catalog = ReportSectionCatalog.model_validate(
            await _call(server, "get_report_sections", {})
        )
        assert [section.id for section in catalog.sections] == [
            "overview",
            "research",
            "settings",
            "metrics",
            "benchmark",
            "equity",
            "ledger",
            "trades",
            "limitations",
            "provenance",
            "ml",
        ]
        assert catalog.templates["standard"] == [
            section.id for section in catalog.sections if section.id != "ml"
        ]
        assert catalog.templates["ml_research"] == ["ml"]
        assert catalog.templates["trades"] == [
            "overview",
            "research",
            "ledger",
            "trades",
            "provenance",
        ]
        args = {
            "run_ids": ["run"],
            "sections": ["overview", "settings", "metrics", "trades"],
            "commentary": [
                {"title": "使用者觀察", "text": "<script>unsafe()</script> 僅為解讀"}
            ],
        }
        report = ReportResponse.model_validate(
            await _call(server, "generate_report", args)
        )
        assert report.scope_count == 1 and report.run_ids == ["run"]
        path = Path(report.path)
        assert path.is_relative_to(workspace.root / "reports")
        assert path.is_file() and report.sha256 == sha256_file(path)
        assert "&lt;script&gt;unsafe()&lt;/script&gt;" in path.read_text()
        assert "<script>unsafe()</script>" not in path.read_text()
        assert "<th>進場時間（UTC）</th>" in path.read_text()
        assert "<th>單邊手續費率</th>" in path.read_text()
        assert "<th>總淨損益</th>" in path.read_text()
        assert "<th>執行識別碼</th>" in path.read_text()
        encoded = re.findall(
            r"data-table-csv='trades-0-0' data-csv='([^']*)'", path.read_text()
        )
        assert len(encoded) == 1
        rows = list(
            csv.DictReader(
                io.StringIO(base64.b64decode(encoded[0]).decode("utf-8-sig"))
            )
        )
        assert rows and "淨損益（報價貨幣）" in rows[0]
        assert "net_pnl" not in rows[0]
        assert list(rows[0])[:4] == [
            "序號",
            "進場時間（UTC）",
            "平倉時間（UTC）",
            "期末估值時間（UTC）",
        ]
        assert "<th>淨損益（報價貨幣）</th>" in path.read_text()
        assert "data-table-json='0:0:trades'" in path.read_text()
        assert "href='#result-0-0'" in path.read_text()
        assert "href='#overview'" in path.read_text()
        assert store.get_artifact(report.artifact_id) is not None
        assert store.get_artifact(report.manifest_artifact_id) is not None
        again = ReportResponse.model_validate(
            await _call(server, "generate_report", args)
        )
        assert again == report
        commentary_only = ReportResponse.model_validate(
            await _call(server, "generate_report", {"run_ids": ["run"], "sections": []})
        )
        assert commentary_only.report_id != report.report_id

    asyncio.run(check())
    assert before == {path: path.read_bytes() for path in before}


@pytest.mark.parametrize(
    ("name", "arguments", "code"),
    [
        ("find_runs", {"limit": 501}, "invalid_history_query"),
        ("get_run_trades", {"run_id": "run", "offset": -1}, "invalid_history_query"),
        (
            "get_run_trades",
            {"run_id": "run", "entry_start": "yesterday"},
            "invalid_history_query",
        ),
        (
            "get_run_trades",
            {"run_id": "run", "scope": "made_up"},
            "unknown_history_scope",
        ),
        ("get_run_equity", {"run_id": "run", "limit": 0}, "invalid_history_query"),
        (
            "get_run_equity",
            {"run_id": "run", "start": "2025-01-01", "end": "2024-01-01"},
            "invalid_history_query",
        ),
        ("get_run_equity", {"run_id": "missing"}, "run_not_found"),
        ("get_run_executions", {"run_id": "missing"}, "run_not_found"),
        ("get_run_account_history", {"run_id": "missing"}, "run_not_found"),
        (
            "get_run_executions",
            {"run_id": "run", "limit": 501},
            "invalid_history_query",
        ),
        (
            "get_run_account_history",
            {"run_id": "run", "offset": -1},
            "invalid_history_query",
        ),
        ("generate_report", {"run_ids": []}, "invalid_report_runs"),
        (
            "generate_report",
            {"run_ids": ["run"], "sections": ["missing"]},
            "invalid_report_sections",
        ),
        (
            "generate_report",
            {"run_ids": ["run"], "sections": ["executions"]},
            "invalid_report_sections",
        ),
        (
            "generate_report",
            {"run_ids": ["run"], "sections": ["account_history"]},
            "invalid_report_sections",
        ),
        (
            "generate_report",
            {"run_ids": ["run"], "commentary": [{"title": "", "text": "x"}]},
            "invalid_report_commentary",
        ),
        ("generate_report", {"run_ids": ["missing"]}, "run_not_found"),
    ],
)
def test_expected_application_errors_match_advertised_union(
    context: tuple[
        WorkspacePaths, SQLiteStore, MCPServer, TradeHistoryService, ReportService
    ],
    name: str,
    arguments: dict[str, Any],
    code: str,
) -> None:
    response = asyncio.run(_call(context[2], name, arguments))
    parsed = (
        ErrorResponse.model_validate(response)
        if name == "generate_report"
        else HistoryQueryError.model_validate(response)
    )
    assert parsed.code == code
    if code == "unknown_history_scope":
        assert response["available_scopes"] == ["full"]


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("get_run_trades", {}),
        ("get_run_trades", {"run_id": "run", "direction": "both"}),
        ("get_run_trades", {"run_id": "run", "status": "filled"}),
        ("get_run_equity", {"run_id": "run", "limit": 1.5}),
        ("get_run_executions", {"run_id": "run", "status": "open"}),
        ("get_run_executions", {"run_id": "run", "side": "long"}),
        ("get_run_account_history", {}),
        ("generate_report", {"run_ids": "run"}),
        (
            "generate_report",
            {
                "run_ids": ["run"],
                "commentary": [{"title": "x", "text": "y", "html": True}],
            },
        ),
    ],
)
def test_invalid_typed_arguments_are_rejected_before_service_dispatch(
    context: tuple[
        WorkspacePaths, SQLiteStore, MCPServer, TradeHistoryService, ReportService
    ],
    name: str,
    arguments: dict[str, Any],
) -> None:
    with pytest.raises(ToolError):
        asyncio.run(context[2].call_tool(name, arguments))
    assert not (context[0].root / "reports").exists()


def test_strict_boundary_rejects_unknown_names_for_every_history_tool(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    server = StrictMCPServer("history-unknown-arguments")
    history.register(
        server,
        TradeHistoryService(workspace=workspace, store=store),
        ReportService(workspace=workspace, store=store),
    )

    async def check() -> None:
        for tool in await server.list_tools():
            with pytest.raises(ToolError, match="Unknown arguments.*unexpected"):
                await server.call_tool(tool.name, {"unexpected": True})

    asyncio.run(check())


@pytest.mark.parametrize(
    "name",
    [
        "find_runs",
        "get_run_trades",
        "get_run_equity",
        "get_run_executions",
        "get_run_account_history",
        "get_report_sections",
        "generate_report",
    ],
)
@pytest.mark.parametrize(
    "broken", [{"success": True}, {"success": False, "error": "missing code"}]
)
def test_service_schema_drift_is_not_returned_as_an_unvalidated_payload(
    context: tuple[
        WorkspacePaths, SQLiteStore, MCPServer, TradeHistoryService, ReportService
    ],
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    broken: dict[str, Any],
) -> None:
    _, _, server, service, reports = context
    target = reports if name in {"generate_report", "get_report_sections"} else service
    monkeypatch.setattr(target, name, lambda *_args, **_kwargs: broken)
    arguments: dict[str, Any] = {}
    if name in {
        "get_run_trades",
        "get_run_equity",
        "get_run_executions",
        "get_run_account_history",
    }:
        arguments["run_id"] = "run"
    if name == "generate_report":
        arguments["run_ids"] = ["run"]
    with pytest.raises(
        UnexpectedToolError, match=f"^Error executing tool {name}$"
    ) as caught:
        asyncio.run(server.call_tool(name, arguments))
    assert isinstance(caught.value.__cause__, ValidationError)
