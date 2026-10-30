"""build_panel through in-process MCP dispatch with MCPServer."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from mcp.server import MCPServer
from mcp_types import CallToolResult
from tests.data_cache import seed_market_cache

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.contracts.ml import BuildPanelResponse, MlContractResponse
from tradingdev.app.ml_service import MLService
from tradingdev.mcp.tools import ml

if TYPE_CHECKING:
    from pathlib import Path

_ROWS = 300
_HOLE = range(100, 124)


def _write_bars(
    workspace: WorkspacePaths, symbol: str, *, drop: range | None = None
) -> None:
    skip = set(drop or ())
    index = np.arange(_ROWS, dtype=float)
    close = 100.0 + 10.0 * np.sin(index / 5.0) + index * 0.01
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=_ROWS, freq="h", tz="UTC"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(_ROWS, 10.0),
        }
    )
    keep = [i for i in range(_ROWS) if i not in skip]
    seed_market_cache(
        workspace.processed_data, frame.iloc[keep].reset_index(drop=True), symbol=symbol
    )


def _dispatch(
    mcp: MCPServer, name: str, arguments: dict[str, object]
) -> dict[str, object]:
    async def call() -> dict[str, object]:
        result = await mcp.call_tool(name, arguments)
        assert isinstance(result, CallToolResult)
        assert result.structured_content is not None
        payload = result.structured_content
        assert isinstance(payload, dict)
        response = payload.get("result", payload)
        assert isinstance(response, dict)
        return response

    return asyncio.run(call())


@pytest.fixture
def service(tmp_path: Path) -> MLService:
    paths = WorkspacePaths(tmp_path / "workspace")
    paths.ensure()
    _write_bars(paths, "BTC/USDT")
    _write_bars(paths, "ETH/USDT", drop=_HOLE)
    return MLService(workspace=paths)


def _arguments(**overrides: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "symbols": ["BTC/USDT", "ETH/USDT"],
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "feature_set": "returns",
        "horizon": 1,
    }
    arguments.update(overrides)
    return arguments


def test_build_panel_returns_a_typed_contract(service: MLService) -> None:
    mcp = MCPServer("ml-panel-test")
    ml.register(mcp, service)
    response = BuildPanelResponse.model_validate(
        _dispatch(mcp, "build_panel", _arguments())
    )
    assert response.success is True
    assert response.panel.n_symbols == 2
    assert response.panel.rows == response.panel.n_symbols * response.panel.n_dates
    assert response.panel.calendar.mode == "intersection"
    assert response.panel.label_column == "target"


def test_build_panel_reports_the_forward_fill_audit(service: MLService) -> None:
    mcp = MCPServer("ml-panel-ffill-test")
    ml.register(mcp, service)
    response = BuildPanelResponse.model_validate(
        _dispatch(
            mcp,
            "build_panel",
            _arguments(calendar_mode="union", missing_policy="forward_fill"),
        )
    )
    assert response.panel.calendar.missing_policy == "forward_fill"
    assert response.panel.calendar.forward_filled["ETH/USDT"] == len(_HOLE)
    assert response.panel.calendar.n_dropped_dates == 0


def test_build_panel_rejects_a_gap_under_the_error_policy(service: MLService) -> None:
    mcp = MCPServer("ml-panel-error-test")
    ml.register(mcp, service)
    response = _dispatch(
        mcp, "build_panel", _arguments(calendar_mode="union", missing_policy="error")
    )
    assert response["success"] is False
    assert response["code"] == "invalid_panel_request"
    assert isinstance(response["error"], str)
    assert "no bar on" in response["error"]


def test_build_panel_rejects_a_single_symbol(service: MLService) -> None:
    mcp = MCPServer("ml-panel-single-test")
    ml.register(mcp, service)
    response = _dispatch(mcp, "build_panel", _arguments(symbols=["BTC/USDT"]))
    assert response["success"] is False
    assert isinstance(response["error"], str)
    assert "at least two distinct symbols" in response["error"]


def test_contract_enumerates_the_calendar_options(tmp_path: Path) -> None:
    mcp = MCPServer("ml-contract-calendar-test")
    ml.register(mcp, MLService(WorkspacePaths(tmp_path / "workspace")))
    contract = MlContractResponse.model_validate(_dispatch(mcp, "get_ml_contract", {}))
    assert contract.calendar_modes == ["intersection", "union"]
    assert contract.missing_policies == ["drop", "forward_fill", "error"]
