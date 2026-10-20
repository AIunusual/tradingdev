"""leaderboard and record=True through in-process MCP dispatch with MCPServer."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
from mcp.server import MCPServer
from mcp_types import CallToolResult

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.contracts.ml import LeaderboardResponse, MlContractResponse
from tradingdev.app.ml_service import MLService
from tradingdev.mcp.tools import ml

if TYPE_CHECKING:
    from pathlib import Path

_ROWS = 300
_GOOD_MODEL = "a" * 16
_BAD_MODEL = "b" * 16


def _write_bars(workspace: WorkspacePaths, symbol: str) -> None:
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
    path = workspace.processed_data / (
        symbol.lower().replace("/", "") + "_1h_2024.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspacePaths:
    paths = WorkspacePaths(tmp_path / "workspace")
    paths.ensure()
    _write_bars(paths, "BTC/USDT")
    return paths


@pytest.fixture
def mcp(workspace: WorkspacePaths) -> MCPServer:
    server = MCPServer("test-ml")
    ml.register(server, MLService(workspace=workspace))
    return server


def _dispatch(mcp: MCPServer, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    async def call() -> dict[str, Any]:
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
def recorded(mcp: MCPServer, workspace: WorkspacePaths) -> str:
    """Build a dataset and record two evaluations of opposite quality."""
    built = _dispatch(
        mcp,
        "build_dataset",
        {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "feature_set": "returns",
        },
    )
    dataset_id = built["dataset"]["dataset_id"]
    frame = pd.read_parquet(workspace.ml_datasets / dataset_id / "dataset.parquet")
    realized = frame["forward_return"].astype(float).tolist()
    for model_id, quality in ((_BAD_MODEL, -1.0), (_GOOD_MODEL, 1.0)):
        _dispatch(
            mcp,
            "evaluate_signal",
            {
                "signal": [value * quality for value in realized],
                "forward_returns": realized,
                "window": 20,
                "record": True,
                "model_id": model_id,
                "dataset_id": dataset_id,
            },
        )
    return str(dataset_id)


def test_contract_lists_the_rankable_metrics(mcp: MCPServer) -> None:
    payload = _dispatch(mcp, "get_ml_contract", {})
    contract = MlContractResponse.model_validate(payload)
    assert "rank_icir" in contract.leaderboard_metrics
    assert "n_periods" not in contract.leaderboard_metrics


def test_evaluate_signal_record_returns_the_stored_entry(
    mcp: MCPServer, workspace: WorkspacePaths, recorded: str
) -> None:
    frame = pd.read_parquet(workspace.ml_datasets / recorded / "dataset.parquet")
    realized = frame["forward_return"].astype(float).tolist()
    payload = _dispatch(
        mcp,
        "evaluate_signal",
        {
            "signal": realized,
            "forward_returns": realized,
            "window": 20,
            "record": True,
            "dataset_id": recorded,
        },
    )
    assert payload["recorded"]["n_candidates"] == 3
    assert payload["recorded"]["replaced"] is False


def test_recording_a_signal_that_is_not_the_dataset_is_refused(
    mcp: MCPServer, recorded: str
) -> None:
    """A record must not be able to claim a dataset it was not measured on."""
    payload = _dispatch(
        mcp,
        "evaluate_signal",
        {
            "signal": [0.1, -0.2, 0.3, -0.4, 0.5] * 40,
            "forward_returns": [0.1, -0.2, 0.3, -0.4, 0.5] * 40,
            "window": 20,
            "record": True,
            "dataset_id": recorded,
        },
    )
    assert payload["success"] is False
    assert "would not describe this dataset" in payload["error"]


def test_recording_with_a_wrong_horizon_is_refused(
    mcp: MCPServer, workspace: WorkspacePaths, recorded: str
) -> None:
    frame = pd.read_parquet(workspace.ml_datasets / recorded / "dataset.parquet")
    realized = frame["forward_return"].astype(float).tolist()
    payload = _dispatch(
        mcp,
        "evaluate_signal",
        {
            "signal": realized,
            "forward_returns": realized,
            "horizon": 7,
            "window": 20,
            "record": True,
            "dataset_id": recorded,
        },
    )
    assert payload["success"] is False
    assert "does not match the dataset's horizon" in payload["error"]


def test_leaderboard_ranks_the_recorded_candidates(
    mcp: MCPServer, recorded: str
) -> None:
    payload = _dispatch(
        mcp, "leaderboard", {"metric": "ic_mean", "dataset_id": recorded}
    )
    board = LeaderboardResponse.model_validate(payload)
    assert board.n_candidates == 2
    assert [row.model_id for row in board.ranked] == [_GOOD_MODEL, _BAD_MODEL]
    assert board.ranked[0].rank == 1
    assert board.caveats


def test_leaderboard_can_query_old_and_rebuilt_dataset_versions(
    mcp: MCPServer, workspace: WorkspacePaths
) -> None:
    records: dict[str, str] = {}
    for version in (1, 2):
        if version == 2:
            source = workspace.processed_data / "btcusdt_1h_2024.parquet"
            frame = pd.read_parquet(source)
            frame["close"] *= np.linspace(1.0, 1.2, len(frame))
            frame.to_parquet(source, index=False)
        built = _dispatch(
            mcp,
            "build_dataset",
            {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "feature_set": "returns",
            },
        )
        assert built["success"] is True
        dataset_id = built["dataset"]["dataset_id"]
        frame = pd.read_parquet(built["dataset"]["path"])
        realized = frame["forward_return"].astype(float).tolist()
        evaluated = _dispatch(
            mcp,
            "evaluate_signal",
            {
                "signal": realized,
                "forward_returns": realized,
                "window": 20,
                "record": True,
                "dataset_id": dataset_id,
                "model_id": _GOOD_MODEL,
            },
        )
        assert evaluated["success"] is True
        records[dataset_id] = evaluated["recorded"]["entry_id"]
    assert len(records) == 2
    unfiltered = _dispatch(mcp, "leaderboard", {"metric": "ic_mean"})
    assert unfiltered["success"] is False
    assert unfiltered["code"] == "invalid_leaderboard_request"
    for dataset_id, entry_id in records.items():
        board = LeaderboardResponse.model_validate(
            _dispatch(
                mcp, "leaderboard", {"metric": "ic_mean", "dataset_id": dataset_id}
            )
        )
        assert board.n_candidates == board.n_ranked == 1
        assert board.ranked[0].dataset_id == dataset_id
        assert board.ranked[0].entry_id == entry_id


@pytest.mark.parametrize("dataset_id", ["", "invalid", "ds_" + "0" * 16])
def test_leaderboard_dataset_filter_refuses_invalid_ids(
    mcp: MCPServer, recorded: str, dataset_id: str
) -> None:
    payload = _dispatch(
        mcp,
        "leaderboard",
        {"metric": "ic_mean", "dataset_id": dataset_id, "allow_mixed": True},
    )
    assert payload["success"] is False
    assert payload["code"] == "invalid_leaderboard_request"
    assert "dataset_id" in payload["error"]


def test_leaderboard_refuses_a_metric_without_a_direction(
    mcp: MCPServer, recorded: str
) -> None:
    payload = _dispatch(mcp, "leaderboard", {"metric": "n_periods"})
    assert payload["success"] is False
    assert "no optimisation direction" in payload["error"]


def test_leaderboard_reports_an_empty_store(mcp: MCPServer) -> None:
    payload = _dispatch(mcp, "leaderboard", {"metric": "ic_mean"})
    assert payload["success"] is False
    assert "no recorded evaluations" in payload["error"]


def test_leaderboard_is_advertised_read_only(mcp: MCPServer) -> None:
    async def tools() -> list[Any]:
        return list(await mcp.list_tools())

    entries = {tool.name: tool for tool in asyncio.run(tools())}
    assert entries["leaderboard"].annotations is not None
    assert entries["leaderboard"].annotations.read_only_hint is True
    assert "dataset_id" in entries["leaderboard"].input_schema["properties"]
    assert "dataset_id" not in entries["leaderboard"].input_schema["required"]
