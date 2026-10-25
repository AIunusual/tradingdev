"""leaderboard and record=True through in-process MCP dispatch with MCPServer."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
from mcp.server import MCPServer
from mcp_types import CallToolResult
from tests.mcp.ml_fixtures import model_pair

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.contracts.ml import LeaderboardPageResponse, MlContractResponse
from tradingdev.app.ml_service import MLService
from tradingdev.domain.ml.leaderboard import EvaluationContext, LeaderboardEntry
from tradingdev.domain.ml.registry import LeaderboardStore
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
    service = MLService(workspace=workspace)
    for model_id, quality in ((_BAD_MODEL, -1.0), (_GOOD_MODEL, 1.0)):
        service.evaluate_signal(
            [
                float((value + 0.002 * np.sin(index)) * quality)
                for index, value in enumerate(realized)
            ],
            realized,
            window=20,
            record=True,
            model_id=model_id,
            dataset_id=dataset_id,
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
    pair = model_pair(MLService(workspace=workspace))
    assert pair["dataset_id"] == recorded
    arguments: dict[str, Any] = {**pair, "window": 20, "record": True}
    payload = _dispatch(
        mcp,
        "evaluate_signal",
        arguments,
    )
    assert payload["success"] is True
    assert payload["model_id"] == pair["model_id"]
    assert payload["dataset_id"] == pair["dataset_id"]
    assert payload["recorded"]["n_candidates"] == 3
    assert payload["recorded"]["replaced"] is False
    repeated = _dispatch(mcp, "evaluate_signal", arguments)
    assert repeated["recorded"]["entry_id"] == payload["recorded"]["entry_id"]
    assert repeated["recorded"]["replaced"] is True
    assert repeated["recorded"]["n_candidates"] == 3


def test_leaderboard_pages_ranked_and_skipped_without_losing_total_counts(
    mcp: MCPServer, workspace: WorkspacePaths, recorded: str
) -> None:
    """Unavailable metrics are still discoverable after ranked rows run out."""
    service = MLService(workspace=workspace)
    frame = pd.read_parquet(workspace.ml_datasets / recorded / "dataset.parquet")
    realized = frame["forward_return"].astype(float).tolist()
    skipped_ids = set()
    for label in ("c", "d", "e"):
        # Perfect rank correlation has zero variance, so Rank ICIR is unavailable.
        result = service.evaluate_signal(
            realized,
            realized,
            window=20,
            record=True,
            model_id=label * 16,
            dataset_id=recorded,
        )
        skipped_ids.add(result["recorded"]["entry_id"])
    ranked_models = []
    received_skipped = set()
    for offset in range(3):
        board = LeaderboardPageResponse.model_validate(
            _dispatch(
                mcp,
                "leaderboard",
                {
                    "metric": "rank_icir",
                    "dataset_id": recorded,
                    "offset": offset,
                    "limit": 1,
                },
            )
        )
        assert board.n_candidates == 5
        assert board.n_ranked == 2 and board.n_skipped == 3
        assert board.offset == offset and board.limit == 1
        assert len(board.skipped) == 1 and board.skipped[0].reason
        assert board.next_offset == (offset + 1 if offset < 2 else None)
        assert board.caveats
        received_skipped.add(board.skipped[0].entry_id)
        if offset < 2:
            assert len(board.ranked) == 1
            assert board.ranked[0].rank == offset + 1
            ranked_models.append(board.ranked[0].model_id)
        else:
            assert board.ranked == []
    assert ranked_models == [_GOOD_MODEL, _BAD_MODEL]
    assert received_skipped == skipped_ids


@pytest.mark.parametrize("offset,expected_models", [(1, [_BAD_MODEL]), (2, [])])
def test_leaderboard_last_page_and_exact_end_keep_totals(
    mcp: MCPServer, recorded: str, offset: int, expected_models: list[str]
) -> None:
    board = LeaderboardPageResponse.model_validate(
        _dispatch(
            mcp,
            "leaderboard",
            {"metric": "ic_mean", "dataset_id": recorded, "offset": offset, "limit": 1},
        )
    )
    assert [row.model_id for row in board.ranked] == expected_models
    assert board.n_candidates == board.n_ranked == 2
    assert board.n_skipped == 0 and board.skipped == []
    assert board.next_offset is None


@pytest.mark.parametrize(
    "page_options",
    [{"offset": -1}, {"offset": 3}, {"limit": 0}, {"limit": -1}, {"limit": 101}],
)
def test_leaderboard_rejects_invalid_page_requests(
    mcp: MCPServer, recorded: str, page_options: dict[str, int]
) -> None:
    payload = _dispatch(
        mcp,
        "leaderboard",
        {"metric": "ic_mean", "dataset_id": recorded, **page_options},
    )
    assert payload["success"] is False
    assert payload["code"] == "invalid_leaderboard_request"
    assert "ranked" not in payload and "skipped" not in payload


def test_leaderboard_ranks_the_recorded_candidates(
    mcp: MCPServer, recorded: str
) -> None:
    payload = _dispatch(
        mcp, "leaderboard", {"metric": "ic_mean", "dataset_id": recorded}
    )
    board = LeaderboardPageResponse.model_validate(payload)
    assert board.n_candidates == 2
    assert [row.model_id for row in board.ranked] == [_GOOD_MODEL, _BAD_MODEL]
    assert board.ranked[0].rank == 1
    assert board.offset == 0 and board.limit == 20 and board.next_offset is None
    assert board.n_ranked == 2 and board.n_skipped == 0
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
        evaluated = MLService(workspace=workspace).evaluate_signal(
            realized,
            realized,
            window=20,
            record=True,
            dataset_id=dataset_id,
            model_id=_GOOD_MODEL,
        )
        assert evaluated["success"] is True
        records[dataset_id] = evaluated["recorded"]["entry_id"]
    assert len(records) == 2
    unfiltered = _dispatch(mcp, "leaderboard", {"metric": "ic_mean"})
    assert unfiltered["success"] is False
    assert unfiltered["code"] == "invalid_leaderboard_request"
    for dataset_id, entry_id in records.items():
        board = LeaderboardPageResponse.model_validate(
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


@pytest.mark.parametrize("n_contexts", [10, 100, 1000])
def test_large_mixed_leaderboard_bounds_success_and_rejection(
    mcp: MCPServer, workspace: WorkspacePaths, n_contexts: int
) -> None:
    entries = [
        LeaderboardEntry(
            entry_id=f"{index:016x}",
            label=f"model-{index}",
            context=EvaluationContext(
                f"S{index:04d}/USDT", "1h", "2024-01-01", "2024-07-01", 1
            ),
            metrics={"ic_mean": 0.5, "ic_t_stat": 4.0, "n_periods": 100.0},
            n_observations=1000,
            model_id=f"{index:016x}",
            dataset_id=f"ds_{index:016x}",
        )
        for index in range(n_contexts)
    ]
    store = LeaderboardStore(workspace.ml_leaderboard)
    store.path.write_text(
        "".join(json.dumps(entry.to_record()) + "\n" for entry in entries),
        encoding="utf-8",
    )
    rejected = _dispatch(mcp, "leaderboard", {"metric": "ic_mean", "limit": 1})
    assert rejected["success"] is False
    assert rejected["code"] == "invalid_leaderboard_request"
    assert f"{n_contexts} distinct contexts" in rejected["error"]
    assert f"{n_contexts - 3} omitted" in rejected["error"]
    assert "showing 3 examples" in rejected["error"]
    assert "dataset_id" in rejected["error"] and "Filter by" in rejected["error"]
    assert len(json.dumps(rejected)) < 2000

    allowed = _dispatch(
        mcp, "leaderboard", {"metric": "ic_mean", "limit": 1, "allow_mixed": True}
    )
    board = LeaderboardPageResponse.model_validate(allowed)
    assert board.n_candidates == board.n_ranked == n_contexts
    assert board.n_skipped == 0 and len(board.ranked) == 1
    assert board.context == "mixed" and board.next_offset == 1
    assert board.caveats
    assert len(json.dumps(allowed)) < 2000

    filtered = LeaderboardPageResponse.model_validate(
        _dispatch(
            mcp,
            "leaderboard",
            {"metric": "ic_mean", "limit": 1, "symbol": "S0000/USDT"},
        )
    )
    assert filtered.n_candidates == filtered.n_ranked == 1
    assert filtered.next_offset is None
    assert filtered.ranked[0].entry_id == entries[0].entry_id
