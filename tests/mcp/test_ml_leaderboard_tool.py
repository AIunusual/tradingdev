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
from tests.data_cache import cached_market_path, seed_market_cache
from tests.mcp.ml_fixtures import model_pair

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.contracts.ml import (
    LeaderboardPageResponse,
    LeaderboardRow,
    LeaderboardSkipped,
    MlContractResponse,
)
from tradingdev.app.ml_service import MLService
from tradingdev.domain.ml.leaderboard import (
    EvaluationContext,
    LeaderboardEntry,
    ProducerSource,
)
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
    seed_market_cache(workspace.processed_data, frame, symbol=symbol)


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


def _signal_rows(board: LeaderboardPageResponse) -> list[LeaderboardRow]:
    """The placed rows of a signal fixture, narrowed to their own model.

    One page shape now carries both sources, so a reader that wants
    ``model_id`` has to say which kind of row it is reading. Every fixture in
    this module records signal evaluations, so a run row here would mean the
    tool answered from the wrong source.
    """
    rows = [row for row in board.ranked if isinstance(row, LeaderboardRow)]
    assert len(rows) == len(board.ranked), "signal fixture returned a run row"
    return rows


def _signal_skipped(board: LeaderboardPageResponse) -> list[LeaderboardSkipped]:
    rows = [row for row in board.skipped if isinstance(row, LeaderboardSkipped)]
    assert len(rows) == len(board.skipped), "signal fixture returned a run row"
    return rows


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

    frame = pd.read_parquet(workspace.ml_datasets / recorded / "dataset.parquet")
    caller = MLService(workspace=workspace).evaluate_signal(
        np.random.default_rng(190).normal(size=len(frame)).tolist(),
        frame["forward_return"].astype(float).tolist(),
        model_id=pair["model_id"],
        dataset_id=pair["dataset_id"],
        window=20,
        record=True,
    )
    assert caller["recorded"]["replaced"] is False
    assert caller["recorded"]["n_candidates"] == 4
    board = LeaderboardPageResponse.model_validate(
        _dispatch(mcp, "leaderboard", {"metric": "hit_rate", "dataset_id": recorded})
    )
    same_model = [
        row for row in _signal_rows(board) if row.model_id == pair["model_id"]
    ]
    assert {row.producer_source for row in same_model} == {
        ProducerSource.BACKEND_MODEL,
        ProducerSource.CALLER_ARRAY,
    }
    assert len(same_model) == 2


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
        assert _signal_skipped(board)[0].producer_source is ProducerSource.CALLER_ARRAY
        assert board.next_offset == (offset + 1 if offset < 2 else None)
        assert board.caveats
        received_skipped.add(_signal_skipped(board)[0].entry_id)
        if offset < 2:
            assert len(board.ranked) == 1
            assert board.ranked[0].rank == offset + 1
            ranked_models.append(_signal_rows(board)[0].model_id)
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
    assert [row.model_id for row in _signal_rows(board)] == expected_models
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
    assert [row.model_id for row in _signal_rows(board)] == [_GOOD_MODEL, _BAD_MODEL]
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
            source = cached_market_path(workspace.processed_data)
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
        placed = _signal_rows(board)[0]
        assert placed.dataset_id == dataset_id
        assert placed.entry_id == entry_id


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


def test_a_source_qualified_metric_ranks_the_same_rows_as_a_bare_one(
    mcp: MCPServer, recorded: str
) -> None:
    """The prefix names the source; it does not change what is measured."""
    bare = _dispatch(mcp, "leaderboard", {"metric": "ic_mean", "dataset_id": recorded})
    qualified = _dispatch(
        mcp, "leaderboard", {"metric": "signal.ic_mean", "dataset_id": recorded}
    )

    assert qualified["ranked"] == bare["ranked"]
    assert qualified["skipped"] == bare["skipped"]
    assert qualified["n_candidates"] == bare["n_candidates"]
    # The answer names the source it read, whichever spelling was asked for.
    assert qualified["metric"] == bare["metric"] == "signal.ic_mean"


def test_every_row_states_the_source_it_was_measured_from(
    mcp: MCPServer, recorded: str
) -> None:
    board = LeaderboardPageResponse.model_validate(
        _dispatch(mcp, "leaderboard", {"metric": "ic_mean", "dataset_id": recorded})
    )

    assert board.source == "signal"
    assert {row.source for row in board.ranked} == {"signal"}


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
    assert _signal_rows(filtered)[0].entry_id == entries[0].entry_id


def test_parameter_surface_opt_in_reaches_service_and_preserves_row_metadata(
    mcp: MCPServer,
    workspace: WorkspacePaths,
    recorded: str,
) -> None:
    frame = pd.read_parquet(workspace.ml_datasets / recorded / "dataset.parquet")
    realized = frame["forward_return"].astype(float).tolist()
    MLService(workspace=workspace).evaluate_signal(
        realized,
        realized,
        dataset_id=recorded,
        window=30,
        record=True,
    )
    args = {"metric": "ic_mean", "dataset_id": recorded}
    assert _dispatch(mcp, "leaderboard", args)["success"] is False
    board = LeaderboardPageResponse.model_validate(
        _dispatch(mcp, "leaderboard", {**args, "allow_parameter_axes": True})
    )
    assert board.parameter_axes is not None
    assert board.parameter_axes["window"] == [20, 30]
    assert board.n_candidates == 3
    rows = _signal_rows(board)
    assert {row.window for row in rows} == {20, 30}
    assert all(row.signal_fingerprint for row in rows)
    assert any("surface" in text for text in board.caveats)


def test_leaderboard_family_filter_reaches_the_service(
    mcp: MCPServer, recorded: str
) -> None:
    rejected = _dispatch(mcp, "leaderboard", {"metric": "icir", "family": "ensemble"})
    assert rejected["success"] is False
    assert rejected["code"] == "invalid_leaderboard_request"

    board = LeaderboardPageResponse.model_validate(
        _dispatch(mcp, "leaderboard", {"metric": "icir", "family": "ml_model"})
    )
    assert board.families == {"ml_model": 2}
    assert {row.method_family for row in _signal_rows(board)} == {"ml_model"}


def test_evaluate_signal_records_a_rule_without_a_model(
    mcp: MCPServer, workspace: WorkspacePaths
) -> None:
    built = _dispatch(
        mcp,
        "build_dataset",
        {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "feature_set": "technical",
        },
    )
    dataset_id = built["dataset"]["dataset_id"]
    frame = pd.read_parquet(workspace.ml_datasets / dataset_id / "dataset.parquet")
    level = float(frame["rsi_14"].median())

    result = _dispatch(
        mcp,
        "evaluate_signal",
        {
            "dataset_id": dataset_id,
            "column": "rsi_14",
            "threshold": level,
            "direction": "below",
            "window": 20,
            "record": True,
        },
    )

    assert result["success"] is True
    assert result["recorded"]["method_family"] == "rule"
    assert result["recorded"]["method_name"] == f"rsi_14 < {level:g}"
    assert "model_id" not in result


def test_evaluate_signal_refuses_a_model_and_a_rule_together(
    mcp: MCPServer, recorded: str
) -> None:
    rejected = _dispatch(
        mcp,
        "evaluate_signal",
        {
            "dataset_id": recorded,
            "model_id": _GOOD_MODEL,
            "column": "rsi_14",
            "threshold": 50.0,
        },
    )
    assert rejected["success"] is False
    assert rejected["code"] == "invalid_signal"


def test_leaderboard_rows_expose_the_training_dataset_field(
    mcp: MCPServer, recorded: str
) -> None:
    """Every row carries the key, so a client never has to guess its presence.

    The fixture records hand-built signals under caller-declared model IDs, so no
    artifact exists to read a training dataset from and the value stays unknown.
    What the value is when an artifact does exist is pinned by the service-level
    tests, not here.
    """
    payload = _dispatch(mcp, "leaderboard", {"metric": "icir"})
    rows = [row for group in ("ranked", "skipped") for row in payload[group]]
    assert rows, "fixture must rank at least one candidate"
    assert all(row["training_dataset_id"] == "" for row in rows)
