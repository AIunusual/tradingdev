"""History queries over real immutable JSON artifacts, without a backtest."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

import pytest

from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.performance import PerformanceStore
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.contracts.history import (
    FindRunsResponse,
    HistoryQueryError,
    RunEquityResponse,
    RunTradesResponse,
)
from tradingdev.app.trade_history_service import HistoryReadError, TradeHistoryService
from tradingdev.domain.execution import ExecutionManifest, OptimizationSpec
from tradingdev.domain.performance.artifacts import (
    FoldStats,
    PerformanceScope,
    ScopeObservations,
    build_artifacts,
)
from tradingdev.domain.strategies.execution import StrategyExecution

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import JsonValue


@pytest.fixture
def storage(tmp_path: Path) -> tuple[WorkspacePaths, SQLiteStore, TradeHistoryService]:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    return workspace, store, TradeHistoryService(workspace=workspace, store=store)


def _trade(*, entry: int, exit_: int, side: int, status: str) -> dict[str, JsonValue]:
    return {
        "entry_idx": entry,
        "exit_idx": exit_,
        "entry_price": 100.0,
        "exit_price": 110.0,
        "direction": side,
        "status": status,
        "size": 2.0,
        "size_quote": 200.0,
        "entry_fees": 0.2,
        "exit_fees": 0.22 if status == "closed" else 0.0,
        "fee": 0.42 if status == "closed" else 0.2,
        "gross_pnl": 20.0 * side,
        "net_pnl": 20.0 * side - (0.42 if status == "closed" else 0.2),
        "custom_recorded_field": "retained",
    }


def _publish(
    storage: tuple[WorkspacePaths, SQLiteStore, TradeHistoryService],
    run_id: str = "run",
    *,
    optimization: bool = False,
    volume: bool = False,
    legacy: bool = False,
    timestamps: bool = True,
    summary: bool = False,
    bundled: bool = False,
    missing_entry: bool = False,
) -> None:
    workspace, store, _ = storage
    params: dict[str, JsonValue] = {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
        "nested": {"lookback": 5, "threshold": 0.5},
        "flag": True,
    }
    config: dict[str, Any] = {
        "strategy": {"id": "fixture", "parameters": params},
        "backtest": {
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "start_date": "2024-01-01",
            "end_date": "2025-06-30",
            "mode": "volume" if volume else "signal",
            "init_cash": None if volume else 1000.0,
        },
    }
    if summary:
        config["validation"] = {"mode": "automatic"}
    search = (
        OptimizationSpec.model_validate(
            {
                "param_ranges": {"fast_period": [10, 11]},
                "optimization_metric": "total_pnl",
                "train_start": "2024-01-01",
                "train_end": "2024-12-31",
                "test_start": "2025-01-01",
                "test_end": "2025-06-30",
            }
        )
        if optimization
        else None
    )
    manifest = ExecutionManifest.create(
        kind="optimization"
        if optimization
        else "walk_forward"
        if summary
        else "backtest",
        config=config,
        strategy_execution=StrategyExecution(
            kind="bundled" if bundled else "generated",
            constructor_kwargs={"config": params} if bundled else params,
        ),
        optimization=search,
    )
    manifest_hash = None if legacy else manifest.manifest_hash
    store.create_run(
        run_id=run_id,
        job_id=run_id,
        strategy_id="fixture",
        artifact_dir=workspace.runs / run_id,
        metrics={"total_pnl": 20.0},
        manifest_hash=manifest_hash,
    )
    if not legacy:
        ExecutionManifestStore(workspace).publish(run_id, manifest)
    trades = [
        _trade(entry=0, exit_=1, side=1, status="closed"),
        _trade(entry=1, exit_=2, side=-1, status="closed"),
        _trade(entry=2, exit_=3, side=1, status="closed" if volume else "open"),
    ]
    if missing_entry:
        trades[0].pop("entry_idx")
    if volume:
        for trade in trades:
            trade.update(entry_slippage=0.1, exit_slippage=0.11, slippage=0.21)
    observations = ScopeObservations(
        init_cash=None if volume else 1000.0,
        equity_curve=[0.0, 10.0, -5.0, 20.0]
        if volume
        else [1000.0, 1010.0, 995.0, 1020.0],
        returns=None if volume else [0.0, 0.01, -0.014851, 0.025125],
        timestamps=[
            "2024-01-01T00:00:00.000000000Z",
            "2024-01-02T00:00:00.000000000Z",
            "2024-01-02T23:59:59.999999999Z",
            "2024-01-03T00:00:00.000000000Z",
        ]
        if timestamps
        else None,
        trades=trades,
    )
    scopes = {}
    raw = {}
    names = ["trial/0/train", "trial/1/train", "test"] if optimization else ["full"]
    for index, name in enumerate(names):
        parameters = (
            {"fast_period": 10 if index == 0 else 11, "nested": {"threshold": 0.7}}
            if optimization
            else params
        )
        scopes[name] = PerformanceScope.model_validate(
            {
                "mode": "volume" if volume else "signal",
                "values": {"total_pnl": 20.0},
                "parameters": parameters,
                "split": "train"
                if name.startswith("trial")
                else "test"
                if optimization
                else "full",
                "metadata": {
                    "strategy_parameters": {"fast_period": 999},
                    "execution_context": config["backtest"],
                },
            }
        )
        raw[name] = observations
    default_scope = "test" if optimization else "full"
    if summary:
        scopes["test_summary"] = PerformanceScope(
            kind="fold_summary",
            mode="signal",
            values={
                "total_pnl": FoldStats(
                    mean=20.0, std=0.0, min=20.0, max=20.0, valid_count=1
                )
            },
            metadata={},
        )
        default_scope = "test_summary"
    artifacts = build_artifacts(
        run_id,
        manifest_hash,
        default_scope,
        scopes,
        raw,
        selected_train_scope="trial/1/train" if optimization else None,
    )
    PerformanceStore(workspace, store).publish(artifacts)


@pytest.mark.parametrize("bundled", [False, True])
def test_find_parameterized_trials_uses_historical_defaults_not_fitted_state(
    storage: tuple[WorkspacePaths, SQLiteStore, TradeHistoryService], bundled: bool
) -> None:
    _publish(storage, optimization=True, bundled=bundled)
    service = storage[2]
    found = FindRunsResponse.model_validate(
        service.find_runs(
            strategy_id="fixture",
            parameters={
                "fast_period": 11,
                "slow_period": 26,
                "nested": {"lookback": 5},
            },
            symbol="BTC/USDT",
            timeframe="1d",
            limit=1,
        )
    )
    assert found.total == 3 and found.matched == 2 and found.next_offset == 1
    assert found.complete and not found.issues
    first = found.runs[0]
    assert first.parameters["nested"] == {"lookback": 5, "threshold": 0.7}
    assert first.parameters["signal_period"] == 9
    assert first.parameter_provenance == "execution_manifest_and_trial"
    assert first.parameters_complete
    second = FindRunsResponse.model_validate(
        service.find_runs(parameters={"fast_period": 11}, offset=1, limit=1)
    )
    assert first.scope != second.runs[0].scope
    assert second.next_offset is None
    assert service.find_runs(parameters={"fast_period": 999})["matched"] == 0
    assert service.find_runs(parameters={"flag": 1})["matched"] == 0
    assert service.find_runs(symbol="ETH/USDT")["matched"] == 0


def test_trade_pagination_filters_and_open_marks_preserve_original_records(
    storage: tuple[WorkspacePaths, SQLiteStore, TradeHistoryService],
) -> None:
    _publish(storage)
    service = storage[2]
    first = RunTradesResponse.model_validate(service.get_run_trades("run", limit=1))
    assert first.total == 3 and first.next_offset == 1
    assert first.trades[0].entry_timestamp == "2024-01-01T00:00:00+00:00"
    filtered = RunTradesResponse.model_validate(
        service.get_run_trades("run", entry_start="2024-01-02", entry_end="2024-01-02")
    )
    assert [trade.trade_id for trade in filtered.trades] == [1, 2]
    assert filtered.matched == 2 and filtered.total == 3
    open_page = RunTradesResponse.model_validate(
        service.get_run_trades("run", direction="long", status="open")
    )
    trade = open_page.trades[0]
    assert trade.trade_id == 2
    assert trade.entry_timestamp == "2024-01-02T23:59:59.999999999+00:00"
    assert trade.exit_timestamp is None and trade.exit_price is None
    assert trade.mark_timestamp == "2024-01-03T00:00:00+00:00"
    assert trade.mark_price == 110
    assert trade.record["exit_price"] == 110
    assert trade.record["custom_recorded_field"] == "retained"
    assert trade.record.get("slippage") is None
    exact = service.get_run_trades("run", entry_end="2024-01-02T08:00:00+08:00")
    assert [trade["trade_id"] for trade in exact["trades"]] == [0, 1]
    assert service.get_run_trades("run", offset=100)["trades"] == []


@pytest.mark.parametrize(
    "bounds",
    [
        {"entry_start": "2024-01-02"},
        {"entry_end": "2024-01-02"},
        {"entry_start": "2024-01-02", "entry_end": "2024-01-02"},
    ],
)
def test_date_filter_excludes_only_legacy_trades_with_unknown_entry_time(
    storage: tuple[WorkspacePaths, SQLiteStore, TradeHistoryService],
    bounds: dict[str, Any],
) -> None:
    _publish(storage, legacy=True, missing_entry=True)
    service = storage[2]
    unfiltered = RunTradesResponse.model_validate(service.get_run_trades("run"))
    assert unfiltered.total == unfiltered.matched == 3
    assert [trade.trade_id for trade in unfiltered.trades] == [0, 1, 2]
    assert unfiltered.trades[0].entry_timestamp is None
    assert "entry_idx" not in unfiltered.trades[0].record
    first = RunTradesResponse.model_validate(
        service.get_run_trades("run", limit=1, **bounds)
    )
    assert (first.total, first.matched, first.next_offset) == (3, 2, 1)
    assert [trade.trade_id for trade in first.trades] == [1]
    second = RunTradesResponse.model_validate(
        service.get_run_trades("run", offset=1, limit=1, **bounds)
    )
    assert [trade.trade_id for trade in second.trades] == [2]
    assert second.next_offset is None
    outside = RunTradesResponse.model_validate(
        service.get_run_trades("run", entry_end="2024-01-01")
    )
    assert outside.total == 3 and outside.matched == 0 and outside.trades == []


@pytest.mark.parametrize("volume", [False, True])
def test_equity_is_saved_aligned_and_has_explicit_capital_basis(
    storage: tuple[WorkspacePaths, SQLiteStore, TradeHistoryService], volume: bool
) -> None:
    _publish(storage, volume=volume)
    service = storage[2]
    page = RunEquityResponse.model_validate(
        service.get_run_equity("run", start="2024-01-02", end="2024-01-02", limit=1)
    )
    assert page.total == 4 and page.matched == 2 and page.next_offset == 1
    assert page.points[0].bar_index == 1
    assert page.points[0].equity == (10.0 if volume else 1010.0)
    assert page.init_cash == (None if volume else 1000.0)
    assert page.equity_basis == ("cumulative_pnl" if volume else "account_equity")
    assert page.points[0].bar_return == (None if volume else 0.01)
    if volume:
        trade = service.get_run_trades("run")["trades"][0]
        assert trade["record"]["entry_slippage"] == 0.1


def test_summary_requires_scalar_scope_and_missing_times_are_not_invented(
    storage: tuple[WorkspacePaths, SQLiteStore, TradeHistoryService],
) -> None:
    _publish(storage, summary=True, timestamps=False)
    service = storage[2]
    error = HistoryQueryError.model_validate(service.get_run_trades("run"))
    assert error.code == "scope_has_no_observations" and error.available_scopes == [
        "full"
    ]
    assert (
        service.get_run_equity("run", scope="missing")["code"]
        == "unknown_history_scope"
    )
    assert service.get_run_trades("run", scope="")["code"] == "unknown_history_scope"
    page = RunTradesResponse.model_validate(service.get_run_trades("run", scope="full"))
    assert page.trades[0].entry_timestamp is None
    assert (
        service.get_run_equity("run", scope="full", start="2024-01-01")["code"]
        == "timestamps_unavailable"
    )
    assert (
        service.get_run_trades("run", scope="full", entry_end="2024-01-02")["code"]
        == "timestamps_unavailable"
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"offset": -1},
        {"limit": 0},
        {"limit": 501},
        {"limit": True},
        {"direction": "both"},
        {"status": "filled"},
        {"entry_start": "not-a-date"},
        {"entry_start": "2025-01-01", "entry_end": "2024-01-01"},
    ],
)
def test_invalid_queries_are_explicit_without_touching_run_data(
    storage: tuple[WorkspacePaths, SQLiteStore, TradeHistoryService],
    arguments: dict[str, Any],
) -> None:
    result = HistoryQueryError.model_validate(
        storage[2].get_run_trades("absent", **arguments)
    )
    assert result.code == "invalid_history_query"


def test_legacy_incomplete_parameters_and_corrupt_runs_remain_visible_issues(
    storage: tuple[WorkspacePaths, SQLiteStore, TradeHistoryService],
) -> None:
    workspace, _, service = storage
    _publish(storage, "good")
    _publish(storage, "legacy", legacy=True)
    _publish(storage, "corrupt")
    (workspace.runs / "corrupt" / "observations.json").write_text("{}")
    page = FindRunsResponse.model_validate(
        service.find_runs(parameters={"fast_period": 12})
    )
    assert not page.complete
    assert [run.run_id for run in page.runs] == ["good"]
    assert {issue.code for issue in page.issues} == {
        "parameters_unavailable",
        "performance_artifact_invalid",
    }
    legacy = RunTradesResponse.model_validate(service.get_run_trades("legacy"))
    assert (
        not legacy.parameters_complete
        and legacy.parameter_provenance == "scope_snapshot"
    )
    assert service.get_run_trades("corrupt")["code"] == "performance_artifact_invalid"
    assert service.get_run_trades("absent")["code"] == "run_not_found"


@pytest.mark.parametrize("damage", ["manifest", "missing_observations", "busy"])
def test_integrity_and_incomplete_publication_are_never_replayed(
    storage: tuple[WorkspacePaths, SQLiteStore, TradeHistoryService], damage: str
) -> None:
    workspace, _, service = storage
    _publish(storage)
    directory = workspace.runs / "run"
    if damage == "manifest":
        path = directory / "manifest.json"
        payload = json.loads(path.read_text())
        payload["strategy_execution"]["constructor_kwargs"]["fast_period"] = 99
        path.write_text(json.dumps(payload))
        expected = "execution_manifest_invalid"
    elif damage == "missing_observations":
        (directory / "observations.json").unlink()
        expected = "performance_artifact_invalid"
    else:
        (directory / ".result-publication").mkdir()
        expected = "performance_artifact_busy"
    assert service.get_run_trades("run")["code"] == expected
    assert service.find_runs()["issues"][0]["code"] == expected


def test_read_only_queries_need_no_strategy_or_pickle_and_return_detached_values(
    storage: tuple[WorkspacePaths, SQLiteStore, TradeHistoryService],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, _, service = storage
    _publish(storage)
    before = {
        path: path.read_bytes() for path in (workspace.runs / "run").glob("*.json")
    }

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "History must not load a strategy, unpickle, or run a backtest"
        )

    monkeypatch.setattr("pickle.load", forbidden)
    for module in (
        "tradingdev.domain.strategies.loader",
        "tradingdev.domain.backtest.engines",
        "tradingdev.app.backtest_service",
    ):
        monkeypatch.setitem(sys.modules, module, None)
    scope = service.load_scope("run")
    scope.parameters["fast_period"] = 100
    assert service.find_runs()["runs"][0]["parameters"]["fast_period"] == 12
    service.get_run_trades("run")["trades"][0]["record"]["entry_price"] = 999
    assert service.get_run_trades("run")["trades"][0]["entry_price"] == 100.0
    service.get_run_equity("run")
    assert before == {path: path.read_bytes() for path in before}
    with pytest.raises(HistoryReadError, match="Unknown run"):
        service.load_scope("missing")
