"""Shared statistical analysis of validated saved run/scope observations."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from tests.strategy_fixtures import GENERATED_PROGRAM

from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.performance import PerformanceStore
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.contracts.statistics import (
    BreakevenSummaryRecord,
    DSRRecord,
    PerformanceSummaryRecord,
    SharpeIntervalRecord,
    StatisticsAnalysisResponse,
    StatisticsQueryError,
)
from tradingdev.app.statistics_service import ALGORITHM_VERSION, StatisticsService
from tradingdev.domain.execution import ExecutionManifest, OptimizationSpec
from tradingdev.domain.performance.artifacts import (
    FoldStats,
    OptimizationSelection,
    PerformanceScope,
    ScopeObservations,
    build_artifacts,
)
from tradingdev.domain.strategies.execution import StrategyExecution

FIRST = pd.Timestamp("2024-01-01", tz="UTC")


def _timestamps(count: int, *, skip_day: int | None = None) -> list[str]:
    days = [index for index in range(count) if index != skip_day]
    return [(FIRST + pd.Timedelta(days=index)).isoformat() for index in days]


def _equity(
    count: int, *, mode: str = "signal", slope: float = 1.0
) -> list[float | None]:
    if mode == "volume":
        return [float(index) * slope for index in range(count)]
    return [1000.0 + float(index) * slope for index in range(count)]


def _trade(*, status: str, pnl: float, fee: float = 0.2) -> dict[str, Any]:
    return {
        "entry_idx": 0,
        "exit_idx": 1,
        "entry_price": 100.0,
        "exit_price": 110.0,
        "direction": 1,
        "status": status,
        "size": 2.0,
        "size_quote": 200.0,
        "fee": fee,
        "net_pnl": pnl,
    }


def _settings(
    *,
    periods_per_year: float | None = 365.0,
    risk_free_rate: float = 0.0,
    frequency: str = "1d",
    capital: float | None = 1000.0,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "initial_cash": capital,
        "frequency": frequency,
        "risk_free_rate": risk_free_rate,
        "return_std_ddof": 1,
        "period_pnl_ddof": 0,
        "missing_dates": "not_filled",
        "trade_scope": "closed",
        "calendar_timezone": "UTC",
    }
    if periods_per_year is not None:
        values["periods_per_year"] = periods_per_year
    return values


def _publish(
    workspace: WorkspacePaths,
    store: SQLiteStore,
    *,
    run_id: str = "run",
    equity: list[float | None] | None = None,
    timestamps: list[str] | None = None,
    trades: list[dict[str, Any]] | None = None,
    mode: str = "signal",
    settings: dict[str, Any] | None = None,
    optimization: bool = False,
    summary: bool = False,
    trial_slopes: dict[str, float] | None = None,
    trial_skip_days: dict[str, int] | None = None,
) -> None:
    equity = _equity(30) if equity is None else equity
    timestamps = _timestamps(len(equity)) if timestamps is None else timestamps
    equity = equity[: len(timestamps)]
    trades = [] if trades is None else trades
    config: dict[str, Any] = {
        "strategy": {"id": "fixture", "parameters": {"fast": 12}},
        "backtest": {
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "mode": mode,
            "init_cash": None if mode == "volume" else 1000.0,
            "periods_per_year": 365.0,
            "risk_free_rate": 0.0,
        },
    }
    if summary:
        config["validation"] = {"mode": "automatic"}
    search = (
        OptimizationSpec.model_validate(
            {
                "param_ranges": {"fast": [10, 11, 12]},
                "optimization_metric": "sharpe_ratio",
                "train_start": "2024-01-01",
                "train_end": "2024-02-01",
                "test_start": "2024-02-02",
                "test_end": "2024-03-01",
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
            kind="generated", program=GENERATED_PROGRAM, constructor_kwargs={"fast": 12}
        ),
        optimization=search,
    )
    store.create_run(
        run_id=run_id,
        job_id=run_id,
        strategy_id="fixture",
        artifact_dir=workspace.runs / run_id,
        metrics={"total_pnl": 30.0},
        manifest_hash=manifest.manifest_hash,
        dataset_id="snapshot-fingerprint",
    )
    ExecutionManifestStore(workspace).publish(run_id, manifest)
    frozen = _settings(capital=None if mode == "volume" else 1000.0)
    if settings is not None:
        frozen = settings
    observations = ScopeObservations(
        init_cash=None if mode == "volume" else 1000.0,
        equity_curve=equity,
        returns=None,
        timestamps=timestamps,
        trades=trades,
    )
    names = (
        ["trial/0/train", "trial/1/train", "trial/2/train", "test"]
        if optimization
        else ["full"]
    )
    scopes: dict[str, PerformanceScope] = {}
    raw: dict[str, ScopeObservations] = {}
    for name in names:
        slope = (trial_slopes or {}).get(name, 2.0 if name == "trial/1/train" else 1.0)
        scalars = _equity(len(equity), mode=mode, slope=slope)
        skip_day = (trial_skip_days or {}).get(name)
        scope_timestamps = (
            _timestamps(len(equity) + 1, skip_day=skip_day)
            if skip_day is not None
            else timestamps
        )
        value = ScopeObservations(
            init_cash=observations.init_cash,
            equity_curve=equity if name == names[0] else scalars,
            returns=None,
            timestamps=scope_timestamps,
            trades=trades,
        )
        raw[name] = value
        scopes[name] = PerformanceScope.model_validate(
            {
                "mode": mode,
                "values": {"total_pnl": 30.0, "sharpe_ratio": 2.0},
                "parameters": {"fast": 12},
                "split": "train" if name.startswith("trial") else "test",
                "trial_index": int(name[6]) if name.startswith("trial") else None,
                "metadata": {
                    "execution_context": config["backtest"],
                    "settings": frozen,
                },
            }
        )
    if summary:
        scopes["test_summary"] = PerformanceScope.model_validate(
            {
                "kind": "fold_summary",
                "mode": mode,
                "values": {
                    "total_pnl": FoldStats(
                        mean=30.0, std=0.0, min=30.0, max=30.0, valid_count=1
                    )
                },
                "metadata": {},
            }
        )
    selection = (
        OptimizationSelection(
            selected_train_scope="trial/1/train",
            objective="sharpe_ratio",
            direction="maximize",
            planned=3,
            raw_attempted=3,
            completed=3,
            objective_available=3,
            objective_unavailable=0,
        )
        if optimization
        else None
    )
    default = "test_summary" if summary else "test" if optimization else "full"
    artifacts = build_artifacts(
        run_id,
        manifest.manifest_hash,
        default,
        scopes,
        raw,
        selected_train_scope="trial/1/train" if optimization else None,
        optimization_selection=selection,
    )
    PerformanceStore(workspace, store).publish(artifacts)


def _summary(response: StatisticsAnalysisResponse) -> PerformanceSummaryRecord:
    assert response.summary.result is not None
    return PerformanceSummaryRecord.model_validate(response.summary.result)


def _breakeven(response: StatisticsAnalysisResponse) -> BreakevenSummaryRecord:
    assert response.breakeven.result is not None
    return BreakevenSummaryRecord.model_validate(response.breakeven.result)


def _interval(response: StatisticsAnalysisResponse) -> SharpeIntervalRecord:
    assert response.inference.interval.result is not None
    return SharpeIntervalRecord.model_validate(response.inference.interval.result)


def _dsr(response: StatisticsAnalysisResponse) -> DSRRecord | None:
    if response.inference.dsr.result is None:
        return None
    return DSRRecord.model_validate(response.inference.dsr.result)


@pytest.fixture
def storage(tmp_path: Path) -> tuple[WorkspacePaths, SQLiteStore, StatisticsService]:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    return workspace, store, StatisticsService(workspace=workspace, store=store)


def test_first_equity_is_not_the_capital_basis(
    storage: tuple[WorkspacePaths, SQLiteStore, StatisticsService],
) -> None:
    workspace, store, service = storage
    _publish(
        workspace,
        store,
        equity=[990.0, 1010.0],
        timestamps=_timestamps(2),
    )
    response = StatisticsAnalysisResponse.model_validate(service.analyze("run"))
    summary = _summary(response)
    assert summary is not None
    assert summary.total_pnl == 10.0
    assert summary.observed_equity_change == 20.0
    assert summary.total_return == pytest.approx(0.01)
    assert response.inference.interval.status == "unavailable"
    problem = response.inference.interval.unavailable
    assert problem is not None and problem.reason == "insufficient_observations"
    assert problem.detail is not None and "2 observations" in problem.detail
    assert response.inference.interval.config.effective_observations == 2
    assert response.algorithm_version == ALGORITHM_VERSION


def test_nonzero_risk_free_uses_independent_daily_return_basis(
    storage: tuple[WorkspacePaths, SQLiteStore, StatisticsService],
) -> None:
    workspace, store, service = storage
    equity = _equity(30, slope=3.0)
    _publish(
        workspace,
        store,
        equity=equity,
        settings=_settings(risk_free_rate=0.05, periods_per_year=365.0),
    )
    response = StatisticsAnalysisResponse.model_validate(service.analyze("run"))
    summary = _summary(response)
    assert summary is not None
    index = pd.DatetimeIndex(_timestamps(30))
    closing = pd.Series(equity, index=index)
    opening = closing.shift(1).fillna(1000.0)
    returns = (closing - opening) / opening
    daily_rate = math.expm1(math.log1p(0.05) / 365.0)
    excess = returns - daily_rate
    expected = float(excess.mean() / excess.std(ddof=1) * math.sqrt(365.0))
    assert summary.sharpe_ratio == pytest.approx(expected, rel=1e-12)
    assert summary.risk_free_rate == 0.05
    assert response.inference.interval.status == "available"
    interval = _interval(response)
    assert interval.valid_reps > 0


def test_closed_and_marked_open_groups_keep_saved_mark(
    storage: tuple[WorkspacePaths, SQLiteStore, StatisticsService],
) -> None:
    workspace, store, service = storage
    trades = [
        _trade(status="closed", pnl=7.8),
        _trade(status="open", pnl=3.8, fee=0.2),
    ]
    _publish(workspace, store, trades=trades)
    response = StatisticsAnalysisResponse.model_validate(service.analyze("run"))
    summary = _summary(response)
    assert summary is not None
    assert summary.closed_trades.total_trades == 1
    assert summary.closed_trades.net_pnl == 7.8
    assert summary.marked_trades.total_trades == 2
    assert summary.marked_trades.net_pnl == pytest.approx(11.6)
    breakeven = _breakeven(response)
    assert breakeven is not None
    # Only the closed trade contributes an exit leg; the open mark does not.
    assert breakeven.traded_notional == pytest.approx(200.0 + 420.0)
    assert breakeven.total_cost == pytest.approx(0.4)


def test_volume_cumulative_pnl_has_no_return_inference(
    storage: tuple[WorkspacePaths, SQLiteStore, StatisticsService],
) -> None:
    workspace, store, service = storage
    _publish(workspace, store, mode="volume", equity=_equity(30, mode="volume"))
    response = StatisticsAnalysisResponse.model_validate(service.analyze("run"))
    summary = _summary(response)
    assert summary is not None
    assert summary.total_return is None
    assert summary.unavailable["total_return"] == "not_applicable"
    for key in ("interval", "psr", "dsr"):
        outcome = getattr(response.inference, key)
        assert outcome.status == "unavailable"
        assert outcome.unavailable is not None
        assert outcome.unavailable.reason == "not_applicable"


def test_missing_annualization_does_not_hide_descriptive_summary(
    storage: tuple[WorkspacePaths, SQLiteStore, StatisticsService],
) -> None:
    workspace, store, service = storage
    _publish(workspace, store, settings=_settings(periods_per_year=None))
    response = StatisticsAnalysisResponse.model_validate(service.analyze("run"))
    summary = _summary(response)
    assert summary is not None
    assert summary.total_pnl == 29.0
    assert summary.sharpe_ratio is None
    assert summary.unavailable["sharpe_ratio"] == "missing_annualization"
    assert response.settings.settings_complete is False
    assert response.inference.interval.unavailable is not None
    assert response.inference.interval.unavailable.reason == "missing_annualization"


def test_gap_is_not_compressed_into_adjacent_bootstrap_periods(
    storage: tuple[WorkspacePaths, SQLiteStore, StatisticsService],
) -> None:
    workspace, store, service = storage
    equity = _equity(30)
    timestamps = _timestamps(30, skip_day=7)
    _publish(workspace, store, equity=equity, timestamps=timestamps)
    response = StatisticsAnalysisResponse.model_validate(service.analyze("run"))
    assert response.summary.status == "available"
    assert response.inference.interval.unavailable is not None
    assert (
        response.inference.interval.unavailable.reason == "observation_gap_unsupported"
    )
    assert response.inference.interval.unavailable.next_step


def test_fold_summary_is_an_explicit_unsupported_scope(
    storage: tuple[WorkspacePaths, SQLiteStore, StatisticsService],
) -> None:
    workspace, store, service = storage
    _publish(workspace, store, summary=True)
    error = StatisticsQueryError.model_validate(service.analyze("run"))
    assert error.success is False
    assert error.code == "fold_summary_unsupported"
    assert "test_summary" in error.available_scopes
    assert error.unavailable is not None and error.unavailable.next_step


def test_corrupt_observation_artifact_is_not_reconstructed(
    storage: tuple[WorkspacePaths, SQLiteStore, StatisticsService],
) -> None:
    workspace, store, service = storage
    _publish(workspace, store)
    path = workspace.runs / "run" / "observations.json"
    path.write_text("{not json", encoding="utf-8")
    error = StatisticsQueryError.model_validate(service.analyze("run"))
    assert error.success is False
    assert error.code == "performance_artifact_invalid"
    assert error.unavailable is not None and error.unavailable.source.startswith("run:")


def test_dsr_only_for_verified_selected_training_scope(
    storage: tuple[WorkspacePaths, SQLiteStore, StatisticsService],
) -> None:
    workspace, store, service = storage
    _publish(workspace, store, optimization=True)
    selected = StatisticsAnalysisResponse.model_validate(
        service.analyze("run", "trial/1/train")
    )
    assert selected.inference.dsr.status == "available"
    dsr = _dsr(selected)
    assert dsr is not None
    assert dsr.n_trials == 3
    assert dsr.trial_scope_count == 3
    assert dsr.search_population == "selected_training_scope"
    test = StatisticsAnalysisResponse.model_validate(service.analyze("run", "test"))
    assert test.inference.dsr.status == "unavailable"
    assert test.inference.dsr.unavailable is not None
    assert test.inference.dsr.unavailable.reason == "not_selected_training_scope"


def test_dsr_is_unavailable_when_a_non_selected_trial_has_a_gap(
    storage: tuple[WorkspacePaths, SQLiteStore, StatisticsService],
) -> None:
    workspace, store, service = storage
    _publish(
        workspace,
        store,
        optimization=True,
        trial_skip_days={"trial/0/train": 7},
    )
    response = StatisticsAnalysisResponse.model_validate(
        service.analyze("run", "trial/1/train")
    )
    outcome = response.inference.dsr
    assert outcome.status == "unavailable"
    assert outcome.unavailable is not None
    assert outcome.unavailable.reason == "observation_gap_unsupported"


def test_source_hash_binds_trial_evidence_used_for_dsr(
    storage: tuple[WorkspacePaths, SQLiteStore, StatisticsService],
) -> None:
    workspace, store, service = storage
    _publish(workspace, store, run_id="run_a", optimization=True)
    _publish(
        workspace,
        store,
        run_id="run_b",
        optimization=True,
        trial_slopes={"trial/0/train": 3.0},
    )
    first = StatisticsAnalysisResponse.model_validate(
        service.analyze("run_a", "trial/1/train")
    )
    second = StatisticsAnalysisResponse.model_validate(
        service.analyze("run_b", "trial/1/train")
    )
    assert _dsr(first) is not None and _dsr(second) is not None
    assert first.source.source_hash != second.source.source_hash


def test_derived_artifact_is_reproducible_and_source_bundle_unchanged(
    storage: tuple[WorkspacePaths, SQLiteStore, StatisticsService],
) -> None:
    workspace, store, service = storage
    _publish(workspace, store)
    source = workspace.runs / "run" / "performance.json"
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    first = StatisticsAnalysisResponse.model_validate(
        service.analyze("run", save_artifact=True)
    )
    second = StatisticsAnalysisResponse.model_validate(
        service.analyze("run", save_artifact=True)
    )
    assert first.source.source_hash == second.source.source_hash
    assert first.artifact is not None and second.artifact is not None
    assert first.artifact.artifact_id == second.artifact.artifact_id
    assert first.artifact.sha256 == second.artifact.sha256
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    artifact = store.get_artifact(first.artifact.artifact_id)
    assert artifact is not None and artifact["sha256"] == first.artifact.sha256
    text = Path(str(artifact["path"])).read_text(encoding="utf-8")
    assert first.source.source_hash in text
    assert "Fixed settings and limits" in text


def test_tampered_published_artifact_is_rejected(
    storage: tuple[WorkspacePaths, SQLiteStore, StatisticsService],
) -> None:
    workspace, store, service = storage
    _publish(workspace, store)
    first = StatisticsAnalysisResponse.model_validate(
        service.analyze("run", save_artifact=True)
    )
    assert first.artifact is not None
    digest_prefix = first.artifact.artifact_id.rsplit(":", 1)[-1]
    path = workspace.runs / "run" / "statistics_analysis" / f"{digest_prefix}.md"
    assert path.exists()
    original = path.read_text(encoding="utf-8")
    path.write_text(original[: len(original) // 2], encoding="utf-8")
    error = StatisticsQueryError.model_validate(
        service.analyze("run", save_artifact=True)
    )
    assert error.success is False
    assert error.code == "statistics_artifact_failed"
