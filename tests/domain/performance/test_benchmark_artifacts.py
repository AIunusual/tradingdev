"""Each saved scope keeps its own aligned baseline, never a shared guess."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.backtest.metrics import calculate_passive_metrics
from tradingdev.domain.backtest.pipeline_result import PipelineResult
from tradingdev.domain.backtest.result import BacktestResult
from tradingdev.domain.benchmark import BENCHMARKS, BenchmarkEvidence
from tradingdev.domain.performance.artifacts import (
    BenchmarkScope,
    PerformanceBundle,
    bundles_from_pipeline,
)

STAMPS = pd.DatetimeIndex(
    pd.to_datetime(
        ["2024-01-01T00:00", "2024-01-02T00:00", "2024-01-03T00:00"], utc=True
    )
)


def _config(*, init_cash: float | None) -> dict[str, Any]:
    return {
        "strategy": {"id": "fixture", "parameters": {}},
        "backtest": {
            "symbol": "ETH/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-01-03T00:00:00",
            "init_cash": init_cash,
            "fees": 0.001,
            "slippage": 0.0,
            "periods_per_year": 8760.0,
        },
        "random_seed": 7,
    }


def _strategy_result(*, init_cash: float | None) -> BacktestResult:
    equity = np.array([1000.0, 1010.0, 1020.0])
    analysis = calculate_passive_metrics(
        equity,
        init_cash=1000.0,
        total_fees=0.0,
        total_volume=0.0,
        timestamps=STAMPS,
        periods_per_year=8760.0,
        frequency="1h",
    )
    return BacktestResult(
        metrics=analysis.metrics,
        equity_curve=equity,
        returns=analysis.returns,
        timestamps=STAMPS.to_numpy(),
        trades=[],
        init_cash=init_cash,
        mode="signal",
        metric_metadata=analysis.metadata,
    )


def _pipeline(
    evidence: BenchmarkEvidence, *, init_cash: float | None = 1000.0
) -> tuple[PipelineResult, dict[str, Any]]:
    result = _strategy_result(init_cash=init_cash)
    return (
        PipelineResult(
            mode="simple",
            backtest_result=result,
            config_snapshot=_config(init_cash=init_cash),
            benchmark=evidence,
        ),
        result.metrics,
    )


def _available(
    closes: list[float], stamps: pd.DatetimeIndex = STAMPS
) -> BenchmarkEvidence:
    return BenchmarkEvidence(
        status="available",
        asset_class="crypto",
        definition=BENCHMARKS["crypto"],
        source="binance_vision",
        market_type="futures/um",
        dataset_id="BTC/USDT:1h:2024-01-01:2024-01-03",
        reused_run_frame=False,
        timestamps=stamps,
        closes=np.asarray(closes, dtype=np.float64),
    )


def test_scope_records_the_baseline_priced_with_the_runs_own_costs() -> None:
    pipeline, projection = _pipeline(_available([100.0, 110.0, 121.0]))
    artifacts = bundles_from_pipeline("run-a", pipeline, projection)
    benchmark = artifacts.performance.benchmarks["full"]

    assert benchmark.status == "available"
    assert benchmark.definition is not None
    assert benchmark.definition.id == "crypto_btc_buy_and_hold"
    assert benchmark.definition.symbol == "BTC/USDT"
    assert benchmark.capital_basis == "run_init_cash"
    assert benchmark.fees == pytest.approx(0.001)
    assert benchmark.slippage == pytest.approx(0.0)
    # Hand-derived: entry 100 with 0.1% commission, so quantity = 1000 / 100.1.
    quantity = 1000 / 100.1
    assert benchmark.equity_curve == pytest.approx(
        [quantity * price for price in (100.0, 110.0, 121.0)]
    )
    assert benchmark.values["total_return"] == pytest.approx(121 / 100 / 1.001 - 1)
    assert benchmark.values["total_fees"] == pytest.approx(quantity * 100 * 0.001)
    assert benchmark.values["total_trades"] == 0
    assert benchmark.coverage is not None
    assert benchmark.coverage.complete
    assert benchmark.coverage.scope_bars == 3
    details = benchmark.metadata["benchmark"]
    assert isinstance(details, dict)
    assert details["exit"] == "not_liquidated_marked_at_last_close"
    # The baseline's own metric definitions are frozen alongside the run's.
    assert "total_return" in artifacts.performance.definitions


def test_baseline_without_initial_capital_uses_one_normalized_unit() -> None:
    pipeline, projection = _pipeline(_available([100.0, 121.0, 110.0]), init_cash=None)
    artifacts = bundles_from_pipeline("run-b", pipeline, projection)
    benchmark = artifacts.performance.benchmarks["full"]
    assert benchmark.capital_basis == "normalized_unit"
    assert benchmark.init_cash == pytest.approx(1.0)
    assert benchmark.values["total_return"] == pytest.approx(110 / 100 / 1.001 - 1)


def test_missing_baseline_bars_are_counted_and_never_back_filled() -> None:
    stamps = pd.DatetimeIndex(STAMPS[:2])
    pipeline, projection = _pipeline(_available([100.0, 110.0], stamps=stamps))
    artifacts = bundles_from_pipeline("run-c", pipeline, projection)
    benchmark = artifacts.performance.benchmarks["full"]
    assert benchmark.coverage is not None
    assert benchmark.coverage.scope_bars == 3
    assert benchmark.coverage.aligned_bars == 2
    assert benchmark.coverage.missing_bars == 1
    assert benchmark.coverage.complete is False
    assert len(benchmark.timestamps or []) == 2


def test_undefined_asset_class_keeps_an_explicit_reason() -> None:
    evidence = BenchmarkEvidence(
        status="undefined",
        asset_class=None,
        reason="undefined_asset_class",
        detail="資產類別 未分類 沒有定義 benchmark",
        source="yahoo_finance",
    )
    pipeline, projection = _pipeline(evidence)
    artifacts = bundles_from_pipeline("run-d", pipeline, projection)
    benchmark = artifacts.performance.benchmarks["full"]
    assert benchmark.status == "undefined"
    assert benchmark.reason == "undefined_asset_class"
    assert benchmark.values == {}
    assert benchmark.equity_curve is None


def test_unavailable_baseline_never_becomes_a_zero_return() -> None:
    evidence = BenchmarkEvidence(
        status="unavailable",
        asset_class="crypto",
        definition=BENCHMARKS["crypto"],
        reason="benchmark_data_unavailable",
        detail="no cached market data",
        source="binance_vision",
    )
    pipeline, projection = _pipeline(evidence)
    artifacts = bundles_from_pipeline("run-e", pipeline, projection)
    benchmark = artifacts.performance.benchmarks["full"]
    assert benchmark.status == "unavailable"
    assert benchmark.values == {}
    assert benchmark.equity_curve is None


def test_no_evidence_records_no_benchmark_block() -> None:
    pipeline, projection = _pipeline(None)  # type: ignore[arg-type]
    artifacts = bundles_from_pipeline("run-f", pipeline, projection)
    assert artifacts.performance.benchmarks == {}


def test_available_scope_cannot_omit_its_series() -> None:
    with pytest.raises(ValueError, match="requires a definition and series"):
        BenchmarkScope.model_validate(
            {
                "status": "available",
                "asset_class": "crypto",
                "capital_basis": "run_init_cash",
            }
        )


def test_absent_scope_cannot_smuggle_values() -> None:
    with pytest.raises(ValueError, match="cannot record values"):
        BenchmarkScope.model_validate(
            {
                "status": "unavailable",
                "reason": "benchmark_data_unavailable",
                "values": {"total_return": 0.0},
            }
        )


def test_metric_failure_degrades_to_unavailable_instead_of_failing_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(*_: Any, **__: Any) -> Any:
        raise ValueError("Equity must contain only finite values")

    monkeypatch.setattr(
        "tradingdev.domain.performance.artifacts.calculate_passive_metrics", _fail
    )
    pipeline, projection = _pipeline(_available([100.0, 110.0, 121.0]))
    artifacts = bundles_from_pipeline("run-h", pipeline, projection)
    benchmark = artifacts.performance.benchmarks["full"]
    assert benchmark.status == "unavailable"
    assert benchmark.reason == "benchmark_metrics_failed"
    assert benchmark.values == {}
    assert benchmark.equity_curve is None
    # The aligned coverage survives, so the reader still sees what was compared.
    assert benchmark.coverage is not None
    assert benchmark.coverage.aligned_bars == 3


def test_bundle_rejects_benchmarks_for_unknown_scopes() -> None:
    with pytest.raises(ValueError, match="scalar result scopes"):
        PerformanceBundle.model_validate(
            {
                "run_id": "run-g",
                "default_scope": "full",
                "scopes": {"full": {"mode": "signal", "values": {}, "metadata": {}}},
                "definitions": {},
                "benchmarks": {"missing": {"status": "undefined", "reason": "x"}},
            }
        )
