"""A real zero-trade trial survives the worker and reaches plateau analysis.

These tests run the bundled KD strategy and the real backtest engine on
synthetic bars, without a job store, worker process, or confirmation. They
pin the contract plateau analysis relies on: a parameter value that never
trades yields an evaluation with an undefined objective rather than an
exception that would fail the whole search, and the persisted trial scopes
feed :func:`trials_from_metrics` with the failed slice kept on its axis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from tradingdev.adapters.execution.strategy_factory import StrategyFactory
from tradingdev.domain.backtest.schemas import BacktestConfig, ParallelConfig
from tradingdev.domain.optimization.grid_search import parameter_grid
from tradingdev.domain.performance.artifacts import scope_from_backtest
from tradingdev.domain.statistics.plateau import (
    DroppedTrialsWarning,
    plateau_summary,
    trials_from_metrics,
)
from tradingdev.mcp.workers import optimization

if TYPE_CHECKING:
    from tradingdev.mcp.workers.optimization import ComboEvaluation

_METRIC = "sharpe_ratio"
_BARS = 24 * 30
# Longer than the data, so the oscillator never forms and nothing trades.
_NEVER_TRADES = 5000
_STRATEGY = {"id": "kd_crossover", "parameters": {}}


def _bars() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=_BARS, freq="h", tz="UTC")
    steps = np.random.default_rng(0).normal(0.0, 0.004, _BARS)
    close = 40_000.0 * np.exp(np.cumsum(steps))
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": 1.0,
        }
    )


def _config() -> BacktestConfig:
    return BacktestConfig(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-30T23:59:59.999999",
        init_cash=10_000.0,
        fees=0.0006,
        slippage=0.0005,
        periods_per_year=365.0,
    )


def _run(params: dict[str, int], frame: pd.DataFrame) -> ComboEvaluation:
    return optimization._run_single_combo(
        _STRATEGY,
        StrategyFactory().resolve_execution(_STRATEGY),
        _config(),
        frame,
        params,
        _METRIC,
        ParallelConfig(),
        42,
    )


def test_zero_trade_combo_returns_an_undefined_objective() -> None:
    evaluation = _run({"k_period": _NEVER_TRADES}, _bars())

    assert evaluation.result.trades == []
    assert evaluation.result.metrics[_METRIC] is None
    assert evaluation.result.metric_metadata["unavailable"][_METRIC] == (
        "zero_denominator"
    )
    assert evaluation.target_value is None
    assert evaluation.parameters == {"k_period": _NEVER_TRADES}


def test_persisted_trial_scopes_feed_plateau_with_the_failed_slice() -> None:
    frame = _bars()
    grid = parameter_grid({"d_period": [3, 5], "k_period": [9, 14, _NEVER_TRADES]})
    evaluations = [_run(params, frame) for params in grid]
    # The worker persists each trial as trial/{index}/train in this form.
    scopes = [
        scope_from_backtest(
            evaluation.result,
            split="train",
            trial_index=index,
            parameters=evaluation.parameters,
        )[0]
        for index, evaluation in enumerate(evaluations)
    ]
    rows = [
        {"params": scope.parameters, _METRIC: scope.values[_METRIC]} for scope in scopes
    ]

    with pytest.warns(DroppedTrialsWarning):
        result = trials_from_metrics(
            rows, params_key="params", metric_key=_METRIC, direction="maximize"
        )
    summary = plateau_summary(result.trials, axes=result.axes)

    assert result.dropped_count == 2
    assert len(result.trials) == 4
    assert result.axes == {
        "d_period": (3.0, 5.0),
        "k_period": (9.0, 14.0, float(_NEVER_TRADES)),
    }
    assert summary.axis_sizes == {"d_period": 2, "k_period": 3}
