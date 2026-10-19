"""Prediction-quality metrics against forward returns, separate from backtests."""

from __future__ import annotations

from tradingdev.domain.ml.evaluation.artifacts import (
    SignalEvaluationBundle,
    SignalMetricDefinitionSnapshot,
    StratumStat,
    build_signal_evaluation,
    strata_records,
)
from tradingdev.domain.ml.evaluation.catalog import (
    SIGNAL_METRIC_CATALOG,
    SignalMetricDefinition,
    signal_metric_catalog,
    summarize_signal_metrics,
)
from tradingdev.domain.ml.evaluation.metrics import (
    IC_METHODS,
    forward_returns,
    hit_rate,
    ic_decay,
    ic_statistics,
    rolling_ic,
    signal_metrics,
    signal_turnover,
)
from tradingdev.domain.ml.evaluation.simulated_returns import (
    SimulatedReturns,
    simulate_strategy_returns,
)
from tradingdev.domain.ml.evaluation.stratified import (
    quantile_strata,
    strata_statistics,
)

__all__ = [
    "IC_METHODS",
    "SIGNAL_METRIC_CATALOG",
    "SignalEvaluationBundle",
    "SignalMetricDefinition",
    "SignalMetricDefinitionSnapshot",
    "SimulatedReturns",
    "StratumStat",
    "build_signal_evaluation",
    "forward_returns",
    "hit_rate",
    "ic_decay",
    "ic_statistics",
    "quantile_strata",
    "rolling_ic",
    "signal_metric_catalog",
    "signal_metrics",
    "signal_turnover",
    "simulate_strategy_returns",
    "strata_records",
    "strata_statistics",
    "summarize_signal_metrics",
]
