"""Signal metric definitions, separate from portfolio-performance metrics.

Inputs are predictions and forward returns, not equity or executed trades. The
``book_*``-free return and risk entries describe the *simulated* long/short
book built from those predictions: costs are excluded and no order was ever
sent, so they are not backtest results either.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class SignalMetricDefinition:
    """Stable meaning and optimisation direction of a signal metric."""

    id: str
    unit: str
    category: str
    provider: str
    description: str
    summary: bool = False
    optimization_direction: Literal["maximize", "minimize"] | None = None
    # Signal metrics describe predictions, never executed volume. The two
    # fields below exist so a signal metric can be stored in the same shape as
    # a portfolio metric; without them a report cannot carry both.
    modes: tuple[str, ...] = ("signal",)
    requires_annualization: bool = False


def _definition(
    metric_id: str,
    unit: str,
    category: str,
    provider: str,
    description: str,
    *,
    summary: bool = False,
    direction: Literal["maximize", "minimize"] | None = None,
    annualized: bool = False,
) -> SignalMetricDefinition:
    return SignalMetricDefinition(
        metric_id,
        unit,
        category,
        provider,
        description,
        summary,
        direction,
        ("signal",),
        annualized,
    )


_DEFINITIONS = [
    _definition(
        "ic_mean",
        "fraction",
        "signal_quality",
        "numpy/pandas",
        "Mean rolling Pearson correlation between the prediction and the "
        "realised forward return.",
        summary=True,
        direction="maximize",
    ),
    _definition(
        "ic_std",
        "fraction",
        "signal_quality",
        "numpy/pandas",
        "Sample standard deviation (ddof=1) of the rolling IC series.",
        direction="minimize",
    ),
    _definition(
        "icir",
        "ratio",
        "signal_quality",
        "numpy/pandas",
        "IC information ratio: mean rolling IC divided by its standard deviation.",
        summary=True,
        direction="maximize",
    ),
    _definition(
        "ic_t_stat",
        "ratio",
        "signal_quality",
        "numpy/pandas",
        "ICIR scaled by the square root of non-overlapping window count; "
        "rolling windows overlap, so the effective sample is reduced.",
        direction="maximize",
    ),
    _definition(
        "ic_positive_rate",
        "fraction",
        "signal_quality",
        "numpy/pandas",
        "Fraction of rolling windows with a strictly positive IC.",
        direction="maximize",
    ),
    _definition(
        "rank_ic_mean",
        "fraction",
        "signal_quality",
        "numpy/pandas",
        "Mean rolling Spearman rank correlation between the prediction and "
        "the realised forward return (Rank IC).",
        summary=True,
        direction="maximize",
    ),
    _definition(
        "rank_ic_std",
        "fraction",
        "signal_quality",
        "numpy/pandas",
        "Sample standard deviation (ddof=1) of the rolling Rank IC series.",
        direction="minimize",
    ),
    _definition(
        "rank_icir",
        "ratio",
        "signal_quality",
        "numpy/pandas",
        "Rank IC information ratio: mean rolling Rank IC divided by its "
        "standard deviation.",
        summary=True,
        direction="maximize",
    ),
    _definition(
        "rank_ic_t_stat",
        "ratio",
        "signal_quality",
        "numpy/pandas",
        "Rank ICIR scaled by the square root of non-overlapping window count.",
        direction="maximize",
    ),
    _definition(
        "rank_ic_positive_rate",
        "fraction",
        "signal_quality",
        "numpy/pandas",
        "Fraction of rolling windows with a strictly positive Rank IC.",
        direction="maximize",
    ),
    _definition(
        "hit_rate",
        "fraction",
        "signal_quality",
        "numpy/pandas",
        "Fraction of bars where the sign of the prediction matches the sign "
        "of the realised forward return; bars with a zero on either side "
        "are excluded.",
        summary=True,
        direction="maximize",
    ),
    _definition(
        "signal_turnover",
        "ratio",
        "signal_stability",
        "numpy/pandas",
        "Mean absolute change of the prediction between consecutive bars; a "
        "proxy for the trading the signal would force.",
        direction="minimize",
    ),
    _definition(
        "strata_monotonicity",
        "ratio",
        "signal_quality",
        "numpy/pandas",
        "Spearman correlation between prediction-quantile rank and the mean "
        "forward return of that quantile; +1 is perfectly monotone.",
        summary=True,
        direction="maximize",
    ),
    _definition(
        "long_short_spread",
        "fraction",
        "signal_quality",
        "numpy/pandas",
        "Mean forward return of the top prediction quantile minus the mean "
        "forward return of the bottom quantile.",
        summary=True,
        direction="maximize",
    ),
    _definition(
        "annual_return",
        "fraction",
        "performance",
        "numpy/pandas",
        "Simulated long/short return compounded over the measured span and "
        "expressed per year; requires positive capital and verified continuous "
        "crypto holding intervals. Unavailable otherwise. Excludes "
        "costs and is not a mark-to-market equity curve.",
        summary=True,
        direction="maximize",
        annualized=True,
    ),
    _definition(
        "max_drawdown",
        "fraction",
        "risk",
        "numpy/pandas",
        "Largest peak-to-trough loss of the simulated long/short equity, as a "
        "positive fraction of the peak; 0.20 means a 20% drawdown.",
        summary=True,
        direction="minimize",
    ),
    _definition(
        "sharpe",
        "ratio",
        "risk_adjusted",
        "numpy/pandas",
        "Mean simulated per-period return divided by its standard deviation "
        "(ddof=1), scaled by the square root of periods per year. Assumes a "
        "zero risk-free rate and excludes costs.",
        summary=True,
        direction="maximize",
        annualized=True,
    ),
    _definition(
        "alpha",
        "fraction",
        "performance",
        "numpy/pandas",
        "Annualised mean simulated return left unexplained by the benchmark's "
        "own return, where the benchmark is holding the measured asset over "
        "the same intervals.",
        summary=True,
        direction="maximize",
        annualized=True,
    ),
    _definition(
        "beta",
        "ratio",
        "exposure",
        "numpy/pandas",
        "Sensitivity of the simulated long/short return to the benchmark's "
        "return over the same intervals; near zero means little directional "
        "market exposure. Exposure is described, not preferred, so it cannot "
        "rank candidates.",
        summary=True,
    ),
    _definition(
        "periods_per_year",
        "count",
        "period",
        "numpy/pandas",
        "Holding intervals in a year for the dataset's timeframe and horizon; "
        "the annualisation these return metrics assumed.",
    ),
    _definition(
        "n_return_periods",
        "count",
        "period",
        "pandas",
        "Holding intervals in the simulated return series; distinct from "
        "n_periods, which counts rolling IC windows.",
    ),
    _definition(
        "n_periods",
        "count",
        "period",
        "pandas",
        "Number of usable rolling IC windows.",
    ),
    _definition(
        "n_effective_periods",
        "count",
        "period",
        "pandas",
        "Non-overlapping window count (bars divided by window); the sample "
        "size behind ic_t_stat and rank_ic_t_stat.",
    ),
]

SIGNAL_METRIC_CATALOG = {definition.id: definition for definition in _DEFINITIONS}


def signal_metric_catalog() -> list[dict[str, Any]]:
    """Describe every supported signal evaluation metric."""
    return [asdict(definition) for definition in SIGNAL_METRIC_CATALOG.values()]


def summarize_signal_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Select a compact view without changing stored or calculated metrics."""
    return {
        key: value
        for key, value in metrics.items()
        if key in SIGNAL_METRIC_CATALOG and SIGNAL_METRIC_CATALOG[key].summary
    }


def definition_snapshot(metric_id: str) -> dict[str, Any]:
    """One signal metric in the shape a stored report expects.

    Field for field the report's own metric definition, so a report can carry
    signal metrics without a second vocabulary. ``modes`` is always
    ``("signal",)``: these metrics describe predictions, never executed volume,
    so a reader cannot mistake them for a backtest result.
    """
    definition = SIGNAL_METRIC_CATALOG.get(metric_id)
    if definition is None:
        raise KeyError(f"unknown signal metric: {metric_id}")
    return {
        "id": definition.id,
        "unit": definition.unit,
        "category": definition.category,
        "provider": definition.provider,
        "description": definition.description,
        "summary": definition.summary,
        "optimization_direction": definition.optimization_direction,
        "modes": list(definition.modes),
        "requires_annualization": definition.requires_annualization,
    }


def report_visibility(metric_id: str) -> bool:
    """Whether a signal metric is shown by default in a user-facing report.

    The rule is mechanical, so a new metric needs no separate discussion: show
    it when it is a summary metric *and* it declares an optimisation direction.
    A metric that cannot say which way is better (``beta`` and the period
    counts) is still stored and can be ticked on, but it is not put in the
    default view, where a reader would read it as quality.
    """
    definition = SIGNAL_METRIC_CATALOG.get(metric_id)
    if definition is None:
        raise KeyError(f"unknown signal metric: {metric_id}")
    return definition.summary and definition.optimization_direction is not None
