"""Signal metric definitions, separate from portfolio-performance metrics.

Inputs are predictions and forward returns, not equity or executed trades.
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


def _definition(
    metric_id: str,
    unit: str,
    category: str,
    provider: str,
    description: str,
    *,
    summary: bool = False,
    direction: Literal["maximize", "minimize"] | None = None,
) -> SignalMetricDefinition:
    return SignalMetricDefinition(
        metric_id,
        unit,
        category,
        provider,
        description,
        summary,
        direction,
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
