"""Semantic metric catalog; presentation never limits calculated results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from tradingdev.domain.performance.sampling import DAILY_EQUITY_METRICS


@dataclass(frozen=True)
class MetricDefinition:
    """Stable meaning and applicability of a calculated metric."""

    id: str
    unit: str
    category: str
    provider: str
    description: str
    summary: bool = False
    optimization_direction: Literal["maximize", "minimize"] | None = None
    modes: tuple[str, ...] = ("signal", "volume")
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
    modes: tuple[str, ...] = ("signal", "volume"),
    requires_annualization: bool = False,
) -> MetricDefinition:
    if metric_id in DAILY_EQUITY_METRICS:
        description += (
            " Requires a recognized bar frequency of one day or finer; "
            "coarser or unknown bars cannot establish daily equity observations "
            "or allocate PnL across calendar boundaries."
        )
    return MetricDefinition(
        metric_id,
        unit,
        category,
        provider,
        description,
        summary,
        direction,
        modes,
        requires_annualization,
    )


_DEFINITIONS = [
    _definition(
        "total_pnl",
        "amount",
        "return",
        "ledger",
        "Net marked-to-market profit, including execution costs.",
        summary=True,
        direction="maximize",
    ),
    _definition(
        "total_return",
        "fraction",
        "return",
        "quantstats",
        "Cumulative return on initial capital.",
        summary=True,
        direction="maximize",
        modes=("signal",),
    ),
    _definition(
        "annual_return",
        "fraction",
        "return",
        "quantstats",
        "Compounded annual growth rate of observed UTC daily returns "
        "using configured days per year.",
        summary=True,
        direction="maximize",
        modes=("signal",),
        requires_annualization=True,
    ),
    _definition(
        "sharpe_ratio",
        "ratio",
        "risk",
        "quantstats",
        "Annualized excess daily return divided by sample standard deviation.",
        summary=True,
        direction="maximize",
        modes=("signal",),
        requires_annualization=True,
    ),
    _definition(
        "sortino_ratio",
        "ratio",
        "risk",
        "quantstats",
        "Annualized return above target divided by downside deviation.",
        summary=True,
        direction="maximize",
        modes=("signal",),
        requires_annualization=True,
    ),
    _definition(
        "calmar_ratio",
        "ratio",
        "risk",
        "quantstats",
        "Compounded annual return divided by maximum observed-day "
        "closing drawdown (daily_max_drawdown).",
        direction="maximize",
        modes=("signal",),
        requires_annualization=True,
    ),
    _definition(
        "annual_volatility",
        "fraction",
        "risk",
        "quantstats",
        "Annualized sample standard deviation of observed daily returns.",
        direction="minimize",
        modes=("signal",),
        requires_annualization=True,
    ),
    _definition(
        "max_drawdown",
        "fraction",
        "risk",
        "quantstats",
        "Largest peak-to-trough loss fraction, reported as nonnegative magnitude.",
        summary=True,
        direction="minimize",
        modes=("signal",),
    ),
    _definition(
        "daily_max_drawdown",
        "fraction",
        "risk",
        "quantstats",
        "Maximum observed-day closing percentage drawdown; "
        "denominator of Calmar ratio.",
        direction="minimize",
        modes=("signal",),
    ),
    _definition(
        "max_drawdown_amount",
        "amount",
        "risk",
        "ledger",
        "Largest peak-to-trough loss amount, including the initial "
        "capital/PnL baseline.",
        summary=True,
        direction="minimize",
    ),
    _definition(
        "total_trades",
        "count",
        "trade",
        "vectorbt",
        "Number of closed trades; open trades excluded.",
        summary=True,
        direction="maximize",
    ),
    _definition(
        "open_trades",
        "count",
        "trade",
        "vectorbt",
        "Number of trades still open at the final mark.",
    ),
    _definition(
        "win_rate",
        "fraction",
        "trade",
        "vectorbt",
        "Fraction of closed trades with positive net PnL, after entry and exit costs.",
        summary=True,
        direction="maximize",
    ),
    _definition(
        "profit_factor",
        "ratio",
        "trade",
        "vectorbt",
        "Closed-trade gross gains divided by absolute gross losses; "
        "all-winning is unbounded.",
        summary=True,
        direction="maximize",
    ),
    _definition(
        "trade_expectancy",
        "amount",
        "trade",
        "vectorbt",
        "Expected net profit per closed trade.",
        direction="maximize",
    ),
    _definition(
        "avg_holding_bars",
        "bars",
        "trade",
        "vectorbt",
        "Mean exit index minus entry index for closed trades.",
    ),
    _definition(
        "total_volume",
        "amount",
        "trade",
        "ledger",
        "Sum of actual entry and exit notional; open marks are not executions.",
        direction="maximize",
    ),
    _definition(
        "total_fees",
        "amount",
        "trade",
        "ledger",
        "Total charged entry and exit commissions (see cost model metadata).",
        direction="minimize",
    ),
    _definition(
        "total_slippage",
        "amount",
        "trade",
        "ledger",
        "Explicit slippage charges in the volume model; signal "
        "slippage is embedded in fill prices.",
    ),
    _definition(
        "n_days", "count", "period", "pandas", "Number of observed UTC calendar dates."
    ),
    _definition(
        "n_months",
        "count",
        "period",
        "pandas",
        "Number of observed UTC calendar months.",
    ),
    _definition(
        "monthly_trades_mean",
        "count",
        "period",
        "pandas",
        "Closed trades divided by observed calendar months.",
        direction="maximize",
    ),
    _definition(
        "monthly_volume_mean",
        "amount",
        "period",
        "pandas",
        "Executed notional divided by observed calendar months.",
        direction="maximize",
    ),
]
for _period in ("daily", "monthly"):
    for _stat in ("mean", "std", "min", "max", "median"):
        _DEFINITIONS.append(
            _definition(
                f"{_period}_pnl_{_stat}",
                "amount",
                "period",
                "numpy/pandas",
                f"{_stat} of net PnL aggregated by observed UTC "
                f"calendar {_period} periods; std uses ddof=0.",
                direction="minimize" if _stat == "std" else "maximize",
            )
        )

METRIC_CATALOG = {definition.id: definition for definition in _DEFINITIONS}


def metric_catalog(mode: str | None = None) -> list[dict[str, Any]]:
    """Describe supported metrics, optionally restricted to an execution mode."""
    return [
        asdict(definition)
        for definition in METRIC_CATALOG.values()
        if mode is None or mode in definition.modes
    ]


def summarize_metrics(metrics: dict[str, Any], mode: str) -> dict[str, Any]:
    """Select a compact view without changing stored or calculated metrics."""
    return {
        key: value
        for key, value in metrics.items()
        if key in METRIC_CATALOG
        and METRIC_CATALOG[key].summary
        and mode in METRIC_CATALOG[key].modes
    }
