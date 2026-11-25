"""Independently check performance from observations and fixed run settings.

The arithmetic does not read an engine's metrics or call its metric calculator.
Shared metric meanings and daily-sampling eligibility follow the performance
catalog. This summary covers nonempty, timestamped runs; it is not a replacement
for the full saved performance artifact or its provenance checks.

Equity and trade net_pnl already include incurred execution costs. Never
subtract those costs again. Total P&L starts at the original capital (zero for
volume-mode cumulative P&L), including any loss before the first observation.
Completed runs under the current execution policy already include terminal
closing fills. Historical or caller-supplied open positions contribute their
last marked value, not hypothetical liquidation proceeds: this function does
not execute trades or invent unexecuted exit costs. Closed-trade statistics and
statistics including marked open trades are returned in separate groups.

Daily/monthly P&L dispersion describes the observed periods (ddof=0).
Sharpe uses daily excess returns and a sample standard deviation (ddof=1),
with the caller's fixed annualisation and risk-free settings. Missing calendar
periods are not filled. Square-root annualisation and sample standard deviations
do not correct serial dependence or guarantee unbiased Sharpe estimates.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tradingdev.domain.performance.sampling import daily_observation_unavailable_reason
from tradingdev.domain.statistics.periods import equity_series
from tradingdev.domain.statistics.sharpe import sharpe_ratio

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt


@dataclass(frozen=True)
class TradeSummary:
    """Net-P&L statistics over the explicitly named group of trade records.

    Break-even records count in total_trades and the win-rate denominator.
    net_pnl sums the group's already-net values. A profit factor without
    losses is unavailable; its reason distinguishes positive gains from 0/0.
    """

    total_trades: int
    winning_trades: int
    losing_trades: int
    net_pnl: float
    win_rate: float | None
    profit_factor: float | None
    unavailable: Mapping[str, str]


@dataclass(frozen=True)
class PerformanceSummary:
    """Independent overview with the settings needed to interpret its values.

    total_pnl and total_return use the original capital/P&L baseline.
    observed_equity_change instead measures last minus first observation;
    it is not a substitute for total P&L. max_drawdown is a nonnegative
    fraction and max_drawdown_amount an amount, both including the baseline.
    Return-based figures are inapplicable to volume mode even if its cumulative
    P&L happens to remain positive. Daily statistics require a declared bar
    frequency of one day or finer; calendar gaps do not determine that frequency.
    return_information_ratio is mean daily return divided by its sample standard
    deviation, scaled by sqrt(periods_per_year), without a risk-free subtraction.
    sharpe_ratio instead uses the run's configured risk-free benchmark.

    closed_trades follows the engine's closed-trade metric scope.
    marked_trades includes every provided record, including open positions
    at their last mark. Neither group proves that the caller supplied a complete
    trade history or observations from the same run. Unavailable values retain
    their reasons; the caller must report those reasons with the values.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    n_bars: int
    n_days: int
    n_months: int
    init_cash: float | None
    frequency: str
    periods_per_year: float | None
    risk_free_rate: float
    total_pnl: float
    observed_equity_change: float
    total_return: float | None
    max_drawdown: float | None
    max_drawdown_amount: float
    daily_pnl_mean: float | None
    daily_pnl_std: float | None
    monthly_pnl_mean: float | None
    monthly_pnl_std: float | None
    sharpe_ratio: float | None
    return_information_ratio: float | None
    closed_trades: TradeSummary
    marked_trades: TradeSummary
    unavailable: Mapping[str, str]


def daily_excess_returns(
    equity_curve: npt.ArrayLike,
    timestamps: npt.ArrayLike,
    *,
    init_cash: float,
    periods_per_year: float,
    risk_free_rate: float,
) -> pd.Series[float]:
    """Daily returns above the run's fixed risk-free rate.

    The first daily opening equity is the run's original capital, not the
    first observation. This is the exact excess-return series used by
    :func:`summarize_performance` for its Sharpe ratio, exposed so inference
    callers do not substitute the generic first-observation return series.
    Missing calendar days are omitted, never filled. The sample standard
    deviation used by the Sharpe functions is ``ddof=1``.
    """
    capital = _positive(init_cash, "init_cash")
    periods = _positive(periods_per_year, "periods_per_year")
    risk_free = _finite(risk_free_rate, "risk_free_rate")
    if risk_free <= -1:
        raise ValueError("risk_free_rate must be greater than -1")
    equity = equity_series(equity_curve, timestamps)
    return _daily_returns(equity, capital, periods, risk_free)[1]


def summarize_performance(
    equity_curve: npt.ArrayLike,
    timestamps: npt.ArrayLike,
    trades: Sequence[Mapping[str, object]],
    *,
    init_cash: float | None,
    frequency: str,
    periods_per_year: float | None,
    risk_free_rate: float,
) -> PerformanceSummary:
    """Recompute a nonempty run's overview from raw observations.

    Pass the run's fixed settings explicitly: positive init_cash for an
    account-equity curve, or None for cumulative P&L with a zero baseline;
    the declared bar frequency; daily periods_per_year (None if unavailable);
    and annual risk_free_rate. Rates may be negative but must exceed -1.
    No setting is inferred from equity or a timestamp gap.

    Trades require finite numeric net_pnl and an explicit status of
    "closed" or "open". Values already include incurred costs. All
    supplied records contribute to marked_trades; only closed records
    contribute to closed_trades.

    Raises:
        ValueError: For invalid settings or trades, or nonfinite derived
            equity changes, period rates or aggregate P&L. Undefined ratios
            are returned with reasons. Equity/timestamp validation and
            conversion errors follow equity_series.
    """
    capital = None if init_cash is None else _positive(init_cash, "init_cash")
    periods = (
        None
        if periods_per_year is None
        else _positive(periods_per_year, "periods_per_year")
    )
    risk_free = _finite(risk_free_rate, "risk_free_rate")
    if risk_free <= -1:
        raise ValueError("risk_free_rate must be greater than -1")
    if not isinstance(frequency, str):
        raise ValueError("frequency must be a declared bar-frequency string")
    sampling_reason = daily_observation_unavailable_reason(frequency)
    equity = equity_series(equity_curve, timestamps)
    values = equity.to_numpy()
    initial = 0.0 if capital is None else capital
    baseline = np.concatenate(([initial], values))
    with np.errstate(over="ignore", invalid="ignore"):
        changes = np.diff(baseline)
        drawdowns = np.maximum.accumulate(baseline) - baseline
    if not np.isfinite(changes).all() or not np.isfinite(drawdowns).all():
        raise ValueError("derived equity changes must be finite")

    daily = equity.resample("D").last().dropna()
    monthly = equity.resample("MS").last().dropna()
    daily_pnl = _period_changes(daily, initial)
    monthly_pnl = _period_changes(monthly, initial)
    unavailable: dict[str, str] = {}
    pnl_stats: dict[str, float | None] = {}
    for prefix, pnl in (("daily", daily_pnl), ("monthly", monthly_pnl)):
        for stat in ("mean", "std"):
            name = f"{prefix}_pnl_{stat}"
            if sampling_reason is not None:
                pnl_stats[name] = None
                unavailable[name] = sampling_reason
            else:
                value = float(pnl.mean() if stat == "mean" else pnl.std(ddof=0))
                pnl_stats[name] = _finite(value, name)

    total_return: float | None = None
    max_drawdown: float | None = None
    sharpe: float | None = None
    return_information: float | None = None
    reason: str | None
    if capital is None:
        reason = "not_applicable"
    elif not ((baseline[:-1] > 0).all() and (values >= 0).all()):
        reason = "nonpositive_capital"
    else:
        reason = None
    if reason is not None:
        for name in (
            "total_return",
            "max_drawdown",
            "sharpe_ratio",
            "return_information_ratio",
        ):
            unavailable[name] = reason
    else:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            bar_returns = changes / baseline[:-1]
        if not np.isfinite(bar_returns).all():
            raise ValueError("derived returns must be finite")
        total_return = _finite(float(values[-1] / initial - 1), "total_return")
        max_drawdown = float(np.max(drawdowns / np.maximum.accumulate(baseline)))
        if sampling_reason is not None:
            unavailable["sharpe_ratio"] = sampling_reason
            unavailable["return_information_ratio"] = sampling_reason
        elif periods is None:
            unavailable["sharpe_ratio"] = "missing_annualization"
            unavailable["return_information_ratio"] = "missing_annualization"
        elif len(daily) < 2:
            unavailable["sharpe_ratio"] = "insufficient_data"
            unavailable["return_information_ratio"] = "insufficient_data"
        else:
            returns, excess = _daily_returns(equity, initial, periods, risk_free)
            sharpe = _ratio(excess, periods, "sharpe_ratio", unavailable)
            return_information = _ratio(
                returns, periods, "return_information_ratio", unavailable
            )

    closed, marked = _trade_groups(trades)
    return PerformanceSummary(
        start=pd.Timestamp(equity.index[0]),
        end=pd.Timestamp(equity.index[-1]),
        n_bars=len(equity),
        n_days=len(daily),
        n_months=len(monthly),
        init_cash=capital,
        frequency=frequency,
        periods_per_year=periods,
        risk_free_rate=risk_free,
        total_pnl=_finite(float(values[-1] - initial), "total_pnl"),
        observed_equity_change=_finite(
            float(values[-1] - values[0]), "observed_equity_change"
        ),
        total_return=total_return,
        max_drawdown=max_drawdown,
        max_drawdown_amount=float(drawdowns.max()),
        daily_pnl_mean=pnl_stats["daily_pnl_mean"],
        daily_pnl_std=pnl_stats["daily_pnl_std"],
        monthly_pnl_mean=pnl_stats["monthly_pnl_mean"],
        monthly_pnl_std=pnl_stats["monthly_pnl_std"],
        sharpe_ratio=sharpe,
        return_information_ratio=return_information,
        closed_trades=_summarize_trades(closed),
        marked_trades=_summarize_trades(marked),
        unavailable=MappingProxyType(unavailable),
    )


def _ratio(
    returns: pd.Series[float],
    periods: float,
    name: str,
    unavailable: dict[str, str],
) -> float | None:
    ratio = sharpe_ratio(returns, periods_per_year=periods)
    if ratio is None:
        unavailable[name] = (
            "zero_denominator" if bool((returns == 0).all()) else "unbounded"
        )
    elif not math.isfinite(ratio):
        unavailable[name] = "unbounded"
        return None
    return ratio


def _daily_returns(
    equity: pd.Series[float], initial: float, periods: float, risk_free: float
) -> tuple[pd.Series[float], pd.Series[float]]:
    """Raw and risk-free-adjusted daily returns on the capital-inclusive basis."""
    daily = equity.resample("D").last().dropna()
    daily_opening = daily.shift(1).fillna(initial)
    returns = _period_changes(daily, initial) / daily_opening
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        daily_rate = _finite(
            float(np.expm1(np.log1p(risk_free) / periods)),
            "daily risk-free rate",
        )
    return returns, returns - daily_rate


def _period_changes(closing: pd.Series[float], initial: float) -> pd.Series[float]:
    changes = closing - closing.shift(1).fillna(initial)
    if not np.isfinite(changes.to_numpy()).all():
        raise ValueError("derived period P&L must be finite")
    return changes


def _trade_groups(
    trades: Sequence[Mapping[str, object]],
) -> tuple[list[float], list[float]]:
    closed: list[float] = []
    marked: list[float] = []
    for position, trade in enumerate(trades):
        pnl = _finite(trade.get("net_pnl"), f"trade {position} net_pnl")
        status = trade.get("status")
        if status not in ("closed", "open"):
            raise ValueError(f"trade {position} requires a closed/open status")
        marked.append(pnl)
        if status == "closed":
            closed.append(pnl)
    return closed, marked


def _summarize_trades(pnls: Sequence[float]) -> TradeSummary:
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit = _finite(sum(wins), "winning trade P&L")
    gross_loss = _finite(-sum(losses), "losing trade P&L")
    unavailable: dict[str, str] = {}
    profit_factor = None
    if not pnls:
        unavailable.update(win_rate="no_trades", profit_factor="no_trades")
    elif gross_loss == 0:
        unavailable["profit_factor"] = (
            "unbounded" if gross_profit > 0 else "zero_denominator"
        )
    else:
        profit_factor = gross_profit / gross_loss
        if not math.isfinite(profit_factor):
            profit_factor = None
            unavailable["profit_factor"] = "unbounded"
    return TradeSummary(
        total_trades=len(pnls),
        winning_trades=len(wins),
        losing_trades=len(losses),
        net_pnl=_finite(sum(pnls), "trade net P&L"),
        win_rate=len(wins) / len(pnls) if pnls else None,
        profit_factor=profit_factor,
        unavailable=MappingProxyType(unavailable),
    )


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{label} must be a finite real number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} must be a finite real number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite real number")
    return result


def _positive(value: object, label: str) -> float:
    result = _finite(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result
