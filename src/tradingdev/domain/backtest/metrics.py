"""Third-party return/risk and closed-trade statistics with explicit semantics."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import quantstats as qs
import vectorbt as vbt
from vectorbt.portfolio.enums import trade_dt

from tradingdev.domain.performance.catalog import METRIC_CATALOG
from tradingdev.domain.performance.sampling import (
    DAILY_EQUITY_METRICS,
    daily_observation_unavailable_reason,
)


@dataclass
class PerformanceAnalysis:
    """Calculated values and the settings required to interpret them."""

    metrics: dict[str, Any]
    metadata: dict[str, Any]
    returns: npt.NDArray[np.float64] | None


def timestamp_index(df: pd.DataFrame) -> pd.DatetimeIndex | None:
    """Use the market timestamps, normalized to UTC, never a synthetic date."""
    values = df.get("timestamp", df.index)
    if "timestamp" not in df and not isinstance(df.index, pd.DatetimeIndex):
        return None
    index = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    if index.hasnans or not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("Market timestamps must be valid, unique, and increasing")
    return index


def normalized_trades(trades: vbt.Trades) -> list[dict[str, Any]]:
    """Export the same vectorbt records used for statistics and reconciliation."""
    result = []
    for record in trades.records_arr:
        entry_notional = float(record["size"] * record["entry_price"])
        is_closed = int(record["status"]) == 1
        fee = float(record["entry_fees"] + record["exit_fees"])
        result.append(
            {
                "direction": 1 if int(record["direction"]) == 0 else -1,
                "entry_idx": int(record["entry_idx"]),
                "exit_idx": int(record["exit_idx"]),
                "entry_price": float(record["entry_price"]),
                "exit_price": float(record["exit_price"]),
                "size": float(record["size"]),
                "size_quote": entry_notional,
                "exit_notional": float(record["size"] * record["exit_price"])
                if is_closed
                else 0.0,
                "entry_fees": float(record["entry_fees"]),
                "exit_fees": float(record["exit_fees"]),
                "fee": fee,
                "gross_pnl": float(record["pnl"]) + fee,
                "net_pnl": float(record["pnl"]),
                "status": "closed" if is_closed else "open",
            }
        )
    return result


def calculate_metrics(
    pf: vbt.Portfolio,
    *,
    periods_per_year: float | None = None,
    risk_free_rate: float = 0.0,
    required_return: float = 0.0,
    frequency: str = "1h",
    timestamps: pd.DatetimeIndex | None = None,
) -> PerformanceAnalysis:
    """Analyze portfolio returns with QuantStats and trades with vectorbt."""
    orders = pf.orders.records_arr
    return _analyze(
        np.asarray(pf.value(), dtype=np.float64),
        pf.trades,
        float(pf.init_cash),
        timestamps,
        float(np.sum(orders["size"] * orders["price"])),
        float(np.sum(orders["fees"])),
        None,
        periods_per_year,
        risk_free_rate,
        required_return,
        frequency,
    )


def calculate_metrics_from_simulation(
    equity_curve: npt.NDArray[np.float64],
    trades: list[dict[str, Any]],
    init_cash: float | None,
    timestamps: pd.DatetimeIndex | None = None,
    *,
    periods_per_year: float | None = None,
    risk_free_rate: float = 0.0,
    required_return: float = 0.0,
    frequency: str = "1h",
) -> PerformanceAnalysis:
    """Normalize an existing ledger into vectorbt records without simulation."""
    records = np.zeros(len(trades), dtype=trade_dt)
    for i, trade in enumerate(trades):
        notional = float(trade["size_quote"])
        records[i] = (
            i,
            0,
            trade["size"],
            trade["entry_idx"],
            trade["entry_price"],
            trade["entry_fees"] + trade.get("entry_slippage", 0.0),
            trade["exit_idx"],
            trade["exit_price"],
            trade["exit_fees"] + trade.get("exit_slippage", 0.0),
            trade["net_pnl"],
            trade["net_pnl"] / notional,
            0 if trade["direction"] == 1 else 1,
            1 if trade["status"] == "closed" else 0,
            i,
        )
    index = timestamps if timestamps is not None else pd.RangeIndex(len(equity_curve))
    wrapper = vbt.ArrayWrapper(index, [0], ndim=1, freq=frequency)
    vbt_trades = vbt.Trades(wrapper, records, close=np.ones(len(equity_curve)))
    return _analyze(
        equity_curve,
        vbt_trades,
        init_cash,
        timestamps,
        sum(float(t["size_quote"] + t["exit_notional"]) for t in trades),
        sum(float(t["fee"]) for t in trades),
        sum(float(t.get("slippage", 0.0)) for t in trades),
        periods_per_year,
        risk_free_rate,
        required_return,
        frequency,
    )


def _analyze(
    equity: npt.NDArray[np.float64],
    trades: vbt.Trades,
    init_cash: float | None,
    timestamps: pd.DatetimeIndex | None,
    total_volume: float,
    total_fees: float,
    total_slippage: float | None,
    periods_per_year: float | None,
    risk_free_rate: float,
    required_return: float,
    frequency: str,
) -> PerformanceAnalysis:
    if not np.all(np.isfinite(equity)):
        raise ValueError("Equity must contain only finite values")
    if timestamps is not None and len(timestamps) != len(equity):
        raise ValueError("Timestamp and equity lengths must match")
    daily_sampling_reason = daily_observation_unavailable_reason(frequency)
    mode = "volume" if init_cash is None else "signal"
    metrics: dict[str, Any] = dict.fromkeys(METRIC_CATALOG)
    unavailable: dict[str, str] = {}
    fallbacks: dict[str, dict[str, str]] = {}
    for metric_id, definition in METRIC_CATALOG.items():
        if mode not in definition.modes:
            unavailable[metric_id] = "not_applicable"
    initial = init_cash if init_cash is not None else 0.0
    metrics.update(
        total_pnl=float(equity[-1]) - initial if len(equity) else 0.0,
        total_volume=total_volume,
        total_fees=total_fees,
        total_slippage=total_slippage,
        total_trades=int(trades.closed.count()),
        open_trades=int(trades.open.count()),
    )
    if total_slippage is None:
        unavailable["total_slippage"] = "embedded_in_fill_prices"
    baseline_equity = np.concatenate(([initial], equity))
    with np.errstate(over="ignore", invalid="ignore"):
        drawdowns = np.maximum.accumulate(baseline_equity) - baseline_equity
        pnl = np.diff(baseline_equity)
    if not np.all(np.isfinite(pnl)) or not np.all(np.isfinite(drawdowns)):
        raise ValueError("Equity differences must be finite")
    metrics["max_drawdown_amount"] = float(np.max(drawdowns))
    _period_metrics(metrics, unavailable, pnl, timestamps, daily_sampling_reason)
    if metrics["total_trades"]:
        closed = trades.closed
        with np.errstate(divide="ignore", invalid="ignore"):
            metrics.update(
                win_rate=float(closed.win_rate()),
                profit_factor=float(closed.profit_factor()),
                trade_expectancy=float(closed.expectancy()),
                avg_holding_bars=float(closed.duration.mean()),
            )
    else:
        for metric_id in (
            "win_rate",
            "profit_factor",
            "trade_expectancy",
            "avg_holding_bars",
        ):
            unavailable[metric_id] = "no_trades"
    returns = None
    if init_cash is not None:
        if len(equity) and np.all(baseline_equity[:-1] > 0) and np.all(equity >= 0):
            with np.errstate(over="ignore", invalid="ignore"):
                returns = np.asarray(pnl / baseline_equity[:-1], dtype=np.float64)
            if not np.all(np.isfinite(returns)):
                raise ValueError("Derived returns must be finite")
            _return_metrics(
                metrics,
                unavailable,
                returns,
                timestamps,
                periods_per_year,
                risk_free_rate,
                required_return,
                daily_sampling_reason,
                fallbacks,
            )
        else:
            for metric_id, definition in METRIC_CATALOG.items():
                if definition.provider == "quantstats":
                    unavailable[metric_id] = (
                        "insufficient_data"
                        if not len(equity)
                        else "nonpositive_capital"
                    )
    for metric_id, value in metrics.items():
        if value is not None and not np.isfinite(value):
            metrics[metric_id] = None
            unavailable[metric_id] = (
                "unbounded" if np.isinf(value) else "zero_denominator"
            )
        elif value is None and metric_id not in unavailable:
            unavailable[metric_id] = "insufficient_data"
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "providers": {
            "quantstats": version("quantstats"),
            "vectorbt": version("vectorbt"),
            **({"numpy": version("numpy")} if fallbacks else {}),
        },
        "mode": mode,
        "settings": {
            "initial_cash": init_cash,
            "periods_per_year": periods_per_year,
            "frequency": frequency,
            "risk_free_rate": risk_free_rate,
            "required_return": required_return,
            "risk_free_rate_per_period": _period_rate(risk_free_rate, periods_per_year)
            if daily_sampling_reason is None
            else None,
            "required_return_per_period": _period_rate(
                required_return, periods_per_year
            )
            if daily_sampling_reason is None
            else None,
            "returns_basis": "simple" if init_cash is not None else "not_applicable",
            "return_sampling": "observed_daily"
            if daily_sampling_reason is None
            else "unavailable",
            "stored_returns_sampling": "bar",
            "drawdown_sampling": "bar",
            "calmar_drawdown_sampling": "observed_daily"
            if daily_sampling_reason is None
            else "unavailable",
            "missing_dates": "not_filled",
            "trade_scope": "closed",
            "calendar_timezone": "UTC",
            "cost_model": "commission_and_separate_notional_slippage_charges"
            if mode == "volume"
            else "commission_and_slippage_adjusted_fill_prices",
            "return_std_ddof": 1,
            "period_pnl_std_ddof": 0,
            "excess_return_adjustment": "configured_rate_before_quantstats",
        },
        "unavailable": unavailable,
        "calculation_fallbacks": fallbacks,
    }
    return PerformanceAnalysis(metrics, metadata, returns)


def _period_rate(annual_rate: float, periods_per_year: float | None) -> float | None:
    if periods_per_year is None:
        return None
    return float(np.expm1(np.log1p(annual_rate) / periods_per_year))


def _return_metrics(
    metrics: dict[str, Any],
    unavailable: dict[str, str],
    returns: npt.NDArray[np.float64],
    timestamps: pd.DatetimeIndex | None,
    periods_per_year: float | None,
    risk_free_rate: float,
    required_return: float,
    daily_sampling_reason: str | None,
    fallbacks: dict[str, dict[str, str]],
) -> None:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        series = pd.Series(returns, index=timestamps)
        metrics["total_return"] = _scalar(qs.stats.comp(series))
        metrics["max_drawdown"] = _maximum_drawdown(series, "max_drawdown", fallbacks)
        if daily_sampling_reason is not None:
            for metric_id in DAILY_EQUITY_METRICS:
                if METRIC_CATALOG[metric_id].provider == "quantstats":
                    unavailable[metric_id] = daily_sampling_reason
            return
        if timestamps is not None:
            daily_returns = series.add(1).groupby(timestamps.normalize()).prod() - 1
            metrics["daily_max_drawdown"] = _maximum_drawdown(
                daily_returns, "daily_max_drawdown", fallbacks
            )
        else:
            unavailable["daily_max_drawdown"] = "missing_timestamps"
        if periods_per_year is None:
            for metric_id, definition in METRIC_CATALOG.items():
                if definition.requires_annualization:
                    unavailable[metric_id] = "missing_annualization"
            return
        if timestamps is None:
            for metric_id, definition in METRIC_CATALOG.items():
                if (
                    definition.requires_annualization
                    or metric_id == "daily_max_drawdown"
                ):
                    unavailable[metric_id] = "missing_timestamps"
            return
        # QuantStats annotates periods as int, but these statistics only divide
        # by it or take its square root. Preserve supported fractional settings;
        # cast is a typing bridge, not an int() conversion or calendar rounding.
        periods = cast("int", periods_per_year)
        if _prepared_as_prices(daily_returns):
            _fallback(fallbacks, "annual_return", "quantstats_returns_price_heuristic")
            metrics["annual_return"] = float(
                np.power(
                    qs.stats.comp(daily_returns) + 1,
                    periods_per_year / len(daily_returns),
                )
                - 1
            )
        else:
            metrics["annual_return"] = _scalar(
                qs.stats.cagr(daily_returns, periods=periods)
            )
        # No observed drawdown has no finite Calmar interpretation. QuantStats
        # otherwise returns infinity here, unlike this metric's saved definition.
        if metrics["daily_max_drawdown"] == 0:
            unavailable["calmar_ratio"] = "zero_denominator"
        else:
            metrics["calmar_ratio"] = float(
                qs.stats.calmar(daily_returns, prepare_returns=False, periods=periods)
            )
        if len(daily_returns) < 2:
            for metric_id in ("sharpe_ratio", "sortino_ratio", "annual_volatility"):
                unavailable[metric_id] = "insufficient_data"
            return
        # QuantStats 0.0.86 ignores negative scalar rf. Adjust explicitly so
        # Sharpe's risk-free rate and Sortino's separate target both keep their
        # established meaning, for positive as well as negative annual rates.
        risk_free = _period_rate(risk_free_rate, periods_per_year)
        target = _period_rate(required_return, periods_per_year)
        assert risk_free is not None and target is not None
        excess = daily_returns - risk_free
        if _prepared_as_prices(excess):
            _fallback(fallbacks, "sharpe_ratio", "quantstats_returns_price_heuristic")
            metrics["sharpe_ratio"] = float(
                excess.mean() / excess.std(ddof=1) * np.sqrt(periods_per_year)
            )
        else:
            metrics["sharpe_ratio"] = _scalar(
                qs.stats.sharpe(excess, rf=0.0, periods=periods)
            )
        excess = daily_returns - target
        if not (excess < 0).any():
            # Keep all-winning returns distinct from a flat zero denominator.
            unavailable["sortino_ratio"] = (
                "unbounded" if excess.mean() > 0 else "zero_denominator"
            )
        else:
            # A negative observation also makes QuantStats' price heuristic
            # unambiguous, so this path never needs a numerical fallback.
            metrics["sortino_ratio"] = _scalar(
                qs.stats.sortino(excess, rf=0.0, periods=periods)
            )
        metrics["annual_volatility"] = _scalar(
            qs.stats.volatility(daily_returns, periods=periods, prepare_returns=False)
        )


def _scalar(value: float | pd.Series) -> float:
    """QuantStats' union annotation also covers DataFrame inputs we never pass."""
    if isinstance(value, pd.Series):
        raise TypeError("Single-series performance statistics must return a scalar")
    return float(value)


def _prepared_as_prices(returns: pd.Series) -> bool:
    """Detect valid returns misread as prices by QuantStats' public API."""
    return bool(returns.min() >= 0 and returns.max() > 1)


def _fallback(
    fallbacks: dict[str, dict[str, str]], metric_id: str, reason: str
) -> None:
    fallbacks[metric_id] = {"provider": "numpy", "reason": reason}


def _maximum_drawdown(
    returns: pd.Series,
    metric_id: str,
    fallbacks: dict[str, dict[str, str]],
) -> float:
    """Use QuantStats without inventing dates or misidentifying large returns."""
    if not isinstance(returns.index, pd.DatetimeIndex):
        # QuantStats subtracts a Timedelta from the first index; a RangeIndex
        # cannot represent that phantom point. A calendar is not needed for MDD.
        _fallback(fallbacks, metric_id, "quantstats_drawdown_requires_timestamps")
        equity = np.concatenate(([1.0], np.asarray(qs.stats.compsum(returns)) + 1))
        return float(np.max(1 - equity / np.maximum.accumulate(equity)))
    if returns.min() >= 0 and returns.max() >= 1:
        # Its drawdown API uses a different heuristic from _prepare_returns:
        # exactly +100% can also be mistaken for a price. Nonnegative returns
        # cannot create a drawdown, regardless of their magnitude.
        _fallback(fallbacks, metric_id, "quantstats_returns_price_heuristic")
        return 0.0
    return abs(float(qs.stats.max_drawdown(returns)))


def _period_metrics(
    metrics: dict[str, Any],
    unavailable: dict[str, str],
    pnl: npt.NDArray[np.float64],
    timestamps: pd.DatetimeIndex | None,
    daily_sampling_reason: str | None,
) -> None:
    if timestamps is None or not len(pnl):
        reason = "missing_timestamps" if timestamps is None else "insufficient_data"
        for metric_id, definition in METRIC_CATALOG.items():
            if definition.category == "period":
                unavailable[metric_id] = reason
        return
    if len(timestamps) != len(pnl):
        raise ValueError("Timestamp and equity lengths must match")
    series = pd.Series(pnl, index=timestamps)
    for prefix, groups in (
        ("daily", timestamps.date),
        ("monthly", timestamps.tz_localize(None).to_period("M")),
    ):
        metrics["n_days" if prefix == "daily" else "n_months"] = len(
            pd.Index(groups).unique()
        )
        values = (
            np.asarray(series.groupby(groups).sum(), dtype=np.float64)
            if daily_sampling_reason is None
            else None
        )
        for stat, func in (
            ("mean", np.mean),
            ("std", np.std),
            ("min", np.min),
            ("max", np.max),
            ("median", np.median),
        ):
            metric_id = f"{prefix}_pnl_{stat}"
            if values is None:
                assert daily_sampling_reason is not None
                unavailable[metric_id] = daily_sampling_reason
            else:
                metrics[metric_id] = float(func(values))
    metrics["monthly_trades_mean"] = metrics["total_trades"] / metrics["n_months"]
    metrics["monthly_volume_mean"] = metrics["total_volume"] / metrics["n_months"]
