"""Analytical computations for the dashboard."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd


def format_metric(value: object, spec: str, suffix: str = "") -> str:
    """Format a finite numeric KPI, keeping unavailable values explicit."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
    ):
        return "N/A"
    return f"{value:{spec}}{suffix}"


def metric_cards(metrics: dict[str, Any], *, mode: str) -> list[tuple[str, str]]:
    """Select compact KPIs without changing a metric's unit by mode."""
    volume_mode = mode == "volume"
    definitions = [
        ("Total P&L", "total_pnl", "+,.2f")
        if volume_mode
        else ("Total Return", "total_return", ".2%"),
        ("Sharpe", "sharpe_ratio", ".3f"),
        ("Max DD (amount)", "max_drawdown_amount", ",.2f")
        if volume_mode
        else ("Max DD", "max_drawdown", ".2%"),
        ("Win Rate", "win_rate", ".1%"),
        ("Closed Trades", "total_trades", ",.0f"),
        ("Volume", "total_volume", ",.2f"),
    ]
    return [
        (label, format_metric(metrics.get(key), spec))
        for label, key, spec in definitions
    ]


def build_equity_series(
    equity_curve: npt.NDArray[np.float64],
    timestamps: npt.NDArray[Any] | None,
) -> pd.Series[float]:
    """Convert raw equity array to a pandas Series with datetime index."""
    if timestamps is not None:
        index: pd.DatetimeIndex | pd.RangeIndex = pd.DatetimeIndex(timestamps)
    else:
        index = pd.RangeIndex(len(equity_curve))
    return pd.Series(equity_curve, index=index, dtype=float)


def build_trades_df(
    trades: list[dict[str, Any]],
    timestamps: npt.NDArray[Any] | None = None,
) -> pd.DataFrame:
    """Map recorded entry/exit bar indices to exact execution timestamps."""
    if not trades:
        return pd.DataFrame(
            columns=[
                "direction",
                "entry_price",
                "exit_price",
                "size_quote",
                "gross_pnl",
                "fee",
                "net_pnl",
                "status",
                "entry_timestamp",
                "exit_timestamp",
                "timestamp",
                "exit_notional",
            ]
        )
    df = pd.DataFrame(trades)
    if "status" not in df:
        df["status"] = None
    for side in ("entry", "exit"):
        time_key = f"{side}_timestamp"
        if time_key not in df.columns:
            recorded = dict(enumerate(timestamps)) if timestamps is not None else {}
            indices = df.get(f"{side}_idx", pd.Series(index=df.index, dtype=float))
            df[time_key] = pd.to_datetime(indices.map(recorded), errors="coerce")
    df["timestamp"] = df["exit_timestamp"].where(df["status"] == "closed")
    return df


def cumulative_pnl(
    equity: pd.Series[float],
    init_cash: float | None,
) -> pd.Series[float]:
    """Compute cumulative PnL (absolute) from equity curve.

    When ``init_cash`` is ``None`` (volume mode), the equity curve
    already represents cumulative P&L from zero.
    """
    if init_cash is None:
        return equity
    return equity - init_cash


def cumulative_pnl_pct(
    equity: pd.Series[float],
    init_cash: float | None,
) -> pd.Series[float]:
    """Compute cumulative PnL as percentage.

    Values are unavailable without a positive capital base.
    """
    if init_cash is None or init_cash <= 0:
        return pd.Series(np.nan, index=equity.index, dtype=float)
    return (equity - init_cash) / init_cash * 100


def consecutive_loss_counts(
    trades_df: pd.DataFrame,
) -> pd.Series[int]:
    """Count frequencies of consecutive-loss streaks.

    Returns a Series where the index is the streak length and
    values are how many times that streak length occurred.
    """
    if trades_df.empty or "net_pnl" not in trades_df.columns:
        return pd.Series(dtype=int)

    closed = trades_df.loc[trades_df["status"] == "closed"]
    is_loss = (closed["net_pnl"] < 0).astype(int).values
    streaks: list[int] = []
    current = 0
    for v in is_loss:
        if v == 1:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)

    if not streaks:
        return pd.Series(dtype=int)

    s = pd.Series(streaks)
    counts = s.value_counts().sort_index()
    counts.index.name = "consecutive_losses"
    counts.name = "frequency"
    return counts


def rolling_mdd_absolute(
    equity: pd.Series[float],
    window_bars: int | pd.Timedelta,
) -> pd.Series[float]:
    """Maximum ordered peak-to-trough loss within each trailing window.

    Integer windows require that many observations. Timedelta windows use
    (t - duration, t] and require duration elapsed since the first observation.
    Missing/nonfinite observations make their window unavailable. Two aggregate
    stacks evict expired peaks in amortized O(n), including dense minute data.
    """
    times: pd.DatetimeIndex | None = None
    if isinstance(window_bars, pd.Timedelta):
        if window_bars <= pd.Timedelta(0) or pd.isna(window_bars):
            raise ValueError("Window duration must be positive")
        if (
            not isinstance(equity.index, pd.DatetimeIndex)
            or equity.index.hasnans
            or not equity.index.is_monotonic_increasing
            or not equity.index.is_unique
        ):
            raise ValueError("Calendar windows require a sorted unique DatetimeIndex")
        times = equity.index
    elif (
        isinstance(window_bars, bool)
        or not isinstance(window_bars, int)
        or window_bars < 1
    ):
        raise ValueError("Window bars must be a positive integer")

    # Each summary is (maximum, minimum, ordered drawdown, invalid count).
    type Summary = tuple[float, float, float, int]

    def combine(left: Summary, right: Summary) -> Summary:
        return (
            max(left[0], right[0]),
            min(left[1], right[1]),
            max(left[2], right[2], left[0] - right[1]),
            left[3] + right[3],
        )

    identity: Summary = (-math.inf, math.inf, 0.0, 0)
    front: list[tuple[Summary, Summary]] = []
    back: list[tuple[Summary, Summary]] = []
    output = np.full(len(equity), np.nan)
    start = 0
    for i, value in enumerate(equity.to_numpy(dtype=float)):
        item: Summary = (
            (float(value), float(value), 0.0, 0)
            if math.isfinite(value)
            else (*identity[:3], 1)
        )
        back.append((item, combine(back[-1][1] if back else identity, item)))
        if times is not None and isinstance(window_bars, pd.Timedelta):
            cutoff = times[i] - window_bars
            target = start
            while times[target] <= cutoff:
                target += 1
            ready = times[i] - times[0] >= window_bars
        else:
            assert isinstance(window_bars, int)
            target = max(0, i - window_bars + 1)
            ready = i + 1 >= window_bars
        while start < target:
            if not front:
                while back:
                    item, _ = back.pop()
                    front.append(
                        (item, combine(item, front[-1][1] if front else identity))
                    )
            front.pop()
            start += 1
        summary = combine(
            front[-1][1] if front else identity, back[-1][1] if back else identity
        )
        if ready and not summary[3]:
            output[i] = summary[2]
    return pd.Series(output, index=equity.index, name=equity.name)


def monthly_volume(
    trades_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate total traded volume by month.

    Returns DataFrame with columns ``month`` (str) and ``volume_quote``.
    """
    if trades_df.empty:
        return pd.DataFrame(columns=["month", "volume_quote"])

    fills = []
    for side, amount_key in (("entry", "size_quote"), ("exit", "exit_notional")):
        time_key = f"{side}_timestamp"
        if time_key not in trades_df or amount_key not in trades_df:
            continue
        records = (
            trades_df.loc[trades_df["status"] == "closed"]
            if side == "exit"
            else trades_df
        )
        fills.append(
            pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(records[time_key], errors="coerce"),
                    "volume_quote": records[amount_key],
                }
            ).dropna()
        )
    if not fills:
        return pd.DataFrame(columns=["month", "volume_quote"])
    df = pd.concat(fills, ignore_index=True)
    df["month"] = df["timestamp"].dt.to_period("M")
    grouped = df.groupby("month")["volume_quote"].sum().reset_index()
    grouped["month"] = grouped["month"].astype(str)
    return pd.DataFrame(grouped)


def filter_by_month(
    equity: pd.Series[float],
    trades_df: pd.DataFrame,
    month: str | None,
) -> tuple[pd.Series[float], pd.DataFrame]:
    """Filter equity and trades to a specific month (YYYY-MM) or all."""
    if month is None or month == "全期間":
        return equity, trades_df

    if hasattr(equity.index, "to_period"):
        mask = equity.index.to_period("M").astype(str) == month
        eq_filtered = equity.loc[mask]
    else:
        eq_filtered = equity

    if not trades_df.empty and "timestamp" in trades_df.columns:
        ts = pd.to_datetime(trades_df["timestamp"])
        t_mask = ts.dt.to_period("M").astype(str) == month
        tr_filtered = trades_df.loc[t_mask]
    else:
        tr_filtered = trades_df

    return eq_filtered, tr_filtered


def available_months(
    timestamps: npt.NDArray[Any] | None,
) -> list[str]:
    """Return list of YYYY-MM strings present in timestamps."""
    if timestamps is None:
        return []
    idx = pd.DatetimeIndex(timestamps)
    months = sorted(idx.to_period("M").unique().astype(str).tolist())
    return months
