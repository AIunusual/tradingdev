"""Volume-mode backtest engine with bar-by-bar simulation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tradingdev.domain.backtest.base_engine import BaseBacktestEngine
from tradingdev.domain.backtest.execution_policy import DEFAULT_VOLUME_POSITION_SIZE
from tradingdev.domain.backtest.metrics import (
    calculate_metrics_from_simulation,
    timestamp_index,
)
from tradingdev.domain.backtest.result import BacktestResult
from tradingdev.shared.utils.logger import setup_logger

logger = setup_logger(__name__)


class VolumeBacktestEngine(BaseBacktestEngine):
    """Bar-by-bar simulation for volume-oriented strategies.

    Equity is tracked as **cumulative P&L from zero** (no init_cash).
    Each trade independently uses a fixed notional ``position_size``.

    An optional **monthly circuit breaker** (``monthly_max_loss``)
    force-closes any open position and halts trading for the remainder
    of the UTC calendar month once the monthly cumulative loss reaches the
    threshold. Monthly P&L is equity minus the previous observed month's last
    close equity (zero for the first month), including costs and unrealised P&L.
    Trading resumes on the first available bar of a new month.

    Behaviour flags (inherited from ``BaseBacktestEngine``):

    * ``re_entry_after_sl`` (default ``True``): After every SL/TP
      exit the simulator immediately re-enters in the direction
      indicated by the current signal.  Set to ``False`` to go flat
      after a stop-out, skip the rest of that bar, and wait for a later shifted
      signal to differ from the stop-out bar's signal, including a change to zero.
    * ``signal_as_position`` (default ``False``): When ``True``,
      ``signal=0`` is interpreted as "go flat" (close any open
      position).  When ``False`` (legacy), ``signal=0`` means
      "do nothing / keep the current position".
    """

    def __init__(
        self,
        fees: float = 0.0006,
        slippage: float = 0.0005,
        freq: str = "1h",
        position_size: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        signal_as_position: bool = False,
        re_entry_after_sl: bool = True,
        monthly_max_loss: float = 1500.0,
        periods_per_year: float | None = None,
        risk_free_rate: float = 0.0,
        required_return: float = 0.0,
    ) -> None:
        super().__init__(
            init_cash=None,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
            required_return=required_return,
            fees=fees,
            slippage=slippage,
            freq=freq,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_as_position=signal_as_position,
            re_entry_after_sl=re_entry_after_sl,
        )
        self._monthly_max_loss = monthly_max_loss

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """Run volume-mode backtest.

        Execution model:
        - Signal from bar *t* is acted upon at bar *t+1* (``shift(1)``).
        - Entry / signal-driven exit prices use bar *t+1*'s **open**.
        - SL / TP checks use bar *t+1*'s **high** and **low**.
        - Equity mark-to-market uses bar *t+1*'s **close**.
        """
        close_s = df["close"].astype(float)
        open_s = df["open"].astype(float) if "open" in df.columns else close_s
        high_s = df["high"].astype(float) if "high" in df.columns else close_s
        low_s = df["low"].astype(float) if "low" in df.columns else close_s
        signal_s = df["signal"].shift(1).fillna(0).astype(int)

        close = close_s.to_numpy()
        open_ = open_s.to_numpy()
        high = high_s.to_numpy()
        low = low_s.to_numpy()
        signal = signal_s.to_numpy()

        prices = np.column_stack((open_, high, low, close))
        if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
            raise ValueError("Prices must be finite and positive")
        n = len(close)
        size_quote = (
            DEFAULT_VOLUME_POSITION_SIZE
            if self._position_size is None
            else self._position_size
        )

        # Dynamic position sizing: per-bar weight if available
        if "size_weight" in df.columns:
            weight_s = df["size_weight"].shift(1).fillna(1.0)
            weights = weight_s.astype(float).to_numpy()
        else:
            weights = np.ones(n, dtype=np.float64)
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("Position weights must be finite and positive")
        sl = self._stop_loss
        tp = self._take_profit
        fee_rate = self._fees + self._slippage

        cash = 0.0
        equity = np.empty(n, dtype=np.float64)
        trades: list[dict[str, Any]] = []

        pos_dir = 0
        entry_price = 0.0
        entry_idx = 0
        pos_qty = 0.0
        blocked_signal: int | None = None

        # Monthly circuit breaker state
        month_start_equity = 0.0
        current_month: tuple[int, int] | None = None
        circuit_breaker_active = False

        # Extract timestamps for monthly tracking
        timestamps = timestamp_index(df)
        has_timestamps = timestamps is not None
        timestamps_arr = timestamps

        logger.info(
            "Running backtest (volume mode): size=%.0f, sl=%s, tp=%s, "
            "monthly_max_loss=%.0f",
            size_quote,
            sl,
            tp,
            self._monthly_max_loss,
        )

        sig_as_pos = self._signal_as_position
        re_entry = self._re_entry_after_sl

        for i in range(n):
            sig = int(signal[i])
            o = float(open_[i])
            c = float(close[i])
            h = float(high[i])
            lo = float(low[i])

            if blocked_signal is not None and sig != blocked_signal:
                blocked_signal = None

            # --- Month boundary check ---
            if has_timestamps:
                ts = pd.Timestamp(timestamps_arr[i])  # type: ignore[index]
                bar_month = (ts.year, ts.month)
                if current_month is not None and bar_month != current_month:
                    # Carry forward the last observed close, even across missing
                    # months. This assigns the gap and only this month's change
                    # in a carried position to the new month.
                    month_start_equity = float(equity[i - 1])
                    circuit_breaker_active = False
                current_month = bar_month

            # --- Circuit breaker: skip all trading, mark-to-market only ---
            if circuit_breaker_active:
                equity[i] = cash  # no open position
                continue

            # --- SL / TP check (uses high/low for intra-bar) ---
            if pos_dir != 0:
                exit_price = _check_sl_tp(
                    pos_dir,
                    entry_price,
                    h,
                    lo,
                    sl,
                    tp,
                )
                if exit_price is not None:
                    trade = _close_position(
                        pos_dir,
                        pos_qty,
                        entry_price,
                        exit_price,
                        self._fees,
                        self._slippage,
                        entry_idx,
                        i,
                    )
                    trades.append(trade)
                    cash += (
                        trade["net_pnl"] + trade["entry_fees"] + trade["entry_slippage"]
                    )
                    pos_dir = 0
                    pos_qty = 0.0

                    # Check circuit breaker before re-entry
                    if (
                        has_timestamps
                        and cash - month_start_equity <= -self._monthly_max_loss
                    ):
                        circuit_breaker_active = True
                    elif re_entry and sig != 0 and size_quote != 0:
                        # Re-entry after SL happens intra-bar; use
                        # close as best estimate of the re-entry fill.
                        eff = size_quote * weights[i]
                        entry_price = c
                        pos_qty = eff / c
                        entry_idx = i
                        cash -= eff * fee_rate
                        pos_dir = sig

                    if not re_entry:
                        # The stop used this bar's high/low. Never travel back to
                        # its open to create a new position, even on a reversal.
                        blocked_signal = sig
                        equity[i] = cash
                        continue

            if circuit_breaker_active:
                equity[i] = cash
                continue

            # --- signal_as_position: signal=0 → close at open ---
            if sig_as_pos and pos_dir != 0 and sig == 0:
                trade = _close_position(
                    pos_dir,
                    pos_qty,
                    entry_price,
                    o,
                    self._fees,
                    self._slippage,
                    entry_idx,
                    i,
                )
                trades.append(trade)
                cash += trade["net_pnl"] + trade["entry_fees"] + trade["entry_slippage"]
                pos_dir = 0
                pos_qty = 0.0

            # --- reversal at open ---
            if pos_dir != 0 and sig != 0 and sig != pos_dir:
                trade = _close_position(
                    pos_dir,
                    pos_qty,
                    entry_price,
                    o,
                    self._fees,
                    self._slippage,
                    entry_idx,
                    i,
                )
                trades.append(trade)
                cash += trade["net_pnl"] + trade["entry_fees"] + trade["entry_slippage"]
                eff = size_quote * weights[i]
                entry_price = o
                pos_qty = eff / o
                entry_idx = i
                cash -= eff * fee_rate
                pos_dir = sig

            # --- new entry from flat at open ---
            elif (
                pos_dir == 0 and sig != 0 and blocked_signal is None and size_quote != 0
            ):
                eff = size_quote * weights[i]
                entry_price = o
                pos_qty = eff / o
                entry_idx = i
                cash -= eff * fee_rate
                pos_dir = sig

            # --- mark-to-market at close ---
            if pos_dir != 0:
                unrealised = pos_dir * pos_qty * (c - entry_price)
                equity[i] = cash + unrealised
            else:
                equity[i] = cash

            # --- Circuit breaker check after trading ---
            # Use the same equity ledger for realised and unrealised changes;
            # closing a carried position must not count its prior-month P&L again.
            effective_monthly = equity[i] - month_start_equity
            if (
                has_timestamps
                and not circuit_breaker_active
                and effective_monthly <= -self._monthly_max_loss
            ):
                circuit_breaker_active = True
                if pos_dir != 0:
                    trade = _close_position(
                        pos_dir,
                        pos_qty,
                        entry_price,
                        c,
                        self._fees,
                        self._slippage,
                        entry_idx,
                        i,
                    )
                    trades.append(trade)
                    cash += (
                        trade["net_pnl"] + trade["entry_fees"] + trade["entry_slippage"]
                    )
                    pos_dir = 0
                    pos_qty = 0.0
                    equity[i] = cash

        if pos_dir != 0:
            trade = _close_position(
                pos_dir,
                pos_qty,
                entry_price,
                float(close[-1]),
                self._fees,
                self._slippage,
                entry_idx,
                n - 1,
            )
            trades.append(trade)
            cash += trade["net_pnl"] + trade["entry_fees"] + trade["entry_slippage"]
            equity[-1] = cash

        analysis = calculate_metrics_from_simulation(
            equity_curve=equity,
            trades=trades,
            init_cash=None,
            timestamps=timestamps,
            periods_per_year=self._periods_per_year,
            risk_free_rate=self._risk_free_rate,
            required_return=self._required_return,
            frequency=self._freq,
        )
        metrics = analysis.metrics
        logger.info(
            "Backtest complete (volume): %d trades, volume=%.0f",
            metrics["total_trades"],
            metrics["total_volume"],
        )
        return BacktestResult(
            metrics=metrics,
            equity_curve=equity,
            trades=trades,
            timestamps=None if timestamps is None else timestamps.to_numpy(),
            init_cash=None,
            mode="volume",
            metric_metadata=analysis.metadata,
            returns=analysis.returns,
        )


def _check_sl_tp(
    pos_dir: int,
    entry_price: float,
    high: float,
    low: float,
    sl: float | None,
    tp: float | None,
) -> float | None:
    """Return exit price if SL or TP was hit, else None."""
    if pos_dir == 1:
        if sl and low <= entry_price * (1 - sl):
            return entry_price * (1 - sl)
        if tp and high >= entry_price * (1 + tp):
            return entry_price * (1 + tp)
    elif pos_dir == -1:
        if sl and high >= entry_price * (1 + sl):
            return entry_price * (1 + sl)
        if tp and low <= entry_price * (1 - tp):
            return entry_price * (1 - tp)
    return None


def _close_position(
    pos_dir: int,
    pos_qty: float,
    entry_price: float,
    exit_price: float,
    fee_rate: float,
    slippage_rate: float,
    entry_idx: int,
    exit_idx: int,
) -> dict[str, Any]:
    """Build a trade record for a closed position."""
    gross_pnl = pos_dir * pos_qty * (exit_price - entry_price)
    exit_value = pos_qty * exit_price
    entry_fee = pos_qty * entry_price * fee_rate
    exit_fee = exit_value * fee_rate
    entry_slippage = pos_qty * entry_price * slippage_rate
    exit_slippage = exit_value * slippage_rate
    net_pnl = gross_pnl - entry_fee - exit_fee - entry_slippage - exit_slippage
    return {
        "direction": pos_dir,
        "size": pos_qty,
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "status": "closed",
        "exit_notional": exit_value,
        "entry_fees": entry_fee,
        "exit_fees": exit_fee,
        "entry_slippage": entry_slippage,
        "exit_slippage": exit_slippage,
        "slippage": entry_slippage + exit_slippage,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "size_quote": pos_qty * entry_price,
        "gross_pnl": gross_pnl,
        "fee": entry_fee + exit_fee,
        "net_pnl": net_pnl,
    }
