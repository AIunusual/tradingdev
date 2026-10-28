"""Signal-mode backtest engine using vectorbt."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import vectorbt as vbt

from tradingdev.domain.backtest.base_engine import BaseBacktestEngine
from tradingdev.domain.backtest.execution_policy import signal_execution_defaults
from tradingdev.domain.backtest.execution_records import extract_execution_records
from tradingdev.domain.backtest.metrics import (
    calculate_metrics,
    calculate_metrics_from_simulation,
    normalized_trades,
    timestamp_index,
)
from tradingdev.domain.backtest.result import BacktestResult
from tradingdev.shared.utils.logger import setup_logger

if TYPE_CHECKING:
    import pandas as pd

logger = setup_logger(__name__)


class SignalBacktestEngine(BaseBacktestEngine):
    """Backtest using vectorbt ``Portfolio.from_signals``.

    Supports all-in or fixed-size positions with optional SL/TP.
    The signal is shifted by 1 bar to avoid look-ahead bias.
    """

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """Run signal-mode backtest.

        Execution uses the **open** price of the bar following the
        signal (after ``shift(1)``) to avoid look-ahead bias.
        """
        init_cash = self._init_cash
        if init_cash is None:
            msg = "SignalBacktestEngine requires init_cash to be set"
            raise ValueError(msg)

        timestamps = timestamp_index(df)
        market = df.copy()
        if timestamps is not None:
            market.index = timestamps
        if market.empty:
            analysis = calculate_metrics_from_simulation(
                np.array([], dtype=np.float64),
                [],
                init_cash,
                timestamps,
                periods_per_year=self._periods_per_year,
                risk_free_rate=self._risk_free_rate,
                required_return=self._required_return,
                frequency=self._freq,
            )
            return BacktestResult(
                metrics=analysis.metrics,
                equity_curve=np.array([], dtype=np.float64),
                init_cash=init_cash,
                metric_metadata=analysis.metadata,
                returns=analysis.returns,
                timestamps=None if timestamps is None else timestamps.to_numpy(),
                execution_records=[],
                account_history=[],
            )
        close = market["close"].astype(float)
        if not np.all(np.isfinite(close)) or (close <= 0).any():
            raise ValueError("Prices must be finite and positive")
        open_ = market["open"].astype(float) if "open" in market.columns else close
        if not np.all(np.isfinite(open_)) or (open_ <= 0).any():
            raise ValueError("Prices must be finite and positive")
        signal = market["signal"].shift(1).fillna(0).astype(int)

        entries = (signal == 1) & (signal.shift(1) != 1)
        exits = (signal != 1) & (signal.shift(1) == 1)
        short_entries = (signal == -1) & (signal.shift(1) != -1)
        short_exits = (signal != -1) & (signal.shift(1) == -1)

        logger.info(
            "Running backtest (signal mode): init_cash=%.0f, fees=%.4f, slippage=%.4f",
            init_cash,
            self._fees,
            self._slippage,
        )

        kwargs: dict[str, Any] = {
            **signal_execution_defaults(),
            "close": close,
            "open": open_,
            "price": open_,
            "entries": entries,
            "exits": exits,
            "short_entries": short_entries,
            "short_exits": short_exits,
            "init_cash": init_cash,
            "fees": self._fees,
            "slippage": self._slippage,
            "freq": self._freq,
            "log": True,
            "use_stops": self._stop_loss is not None or self._take_profit is not None,
        }

        if self._position_size is not None:
            kwargs["size"] = self._position_size
            kwargs["size_type"] = "value"

        if self._stop_loss is not None:
            kwargs["sl_stop"] = self._stop_loss

        if self._take_profit is not None:
            kwargs["tp_stop"] = self._take_profit

        pf = vbt.Portfolio.from_signals(**kwargs)

        analysis = calculate_metrics(
            pf,
            timestamps=timestamps,
            periods_per_year=self._periods_per_year,
            risk_free_rate=self._risk_free_rate,
            required_return=self._required_return,
            frequency=self._freq,
        )
        metrics = analysis.metrics
        logger.info(
            "Backtest complete: %d trades",
            metrics["total_trades"],
        )

        equity_curve = np.asarray(pf.value(), dtype=np.float64)
        trades = normalized_trades(pf.trades)
        execution_records, account_history = extract_execution_records(
            pf, market, timestamps
        )

        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_curve,
            trades=trades,
            timestamps=None if timestamps is None else timestamps.to_numpy(),
            init_cash=init_cash,
            mode="signal",
            metric_metadata=analysis.metadata,
            returns=analysis.returns,
            execution_records=execution_records,
            account_history=account_history,
        )
