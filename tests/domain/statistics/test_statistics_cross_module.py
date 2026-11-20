"""Synthetic cross-module cases with explicit, independently derived settings.

These cases use tiny hand-computable observations. Their expected values are
derived by hand from the definitions in each module's docstring, never read
back from the functions under test. They pin the boundaries the historical
KD fixture cannot cover: a first bar after fees, marked-open trades, no
trades, missing annualisation, and volume (zero-baseline) mode.

The paired-comparison module already has its own tests. This file does not
import functions from a future PR, and does not duplicate each module's full
parameter matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from tradingdev.domain.statistics.breakeven import summarize_breakeven
from tradingdev.domain.statistics.performance import (
    PerformanceSummary,
    summarize_performance,
)
from tradingdev.domain.statistics.periods import period_returns

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any


_DAYS = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")


def _summary(
    equity: Sequence[float],
    trades: Sequence[Mapping[str, Any]],
    *,
    init_cash: float | None = 100.0,
    periods_per_year: float | None = 365.0,
) -> PerformanceSummary:
    return summarize_performance(
        equity,
        _DAYS[: len(equity)],
        trades,
        init_cash=init_cash,
        frequency="1d",
        periods_per_year=periods_per_year,
        risk_free_rate=0.0,
    )


def _trade(pnl: float, status: str, *, size: float = 10.0) -> dict[str, object]:
    return {
        "net_pnl": pnl,
        "status": status,
        "size_quote": size,
        "entry_price": 100.0,
        "exit_price": 110.0,
        "fee": 0.1,
    }


class TestOpeningEquityBaseline:
    def test_period_returns_start_at_first_observation_not_init_cash(self) -> None:
        # Equity 100 -> 110 -> 99. Period returns compare each close with
        # the previous close; the first bar has no earlier observation.
        returns = period_returns([100.0, 110.0, 99.0], _DAYS[:3], "daily")

        # Hand: 110/100-1 = 0.10; 99/110-1 = -0.10 exactly.
        assert len(returns) == 3
        assert returns.iloc[1] == pytest.approx(0.10, rel=0, abs=1e-12)
        assert returns.iloc[2] == pytest.approx(-0.10, rel=0, abs=1e-12)

    def test_performance_total_pnl_starts_at_the_declared_capital(self) -> None:
        # Equity 100 -> 110 -> 99 means capital 100 lost 1 overall, even
        # though observed last-minus-first is -1 too here; use a case where
        # the first bar is already net of fees to separate the two baselines.
        summary = _summary([99.0, 110.0, 101.0], [], init_cash=100.0)

        # total_pnl = last - init_cash = 101 - 100 = +1 (includes the fee
        # already reflected in the first observation).
        # observed_equity_change = last - first = 101 - 99 = +2.
        assert summary.total_pnl == pytest.approx(1.0, rel=0, abs=1e-12)
        assert summary.observed_equity_change == pytest.approx(2.0, rel=0, abs=1e-12)
        assert summary.total_return == pytest.approx(0.01, rel=0, abs=1e-12)


class TestTradeGroups:
    def test_closed_only_vs_marked_open(self) -> None:
        trades = [_trade(10.0, "closed"), _trade(-2.0, "closed"), _trade(5.0, "open")]
        summary = _summary([100.0, 110.0, 108.0, 113.0], trades)

        assert summary.closed_trades.total_trades == 2
        assert summary.closed_trades.net_pnl == pytest.approx(8.0, rel=0, abs=1e-12)
        assert summary.closed_trades.win_rate == pytest.approx(0.5)
        assert summary.marked_trades.total_trades == 3
        assert summary.marked_trades.net_pnl == pytest.approx(13.0, rel=0, abs=1e-12)

    def test_zero_trades_reports_reasons_not_zero_ratios(self) -> None:
        summary = _summary([100.0, 101.0, 102.0], [], init_cash=100.0)

        assert summary.closed_trades.total_trades == 0
        assert summary.closed_trades.win_rate is None
        assert summary.closed_trades.unavailable["win_rate"] == "no_trades"
        assert summary.closed_trades.unavailable["profit_factor"] == "no_trades"


class TestUnavailableSettings:
    def test_missing_annualisation_keeps_a_reason(self) -> None:
        summary = _summary([100.0, 110.0, 101.0, 120.0], [], periods_per_year=None)

        assert summary.sharpe_ratio is None
        assert summary.unavailable["sharpe_ratio"] == "missing_annualization"
        assert summary.unavailable["return_information_ratio"] == (
            "missing_annualization"
        )
        # P&L does not need annualisation and stays available.
        assert summary.total_pnl == pytest.approx(20.0, rel=0, abs=1e-12)

    def test_volume_mode_uses_zero_baseline_and_no_return_ratios(self) -> None:
        # Volume-mode cumulative P&L has no capital to divide by.
        summary = _summary([0.0, 50.0, -20.0, 40.0], [], init_cash=None)

        assert summary.total_pnl == pytest.approx(40.0, rel=0, abs=1e-12)
        assert summary.total_return is None
        assert summary.unavailable["total_return"] == "not_applicable"
        assert summary.max_drawdown is None
        assert summary.sharpe_ratio is None


class TestBreakevenNotional:
    def test_open_trade_has_no_phantom_exit_leg(self) -> None:
        # Closed: size 10, entry 100, exit 110 -> notional 10*(1+1.1) = 21.
        # Open: only the entry leg -> notional 10. Total 31, not 42.
        summary = summarize_breakeven(
            [_trade(10.0, "closed"), _trade(5.0, "open")],
            start=_DAYS[0],
            end=_DAYS[-1],
            capital=100.0,
        )

        assert summary.traded_notional == pytest.approx(31.0, rel=0, abs=1e-12)
        assert summary.total_cost == pytest.approx(0.2, rel=0, abs=1e-12)

    def test_zero_trades_have_zero_notional(self) -> None:
        summary = summarize_breakeven([], start=_DAYS[0], end=_DAYS[-1], capital=100.0)

        assert summary.traded_notional == 0.0
        assert summary.total_trades == 0
