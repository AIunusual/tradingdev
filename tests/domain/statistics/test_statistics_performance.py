"""Tests for the engine-independent performance summary."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from tradingdev.domain.backtest.metrics import calculate_metrics_from_simulation
from tradingdev.domain.backtest.signal_engine import SignalBacktestEngine
from tradingdev.domain.backtest.volume_engine import VolumeBacktestEngine
from tradingdev.domain.statistics.performance import (
    PerformanceSummary,
    daily_excess_returns,
    summarize_performance,
)
from tradingdev.domain.statistics.periods import period_pnl

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def _summary(
    equity: npt.ArrayLike,
    timestamps: npt.ArrayLike,
    trades: Sequence[Mapping[str, object]],
    *,
    init_cash: float | None = 100.0,
    frequency: str = "1h",
    periods_per_year: float | None = 365.0,
    risk_free_rate: float = 0.0,
) -> PerformanceSummary:
    """Supply this fixture's explicit settings to the public API."""
    return summarize_performance(
        equity,
        timestamps,
        trades,
        init_cash=init_cash,
        frequency=frequency,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )


def _hourly_2024() -> tuple[npt.NDArray[np.float64], pd.DatetimeIndex]:
    timestamps = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h")
    rng = np.random.default_rng(7)
    equity = 10_000.0 + np.cumsum(rng.normal(0.5, 5.0, len(timestamps)))
    return equity, timestamps


class TestFullYearSummary:
    def test_counts_366_days_and_12_months(self) -> None:
        equity, timestamps = _hourly_2024()
        summary = _summary(equity, timestamps, [], init_cash=10_000.0)

        assert summary.n_bars == 8784
        assert summary.n_days == 366
        assert summary.n_months == 12
        assert summary.start == pd.Timestamp("2024-01-01", tz="UTC")
        assert summary.end == pd.Timestamp("2024-12-31 23:00", tz="UTC")

    def test_daily_and_monthly_pnl_add_up_to_total_pnl(self) -> None:
        equity, timestamps = _hourly_2024()
        summary = _summary(equity, timestamps, [], init_cash=10_000.0)

        assert summary.total_pnl == pytest.approx(equity[-1] - 10_000.0)
        assert summary.daily_pnl_mean is not None
        assert summary.monthly_pnl_mean is not None
        assert summary.daily_pnl_mean * summary.n_days == pytest.approx(
            summary.total_pnl
        )
        assert summary.monthly_pnl_mean * summary.n_months == pytest.approx(
            summary.total_pnl
        )

    def test_std_is_measured_on_daily_not_per_bar_pnl(self) -> None:
        equity, timestamps = _hourly_2024()
        summary = _summary(equity, timestamps, [], init_cash=10_000.0)

        daily = period_pnl(equity, timestamps, "daily")
        daily.iloc[0] += equity[0] - 10_000.0
        assert summary.daily_pnl_std == pytest.approx(float(daily.std(ddof=0)))
        per_bar_std = float(np.diff(equity).std(ddof=1))
        assert summary.daily_pnl_std is not None
        assert summary.daily_pnl_std > 3 * per_bar_std

    def test_sharpe_annualizes_daily_returns(self) -> None:
        equity, timestamps = _hourly_2024()
        daily_equity = pd.Series(equity, index=timestamps).resample("D").last()
        returns = daily_equity / daily_equity.shift(1).fillna(10_000.0) - 1.0
        expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)

        summary = _summary(
            equity, timestamps, [], init_cash=10_000.0, periods_per_year=252
        )

        assert summary.sharpe_ratio == pytest.approx(expected)


class TestEquityStatistics:
    def test_drawdown_and_return_from_known_curve(self) -> None:
        timestamps = pd.date_range("2024-01-01", periods=5, freq="D")
        summary = _summary([100.0, 120.0, 90.0, 110.0, 130.0], timestamps, [])

        assert summary.total_pnl == pytest.approx(30.0)
        assert summary.total_return == pytest.approx(0.3)
        assert summary.max_drawdown_amount == pytest.approx(30.0)
        assert summary.max_drawdown == pytest.approx(0.25)

    def test_cumulative_pnl_curve_has_no_return_figures(self) -> None:
        timestamps = pd.date_range("2024-01-01", periods=4, freq="D")
        summary = _summary([0.0, 50.0, -20.0, 40.0], timestamps, [], init_cash=None)

        assert summary.total_pnl == pytest.approx(40.0)
        assert summary.max_drawdown_amount == pytest.approx(70.0)
        assert summary.total_return is None
        assert summary.max_drawdown is None
        assert summary.sharpe_ratio is None

    def test_single_day_has_defined_population_dispersion_but_no_sharpe(self) -> None:
        timestamps = pd.date_range("2024-01-01", periods=3, freq="h")
        summary = _summary([100.0, 101.0, 102.0], timestamps, [])

        assert summary.n_days == 1
        assert summary.daily_pnl_std == 0.0
        assert summary.monthly_pnl_std == 0.0
        assert summary.sharpe_ratio is None

    def test_flat_curve_has_no_sharpe(self) -> None:
        timestamps = pd.date_range("2024-01-01", periods=5, freq="D")
        summary = _summary([100.0] * 5, timestamps, [])

        assert summary.sharpe_ratio is None
        assert summary.max_drawdown_amount == 0.0


class TestTimestamps:
    def test_aware_timestamps_across_a_dst_change(self) -> None:
        # Local Berlin times either side of the 2024-03-31 DST change.
        summary = _summary(
            [100.0, 101.0],
            ["2024-03-30T12:00:00+01:00", "2024-03-31T12:00:00+02:00"],
            [],
        )

        assert summary.start == pd.Timestamp("2024-03-30 11:00", tz="UTC")
        assert summary.end == pd.Timestamp("2024-03-31 10:00", tz="UTC")
        assert summary.n_days == 2
        assert summary.daily_pnl_mean == pytest.approx(0.5)


class TestTradeStatistics:
    _timestamps = pd.date_range("2024-01-01", periods=2, freq="D")

    def test_counts_wins_losses_and_breakeven(self) -> None:
        trades: list[dict[str, object]] = [
            {"status": "closed", "net_pnl": 30.0},
            {"status": "closed", "net_pnl": -10.0},
            {"status": "closed", "net_pnl": 0.0},
            {"status": "closed", "net_pnl": np.float64(20.0)},
            {"status": "closed", "net_pnl": np.int64(-15)},
        ]
        summary = _summary([100.0, 125.0], self._timestamps, trades)

        assert summary.closed_trades.total_trades == 5
        assert summary.closed_trades.winning_trades == 2
        assert summary.closed_trades.losing_trades == 2
        assert summary.closed_trades.win_rate == pytest.approx(0.4)
        assert summary.closed_trades.profit_factor == pytest.approx(50.0 / 25.0)
        assert summary.marked_trades.net_pnl == pytest.approx(25.0)

    def test_no_trades(self) -> None:
        summary = _summary([100.0, 100.0], self._timestamps, [])

        assert summary.closed_trades.total_trades == 0
        assert summary.closed_trades.win_rate is None
        assert summary.closed_trades.profit_factor is None
        assert summary.marked_trades.net_pnl == 0.0

    def test_no_losing_trades_leaves_profit_factor_undefined(self) -> None:
        trades = [
            {"status": "closed", "net_pnl": 5.0},
            {"status": "closed", "net_pnl": 0.0},
        ]
        summary = _summary([100.0, 105.0], self._timestamps, trades)

        assert summary.closed_trades.win_rate == pytest.approx(0.5)
        assert summary.closed_trades.profit_factor is None

    @pytest.mark.parametrize(
        "trade",
        [
            {},
            {"status": "closed", "net_pnl": None},
            {"status": "closed", "net_pnl": "1.0"},
            {"status": "closed", "net_pnl": True},
        ],
    )
    def test_rejects_trade_without_numeric_net_pnl(
        self, trade: dict[str, object]
    ) -> None:
        with pytest.raises(
            ValueError, match="trade 0 net_pnl must be a finite real number"
        ):
            _summary([100.0, 100.0], self._timestamps, [trade])

    def test_rejects_non_finite_net_pnl(self) -> None:
        with pytest.raises(ValueError, match="finite real number"):
            _summary(
                [100.0, 100.0],
                self._timestamps,
                [{"status": "closed", "net_pnl": float("inf")}],
            )

    @pytest.mark.parametrize(
        "periods_per_year", [0, -365.0, math.nan, math.inf, -math.inf, True, "365"]
    )
    def test_rejects_invalid_periods_per_year(self, periods_per_year: object) -> None:
        # The curve has a finite Sharpe, so only the setting can make it
        # non-finite; NaN and +inf used to pass the ``<= 0`` check.
        with pytest.raises(ValueError, match="periods_per_year"):
            _summary(
                [100.0, 110.0, 100.0, 120.0],
                pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC"),
                [],
                periods_per_year=periods_per_year,  # type: ignore[arg-type]
            )


def test_initial_costs_are_not_discarded_or_charged_twice() -> None:
    # Original cash 100, one unit of entry cost and one of exit cost.
    result = _summary(
        [99.0, 98.0],
        pd.date_range("2024-01-01", periods=2, freq="D"),
        [{"net_pnl": -2.0, "fee": 2.0, "status": "closed"}],
    )
    assert result.total_pnl == -2.0
    assert result.observed_equity_change == -1.0
    assert result.total_return == pytest.approx(-0.02)
    assert result.max_drawdown_amount == 2.0
    assert result.max_drawdown == pytest.approx(0.02)
    assert result.daily_pnl_mean == -1.0
    assert result.daily_pnl_std == 0.0
    assert result.closed_trades.net_pnl == result.total_pnl


@pytest.mark.parametrize("daily_rate", [0.0, 0.001, -0.001])
def test_excess_sharpe_and_return_information_ratio_are_distinct(
    daily_rate: float,
) -> None:
    returns = np.array([0.02, -0.01, 0.03])
    annual_rate = (1.0 + daily_rate) ** 252 - 1.0
    result = _summary(
        100.0 * np.cumprod(1.0 + returns),
        pd.date_range("2024-01-01", periods=3, freq="D"),
        [],
        periods_per_year=252,
        risk_free_rate=annual_rate,
    )
    assert result.sharpe_ratio == pytest.approx(
        float((returns.mean() - daily_rate) / returns.std(ddof=1) * np.sqrt(252))
    )
    assert result.return_information_ratio == pytest.approx(
        float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
    )
    assert result.risk_free_rate == annual_rate
    assert result.periods_per_year == 252


def test_closed_and_marked_trade_scopes_keep_their_own_values() -> None:
    result = _summary(
        [100.0, 85.0],
        pd.date_range("2024-01-01", periods=2, freq="D"),
        [
            {"net_pnl": 10.0, "status": "closed"},
            {"net_pnl": -5.0, "status": "closed"},
            {"net_pnl": 0.0, "status": "closed"},
            {"net_pnl": -20.0, "status": "open"},
        ],
    )
    assert result.closed_trades.total_trades == 3
    assert result.closed_trades.win_rate == pytest.approx(1 / 3)
    assert result.closed_trades.profit_factor == 2.0
    assert result.closed_trades.net_pnl == 5.0
    assert result.marked_trades.total_trades == 4
    assert result.marked_trades.win_rate == 0.25
    assert result.marked_trades.profit_factor == 0.4
    assert result.marked_trades.net_pnl == result.total_pnl == -15.0


@pytest.mark.parametrize("status", [None, "OPEN", "partial", True])
def test_trade_status_is_required_and_not_guessed(status: object) -> None:
    trade: dict[str, object] = {"net_pnl": 1.0}
    if status is not None:
        trade["status"] = status
    with pytest.raises(ValueError, match="trade 0 requires a closed/open status"):
        _summary([100.0], pd.DatetimeIndex(["2024-01-01"]), [trade])


def test_positive_cumulative_pnl_is_not_mistaken_for_account_capital() -> None:
    result = _summary(
        [10.0, 20.0, 15.0],
        pd.date_range("2024-01-01", periods=3, freq="D"),
        [],
        init_cash=None,
    )
    assert result.total_pnl == 15.0
    assert result.observed_equity_change == 5.0
    assert result.max_drawdown_amount == 5.0
    assert result.total_return is None
    assert result.max_drawdown is None
    assert result.sharpe_ratio is None
    assert result.return_information_ratio is None
    assert result.unavailable["total_return"] == "not_applicable"


@pytest.mark.parametrize("frequency", ["1w", "2d", "unknown"])
def test_daily_statistics_need_declared_sampling(frequency: str) -> None:
    result = _summary(
        [100.0, 105.0, 102.0],
        pd.date_range("2024-01-01", periods=3, freq="D"),
        [],
        frequency=frequency,
    )
    assert result.total_pnl == 2.0
    assert result.total_return == pytest.approx(0.02)
    assert result.n_days == 3
    assert result.daily_pnl_mean is None
    assert result.daily_pnl_std is None
    assert result.monthly_pnl_mean is None
    assert result.sharpe_ratio is None
    assert result.return_information_ratio is None
    assert result.unavailable["sharpe_ratio"] == (
        "unknown_bar_frequency"
        if frequency == "unknown"
        else "unsupported_daily_sampling"
    )


def test_declared_daily_data_can_have_calendar_gaps() -> None:
    result = _summary(
        [110.0, 121.0],
        ["2024-01-01", "2024-01-04"],
        [],
        frequency="1d",
    )
    assert result.n_days == 2
    assert result.daily_pnl_mean == 10.5
    assert result.daily_pnl_std == 0.5
    assert result.monthly_pnl_mean == 21.0
    assert result.monthly_pnl_std == 0.0


def test_missing_annualization_does_not_hide_other_metrics() -> None:
    result = _summary(
        [100.0, 105.0],
        ["2024-01-01", "2024-01-02"],
        [],
        periods_per_year=None,
    )
    assert result.total_pnl == 5.0
    assert result.daily_pnl_mean == 2.5
    assert result.sharpe_ratio is None
    assert result.return_information_ratio is None
    assert result.unavailable["sharpe_ratio"] == "missing_annualization"


def test_terminal_zero_equity_preserves_complete_loss() -> None:
    result = _summary([90.0, 0.0], ["2024-01-01", "2024-01-02"], [])
    assert result.total_pnl == -100.0
    assert result.total_return == -1.0
    assert result.max_drawdown == 1.0
    assert result.max_drawdown_amount == 100.0


@pytest.mark.parametrize(
    ("equity", "reason"),
    [
        ([200.0, 400.0], "unbounded"),
        ([50.0, 25.0], "unbounded"),
        ([100.0, 100.0], "zero_denominator"),
    ],
)
def test_constant_return_reasons_match_the_engine(
    equity: list[float], reason: str
) -> None:
    timestamps = pd.date_range("2024-01-01", periods=2, tz="UTC")
    summary = _summary(equity, timestamps, [], periods_per_year=252.0)
    engine = calculate_metrics_from_simulation(
        np.asarray(equity),
        [],
        100.0,
        timestamps,
        periods_per_year=252.0,
        frequency="1d",
    )
    assert engine.metrics["sharpe_ratio"] is None
    assert engine.metadata["unavailable"]["sharpe_ratio"] == reason
    assert summary.sharpe_ratio is None
    assert summary.return_information_ratio is None
    assert summary.unavailable["sharpe_ratio"] == reason
    assert summary.unavailable["return_information_ratio"] == reason


@pytest.mark.parametrize("capital", [0.0, -1.0, math.inf, True, "100"])
def test_initial_capital_must_be_explicit_and_valid(capital: object) -> None:
    with pytest.raises(ValueError, match="init_cash"):
        _summary([100.0], ["2024-01-01"], [], init_cash=capital)  # type: ignore[arg-type]


@pytest.mark.parametrize("rate", [-1.0, -2.0, math.nan, True, "0"])
def test_risk_free_rate_validation(rate: object) -> None:
    with pytest.raises(ValueError, match="risk_free_rate"):
        _summary([100.0], ["2024-01-01"], [], risk_free_rate=rate)  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", ["signal", "volume"])
def test_summary_independently_reconciles_current_engine_outputs(mode: str) -> None:
    prices = [100.0, 100.0, 110.0, 100.0, 120.0]
    timestamps = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    market = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "signal": [1, 0, 1, 1, 1],
        }
    )
    engine = (
        SignalBacktestEngine(
            init_cash=1000.0,
            position_size=100.0,
            fees=0.01,
            slippage=0.002,
            freq="1d",
            periods_per_year=252,
            risk_free_rate=0.02,
        )
        if mode == "signal"
        else VolumeBacktestEngine(
            position_size=100.0,
            fees=0.01,
            slippage=0.002,
            freq="1d",
            signal_as_position=True,
        )
    )
    run = engine.run(market)
    result = _summary(
        run.equity_curve,
        timestamps,
        run.trades,
        init_cash=run.init_cash,
        frequency="1d",
        periods_per_year=252,
        risk_free_rate=0.02,
    )
    for metric in (
        "total_pnl",
        "total_return",
        "max_drawdown",
        "max_drawdown_amount",
        "daily_pnl_mean",
        "daily_pnl_std",
        "monthly_pnl_mean",
        "monthly_pnl_std",
        "sharpe_ratio",
    ):
        actual = getattr(result, metric)
        expected = run.metrics[metric]
        if expected is None:
            assert actual is None, metric
        else:
            assert actual == pytest.approx(expected), metric
    assert result.closed_trades.total_trades == run.metrics["total_trades"]
    assert result.marked_trades.total_trades == len(run.trades) == 2
    assert result.closed_trades == result.marked_trades
    assert result.marked_trades.net_pnl == pytest.approx(result.total_pnl)
    assert result.closed_trades.win_rate == pytest.approx(run.metrics["win_rate"])


def test_daily_excess_returns_start_from_original_capital_not_first_observation() -> (
    None
):
    equity = [990.0, 1010.0, 1005.0]
    timestamps = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    actual = daily_excess_returns(
        equity,
        timestamps,
        init_cash=1000.0,
        periods_per_year=365.0,
        risk_free_rate=0.02,
    )
    closing = pd.Series(equity, index=timestamps)
    opening = closing.shift(1).fillna(1000.0)
    expected = (closing - opening) / opening - math.expm1(math.log1p(0.02) / 365.0)
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=1e-15)
    # A generic first-observation baseline would give 0 for day 1 and omit
    # the 10-point loss that happened before the first observation.
    assert actual.iloc[0] == pytest.approx(-0.01 - math.expm1(math.log1p(0.02) / 365.0))
