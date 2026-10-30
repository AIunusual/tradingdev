"""Monthly loss uses observed equity changes, not whole carried-trade P&L."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.backtest.volume_engine import VolumeBacktestEngine


def monthly_market(
    closes: list[float],
    *,
    direction: int = 1,
    dates: list[str] | None = None,
    executed_signals: list[int] | None = None,
) -> pd.DataFrame:
    """Use one unit at entry; explicit signals describe the execution bar."""
    prices = np.array(closes, dtype=float)
    opens = np.concatenate(([prices[0]], prices[:-1]))
    signals = executed_signals or [0] + [direction] * (len(prices) - 1)
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(dates, utc=True, format="ISO8601")
            if dates is not None
            else pd.date_range("2024-01-30", periods=len(prices), freq="D", tz="UTC"),
            "open": opens,
            "close": prices,
            "high": np.maximum(opens, prices),
            "low": np.minimum(opens, prices),
            "volume": 1.0,
            "signal": signals[1:] + [0],
        }
    )


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize(
    "dates",
    [
        ["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"],
        ["2024-01-30", "2024-01-31", "2024-04-01", "2024-04-02"],
        ["2024-12-30", "2024-12-31", "2025-01-01", "2025-01-02"],
    ],
)
@pytest.mark.parametrize("costs", [False, True])
def test_carried_profit_cannot_offset_new_month_loss(
    direction: int, dates: list[str], costs: bool
) -> None:
    df = monthly_market(
        [100, 100 + direction * 20, 100 + direction * 10, 100],
        direction=direction,
        dates=dates,
    )
    result = VolumeBacktestEngine(
        position_size=100,
        fees=0.01 if costs else 0,
        slippage=0.02 if costs else 0,
        monthly_max_loss=5,
        freq="1d",
    ).run(df)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert (trade["entry_idx"], trade["exit_idx"]) == (1, 2)
    assert trade["entry_price"] == 100
    assert trade["exit_price"] == 100 + direction * 10
    rate = 0.03 if costs else 0
    final = 10 - (200 + direction * 10) * rate
    np.testing.assert_allclose(result.equity_curve, [0, 20 - 100 * rate, final, final])
    assert trade["net_pnl"] == pytest.approx(final)
    assert result.metrics["total_pnl"] == pytest.approx(final)


@pytest.mark.parametrize("direction", [1, -1])
def test_carried_loss_is_not_charged_again_in_new_month(direction: int) -> None:
    df = monthly_market(
        [100, 100 - direction * 10, 100 - direction * 20, 100 - direction * 25, 100],
        direction=direction,
    )
    result = VolumeBacktestEngine(
        position_size=100, fees=0, slippage=0, monthly_max_loss=15
    ).run(df)
    assert [(t["entry_idx"], t["exit_idx"]) for t in result.trades] == [(1, 3)]
    # Exactly the monthly threshold triggers; lifetime loss at index 2 does not.
    np.testing.assert_allclose(result.equity_curve, [0, -10, -20, -25, -25])


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("reverse", [False, True])
def test_month_open_exit_and_reversal_use_month_equity(
    direction: int, reverse: bool
) -> None:
    next_signal = -direction if reverse else 0
    df = monthly_market(
        [100, 100 + direction * 20, 100 + direction * 10, 100, 100],
        executed_signals=[0, direction, next_signal, direction, direction],
    )
    # A gap at the new month's open belongs to the new month too.
    df.loc[2, "open"] = 100 + direction * 10
    result = VolumeBacktestEngine(
        position_size=100,
        fees=0,
        slippage=0,
        monthly_max_loss=5,
        signal_as_position=True,
    ).run(df)
    expected = [(1, 2), (2, 2)] if reverse else [(1, 2)]
    assert [(t["entry_idx"], t["exit_idx"]) for t in result.trades] == expected
    np.testing.assert_allclose(result.equity_curve, [0, 20, 10, 10, 10])
    assert sum(t["net_pnl"] for t in result.trades) == pytest.approx(10)


def test_only_current_month_costs_count_and_new_entry_cost_can_trip_breaker() -> None:
    df = monthly_market([100] * 5, executed_signals=[0, 1, 0, 1, 1])
    result = VolumeBacktestEngine(
        position_size=100,
        fees=0.01,
        slippage=0.02,
        monthly_max_loss=5,
        signal_as_position=True,
    ).run(df)
    # January entry costs 3; February exit costs 3, so trading may resume.
    # The next entry costs another 3 and trips February's limit, then exit costs 3.
    assert [(t["entry_idx"], t["exit_idx"]) for t in result.trades] == [(1, 2), (3, 3)]
    np.testing.assert_allclose(result.equity_curve, [0, -3, -6, -12, -12])
    assert result.metrics["total_fees"] == pytest.approx(4)
    assert result.metrics["total_slippage"] == pytest.approx(8)


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("stop", ["stop_loss", "take_profit"])
@pytest.mark.parametrize("reentry", [False, True])
def test_carried_stop_uses_month_loss_before_reentry(
    direction: int, stop: str, reentry: bool
) -> None:
    previous_profit = 2 if stop == "stop_loss" else 4
    stop_profit = -4 if stop == "stop_loss" else 25
    df = monthly_market(
        [100, 100 + direction * previous_profit, 100 + direction * stop_profit, 100],
        direction=direction,
    )
    result = VolumeBacktestEngine(
        position_size=100,
        fees=0,
        slippage=0,
        monthly_max_loss=5,
        re_entry_after_sl=reentry,
        stop_loss=0.04 if stop == "stop_loss" else None,
        take_profit=0.25 if stop == "take_profit" else None,
    ).run(df)
    assert result.trades[0]["exit_idx"] == 2
    assert result.trades[0]["net_pnl"] == pytest.approx(stop_profit)
    if stop == "stop_loss" or not reentry:
        assert len(result.trades) == 1
        np.testing.assert_allclose(result.equity_curve[2:], stop_profit)
    else:
        assert len(result.trades) == 2
        assert result.trades[1]["entry_idx"] == 2
        assert result.trades[1]["entry_price"] == 100 + direction * stop_profit


@pytest.mark.parametrize("resumption", ["2024-03-01", "2024-05-01"])
def test_breaker_resumes_at_next_observed_month(resumption: str) -> None:
    df = monthly_market(
        [100, 120, 110, 100, 100, 102],
        dates=[
            "2024-01-30",
            "2024-01-31",
            "2024-02-01",
            "2024-02-02",
            resumption,
            str(pd.Timestamp(resumption) + pd.Timedelta(days=1)),
        ],
    )
    result = VolumeBacktestEngine(
        position_size=100, fees=0, slippage=0, monthly_max_loss=5
    ).run(df)
    assert [(t["entry_idx"], t["exit_idx"]) for t in result.trades] == [(1, 2), (4, 5)]
    np.testing.assert_allclose(result.equity_curve, [0, 20, 10, 10, 10, 12])


def test_without_timestamps_monthly_breaker_remains_unavailable() -> None:
    df = monthly_market([100, 120, 110, 100]).drop(columns="timestamp")
    result = VolumeBacktestEngine(
        position_size=100, fees=0, slippage=0, monthly_max_loss=5
    ).run(df)
    assert result.trades[0]["exit_idx"] == 3
    np.testing.assert_allclose(result.equity_curve, [0, 20, 10, 0])
