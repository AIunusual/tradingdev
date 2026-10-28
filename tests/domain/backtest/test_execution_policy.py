"""Execution-policy regressions measured through trades, costs, and equity."""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt

from tradingdev.domain.backtest.signal_engine import SignalBacktestEngine
from tradingdev.domain.backtest.volume_engine import VolumeBacktestEngine


def _market(executed_signals: list[int]) -> pd.DataFrame:
    """Express expected shifted signals directly, starting with the empty bar."""
    assert executed_signals[0] == 0
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-31 20:00", periods=len(executed_signals), freq="h", tz="UTC"
            ),
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "signal": executed_signals[1:] + [0],
        }
    )


class _StopSettings(TypedDict, total=False):
    stop_loss: float
    take_profit: float


def _trigger(df: pd.DataFrame, direction: int, stop: str) -> _StopSettings:
    downward = (direction == 1) == (stop == "stop_loss")
    df.loc[2, "low" if downward else "high"] = 90.0 if downward else 110.0
    if stop == "stop_loss":
        return {"stop_loss": 0.05}
    return {"take_profit": 0.05}


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("stop", ["stop_loss", "take_profit"])
def test_stop_waits_for_changed_signal_without_repeated_entries(
    direction: int, stop: str
) -> None:
    df = _market([0, direction, direction, direction, direction, direction])
    engine = VolumeBacktestEngine(
        position_size=200,
        fees=0.01,
        slippage=0.02,
        re_entry_after_sl=False,
        **_trigger(df, direction, stop),
    )
    result = engine.run(df)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert (trade["entry_idx"], trade["exit_idx"]) == (1, 2)
    exit_price = 100 * (1 + direction * (-0.05 if stop == "stop_loss" else 0.05))
    gross = direction * 2 * (exit_price - 100)
    expected_fees = (200 + 2 * exit_price) * 0.01
    expected_slippage = (200 + 2 * exit_price) * 0.02
    expected = gross - expected_fees - expected_slippage
    assert trade["exit_price"] == pytest.approx(exit_price)
    assert result.metrics["total_fees"] == pytest.approx(expected_fees)
    assert result.metrics["total_slippage"] == pytest.approx(expected_slippage)
    assert result.metrics["total_pnl"] == pytest.approx(expected)
    assert result.equity_curve[1] == pytest.approx(-6.0)
    np.testing.assert_allclose(result.equity_curve[2:], expected)


@pytest.mark.parametrize("stop", ["stop_loss", "take_profit"])
@pytest.mark.parametrize(
    ("signals", "second_entry", "direction"),
    [
        ([0, 1, 1, 1, 0, 1, 1], 5, 1),
        ([0, 1, 1, -1, -1], 3, -1),
        ([0, 1, -1, -1, 0, -1], 5, -1),
    ],
)
def test_stop_releases_wait_only_on_a_later_changed_signal(
    signals: list[int], second_entry: int, direction: int, stop: str
) -> None:
    df = _market(signals)
    result = VolumeBacktestEngine(
        position_size=200,
        fees=0,
        slippage=0,
        re_entry_after_sl=False,
        **_trigger(df, 1, stop),
    ).run(df)
    assert len(result.trades) == 2
    first, second = result.trades
    assert first["exit_idx"] == 2
    assert second["entry_idx"] == second_entry
    assert second["direction"] == direction
    assert second["entry_price"] == 100.0
    assert result.metrics["total_fees"] == result.metrics["total_slippage"] == 0


def test_enabled_reentry_preserves_close_fill_and_mark_to_market() -> None:
    df = _market([0, 1, 1, 1])
    _trigger(df, 1, "stop_loss")
    df.loc[2, ["close", "high"]] = 110.0
    df.loc[3, ["open", "low", "high", "close"]] = 120.0
    result = VolumeBacktestEngine(
        position_size=200,
        stop_loss=0.05,
        fees=0,
        slippage=0,
        re_entry_after_sl=True,
    ).run(df)
    assert len(result.trades) == 2
    assert result.trades[1]["entry_idx"] == 2
    assert result.trades[1]["entry_price"] == 110.0
    assert result.equity_curve[2] == pytest.approx(-10.0)
    assert result.metrics["total_pnl"] == pytest.approx(-10 + 200 / 110 * 10)


@pytest.mark.parametrize("change_signal", [False, True])
def test_month_reset_does_not_override_stop_signal_wait(change_signal: bool) -> None:
    signals = [0, 1, 1, -1, -1, -1] if change_signal else [0, 1, 1, 1, 1, 1]
    df = _market(signals)
    _trigger(df, 1, "stop_loss")
    result = VolumeBacktestEngine(
        position_size=200,
        stop_loss=0.05,
        fees=0,
        slippage=0,
        re_entry_after_sl=False,
        monthly_max_loss=9,
    ).run(df)
    assert result.trades[0]["exit_idx"] == 2
    if change_signal:
        assert len(result.trades) == 2
        assert result.trades[1]["entry_idx"] == 4  # first February bar
        assert result.trades[1]["direction"] == -1
    else:
        assert len(result.trades) == 1
    np.testing.assert_allclose(result.equity_curve[2:], -10)


@pytest.mark.parametrize("size", [None, 0.0, 200.0])
@pytest.mark.parametrize("fee", [0.0, 0.01])
def test_volume_zero_means_no_orders_and_none_keeps_default(
    size: float | None, fee: float
) -> None:
    df = _market([0, 1, -1, 1])
    result = VolumeBacktestEngine(position_size=size, fees=fee, slippage=fee).run(df)
    if size == 0:
        assert result.trades == []
        np.testing.assert_array_equal(result.equity_curve, np.zeros(4))
        for key in (
            "total_trades",
            "total_pnl",
            "total_volume",
            "total_fees",
            "total_slippage",
        ):
            assert result.metrics[key] == 0
        assert result.metric_metadata["unavailable"]["win_rate"] == "no_trades"
    else:
        assert len(result.trades) == 3
        assert all(trade["size_quote"] == 200 for trade in result.trades)
        assert result.metrics["total_volume"] == 1200
        assert result.metrics["total_fees"] == pytest.approx(1200 * fee)
        assert result.metrics["total_slippage"] == pytest.approx(1200 * fee)


def test_signal_zero_also_has_no_orders() -> None:
    result = SignalBacktestEngine(
        init_cash=1000, position_size=0, fees=0, slippage=0
    ).run(_market([0, 1, -1, 1]))
    assert result.trades == []
    assert result.execution_records == []
    np.testing.assert_array_equal(result.equity_curve, np.full(4, 1000))


@pytest.mark.parametrize("position_size", [None, 200.0])
def test_disclosed_signal_policy_ignores_mutation_of_pinned_global_defaults(
    monkeypatch: pytest.MonkeyPatch, position_size: float | None
) -> None:
    df = _market([0, 1, 1, -1, 0])
    df.loc[2, "close"] = 90.0
    engine = SignalBacktestEngine(
        init_cash=1000,
        position_size=position_size,
        stop_loss=0.05,
        fees=0.01,
        slippage=0.02,
    )
    baseline = engine.run(df)
    mutations: dict[str, Any] = {
        "size": 1,
        "size_type": "percent",
        "fixed_fees": 50,
        "reject_prob": 1,
        "min_size": 50,
        "max_size": 1,
        "size_granularity": 1,
        "lock_cash": True,
        "allow_partial": False,
        "val_price": -np.inf,
        "accumulate": True,
        "sl_trail": True,
        "tp_stop": 0.01,
        "stop_entry_price": "fillprice",
        "stop_exit_price": "stopmarket",
        "upon_stop_exit": "reverse",
        "upon_opposite_entry": "ignore",
        "use_stops": False,
    }
    for key, value in mutations.items():
        monkeypatch.setitem(vbt.settings["portfolio"], key, value)
    result = engine.run(df)
    assert result.trades == baseline.trades
    assert result.execution_records == baseline.execution_records
    np.testing.assert_array_equal(result.equity_curve, baseline.equity_curve)


def test_signal_stop_uses_entry_close_and_ignores_exit_slippage() -> None:
    df = _market([0, 1, 1, 1])
    df.loc[1, ["close", "high"]] = 120.0
    df.loc[2, "open"] = 110.0
    result = SignalBacktestEngine(
        init_cash=1000, position_size=200, fees=0.01, slippage=0.02, stop_loss=0.1
    ).run(df)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade["entry_price"] == 102.0
    assert trade["exit_idx"] == 2
    assert trade["exit_price"] == 108.0  # 120 * 0.9, no stop exit slippage
    assert trade["entry_fees"] == pytest.approx(2 * 102 * 0.01)
    assert trade["exit_fees"] == pytest.approx(2 * 108 * 0.01)


def test_signal_stop_does_not_consume_market_high_low() -> None:
    df = _market([0, 1, 1, 1])
    df.loc[2, "low"] = 1.0
    result = SignalBacktestEngine(
        init_cash=1000, position_size=200, fees=0, slippage=0, stop_loss=0.05
    ).run(df)
    assert len(result.trades) == 1
    assert result.trades[0]["status"] == "open"


def test_fixed_short_notional_is_not_a_no_leverage_guarantee() -> None:
    result = SignalBacktestEngine(
        init_cash=1000, position_size=2000, fees=0, slippage=0
    ).run(_market([0, -1, 0]))
    assert result.trades[0]["size_quote"] == 2000
    assert result.trades[0]["direction"] == -1
