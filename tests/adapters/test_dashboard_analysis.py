"""Dashboard KPIs and trade charts follow the normalized performance ledger."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from tradingdev.adapters.dashboard.analysis import (
    build_trades_df,
    consecutive_loss_counts,
    cumulative_pnl_pct,
    filter_by_month,
    metric_cards,
    monthly_volume,
    rolling_mdd_absolute,
)


def test_signal_cards_preserve_ratio_and_fraction_units_and_positive_drawdown() -> None:
    cards = dict(
        metric_cards(
            {
                "total_return": 0.125,
                "sharpe_ratio": 1.25,
                "max_drawdown": 0.05,
                "max_drawdown_amount": 500.0,
                "win_rate": 0.75,
                "total_trades": 4,
                "total_volume": 2000.0,
            },
            mode="signal",
        )
    )

    assert cards == {
        "Total Return": "12.50%",
        "Sharpe": "1.250",
        "Max DD": "5.00%",
        "Win Rate": "75.0%",
        "Closed Trades": "4",
        "Volume": "2,000.00",
    }


def test_volume_cards_use_amount_drawdown_and_unavailable_sharpe() -> None:
    cards = dict(
        metric_cards(
            {
                "total_pnl": -0.5,
                "sharpe_ratio": None,
                "max_drawdown": None,
                "max_drawdown_amount": 25.5,
                "total_trades": 0,
                "win_rate": None,
                "total_volume": 200.0,
            },
            mode="volume",
        )
    )

    assert cards["Total P&L"] == "-0.50"
    assert cards["Sharpe"] == "N/A"
    assert cards["Max DD (amount)"] == "25.50"
    assert cards["Win Rate"] == "N/A"
    assert cards["Closed Trades"] == "0"


@pytest.mark.parametrize("mode", ["signal", "volume"])
@pytest.mark.parametrize(
    "value", [None, float("nan"), float("inf"), -float("inf"), True]
)
def test_cards_never_format_unavailable_values_as_numbers(
    mode: str, value: float | bool | None
) -> None:
    metrics = dict.fromkeys(
        (
            "total_pnl",
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "max_drawdown_amount",
            "win_rate",
            "total_trades",
            "total_volume",
        ),
        value,
    )

    assert {text for _, text in metric_cards(metrics, mode=mode)} == {"N/A"}


def _trades() -> list[dict[str, Any]]:
    return [
        {
            "entry_idx": 1,
            "exit_idx": 2,
            "size_quote": 100.0,
            "exit_notional": 120.0,
            "net_pnl": 19.0,
            "status": "closed",
        },
        {
            "entry_idx": 3,
            "exit_idx": 3,
            "size_quote": 50.0,
            "exit_notional": 75.0,
            "net_pnl": 24.0,
            "status": "open",
        },
    ]


def test_execution_times_and_monthly_volume_use_recorded_indices_and_fill_amounts() -> (
    None
):
    timestamps = pd.to_datetime(
        ["2024-01-01", "2024-01-31", "2024-02-01", "2024-02-02"]
    ).to_numpy()
    trades = build_trades_df(_trades(), timestamps)

    assert trades.loc[0, "entry_timestamp"] == pd.Timestamp("2024-01-31")
    assert trades.loc[0, "timestamp"] == pd.Timestamp("2024-02-01")
    assert pd.isna(trades.loc[1, "timestamp"])
    volume = monthly_volume(trades)
    assert volume.to_dict("records") == [
        {"month": "2024-01", "volume_quote": 100.0},
        {"month": "2024-02", "volume_quote": 170.0},
    ]
    equity = pd.Series([100.0, 99.0, 119.0, 143.0], index=pd.DatetimeIndex(timestamps))
    _, january_trades = filter_by_month(equity, trades, "2024-01")
    _, february_trades = filter_by_month(equity, trades, "2024-02")
    assert january_trades.empty
    assert february_trades["net_pnl"].tolist() == [19.0]


def test_missing_trade_indices_do_not_invent_execution_dates() -> None:
    trades = build_trades_df(
        [{"net_pnl": 1.0, "status": "closed", "size_quote": 100.0}],
        pd.date_range("2024-01-01", periods=2).to_numpy(),
    )

    assert trades["timestamp"].isna().all()
    assert monthly_volume(trades).empty


def test_empty_trade_ledger_supports_dashboard_helpers() -> None:
    trades = build_trades_df([])

    assert monthly_volume(trades).empty
    assert consecutive_loss_counts(trades).empty


def test_losing_streaks_exclude_open_trades_and_stop_at_breakeven() -> None:
    trades = pd.DataFrame(
        {
            "net_pnl": [-1.0, -5.0, -2.0, 0.0, -1.0, 2.0],
            "status": ["closed", "open", "closed", "closed", "closed", "closed"],
        }
    )

    assert consecutive_loss_counts(trades).to_dict() == {1: 1, 2: 1}


@pytest.mark.parametrize("capital", [None, 0.0, -1.0])
def test_percentage_curve_is_unavailable_without_positive_capital(
    capital: float | None,
) -> None:
    equity = pd.Series([0.0, 1.0, 2.0])

    assert cumulative_pnl_pct(equity, capital).isna().all()


def test_percentage_curve_uses_initial_capital() -> None:
    equity = pd.Series([100.0, 110.0, 90.0])

    np.testing.assert_allclose(cumulative_pnl_pct(equity, 100.0), [0.0, 10.0, -10.0])


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([100.0, 50.0, 60.0, 70.0], [50.0, 0.0]),
        ([10.0, 20.0, 30.0, 40.0], [0.0, 0.0]),
        ([40.0, 30.0, 20.0, 10.0], [20.0, 20.0]),
        ([20.0, 20.0, 20.0, 20.0], [0.0, 0.0]),
        ([10.0, 40.0, 30.0, 20.0], [10.0, 20.0]),
    ],
)
def test_rolling_drawdown_uses_only_ordered_pairs_in_each_window(
    values: list[float], expected: list[float]
) -> None:
    result = rolling_mdd_absolute(pd.Series(values), 3)
    assert result.iloc[:2].isna().all()
    np.testing.assert_allclose(result.iloc[2:], expected)


def test_rolling_drawdown_matches_independent_pairwise_reference() -> None:
    values = np.random.default_rng(72).normal(size=100)
    for window in (1, 7, 30):
        expected = [
            max(
                values[a] - values[b]
                for a in range(i - window + 1, i + 1)
                for b in range(a, i + 1)
            )
            for i in range(window - 1, len(values))
        ]
        np.testing.assert_allclose(
            rolling_mdd_absolute(pd.Series(values), window).iloc[window - 1 :], expected
        )


@pytest.mark.parametrize("frequency", ["2min", "90min", "7D", "MS"])
def test_calendar_drawdown_uses_elapsed_time_for_every_frequency(
    frequency: str,
) -> None:
    index = pd.date_range("2024-01-01", "2024-03-01", freq=frequency, tz="UTC")
    # A peak before the current window must disappear even with sparse bars.
    equity = pd.Series(50.0, index=index)
    equity.iloc[0] = 100.0
    result = rolling_mdd_absolute(equity, pd.Timedelta(days=30))
    assert result.loc[index < index[0] + pd.Timedelta(days=30)].isna().all()
    assert result.iloc[-1] == 0.0


def test_calendar_drawdown_handles_missing_bars_and_excludes_left_boundary() -> None:
    equity = pd.Series(
        [100.0, 50.0, 80.0, 60.0, 70.0],
        index=pd.to_datetime(
            ["2024-01-01", "2024-01-04", "2024-01-31", "2024-02-02", "2024-02-04"],
            utc=True,
        ),
    )
    result = rolling_mdd_absolute(equity, pd.Timedelta(days=30))
    np.testing.assert_allclose(
        result, [np.nan, np.nan, 0.0, 20.0, 20.0], equal_nan=True
    )


def test_rolling_drawdown_invalid_values_do_not_become_zero_risk() -> None:
    result = rolling_mdd_absolute(pd.Series([10.0, np.nan, 20.0, 15.0, 12.0]), 2)
    np.testing.assert_allclose(
        result, [np.nan, np.nan, np.nan, 5.0, 3.0], equal_nan=True
    )
    with pytest.raises(ValueError, match="DatetimeIndex"):
        rolling_mdd_absolute(pd.Series([10.0, 20.0]), pd.Timedelta(days=30))
