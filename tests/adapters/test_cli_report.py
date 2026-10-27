"""CLI report behavior for available and unavailable numerical metrics."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from tradingdev.adapters.cli.report import format_metrics_report
from tradingdev.domain.backtest.signal_engine import SignalBacktestEngine


def _metrics() -> dict[str, Any]:
    return {
        "total_pnl": 1250.0,
        "total_return": 0.125,
        "annual_return": 0.25,
        "max_drawdown": 0.05,
        "max_drawdown_amount": 500.0,
        "sharpe_ratio": 1.25,
        "sortino_ratio": 1.5,
        "calmar_ratio": 5.0,
        "annual_volatility": 0.15,
        "win_rate": 0.75,
        "profit_factor": 3.0,
        "trade_expectancy": 312.5,
        "total_fees": 2.0,
        "total_trades": 4,
        "total_volume": 2000.0,
        "daily_pnl_mean": 625.0,
        "daily_pnl_std": 125.0,
        "daily_pnl_min": -100.0,
        "daily_pnl_max": 1000.0,
        "daily_pnl_median": 625.0,
        "n_days": 2,
        "monthly_pnl_mean": 1250.0,
        "monthly_pnl_std": 0.0,
        "monthly_pnl_min": 1250.0,
        "monthly_pnl_max": 1250.0,
        "monthly_pnl_median": 1250.0,
        "monthly_trades_mean": 4.0,
        "monthly_volume_mean": 2000.0,
        "n_months": 1,
    }


def _values(report: str) -> dict[str, str]:
    return {
        key.strip(): value.strip()
        for line in report.splitlines()
        if ":" in line
        for key, value in [line.split(":", 1)]
    }


def test_report_preserves_finite_numbers_and_explicit_units() -> None:
    report = format_metrics_report(_metrics())
    values = _values(report)

    assert values["Total P&L"] == "+1,250.00"
    assert values["Total Return"] == "12.50%"
    assert values["Annual Return"] == "25.00%"
    assert values["Max Drawdown"] == "5.00%"
    assert values["Max Drawdown (amount)"] == "500.00"
    assert values["Sharpe Ratio"] == "1.2500"
    assert values["Sortino Ratio"] == "1.5000"
    assert values["Calmar Ratio"] == "5.0000"
    assert values["Annual Volatility"] == "15.00%"
    assert values["Win Rate"] == "75.00%"
    assert values["Trade Expectancy"] == "+312.50"
    assert values["Total Fees"] == "2.00"
    assert values["Total Trades"] == "4"
    assert values["Total Volume"] == "2,000.00"
    assert values["Daily P&L min"] == "-100.00"
    assert values["Monthly Volume"] == "2,000.00"
    assert values["Monthly P&L std"] == "0.00"
    assert values["Days"] == "2"
    assert values["Months"] == "1"
    assert "quote currency" in report
    assert "Est. Monthly Vol" not in report


def test_large_annual_return_uses_percentage_scale_in_cli_report() -> None:
    metrics = _metrics() | {"annual_return": 493.04577519114036}
    assert _values(format_metrics_report(metrics))["Annual Return"] == "49304.58%"
    assert metrics["annual_return"] == 493.04577519114036


def test_volume_report_keeps_return_metrics_unavailable_and_drawdown_in_own_field() -> (
    None
):
    metrics = _metrics()
    metrics.update(
        dict.fromkeys(
            (
                "total_return",
                "annual_return",
                "max_drawdown",
                "sharpe_ratio",
                "sortino_ratio",
                "calmar_ratio",
                "annual_volatility",
            )
        )
    )
    values = _values(format_metrics_report(metrics, "volume"))

    assert values["Max Drawdown (amount)"] == "500.00"
    assert values["Total P&L"] == "+1,250.00"
    assert values["Total Return"] == "N/A"
    assert values["Max Drawdown"] == "N/A"
    assert values["Sharpe Ratio"] == "N/A"
    assert values["Win Rate"] == "75.00%"


@pytest.mark.parametrize("mode", ["signal", "volume"])
@pytest.mark.parametrize(
    "value", [None, float("inf"), float("-inf"), float("nan"), True]
)
def test_report_marks_unavailable_metrics_without_numeric_formatting(
    mode: str, value: float | bool | None
) -> None:
    metrics = dict.fromkeys(_metrics(), value)
    values = _values(format_metrics_report(metrics, mode))

    assert values
    assert set(values.values()) == {"N/A"}


@pytest.mark.parametrize("mode", ["signal", "volume"])
def test_report_marks_missing_metrics_as_unavailable(mode: str) -> None:
    values = _values(format_metrics_report({}, mode))

    assert values
    assert set(values.values()) == {"N/A"}


def test_monthly_volume_uses_calculated_value_without_estimating_from_days() -> None:
    metrics = _metrics()
    metrics["n_days"] = 0
    metrics["monthly_volume_mean"] = 1234.5

    assert _values(format_metrics_report(metrics))["Monthly Volume"] == "1,234.50"


def test_weekly_backtest_report_keeps_observed_totals_and_marks_daily_metrics_na() -> (
    None
):
    prices = [100.0, 100.0, 110.0, 121.0, 110.0, 115.0, 130.0, 130.0]
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=8, freq="7D", tz="UTC"),
            "open": prices,
            "close": prices,
            "signal": [1, 1, 1, 1, 1, 0, 0, 0],
        }
    )
    result = SignalBacktestEngine(
        init_cash=100.0, fees=0.0, slippage=0.0, freq="7D", periods_per_year=365
    ).run(frame)
    values = _values(format_metrics_report(result.metrics))

    assert values["Total Return"] == "30.00%"
    assert values["Total P&L"] == "+30.00"
    assert values["Total Trades"] == "1"
    assert values["Annual Return"] == "N/A"
    assert values["Sharpe Ratio"] == "N/A"
    assert values["Annual Volatility"] == "N/A"
    for period in ("Daily", "Monthly"):
        for statistic in ("mean", "std", "min", "max", "median"):
            assert values[f"{period} P&L {statistic}"] == "N/A"
