"""The research and product buy-and-hold baselines are different measurements.

Expected values are computed by hand from the price path 100 -> 110 -> 121
and a 1% per-side commission, not taken from either implementation.
:func:`tradingdev.domain.benchmark.buy_and_hold` invests at the first close
and liquidates at the last, charging both fills; the statistics package
builds a cost-free per-period return series from caller-supplied prices.
Neither series may be described as the other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.benchmark import buy_and_hold
from tradingdev.domain.statistics import benchmark as research_benchmark
from tradingdev.domain.statistics.benchmark import buy_and_hold_returns


def test_module_docstring_keeps_the_current_product_cost_rule() -> None:
    # The product baseline changed to entry and exit fills in the terminal
    # liquidation work; a stale "entry only" claim makes this module's
    # comparison with the product baseline wrong for readers.
    doc = research_benchmark.__doc__ or ""
    assert "both fills" in doc
    assert "entry only" not in doc
    assert "no costs at all" in doc


def test_product_baseline_charges_both_fills() -> None:
    # init_cash 101 and a 1% fee make the funded entry price 101, so the
    # position is exactly one unit: entry 100, exit 110.
    result = buy_and_hold(np.array([100.0, 110.0]), 101.0, fees=0.01, slippage=0.0)
    record = result.trade_record()

    assert float(record["entry_fees"]) == pytest.approx(1.0)
    assert float(record["exit_fees"]) == pytest.approx(1.1)
    assert float(record["fee"]) == pytest.approx(2.1)
    assert float(record["gross_pnl"]) == pytest.approx(10.0)
    assert float(record["net_pnl"]) == pytest.approx(7.9)
    # Final equity is the exit proceeds after the exit commission: 110 - 1.1.
    assert float(result.equity[-1]) == pytest.approx(108.9)


def test_research_baseline_charges_no_costs() -> None:
    timestamps = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")

    returns = buy_and_hold_returns(
        np.array([100.0, 110.0, 121.0]),
        timestamps=timestamps,
        frequency="daily",
    )

    # The first period uses the first price as its opening equity; each later
    # period is closing / previous closing - 1, with no fee or slippage.
    assert returns.to_numpy().tolist() == pytest.approx([0.0, 0.1, 0.1])


def test_the_two_baselines_disagree_on_the_same_prices() -> None:
    prices = np.array([100.0, 110.0])
    product = buy_and_hold(prices, 101.0, fees=0.01, slippage=0.0)
    timestamps = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    research = buy_and_hold_returns(prices, timestamps=timestamps, frequency="daily")

    # Cost-free holding gains 10%; the product baseline nets 7.9 after the
    # 2.1 of entry and exit fees. The two are not the same measurement.
    cost_free_total_return = float(np.prod(1.0 + research.to_numpy()) - 1.0)
    product_net_return = float(product.trade_record()["net_pnl"]) / 101.0
    assert cost_free_total_return == pytest.approx(0.1)
    assert product_net_return == pytest.approx(7.9 / 101.0)
    assert cost_free_total_return > product_net_return
