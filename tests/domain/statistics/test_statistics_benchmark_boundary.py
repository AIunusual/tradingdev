"""The research and product buy-and-hold baselines are different measurements.

Every expected value below is derived from the input specification alone:
prices 100 -> 110 -> 121, a 1% per-side commission and init_cash 101.
The derivation is written out as arithmetic over those inputs, so a reader
can check it without running either implementation. Nothing here is copied
from a program's output.

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

# Input specification (human-defined, from the terminal-liquidation contract).
_PRICES = (100.0, 110.0, 121.0)
_FEE_RATE = 0.01
_INIT_CASH = 101.0

# Derived by hand before running either module. The funded entry price is
# 100 * (1 + 0.01) = 101, so 101 of capital buys exactly one unit.
_QUANTITY = _INIT_CASH / (_PRICES[0] * (1.0 + _FEE_RATE))
_EXPECTED_ENTRY_FEE = _QUANTITY * _PRICES[0] * _FEE_RATE
_EXPECTED_EXIT_FEE = _QUANTITY * _PRICES[1] * _FEE_RATE
_EXPECTED_TOTAL_FEE = _EXPECTED_ENTRY_FEE + _EXPECTED_EXIT_FEE
_EXPECTED_GROSS_PNL = _QUANTITY * (_PRICES[1] - _PRICES[0])
_EXPECTED_NET_PNL = _EXPECTED_GROSS_PNL - _EXPECTED_TOTAL_FEE
_EXPECTED_FINAL_EQUITY = _QUANTITY * _PRICES[1] - _EXPECTED_EXIT_FEE
_EXPECTED_COST_FREE_RETURN = _PRICES[1] / _PRICES[0] - 1.0
_EXPECTED_PERIOD_RETURNS = [
    0.0,
    _PRICES[1] / _PRICES[0] - 1.0,
    _PRICES[2] / _PRICES[1] - 1.0,
]


def test_module_docstring_keeps_the_current_product_cost_rule() -> None:
    # The product baseline changed to entry and exit fills in the terminal
    # liquidation work; a stale "entry only" claim makes this module's
    # comparison with the product baseline wrong for readers.
    doc = research_benchmark.__doc__ or ""
    assert "both fills" in doc
    assert "entry only" not in doc
    assert "no costs at all" in doc


def test_product_baseline_charges_both_fills() -> None:
    result = buy_and_hold(
        np.array(_PRICES[:2]), _INIT_CASH, fees=_FEE_RATE, slippage=0.0
    )
    record = result.trade_record()

    assert float(record["entry_fees"]) == pytest.approx(_EXPECTED_ENTRY_FEE)
    assert float(record["exit_fees"]) == pytest.approx(_EXPECTED_EXIT_FEE)
    assert float(record["fee"]) == pytest.approx(_EXPECTED_TOTAL_FEE)
    assert float(record["gross_pnl"]) == pytest.approx(_EXPECTED_GROSS_PNL)
    assert float(record["net_pnl"]) == pytest.approx(_EXPECTED_NET_PNL)
    assert float(result.equity[-1]) == pytest.approx(_EXPECTED_FINAL_EQUITY)


def test_research_baseline_charges_no_costs() -> None:
    timestamps = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")

    returns = buy_and_hold_returns(
        np.array(_PRICES), timestamps=timestamps, frequency="daily"
    )

    # The first period uses the first price as its opening equity; each later
    # period is closing / previous closing - 1, with no fee or slippage.
    assert returns.to_numpy().tolist() == pytest.approx(_EXPECTED_PERIOD_RETURNS)


def test_the_two_baselines_disagree_on_the_same_prices() -> None:
    prices = np.array(_PRICES[:2])
    product = buy_and_hold(prices, _INIT_CASH, fees=_FEE_RATE, slippage=0.0)
    timestamps = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    research = buy_and_hold_returns(prices, timestamps=timestamps, frequency="daily")

    # Cost-free holding gains 10%; the product baseline nets 7.9 after the
    # 2.1 of entry and exit fees. The two are not the same measurement.
    cost_free_total_return = float(np.prod(1.0 + research.to_numpy()) - 1.0)
    product_net_return = float(product.trade_record()["net_pnl"]) / _INIT_CASH
    assert cost_free_total_return == pytest.approx(_EXPECTED_COST_FREE_RETURN)
    assert product_net_return == pytest.approx(_EXPECTED_NET_PNL / _INIT_CASH)
    assert cost_free_total_return > product_net_return
