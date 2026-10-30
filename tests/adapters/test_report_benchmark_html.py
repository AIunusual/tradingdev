"""The benchmark section states saved evidence without repeating whole series."""

from __future__ import annotations

from typing import Any

import pandas as pd

from tradingdev.adapters.reporting.html import _benchmark
from tradingdev.adapters.reporting.tables import _benchmark_row


def _bars(count: int) -> list[str]:
    return [
        stamp.isoformat().replace("+00:00", "Z")
        for stamp in pd.date_range("2024-01-01", periods=count, freq="h", tz="UTC")
    ]


def _scope(**changes: Any) -> dict[str, Any]:
    scope: dict[str, Any] = {
        "scope_id": "full",
        "kind": "backtest",
        "observations": {
            "init_cash": 1000.0,
            "timestamps": _bars(3),
            "equity_curve": [1000.0, 1010.0, 1020.0],
        },
        "values": {"total_return": 0.02, "total_pnl": 20.0},
        "definitions": {
            "total_return": {"unit": "fraction"},
            "total_pnl": {"unit": "amount"},
        },
        "benchmark": None,
    }
    scope.update(changes)
    return scope


def _available(count: int = 3, **changes: Any) -> dict[str, Any]:
    stamps = _bars(count)
    benchmark: dict[str, Any] = {
        "status": "available",
        "asset_class": "crypto",
        "definition": {
            "id": "crypto_btc_buy_and_hold",
            "asset_class": "crypto",
            "symbol": "BTC/USDT",
            "label": "BTC/USDT 買進持有",
            "weighting": "buy_and_hold",
            "rationale": "加密貨幣策略的機會成本基準是持有比特幣。",
        },
        "source": "binance_vision",
        "market_type": "futures/um",
        "dataset_id": "BTC/USDT:1h:2024-01-01:2024-01-03",
        "reused_run_frame": True,
        "init_cash": 1000.0,
        "capital_basis": "run_init_cash",
        "fees": 0.001,
        "slippage": 0.0,
        "coverage": {
            "scope_bars": count,
            "aligned_bars": count,
            "missing_bars": 0,
            "complete": True,
            "start": stamps[0],
            "end": stamps[-1],
        },
        "timestamps": stamps,
        "equity_curve": [1000.0 + index for index in range(count)],
        "returns": [0.0 for _ in range(count)],
        "values": {"total_return": 0.1, "total_pnl": 100.0},
        "metadata": {"benchmark": {"weighting": "buy_and_hold"}},
    }
    benchmark.update(changes)
    return benchmark


def _cells(row: str) -> list[str]:
    return [cell.split(">", 1)[1].split("<", 1)[0] for cell in row.split("<td")[1:]]


def test_benchmark_details_do_not_repeat_the_full_series() -> None:
    count = 1000
    stamps = _bars(count)
    scope = _scope(
        observations={
            "init_cash": 1000.0,
            "timestamps": stamps,
            "equity_curve": [1000.0 for _ in range(count)],
        },
        benchmark=_available(count),
    )
    fragment = _benchmark(scope)
    details = fragment.split("<pre>")[-1].split("</pre>")[0]
    # The chart already carries both paths; the inspectable block must not
    # repeat 1,000 timestamps, closes and returns as indented JSON.
    assert "equity_curve" not in details
    assert "timestamps" not in details
    assert "returns" not in details
    assert "coverage" in details
    assert fragment.count("<polyline") >= 2


def test_fold_summary_does_not_promise_that_a_rerun_adds_a_baseline() -> None:
    fragment = _benchmark(_scope(kind="fold_summary", observations=None))
    assert "各折彙總" in fragment
    assert "重新執行" not in fragment


def test_scope_without_saved_evidence_names_the_optimization_gap() -> None:
    fragment = _benchmark(_scope())
    assert "沒有保存 benchmark 證據" in fragment
    assert "參數搜尋" in fragment


def test_normalized_baseline_withholds_amounts_instead_of_mixing_units() -> None:
    fragment = _benchmark(
        _scope(benchmark=_available(capital_basis="normalized_unit", init_cash=1.0))
    )
    row = fragment.split("<th>總淨損益</th>", 1)[1].split("</tr>", 1)[0]
    cells = _cells(row)
    assert cells[0] != "—"  # the strategy's realized amount stays visible
    assert cells[1] == "—"  # a one-unit baseline amount is not comparable
    assert cells[2] == "—"  # so no difference is computed


def test_partial_baseline_never_subtracts_returns_from_different_periods() -> None:
    scope = _scope(
        benchmark=_available(
            coverage={
                "scope_bars": 3,
                "aligned_bars": 2,
                "missing_bars": 1,
                "complete": False,
            }
        )
    )
    row = _benchmark_row(scope)
    assert row["excess_return"] is None
    assert row["benchmark_coverage"] == "2/3"
    fragment = _benchmark(scope)
    result_row = fragment.split("<th>累積報酬率</th>", 1)[1].split("</tr>", 1)[0]
    assert _cells(result_row)[2] == "—"
