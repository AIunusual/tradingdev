"""Chronological reporting keeps native transitions, units and missing evidence."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from tradingdev.domain.backtest.execution_records import (
    AccountSnapshot,
    AccountState,
    ExecutionRecord,
    MarketReference,
)
from tradingdev.domain.backtest.signal_engine import SignalBacktestEngine
from tradingdev.domain.presentation.ledger import (
    LEDGER_COLUMNS,
    build_ledger,
    ledger_value_label,
)


def _execution(
    *,
    execution_id: int = 0,
    bar: int = 1,
    cash: float = 1000.0,
    position: float = 0.0,
    quantity: float = 2.0,
    price: float = 100.0,
    fees: float | None = 0.2,
    **updates: Any,
) -> ExecutionRecord:
    after_position = position + quantity
    after_cash = cash - quantity * price - (fees or 0.0)
    return ExecutionRecord.model_validate(
        {
            "execution_id": execution_id,
            "order_id": execution_id,
            "bar_index": bar,
            "timestamp": f"2024-01-0{bar + 1}T00:00:00+00:00",
            "status": "filled",
            "status_info": None,
            "side": "buy" if quantity > 0 else "sell",
            "requested_size": quantity,
            "requested_size_kind": "finite",
            "requested_size_type": "amount",
            "requested_direction": "both",
            "requested_price": price,
            "requested_fee_rate": 0.001,
            "requested_fixed_fees": 0.0,
            "requested_slippage": 0.0,
            "filled_size": abs(quantity),
            "filled_price": price,
            "fees": fees,
            "market": MarketReference(
                open=price, high=price + 10.0, low=price - 10.0, close=price + 5.0
            ),
            "valuation_price": price,
            "before": AccountSnapshot(
                cash=cash,
                position=position,
                free_cash=cash,
                debt=0.0,
                equity=cash + position * price,
            ),
            "after": AccountSnapshot(
                cash=after_cash,
                position=after_position,
                free_cash=after_cash,
                debt=0.0,
                equity=after_cash + after_position * price,
            ),
            **updates,
        }
    )


def _state(
    bar: int, *, cash: float, position: float, price: float = 105.0
) -> AccountState:
    return AccountState(
        bar_index=bar,
        timestamp=f"2024-01-0{bar + 1}T00:00:00+00:00",
        cash=cash,
        free_cash=cash,
        position=position,
        asset_value=position * price,
        equity=cash + position * price,
        mark_price=price,
    )


def test_one_timeline_separates_execution_from_close_revaluation() -> None:
    buy = _execution()
    states = [
        _state(0, cash=1000.0, position=0.0),
        _state(1, cash=buy.after.cash, position=buy.after.position),
        _state(2, cash=buy.after.cash, position=buy.after.position, price=115.0),
    ]
    projection = build_ledger([buy], states)
    first, fill, close, next_close = projection.rows
    assert [row["event_type"] for row in projection.rows] == [
        "bar_close",
        "execution",
        "bar_close",
        "bar_close",
    ]
    assert [row["sequence"] for row in projection.rows] == [1, 2, 3, 4]
    assert first["before.cash"] is first["delta.cash"] is None
    assert first["comparison_basis"] == "unavailable"
    assert fill["requested_size_unit"] == "base_asset"
    assert fill["notional"] == 200.0
    assert fill["expected_cash_delta"] == pytest.approx(-200.2)
    assert fill["before.cash"] == 1000.0
    assert fill["delta.cash"] == pytest.approx(-200.2)
    assert fill["after.cash"] == pytest.approx(799.8)
    assert fill["delta.position"] == 2.0
    assert close["comparison_basis"] == "last_execution"
    assert close["before.valuation_price"] == 100.0
    assert close["after.valuation_price"] == 105.0
    assert close["delta.equity"] == 10.0
    assert close["delta.cash"] == close["delta.position"] == 0.0
    assert close["status"] is close["filled_size"] is close["fees"] is None
    assert close["before.debt"] == 0.0
    assert close["after.debt"] is close["delta.debt"] is None
    assert next_close["comparison_basis"] == "previous_bar_close"
    assert next_close["delta.equity"] == 20.0
    assert next_close["market.open"] is None
    assert next_close["market.close"] == 115.0
    assert all(row["reconciliation"] == "matched" for row in projection.rows)
    assert (
        projection.execution_availability
        == projection.account_availability
        == "available"
    )


def test_reversal_is_one_fill_with_cash_credit_and_signed_position_change() -> None:
    record = _execution(cash=799.8, position=2.0, quantity=-5.0, price=90.0, fees=0.45)
    projection = build_ledger([record], None)
    [row] = projection.rows
    assert row["side"] == "sell"
    assert row["filled_size"] == 5.0
    assert row["signed_filled_size"] == row["delta.position"] == -5.0
    assert row["before.position"] == 2.0
    assert row["after.position"] == -3.0
    assert row["notional"] == 450.0
    assert row["delta.cash"] == pytest.approx(449.55)
    assert row["reconciliation"] == "matched"
    assert projection.account_availability == "not_recorded"


def test_multiple_attempts_in_same_bar_keep_native_order_before_one_close() -> None:
    buy = _execution(execution_id=3)
    sell = _execution(
        execution_id=8, cash=buy.after.cash, position=2.0, quantity=-1.0, fees=0.1
    )
    state = _state(1, cash=sell.after.cash, position=1.0)
    rows = build_ledger([sell, buy], [state]).rows
    assert [row["execution_id"] for row in rows] == [3, 8, None]
    assert [row["sequence"] for row in rows] == [1, 2, 3]
    assert rows[-1]["before.cash"] == sell.after.cash
    assert rows[-1]["before.position"] == 1.0
    assert rows[-1]["delta.position"] == 0.0


@pytest.mark.parametrize("status", ["ignored", "rejected"])
def test_unfilled_attempt_does_not_fabricate_fill_details(status: str) -> None:
    before = AccountSnapshot(
        cash=1000.0, position=0.0, free_cash=1000.0, debt=0.0, equity=1000.0
    )
    record = _execution(
        status=status,
        status_info="randomevent",
        order_id=None,
        side=None,
        filled_size=None,
        filled_price=None,
        fees=None,
        before=before,
        after=before,
        requested_size=None,
        requested_size_kind="positive_infinity",
    )
    [row] = build_ledger([record], []).rows
    assert row["status"] == status
    assert row["requested_size"] is None
    assert row["requested_size_kind"] == "positive_infinity"
    assert row["requested_size_unit"] == "base_asset"
    for key in (
        "side",
        "filled_size",
        "filled_price",
        "fees",
        "notional",
        "expected_fees",
        "expected_cash_delta",
    ):
        assert row[key] is None
    assert row["delta.cash"] == row["delta.position"] == 0.0
    assert row["reconciliation"] == "matched"


def test_native_dust_normalization_is_retained_without_rounded_validation() -> None:
    before = AccountSnapshot(
        cash=1e-12, position=0.0, free_cash=1e-12, debt=0.0, equity=1e-12
    )
    after = AccountSnapshot(cash=0.0, position=0.0, free_cash=0.0, debt=0.0, equity=0.0)
    record = _execution(
        status="rejected",
        order_id=None,
        side=None,
        filled_size=None,
        filled_price=None,
        fees=None,
        before=before,
        after=after,
    )
    [row] = build_ledger([record], None).rows
    assert row["delta.cash"] == row["cash_residual"] == -1e-12
    assert row["reconciliation"] == "matched"


@pytest.mark.parametrize(
    ("size_type", "unit"),
    [
        ("amount", "base_asset"),
        ("targetamount", "base_asset"),
        ("value", "quote_currency"),
        ("targetvalue", "quote_currency"),
        ("percent", "ratio"),
        ("targetpercent", "ratio"),
        ("unknown_future_type", None),
    ],
)
def test_requested_size_unit_is_from_saved_type_not_fill_size(
    size_type: str, unit: str | None
) -> None:
    [row] = build_ledger([_execution(requested_size_type=size_type)], None).rows
    assert row["requested_size_unit"] == unit
    if unit is None:
        assert "單位未推測" in str(row["reconciliation_note"])


def test_missing_streams_differ_from_recorded_empty_and_volume_unsupported() -> None:
    legacy = build_ledger(None, None)
    empty = build_ledger([], [])
    volume = build_ledger(None, None, mode="volume")
    assert legacy.rows == empty.rows == volume.rows == []
    assert (
        legacy.execution_availability == legacy.account_availability == "not_recorded"
    )
    assert empty.execution_availability == empty.account_availability == "available"
    assert (
        volume.execution_availability
        == volume.account_availability
        == "unsupported_volume_accounting"
    )


def test_account_only_history_keeps_balance_changes_but_invents_no_orders() -> None:
    rows = build_ledger(
        None,
        [_state(0, cash=1000.0, position=0.0), _state(1, cash=500.0, position=5.0)],
    ).rows
    assert len(rows) == 2
    assert rows[-1]["delta.cash"] == -500.0
    assert rows[-1]["delta.position"] == 5.0
    for row in rows:
        assert row["event_type"] == "bar_close"
        assert (
            row["execution_id"] is row["filled_price"] is row["requested_size"] is None
        )
        assert row["cash_residual"] is row["position_residual"] is None


@pytest.mark.parametrize(
    ("fee_difference", "expected"), [(1e-9, "matched"), (0.0001, "mismatch")]
)
def test_reconciliation_uses_full_precision_and_published_tolerance(
    fee_difference: float, expected: str
) -> None:
    record = _execution(fees=0.2 + fee_difference)
    original = record.model_dump_json()
    [row] = build_ledger([record], None).rows
    assert row["reconciliation"] == expected
    assert row["fees_residual"] == pytest.approx(fee_difference)
    assert row["fees"] == record.fees
    assert record.model_dump_json() == original
    # The mismatch is materially smaller than the human display's two decimals.
    assert round(float(str(row["fees_residual"])), 2) == 0.0
    json.dumps(row, allow_nan=False)


def test_cash_reconciliation_uses_transaction_scale_for_large_all_in_fills() -> None:
    record = _execution(cash=1e10, quantity=1e8, fees=0.0, requested_fee_rate=0.0)
    altered = record.after.model_copy(
        update={"cash": 1e-7, "equity": record.after.equity + 1e-7}
    )
    validated = ExecutionRecord.model_validate(
        {**record.model_dump(), "after": altered}
    )
    [row] = build_ledger([validated], None).rows
    assert row["cash_residual"] == 1e-7
    assert row["reconciliation"] == "matched"


def test_column_schema_is_complete_time_first_and_preserves_missing_market_data() -> (
    None
):
    record = _execution(timestamp=None, market=MarketReference(close=105.0))
    [row] = build_ledger([record], None).rows
    assert (
        row["timestamp"]
        is row["market.open"]
        is row["market.high"]
        is row["market.low"]
        is None
    )
    keys = [column.key for column in LEDGER_COLUMNS]
    assert keys[:2] == ["timestamp", "sequence"]
    assert len(keys) == len(set(keys))
    assert set(row) == set(keys)
    assert all(column.label != column.key for column in LEDGER_COLUMNS)
    assert (
        next(
            column.label for column in LEDGER_COLUMNS if column.key == "delta.position"
        )
        == "持倉數量變動（標的單位，負數為減少）"
    )
    assert ledger_value_label("event_type", "bar_close") == "K 線收盤估值"
    assert ledger_value_label("requested_size_unit", "ratio") == "比例（1 表示 100%）"
    assert ledger_value_label("unknown", "future") == "future"
    assert ledger_value_label("unknown", None) == "未記錄／不適用"


@pytest.mark.parametrize("capital", [1000.0, 1e10])
@pytest.mark.parametrize("fee_rate", [0.001, 1e-8])
def test_actual_engine_orders_reconcile_with_slippage_reversals_and_closes(
    capital: float,
    fee_rate: float,
) -> None:
    market = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, tz="UTC"),
            "open": [100.0, 110.0, 90.0, 95.0, 105.0],
            "close": [105.0, 115.0, 95.0, 100.0, 110.0],
            "signal": [1, -1, 0, 1, 1],
        }
    )
    result = SignalBacktestEngine(init_cash=capital, fees=fee_rate, slippage=0.01).run(
        market
    )
    projection = build_ledger(result.execution_records, result.account_history)
    assert result.execution_records is not None
    assert len(projection.rows) == len(result.execution_records) + 5
    assert all(row["reconciliation"] == "matched" for row in projection.rows)
    json.dumps(projection.rows, allow_nan=False)
    if capital == 1e10 and fee_rate == 1e-8:
        first = result.execution_records[0]
        [original] = build_ledger([first], None).rows
        assert abs(float(str(original["fees_residual"]))) > 1e-8
        assert original["fees_residual"] == pytest.approx(-9.073486495e-7)
        assert first.filled_size is not None and first.filled_price is not None
        notional = first.filled_size * first.filled_price
        altered = first.model_copy(
            update={"requested_fee_rate": fee_rate + 0.0001 / notional}
        )
        [mismatch] = build_ledger([altered], None).rows
        # This difference remains below two-decimal display precision but well
        # above native roundoff. A notional * 1e-10 tolerance would hide it.
        assert 0.00009 < abs(float(str(mismatch["fees_residual"]))) < 0.0002
        assert mismatch["reconciliation"] == "mismatch"


def test_unused_large_cash_cannot_hide_a_small_trade_fee_error() -> None:
    record = _execution(cash=1e10, fees=0.200001)
    [row] = build_ledger([record], None).rows
    assert row["fees_residual"] == pytest.approx(1e-6)
    assert row["reconciliation"] == "mismatch"


def test_fixed_fees_are_included_and_price_mismatches_are_detected() -> None:
    [row] = build_ledger([_execution(fees=0.3, requested_fixed_fees=0.1)], None).rows
    assert row["expected_fees"] == pytest.approx(0.3)
    assert row["reconciliation"] == "matched"
    [mismatch] = build_ledger([_execution(requested_slippage=0.000001)], None).rows
    assert mismatch["fill_price_residual"] == pytest.approx(-0.0001)
    assert mismatch["reconciliation"] == "mismatch"
