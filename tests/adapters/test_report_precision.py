"""Human reconciliation must not rely on rounded displayed asset quantities."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from tradingdev.adapters.reporting.csv_export import table_csv
from tradingdev.adapters.reporting.tables import ledger_table, render_table, trade_table
from tradingdev.domain.presentation.ledger import LEDGER_COLUMNS, LedgerProjection


def ledger_row(**values: Any) -> dict[str, Any]:
    row = dict.fromkeys(column.key for column in LEDGER_COLUMNS)
    row.update(values)
    return row


def test_full_precision_reconciles_all_cash_fill_with_rounded_quantity() -> None:
    row = ledger_row(
        sequence=52,
        event_type="execution",
        status="filled",
        filled_size=0.10173757490229546,
        filled_price=98252.8,
        fees=3.998400639744432,
        notional=9996.001599360256,
        **{"before.cash": 10000.0, "delta.cash": -10000.0, "after.cash": 0.0},
    )
    spec = ledger_table(LedgerProjection([row], "available", "available", ()))
    exact = spec["full_precision_values"]
    quantity = Decimal(exact["filled_size"][0])
    price = Decimal(exact["filled_price"][0])
    fee = Decimal(exact["fees"][0])
    cash_delta = Decimal(exact["delta.cash"][0])
    assert abs(quantity * price + fee + cash_delta) < Decimal("1e-11")
    assert Decimal("0.10") * price + Decimal("4.00") == Decimal("9829.280")
    assert next(c for c in spec["columns"] if c["field"] == "notional")["visible"]
    cash_preset = next(p for p in spec["column_presets"] if p["label"] == "現金與持倉")
    assert "notional" in cash_preset["fields"]
    html = render_table(spec, identifier="ledger", key="0:0:ledger")
    assert (
        "data-full='0.10173757490229546' title='0.10173757490229546'>0.10</td>" in html
    )
    assert "data-full='3.998400639744432' title='3.998400639744432'>4.00</td>" in html
    assert html.count("data-full-precision") == 1
    assert "再核算數量 × 價格與費用" in html
    assert "0.10173757490229546" in table_csv(spec).decode("utf-8-sig")
    assert spec["rows"][0]["filled_size"] == row["filled_size"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0004, "0.04%"),
        (0.30000000000000004, "30.000000000000004%"),
        (1e-30, "1E-28%"),
        (-0.0, "-0E+1%"),
        (None, None),
    ],
)
def test_percentage_text_preserves_decimal_digits_and_unit(
    value: float | None,
    expected: str | None,
) -> None:
    row = ledger_row(requested_fee_rate=value)
    spec = ledger_table(LedgerProjection([row], "available", "available", ()))
    assert spec["full_precision_values"]["requested_fee_rate"] == [expected]
    assert spec["rows"][0]["requested_fee_rate"] == value


def test_large_integer_uses_exact_text_and_disables_lossy_browser_sorting() -> None:
    large = 9007199254740993
    scope = {
        "observations": {
            "timestamps": None,
            "trades": [
                {"entry_idx": large, "direction": 1, "status": "open", "size": 1e-12},
                {
                    "entry_idx": large - 1,
                    "direction": -1,
                    "status": "closed",
                    "size": None,
                },
            ],
        }
    }
    spec = trade_table(scope)
    assert spec["full_precision_values"]["entry_idx"] == [str(large), str(large - 1)]
    assert (
        next(c for c in spec["columns"] if c["field"] == "entry_idx")["sortable"]
        is False
    )
    assert spec["full_precision_values"]["size"] == ["1e-12", None]
    assert spec["rows"][0]["entry_idx"] == large
    assert f"<td>{large}</td>" in render_table(spec, identifier="t", key="0:0:trades")


def test_empty_precision_projection_keeps_columns_and_no_fabricated_values() -> None:
    spec = ledger_table(LedgerProjection([], "available", "available", ()))
    assert spec["full_precision_values"]["filled_size"] == []
    html = render_table(spec, identifier="ledger", key="0:0:ledger")
    assert "共 0 筆" in html
    assert "<tbody></tbody>" in html


def test_number_column_also_preserves_unsafe_integer_units_in_default_display() -> None:
    large = 9007199254740993
    scope = {
        "observations": {
            "timestamps": None,
            "trades": [
                {"size": large, "status": "open"},
                {"size": large - 1, "status": "open"},
                {"size": 1.234, "status": "open"},
            ],
        }
    }
    spec = trade_table(scope)
    assert spec["full_precision_values"]["size"] == [
        str(large),
        str(large - 1),
        "1.234",
    ]
    assert next(c for c in spec["columns"] if c["field"] == "size")["sortable"] is False
    html = render_table(spec, identifier="t", key="0:0:trades")
    assert f"title='{large}'>{large}</td>" in html
    assert f"title='{large - 1}'>{large - 1}</td>" in html
    assert "title='1.234'>1.23</td>" in html


def comparison_spec(values: list[Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from tradingdev.adapters.reporting.tables import comparison_table

    payload = {
        "runs": [
            {
                "identity": {"run_id": "run", "strategy_id": "strategy"},
                "selected_train_scope": None,
                "default_scope": "0",
                "scopes": [
                    {
                        "scope_id": str(i),
                        "observations": None,
                        "metadata": {},
                        "kind": "backtest",
                        "split": None,
                        "parameters": {"choice": value},
                        "values": {},
                        "definitions": {},
                    }
                    for i, value in enumerate(values)
                ],
            }
        ]
    }
    return comparison_table(payload, navigate=True), payload


def test_nullable_comparison_parameter_can_switch_numeric_precision() -> None:
    spec, original = comparison_spec([0.10173757490229546, None])
    item = next(c for c in spec["columns"] if c["field"] == "parameter:choice")
    assert item["kind"] == "number"
    assert spec["full_precision_values"]["parameter:choice"] == [
        "0.10173757490229546",
        None,
    ]
    html = render_table(spec, identifier="comparison", key="comparison")
    assert "title='0.10173757490229546'>0.10</td>" in html
    assert original["runs"][0]["scopes"][1]["parameters"]["choice"] is None


@pytest.mark.parametrize(
    "values,expected",
    [
        ([0.10173757490229546, "automatic"], ["0.10173757490229546", "automatic"]),
        ([9007199254740993, "automatic"], ["9007199254740993", "automatic"]),
        ([0.10173757490229546, False], ["0.10173757490229546", "false"]),
    ],
)
def test_mixed_parameter_columns_preserve_exact_backend_text(
    values: list[Any],
    expected: list[str],
) -> None:
    spec, original = comparison_spec(values)
    item = next(c for c in spec["columns"] if c["field"] == "parameter:choice")
    assert item["kind"] == "text"
    assert [r["parameter:choice"] for r in spec["rows"]] == expected
    html = render_table(spec, identifier="comparison", key="comparison")
    assert all(f"<td>{value}</td>" in html for value in expected)
    assert [s["parameters"]["choice"] for s in original["runs"][0]["scopes"]] == values
