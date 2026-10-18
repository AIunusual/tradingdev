"""Readable exports share table order without losing precision or executing formulas."""

from __future__ import annotations

import codecs
import csv
import io
import json
from typing import Any

import pytest

from tradingdev.adapters.reporting.csv_export import table_csv
from tradingdev.adapters.reporting.tables import column, ledger_table, trade_table
from tradingdev.domain.presentation.ledger import LedgerProjection


def read_csv(content: bytes) -> list[list[str]]:
    assert content.startswith(codecs.BOM_UTF8)
    return list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))


def test_csv_uses_chosen_order_not_dictionary_or_alphabetical_order() -> None:
    spec = {
        "columns": [
            column("timestamp", "時間"),
            column("sequence", "序號", "integer"),
            column("amount", "金額", "number"),
            column("rate", "費率", "percent"),
        ],
        "rows": [
            {
                "amount": 12.3456789012345,
                "sequence": 1,
                "rate": 0.0004,
                "timestamp": "2024-01-01T00:00:00+00:00",
            }
        ],
    }
    rows = read_csv(table_csv(spec))
    assert rows[0] == ["時間", "序號", "金額", "費率（比例，1 = 100%）"]
    assert rows[1] == ["2024-01-01T00:00:00+00:00", "1", "12.3456789012345", "0.0004"]


def test_ledger_export_does_not_collapse_distinct_nanosecond_events() -> None:
    from tradingdev.domain.presentation.ledger import LEDGER_COLUMNS

    rows = []
    for sequence, tail in enumerate(("123456789", "123456790"), 1):
        row: dict[str, Any] = dict.fromkeys(c.key for c in LEDGER_COLUMNS)
        row.update(
            timestamp=f"2024-01-01T08:00:00.{tail}+08:00",
            sequence=sequence,
            event_type="bar_close",
        )
        rows.append(row)
    spec = ledger_table(LedgerProjection(rows, "available", "available", ()))
    exported = read_csv(table_csv(spec))
    assert exported[1][0] == "2024-01-01T00:00:00.123456789+00:00"
    assert exported[2][0] == "2024-01-01T00:00:00.123456790+00:00"


@pytest.mark.parametrize(
    "text",
    ["=1+1", "+1", "-formula", "@SUM(A1)", "  =1", "\tplain", "\rplain", "\nplain"],
)
def test_formula_like_text_and_headers_are_escaped(text: str) -> None:
    spec = {"columns": [column("text", text)], "rows": [{"text": text}]}
    assert read_csv(table_csv(spec)) == [["'" + text], ["'" + text]]


def test_quoted_unicode_newlines_null_and_numeric_negatives_survive() -> None:
    values: list[Any] = ['中文,"引號"\n下一行', None, -1.23456789, 0.0, False, 2]
    spec = {"columns": [column("value", "值")], "rows": [{"value": x} for x in values]}
    assert read_csv(table_csv(spec))[1:] == [
        ['中文,"引號"\n下一行'],
        [""],
        ["-1.23456789"],
        ["0.0"],
        ["否"],
        ["2"],
    ]


def test_hidden_columns_and_every_row_are_exported() -> None:
    spec = {
        "columns": [
            column("sequence", "序號", "integer"),
            column("secret", "補充資料", visible=False),
        ],
        "rows": [{"sequence": n, "secret": n / 3} for n in range(1, 101)],
    }
    result = read_csv(table_csv(spec))
    assert result[0] == ["序號", "補充資料"]
    assert len(result) == 101 and result[-1] == ["100", str(100 / 3)]


def test_empty_export_still_explains_its_schema() -> None:
    assert read_csv(
        table_csv({"columns": [column("sequence", "序號")], "rows": []})
    ) == [["序號"]]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_numbers_fail_instead_of_exporting_ambiguous_values(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        table_csv({"columns": [column("n", "值")], "rows": [{"n": value}]})


def test_unsupported_values_fail() -> None:
    with pytest.raises(TypeError, match="Unsupported"):
        table_csv({"columns": [column("n", "值")], "rows": [{"n": object()}]})


@pytest.mark.parametrize("status", ["open", "closed"])
def test_paired_trade_exports_share_static_columns_and_keep_all_source_fields(
    status: str,
) -> None:
    trade = {
        "entry_idx": 0,
        "exit_idx": 1,
        "direction": -1,
        "status": status,
        "entry_price": 101.23456789,
        "exit_price": 100.12345678,
        "net_pnl": 1.123456789,
        "extra": {"unknown.key": "=value"},
    }
    scope = {
        "observations": {
            "timestamps": ["2024-01-01T08:00:00+08:00", "2024-01-02T08:00:00+08:00"],
            "trades": [trade],
        }
    }
    spec = trade_table(scope)
    exported = read_csv(table_csv(spec))
    assert exported[0] == [item["title"] for item in spec["columns"]]
    row = dict(zip(exported[0], exported[1], strict=True))
    assert exported[0][:2] == ["序號", "進場時間（UTC）"]
    assert row["進場時間（UTC）"] == "2024-01-01T00:00:00+00:00"
    assert row["淨損益（報價貨幣）"] == "1.123456789"
    assert json.loads(row["原始資料（JSON）"]) == trade
    assert row["平倉價格（報價貨幣／標的單位）"] == (
        "100.12345678" if status == "closed" else ""
    )
    assert row["期末估值價格（報價貨幣／標的單位）"] == (
        "100.12345678" if status == "open" else ""
    )
    assert scope["observations"]["trades"] == [trade]
