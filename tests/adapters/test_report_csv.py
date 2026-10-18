"""CSV bytes preserve saved evidence while presenting readable Chinese fields."""

from __future__ import annotations

import codecs
import csv
import io
import json
import math
from typing import Any, cast

import pytest

from tradingdev.adapters.reporting.csv_export import RecordDataset, render_records_csv


def _read_csv(content: bytes) -> list[dict[str, str]]:
    assert content.startswith(codecs.BOM_UTF8)
    text = content.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text, newline="")))


def test_trade_csv_localizes_all_fields_from_both_engines_without_rounding() -> None:
    trade: dict[str, Any] = {
        "direction": -1,
        "entry_idx": 0,
        "exit_idx": 42,
        "entry_price": 123.45678901234567,
        "exit_price": 100.00000000000001,
        "size": 1e-12,
        "size_quote": 123.45678901234567e-12,
        "exit_notional": 0.0,
        "entry_fees": 1e-15,
        "exit_fees": 0.0,
        "fee": 1e-15,
        "gross_pnl": -0.000000123456789,
        "net_pnl": -0.00000012345679,
        "status": "open",
        "entry_slippage": 0.00012345678901234567,
        "exit_slippage": 0.0,
        "slippage": 0.00012345678901234567,
    }
    expected = {
        "交易方向": "做空",
        "進場 K 線索引（從 0 起算）": "0",
        "平倉／期末估值 K 線索引（從 0 起算）": "42",
        "進場價格（報價貨幣／標的單位）": str(trade["entry_price"]),
        "平倉／期末估值價格（報價貨幣／標的單位）": str(trade["exit_price"]),
        "數量（標的單位）": "1e-12",
        "進場名目金額（報價貨幣）": str(trade["size_quote"]),
        "平倉成交金額（報價貨幣）": "0.0",
        "進場手續費（報價貨幣）": "1e-15",
        "平倉手續費（報價貨幣）": "0.0",
        "手續費合計（報價貨幣）": "1e-15",
        "毛損益（報價貨幣）": str(trade["gross_pnl"]),
        "淨損益（報價貨幣）": str(trade["net_pnl"]),
        "持倉狀態": "未平倉",
        "進場滑價成本（報價貨幣）": str(trade["entry_slippage"]),
        "平倉滑價成本（報價貨幣）": "0.0",
        "滑價成本合計（報價貨幣）": str(trade["slippage"]),
    }
    content = render_records_csv([trade], dataset="trades")
    assert _read_csv(content) == [expected]
    assert content.endswith(b"\r\n") and content.count(b"\r\n") == 2


@pytest.mark.parametrize("dataset", ["execution_records", "account_history"])
def test_native_records_keep_all_rows_full_values_and_localized_units(
    dataset: RecordDataset,
) -> None:
    before = {"cash": 999.1234567890123, "position": -0.12345678901234568}
    records: list[dict[str, Any]] = [
        {
            "execution_id": 9007199254740993,
            "bar_index": 0,
            "timestamp": "2024-01-02T08:00:00.123456789+08:00",
            "status": "filled",
            "side": "buy",
            "requested_size": 0.12345678901234568,
            "requested_size_type": "percent",
            "requested_size_kind": "finite",
            "requested_direction": "both",
            "requested_fee_rate": 0.00012345678901234567,
            "requested_slippage": 1e-12,
            "market": {"open": 100.123456789, "close": 101.987654321},
            "before": before,
            "after": {"cash": 988.1234567890123, "position": 0.012345678901234568},
            "mark_price": 101.987654321,
            "status_info": None,
            "fees": 0.0,
        },
        {"status": "rejected", "status_info": "future_reason", "side": None},
        {"status": "ignored", "requested_size_kind": "positive_infinity"},
    ]
    rows = _read_csv(render_records_csv(records, dataset=dataset))
    assert len(rows) == len(records)
    assert rows[0]["執行紀錄編號"] == "9007199254740993"
    assert rows[0]["K 線時間（含保存時區）"] == "2024-01-02T08:00:00.123456789+08:00"
    assert rows[0]["市場開盤價（報價貨幣／標的單位）"] == "100.123456789"
    assert rows[0]["市場收盤價（報價貨幣／標的單位）"] == "101.987654321"
    assert rows[0]["收盤估值價格（報價貨幣／標的單位）"] == "101.987654321"
    assert rows[0]["執行前現金（報價貨幣）"] == str(before["cash"])
    assert rows[0]["執行前總持倉（標的單位）"] == str(before["position"])
    assert rows[0]["單邊手續費率（比例，1 = 100%）"] == "0.00012345678901234567"
    assert rows[0]["滑價率（比例，1 = 100%）"] == "1e-12"
    assert rows[0]["委託規模（單位依規模類型；比例以 1 = 100% 表示）"] == str(
        records[0]["requested_size"]
    )
    assert rows[0]["委託規模類型"] == "可用資源比例"
    assert rows[0]["委託規模數值類別"] == "有限數值"
    assert rows[0]["允許的持倉方向"] == "允許做多與做空"
    assert rows[0]["買賣方向"] == "買入"
    assert rows[0]["狀態原因"] == ""
    assert rows[0]["成交手續費（報價貨幣）"] == "0.0"
    assert rows[1]["執行狀態"] == "已拒絕"
    assert rows[1]["狀態原因"] == "未識別選項：future_reason"
    assert rows[1]["買賣方向"] == rows[1]["執行紀錄編號"] == ""
    assert rows[2]["執行狀態"] == "已忽略"
    assert rows[2]["委託規模數值類別"] == "正無限數值"


@pytest.mark.parametrize(
    "value",
    ["=1+1", "+SUM(1,2)", "-12", "@SUM(A1)", "\ttext", "\rtext", "\ntext", "  =1"],
)
def test_csv_protects_formula_text_without_changing_numeric_negatives(
    value: str,
) -> None:
    rows = _read_csv(
        render_records_csv(
            [{"note": value, "net_pnl": -12.345678901234567}], dataset="trades"
        )
    )
    assert rows == [
        {
            '未識別欄位（["note"]）': "'" + value,
            "淨損益（報價貨幣）": "-12.345678901234567",
        }
    ]


def test_csv_round_trips_escaping_unicode_structures_and_sparse_unknown_fields() -> (
    None
):
    unusual_key = '自訂,欄位"\r\n<script>🦊'
    unusual_text = '原始,值"\r\n</script>🦊'
    structure = {"a.b": [1, 9007199254740993, {"nested": unusual_text}]}
    trades = [
        {unusual_key: unusual_text, "extension": structure, "enabled": True},
        {"other": [], "empty": {}, "enabled": False, "missing": None},
    ]
    rows = _read_csv(render_records_csv(trades, dataset="trades"))
    key = f"未識別欄位（{json.dumps([unusual_key], ensure_ascii=False)}）"
    assert rows[0][key] == unusual_text
    assert json.loads(rows[0]['未識別欄位（["extension"]）']) == structure
    assert rows[0]['未識別欄位（["enabled"]）'] == "是"
    assert rows[1]['未識別欄位（["enabled"]）'] == "否"
    assert rows[1]['未識別欄位（["other"]）'] == "[]"
    assert rows[1]['未識別欄位（["empty"]）'] == "{}"
    assert rows[1]['未識別欄位（["extension"]）'] == ""
    assert rows[0]['未識別欄位（["missing"]）'] == ""


def test_native_csv_distinguishes_literal_dot_keys_from_nested_field_paths() -> None:
    records: list[dict[str, Any]] = [
        {
            "before": {"cash": 10.123456789, "a.b": 1},
            "before.cash": "literal dotted field",
            "before.a": {"b": 2},
            "__proto__": "retained",
            "extension": {},
            "array": ["all", "values"],
        },
        {"before": 9},
    ]
    content = render_records_csv(records, dataset="execution_records")
    parsed = list(csv.reader(io.StringIO(content.decode("utf-8-sig"), newline="")))
    assert len(parsed[0]) == len(set(parsed[0])) == 8
    assert all(len(row) == len(parsed[0]) for row in parsed)
    first, second = _read_csv(content)
    assert first["執行前現金（報價貨幣）"] == "10.123456789"
    assert first['未識別欄位（["before.cash"]）'] == "literal dotted field"
    assert first['未識別欄位（["before","a.b"]）'] == "1"
    assert first['未識別欄位（["before.a","b"]）'] == "2"
    assert first['未識別欄位（["__proto__"]）'] == "retained"
    assert first['未識別欄位（["extension"]）'] == "{}"
    assert first['未識別欄位（["array"]）'] == '["all","values"]'
    assert second['未識別欄位（["before"]）'] == "9"


def test_trade_known_and_unknown_enums_are_explicit() -> None:
    rows = _read_csv(
        render_records_csv(
            [
                {"status": "closed", "direction": 1},
                {"status": "future_status", "direction": "future_direction"},
                {"status": None, "direction": None},
            ],
            dataset="trades",
        )
    )
    assert rows == [
        {"持倉狀態": "已平倉", "交易方向": "做多"},
        {
            "持倉狀態": "未識別選項：future_status",
            "交易方向": "未識別選項：future_direction",
        },
        {"持倉狀態": "", "交易方向": ""},
    ]


@pytest.mark.parametrize("dataset", ["trades", "execution_records", "account_history"])
def test_empty_csv_is_only_a_utf8_bom(dataset: RecordDataset) -> None:
    assert render_records_csv([], dataset=dataset) == codecs.BOM_UTF8


def test_empty_trade_objects_still_export_one_row_per_saved_record() -> None:
    assert _read_csv(render_records_csv([{}, {}], dataset="trades")) == [
        {"序號": "1"},
        {"序號": "2"},
    ]


def test_unknown_dataset_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="Unknown CSV dataset"):
        render_records_csv([], dataset=cast("RecordDataset", "unknown"))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, {"nested": math.inf}])
def test_nonfinite_values_fail_instead_of_exporting_invalid_saved_evidence(
    value: Any,
) -> None:
    with pytest.raises(ValueError):
        render_records_csv([{"unknown": value}], dataset="trades")


def test_unsupported_values_fail_explicitly() -> None:
    with pytest.raises(TypeError, match="Unsupported CSV value type"):
        render_records_csv([{"unknown": object()}], dataset="trades")
