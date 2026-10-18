"""Localized CSV files with full numeric precision from saved observations."""

from __future__ import annotations

import csv
import io
import json
import math
from collections import Counter
from typing import Any, Literal

from tradingdev.adapters.reporting.labels import RECORD_LABELS, enum_label

type RecordDataset = Literal["trades", "execution_records", "account_history"]
type FieldPath = tuple[str, ...]

_TRADE_LABELS = {
    "direction": "交易方向",
    "entry_idx": "進場 K 線索引（從 0 起算）",
    "exit_idx": "平倉／期末估值 K 線索引（從 0 起算）",
    "entry_price": "進場價格（報價貨幣／標的單位）",
    "exit_price": "平倉／期末估值價格（報價貨幣／標的單位）",
    "size": "數量（標的單位）",
    "size_quote": "進場名目金額（報價貨幣）",
    "exit_notional": "平倉成交金額（報價貨幣）",
    "entry_fees": "進場手續費（報價貨幣）",
    "exit_fees": "平倉手續費（報價貨幣）",
    "fee": "手續費合計（報價貨幣）",
    "gross_pnl": "毛損益（報價貨幣）",
    "net_pnl": "淨損益（報價貨幣）",
    "status": "持倉狀態",
    "entry_slippage": "進場滑價成本（報價貨幣）",
    "exit_slippage": "平倉滑價成本（報價貨幣）",
    "slippage": "滑價成本合計（報價貨幣）",
}
_RECORD_CSV_LABELS = {
    key: label for key, label in RECORD_LABELS.items() if key not in {"bar UTC", "raw"}
} | {
    **{
        key: f"{RECORD_LABELS[key]}（報價貨幣／標的單位）"
        for key in (
            "market.open",
            "market.high",
            "market.low",
            "market.close",
            "requested_price",
            "filled_price",
            "valuation_price",
            "mark_price",
        )
    },
    "requested_fee_rate": "單邊手續費率（比例，1 = 100%）",
    "requested_slippage": "滑價率（比例，1 = 100%）",
    "requested_size": "委託規模（單位依規模類型；比例以 1 = 100% 表示）",
    "timestamp": "K 線時間（含保存時區）",
}
_RECORD_ENUMS = {
    "status",
    "status_info",
    "side",
    "requested_size_kind",
    "requested_size_type",
    "requested_direction",
}


def _flatten(record: dict[str, Any], prefix: FieldPath = ()) -> dict[FieldPath, Any]:
    result: dict[FieldPath, Any] = {}
    for key, value in record.items():
        path = (*prefix, key)
        if isinstance(value, dict) and value:
            result.update(_flatten(value, path))
        else:
            result[path] = value
    return result


def _path_text(path: FieldPath) -> str:
    return json.dumps(path, ensure_ascii=False, separators=(",", ":"))


def _headers(columns: list[FieldPath], dataset: RecordDataset) -> list[str]:
    labels: dict[FieldPath, str] = (
        {(key,): label for key, label in _TRADE_LABELS.items()}
        if dataset == "trades"
        else {tuple(key.split(".")): label for key, label in _RECORD_CSV_LABELS.items()}
    )
    candidates = [
        labels.get(path, f"未識別欄位（{_path_text(path)}）") for path in columns
    ]
    counts = Counter(candidates)
    used: set[str] = set()
    headers = []
    for path, label in zip(columns, candidates, strict=True):
        header = (
            f"{label}（欄位路徑：{_path_text(path)}）" if counts[label] > 1 else label
        )
        while header in used:
            header += f"（欄位路徑：{_path_text(path)}）"
        headers.append(header)
        used.add(header)
    return headers


def _display_value(path: FieldPath, value: Any, dataset: RecordDataset) -> Any:
    if value is None:
        return None
    if dataset == "trades":
        if path == ("status",) and isinstance(value, str):
            return {"closed": "已平倉", "open": "未平倉"}.get(
                value, f"未識別選項：{value}"
            )
        if path == ("direction",):
            if not isinstance(value, bool) and isinstance(value, int | float):
                if value == 1:
                    return "做多"
                if value == -1:
                    return "做空"
            return f"未識別選項：{value}"
    elif len(path) == 1:
        key = path[0]
        if key in _RECORD_ENUMS and isinstance(value, str):
            return enum_label(key, value)
    return value


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, str):
        # CSV quoting alone does not prevent spreadsheet formula interpretation.
        # Protect control/space-prefixed formulas as well as direct prefixes.
        if value.startswith(("\t", "\r", "\n")) or value.lstrip().startswith(
            ("=", "+", "-", "@")
        ):
            return "'" + value
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("CSV numeric values must be finite")
        return str(value)
    if isinstance(value, dict | list):
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
    raise TypeError(f"Unsupported CSV value type: {type(value).__name__}")


def render_records_csv(
    records: list[dict[str, Any]], *, dataset: RecordDataset
) -> bytes:
    """Encode every saved row with Chinese headers and unrounded numeric values.

    Trade extensions retain structured values as JSON cells. Native record objects
    flatten by tuple paths so literal dotted keys cannot overwrite nested fields.
    The UTF-8 BOM helps spreadsheet programs recognize Chinese text offline.
    """
    if dataset not in {"trades", "execution_records", "account_history"}:
        raise ValueError(f"Unknown CSV dataset: {dataset}")
    rows: list[dict[FieldPath, Any]] = [
        {(key,): value for key, value in record.items()}
        if dataset == "trades"
        else _flatten(record)
        for record in records
    ]
    columns = sorted({path for row in rows for path in row})
    output = io.StringIO(newline="")
    writer = csv.writer(output, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
    if columns:
        writer.writerow([_cell_text(header) for header in _headers(columns, dataset)])
        for row in rows:
            writer.writerow(
                [
                    _cell_text(_display_value(path, row.get(path), dataset))
                    for path in columns
                ]
            )
    elif rows:
        writer.writerow(["序號"])
        writer.writerows([index] for index in range(1, len(rows) + 1))
    return output.getvalue().encode("utf-8-sig")
