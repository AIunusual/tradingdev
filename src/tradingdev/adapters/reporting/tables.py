"""One readable projection shared by interactive, static and CSV report tables."""

from __future__ import annotations

import base64
import json
import re
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pandas as pd

from tradingdev.adapters.presentation.html import escape
from tradingdev.adapters.reporting.csv_export import table_csv
from tradingdev.adapters.reporting.labels import (
    enum_label,
    metric_label,
    scope_name,
    unit_label,
)
from tradingdev.domain.presentation.ledger import LEDGER_COLUMNS, ledger_value_label

if TYPE_CHECKING:
    from tradingdev.domain.presentation.ledger import LedgerProjection


def column(
    field: str,
    title: str,
    kind: str = "text",
    *,
    group: str = "基本資料",
    visible: bool = True,
    width: int | None = None,
) -> dict[str, Any]:
    return {
        "field": field,
        "title": title,
        "kind": kind,
        "group": group,
        "visible": visible,
        **({"width": width} if width is not None else {}),
    }


def utc(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC").isoformat()


def _cell(value: Any, kind: str) -> str:
    if value is None:
        return "—"
    if isinstance(value, float | int) and not isinstance(value, bool):
        if type(value) is int and abs(value) > 2**53 - 1 and kind != "percent":
            return str(value)
        if kind == "percent":
            return f"{value:z.2%}"
        if kind == "integer":
            return str(value)
        return f"{value:z.2f}"
    if isinstance(value, dict | list):
        return escape(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return escape(value)


def _precise_cell(value: Any, kind: str) -> str | None:
    if not isinstance(value, float | int) or isinstance(value, bool):
        return None
    if kind == "percent":
        # Shift decimal places without float multiplication or Decimal rounding.
        parts = Decimal(str(value)).as_tuple()
        assert isinstance(parts.exponent, int)
        return str(Decimal((parts.sign, parts.digits, parts.exponent + 2))) + "%"
    return str(value)


def _with_precision(spec: dict[str, Any]) -> dict[str, Any]:
    """Supply source decimal text before a browser parses numbers as float64."""
    exact = {}
    for item in spec["columns"]:
        if item["kind"] not in {"number", "integer", "percent"}:
            continue
        values = [row.get(item["field"]) for row in spec["rows"]]
        exact[item["field"]] = [_precise_cell(value, item["kind"]) for value in values]
        if any(type(value) is int and abs(value) > 2**53 - 1 for value in values):
            # Distinct source integers can collapse to the same JavaScript Number.
            item["sortable"] = False
    return {**spec, "full_precision_values": exact}


def render_table(spec: dict[str, Any], *, identifier: str, key: str) -> str:
    """Keep full semantic HTML available even when JavaScript cannot execute."""
    columns, rows = spec["columns"], spec["rows"]
    headers = "".join(f"<th>{escape(c['title'])}</th>" for c in columns)
    body = []
    for row in rows:
        cells = []
        for item in columns:
            value = row.get(item["field"])
            cell = _cell(value, item["kind"])
            exact = _precise_cell(value, item["kind"])
            attributes = (
                f" data-full='{escape(exact)}' title='{escape(exact)}'"
                if exact is not None and item["kind"] in {"number", "percent"}
                else ""
            )
            if item["field"] == "sequence" and spec.get("navigate_field"):
                destination = row.get(spec["navigate_field"])
                cell = (
                    f"<a href='{escape(destination)}' "
                    f"aria-label='查看結果 {escape(value)} 的明細'>"
                    f"{cell} · 查看明細 →</a>"
                    if isinstance(destination, str)
                    and re.fullmatch(r"#result-\d+-\d+", destination)
                    else f"{cell} · 無可用明細"
                )
            cells.append(f"<td{attributes}>{cell}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    encoded = base64.b64encode(table_csv(spec)).decode("ascii")
    json_encoded = base64.b64encode(
        json.dumps(
            spec,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).decode("ascii")
    return (
        f"<div class='report-table' id='{identifier}'>"
        f"<div class='controls'><button data-table-csv='{identifier}' "
        f"data-csv='{encoded}'>下載全部紀錄 CSV（中文欄名、完整精度）</button>"
        f"<button data-table-json='{escape(key)}' data-json='{json_encoded}'>"
        "下載表格 JSON（完整精度）"
        "</button></div>"
        "<p class='precision-controls'><label><input type='checkbox' "
        "data-full-precision> 顯示完整精度</label>"
        "（同步套用互動表與文字版；不影響下載資料）</p>"
        "<p class='muted'>預設四捨五入至小數點後兩位；"
        "請勾選完整精度後再核算數量 × 價格與費用。"
        "例如顯示 0.10 的成交量不一定等於精確的 0.1。</p>"
        + (
            "<p class='table-navigation'><strong>如何查看詳細結果：</strong>"
            "點選每列的「查看明細 →」，開啟這組結果的詳細內容。"
            "互動表格也可直接點選資料列。</p>"
            if spec.get("navigate_field")
            else ""
        )
        + f"<div class='bokeh-table' data-table-key='{escape(key)}'></div>"
        "<details class='static-table' open><summary>"
        "文字版完整表格（不需 JavaScript）</summary>"
        f"<p>共 {len(rows)} 筆；全部欄位與資料均保留。</p>"
        "<noscript data-static-table>"
        f"<div class='scroll'><table id='{identifier}-static'>"
        f"<thead><tr>{headers}</tr>"
        f"</thead><tbody>{''.join(body)}</tbody></table></div></noscript></details>"
        "<p class='muted'>兩位小數模式下，較小數值可能顯示為 0.00；"
        "完整精度使用保存數值的十進位表示，可能包含浮點運算尾差；"
        "若欄位過窄，可拖寬欄位或將滑鼠停在數值上閱讀完整值。"
        "超出瀏覽器安全整數範圍的欄位保留精確文字並停用排序。"
        "空白資料以 — 表示，不代表零。CSV／JSON 保留全部欄位、全部紀錄與完整精度，"
        "不受篩選、排序或欄位勾選影響。CSV 的百分比保留原始比例，1 表示 100%；"
        "公式字元開頭的文字加單引號，避免試算表執行公式。</p></div>"
    )


def _benchmark_row(scope: dict[str, Any]) -> dict[str, Any]:
    """Summarize a saved baseline; absent evidence stays empty, not zero."""
    benchmark = scope.get("benchmark")
    if not isinstance(benchmark, dict):
        return {
            "benchmark_symbol": None,
            "benchmark_return": None,
            "excess_return": None,
            "benchmark_coverage": None,
        }
    definition = benchmark.get("definition")
    definition = definition if isinstance(definition, dict) else {}
    values = benchmark.get("values")
    values = values if isinstance(values, dict) else {}
    reference = values.get("total_return")
    reference = (
        reference
        if isinstance(reference, int | float) and not isinstance(reference, bool)
        else None
    )
    strategy = scope["values"].get("total_return")
    strategy = (
        strategy
        if isinstance(strategy, int | float) and not isinstance(strategy, bool)
        else None
    )
    coverage = benchmark.get("coverage")
    coverage = coverage if isinstance(coverage, dict) else {}
    return {
        "benchmark_symbol": definition.get("symbol"),
        "benchmark_return": reference,
        "excess_return": None
        if strategy is None or reference is None or coverage.get("complete") is not True
        else float(strategy) - float(reference),
        "benchmark_coverage": (
            f"{coverage['aligned_bars']}/{coverage['scope_bars']}"
            if "aligned_bars" in coverage and "scope_bars" in coverage
            else None
        ),
    }


def comparison_table(payload: dict[str, Any], *, navigate: bool) -> dict[str, Any]:
    """Compare every saved result without combining train/test or fold equity."""
    columns = [
        column(
            "sequence",
            "結果序號／明細" if navigate else "結果序號",
            "integer",
            width=190 if navigate else 85,
        ),
        column("period", "涵蓋日期（UTC；精確時間可勾選）", width=260),
        column("start", "開始時間（UTC）", visible=False),
        column("end", "結束時間（UTC）", visible=False),
        column("run_id", "執行識別碼", visible=False),
        column("strategy", "策略", visible=False),
        column("scope", "結果範圍", visible=False),
        column("role", "用途", width=85),
        column("selected", "選定結果", width=130),
        column("symbol", "交易標的", visible=False),
        column("timeframe", "K 線週期", visible=False),
    ]
    parameters = sorted(
        {
            key
            for run in payload["runs"]
            for scope in run["scopes"]
            for key in scope["parameters"]
        }
    )
    for key in parameters:
        values = [
            scope["parameters"][key]
            for run in payload["runs"]
            for scope in run["scopes"]
            if key in scope["parameters"] and scope["parameters"][key] is not None
        ]
        kind = (
            "integer"
            if all(type(value) is int for value in values)
            else "number"
            if all(type(value) in (int, float) for value in values)
            else "text"
        )
        columns.append(
            column(
                "parameter:" + key, f"策略參數（{key}）", kind, group="參數", width=155
            )
        )
    metrics = (
        "total_return",
        "annual_return",
        "max_drawdown",
        "sharpe_ratio",
        "total_trades",
        "win_rate",
        "total_pnl",
    )
    # Keep historical definitions separate if even one scope uses another unit.
    metric_units: dict[str, set[str]] = {}
    for run in payload["runs"]:
        for scope in run["scopes"]:
            for metric in metrics:
                if metric in scope["values"]:
                    metric_units.setdefault(metric, set()).add(
                        scope["definitions"][metric]["unit"]
                    )
    for metric, units in metric_units.items():
        for unit in sorted(units):
            kind = (
                "percent"
                if unit == "fraction"
                else "integer"
                if unit == "count"
                else "number"
            )
            columns.append(
                column(
                    f"metric:{metric}:{unit}",
                    f"{metric_label(metric)}（{unit_label(unit)}）",
                    kind,
                    group="績效",
                    visible=metric
                    in {"total_return", "max_drawdown", "sharpe_ratio", "total_trades"},
                    width=165,
                )
            )
    columns.append(column("benchmark_symbol", "基準標的", group="基準", width=110))
    columns.append(
        column("benchmark_coverage", "基準覆蓋 K 線", group="基準", width=130)
    )
    columns.append(
        column(
            "benchmark_return",
            "基準報酬（買進持有）",
            "percent",
            group="基準",
            width=165,
        )
    )
    columns.append(
        column(
            "excess_return",
            "超額報酬（策略−基準）",
            "percent",
            group="基準",
            width=165,
        )
    )
    rows: list[dict[str, Any]] = []
    text_parameters = {
        item["field"].removeprefix("parameter:")
        for item in columns
        if item["field"].startswith("parameter:") and item["kind"] == "text"
    }
    for ri, run in enumerate(payload["runs"]):
        for si, scope in enumerate(run["scopes"]):
            obs = scope["observations"]
            timestamps = obs["timestamps"] if obs else None
            saved_context = scope["metadata"].get("execution_context")
            context = saved_context if isinstance(saved_context, dict) else {}
            row = {
                "sequence": len(rows) + 1,
                "start": utc(timestamps[0]) if timestamps else None,
                "end": utc(timestamps[-1]) if timestamps else None,
                "period": (
                    f"{str(utc(timestamps[0]))[:10]} ～ {str(utc(timestamps[-1]))[:10]}"
                )
                if timestamps
                else None,
                "run_id": run["identity"]["run_id"],
                "strategy": run["identity"]["strategy_id"],
                "scope": scope_name(scope["scope_id"]),
                "role": "各折摘要"
                if scope["kind"] == "fold_summary"
                else enum_label("split", scope["split"] or "full"),
                "selected": "選定訓練結果"
                if scope["scope_id"] == run["selected_train_scope"]
                else "預設結果"
                if scope["scope_id"] == run["default_scope"]
                else "",
                "symbol": context.get("symbol"),
                "timeframe": context.get("timeframe"),
                "detail": f"#result-{ri}-{si}" if navigate else None,
            }
            row.update(
                {
                    "parameter:" + key: json.dumps(value, ensure_ascii=False)
                    if key in text_parameters
                    and value is not None
                    and not isinstance(value, str)
                    else value
                    for key, value in scope["parameters"].items()
                }
            )
            for metric, value in scope["values"].items():
                if metric in metric_units:
                    unit = scope["definitions"][metric]["unit"]
                    row[f"metric:{metric}:{unit}"] = (
                        value if not isinstance(value, dict) else None
                    )
            row.update(_benchmark_row(scope))
            rows.append(row)
    if fixed := payload.get("fixed_comparison"):
        by_run = {row["run_id"]: row for row in rows}
        rows = []
        for sequence, candidate in enumerate(fixed["candidates"], 1):
            row = dict(by_run.get(candidate["run_id"], {}))
            row.update(
                sequence=sequence,
                candidate=candidate["label"],
                status={
                    "succeeded": "成功",
                    "failed": "失敗／無結果",
                    "cancelled": "已取消／無結果",
                    "not_run": "未執行／無結果",
                }[candidate["status"]],
                rationale=candidate["rationale"],
                reason=candidate["reason"],
                effective_parameters=json.dumps(
                    candidate["parameters"], ensure_ascii=False, sort_keys=True
                ),
                run_id=candidate["run_id"],
            )
            rows.append(row)
        metric_columns = [
            item for item in columns if item["field"].startswith("metric:")
        ]
        other_columns = [item for item in columns[1:] if item not in metric_columns]
        for item in other_columns:
            item["visible"] = False
        for item in metric_columns:
            item["width"] = 145
            if item["field"].startswith("metric:total_trades:"):
                item["visible"] = False
        columns = [
            columns[0],
            column("candidate", "固定候選", width=145),
            column("status", "執行狀態", width=145),
            column("effective_parameters", "完整有效參數", width=190),
            *metric_columns,
            column("reason", "無結果原因", width=230),
            column("rationale", "執行前比較理由", width=300, visible=False),
            *other_columns,
        ]
    return _with_precision(
        {
            "columns": columns,
            "rows": rows,
            "navigate_field": "detail" if navigate else None,
            **(
                {
                    "height": max(160, 50 + 32 * len(rows)),
                    "search_placeholder": "固定候選、狀態、參數或原因…",
                }
                if fixed
                else {}
            ),
        }
    )


def trade_table(scope: dict[str, Any]) -> dict[str, Any]:
    """Paired P&L is supplementary to, and distinct from, the execution ledger."""
    columns = [
        column("sequence", "序號", "integer"),
        column("entry_time", "進場時間（UTC）"),
        column("exit_time", "平倉時間（UTC）"),
        column("mark_time", "期末估值時間（UTC）"),
        column("direction", "交易方向"),
        column("status", "持倉狀態"),
        column("entry_price", "進場價格（報價貨幣／標的單位）", "number"),
        column("exit_price", "平倉價格（報價貨幣／標的單位）", "number"),
        column("mark_price", "期末估值價格（報價貨幣／標的單位）", "number"),
        column("size", "數量（標的單位）", "number"),
        column("entry_fees", "進場手續費（報價貨幣）", "number", visible=False),
        column("exit_fees", "平倉手續費（報價貨幣）", "number", visible=False),
        column("fee", "手續費合計（報價貨幣）", "number"),
        column("gross_pnl", "毛損益（報價貨幣）", "number", visible=False),
        column("net_pnl", "淨損益（報價貨幣）", "number"),
        column("entry_idx", "進場 K 線索引（從 0 起算）", "integer", visible=False),
        column(
            "exit_idx", "平倉／期末估值 K 線索引（從 0 起算）", "integer", visible=False
        ),
        column("raw", "原始資料（JSON）", visible=False),
    ]
    observations = scope["observations"]
    timestamps = observations["timestamps"]
    rows = []
    for index, trade in enumerate(observations["trades"]):
        status = trade.get("status")

        def stamp(key: str, record: dict[str, Any] = trade) -> str | None:
            bar = record.get(key)
            return utc(timestamps[bar]) if timestamps and isinstance(bar, int) else None

        rows.append(
            {
                **trade,
                "sequence": index + 1,
                "entry_time": stamp("entry_idx"),
                "exit_time": stamp("exit_idx") if status == "closed" else None,
                "mark_time": stamp("exit_idx") if status == "open" else None,
                "exit_price": trade.get("exit_price") if status == "closed" else None,
                "mark_price": trade.get("exit_price") if status == "open" else None,
                "direction": "做多"
                if trade.get("direction") == 1
                else "做空"
                if trade.get("direction") == -1
                else "未知"
                if trade.get("direction") is None
                else f"未識別方向：{trade['direction']}",
                "status": {"open": "未平倉", "closed": "已平倉"}.get(
                    status, "未知" if status is None else f"未識別狀態：{status}"
                ),
                "filter_status": status,
                "raw": json.dumps(
                    trade, ensure_ascii=False, allow_nan=False, sort_keys=True
                ),
            }
        )
    return _with_precision(
        {
            "columns": columns,
            "rows": rows,
            "filter_field": "filter_status",
            "event_options": [
                {"value": "closed", "label": "已平倉"},
                {"value": "open", "label": "未平倉"},
            ],
        }
    )


def ledger_table(projection: LedgerProjection) -> dict[str, Any]:
    """Translate only presentation values; retain all observed numeric precision."""
    rates = {"requested_fee_rate", "requested_slippage"}
    columns = [
        column(
            item.key,
            item.label.replace("（比例）", "") if item.key in rates else item.label,
            "percent" if item.key in rates else item.kind,
            group=item.group,
            visible=item.default_visible,
        )
        for item in LEDGER_COLUMNS
    ]
    enums = {
        "status",
        "status_info",
        "side",
        "requested_size_kind",
        "requested_size_type",
        "requested_direction",
    }
    projection_enums = {
        "event_type",
        "comparison_basis",
        "requested_size_unit",
        "reconciliation",
    }
    rows = []
    for record in projection.rows:
        row = dict(record)
        for key in enums:
            if isinstance(row[key], str):
                row[key] = enum_label(key, str(row[key]))
        for key in projection_enums:
            if row[key] is not None:
                row[key] = ledger_value_label(key, row[key])
        stamp = record["timestamp"]
        row["timestamp"] = utc(stamp) if isinstance(stamp, str) else None
        row["filter_status"] = (
            "bar_close" if record["event_type"] == "bar_close" else record["status"]
        )
        rows.append(row)
    return _with_precision(
        {
            "columns": columns,
            "rows": rows,
            "filter_field": "filter_status",
            "event_options": [
                {"value": "filled", "label": "已成交"},
                {"value": "ignored", "label": "未成交：忽略"},
                {"value": "rejected", "label": "未成交：拒絕"},
                {"value": "bar_close", "label": "收盤估值"},
            ],
            "column_presets": [
                {
                    "label": "價格與費用",
                    "fields": [
                        "timestamp",
                        "sequence",
                        "status",
                        "side",
                        "market.open",
                        "market.close",
                        "requested_size",
                        "requested_size_kind",
                        "requested_size_unit",
                        "requested_size_type",
                        "requested_price",
                        "filled_size",
                        "filled_price",
                        "notional",
                        "requested_fee_rate",
                        "requested_fixed_fees",
                        "requested_slippage",
                        "fees",
                        "expected_fees",
                        "fees_residual",
                        "fill_price_residual",
                    ],
                },
                {
                    "label": "現金與持倉",
                    "fields": [
                        "timestamp",
                        "sequence",
                        "event_type",
                        "status",
                        "side",
                        "filled_size",
                        "filled_price",
                        "notional",
                        "fees",
                        "before.cash",
                        "delta.cash",
                        "after.cash",
                        "before.position",
                        "delta.position",
                        "after.position",
                        "after.equity",
                        "reconciliation",
                    ],
                },
                {
                    "label": "核對差額",
                    "fields": [
                        "timestamp",
                        "sequence",
                        "event_type",
                        "status",
                        "reconciliation",
                        "cash_residual",
                        "position_residual",
                        "fees_residual",
                        "fill_price_residual",
                        "equity_residual",
                        "reconciliation_note",
                    ],
                },
            ],
            "notes": list(projection.notes),
            "execution_availability": projection.execution_availability,
            "account_availability": projection.account_availability,
        }
    )
