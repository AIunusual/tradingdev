"""Deterministic, self-contained HTML/SVG rendering of saved research evidence."""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from typing import Any, Literal

from tradingdev.adapters.presentation.html import (
    details_html,
    escape,
    render_html_document,
    value_html,
)
from tradingdev.adapters.reporting.columns import COLUMN_ORDER, RecordDataset
from tradingdev.adapters.reporting.csv_export import render_records_csv
from tradingdev.adapters.reporting.labels import (
    RECORD_LABELS,
    enum_label,
    metric_label,
    reason_label,
    unit_label,
)
from tradingdev.domain.presentation.reports import REPORT_SECTIONS

_STYLE = """
.metric{font-variant-numeric:tabular-nums}
svg{width:100%;height:auto;display:block}svg text{font-size:12px;fill:#52657a}
.two{display:grid;grid-template-columns:1fr 1fr;gap:20px}.scope[hidden]{display:none}
@media(max-width:750px){.two{grid-template-columns:1fr}}
@media print{.scope[hidden]{display:block}}
"""
_SCRIPT = """
'use strict';
const select=document.getElementById('scope-select');
if(select){select.addEventListener('change',()=>{
 document.querySelectorAll('.scope').forEach(el=>{el.hidden=el.dataset.scope!==select.value;});
});}
document.querySelectorAll('[data-filter]').forEach(input=>{
 input.addEventListener('input',()=>{
  const table=document.getElementById(input.dataset.filter);
  const query=input.value.toLowerCase();
  table.querySelectorAll('tbody tr').forEach(row=>{
   row.hidden=!row.textContent.toLowerCase().includes(query);
  });
 });
});
document.querySelectorAll('table[data-sortable] th button').forEach(button=>{
 button.addEventListener('click',()=>{
  const table=button.closest('table'),body=table.tBodies[0];
  const index=Number(button.dataset.column);
  const direction=button.dataset.direction==='asc'?-1:1;
  const rows=Array.from(body.rows);
  rows.sort((a,b)=>{
   const x=a.cells[index].dataset.sortValue??a.cells[index].textContent.trim();
   const y=b.cells[index].dataset.sortValue??b.cells[index].textContent.trim();
   const nx=Number(x),ny=Number(y);
   const diff=x!==''&&y!==''&&Number.isFinite(nx)&&Number.isFinite(ny)
    ?nx-ny:x.localeCompare(y);
   return diff*direction;
  });
  rows.forEach(row=>body.appendChild(row));
  button.dataset.direction=direction===1?'asc':'desc';
 });
});
document.querySelectorAll('[data-download]').forEach(button=>{
 button.addEventListener('click',()=>{
  const [runIndex,scopeIndex]=button.dataset.download.split(':').map(Number);
  const dataset=button.dataset.dataset;
  const bytes=Uint8Array.from(atob(button.dataset.csv),char=>char.charCodeAt(0));
  const blob=new Blob([bytes],{type:'text/csv;charset=utf-8'});
  const url=URL.createObjectURL(blob);
  const link=document.createElement('a');link.href=url;
  link.download=dataset+'_'+runIndex+'_'+scopeIndex+'.csv';link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
 });
});
"""


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _format_metric(value: Any, unit: str) -> str:
    if value is None:
        return "未知／未記錄"
    if isinstance(value, dict):
        labels = {
            "mean": "平均值",
            "std": "標準差",
            "min": "最小值",
            "max": "最大值",
            "valid_count": "有效折數",
        }
        return (
            "<ul>"
            + "".join(
                f"<li>{escape(labels.get(key, '未識別統計：' + key))}："
                + _format_metric(item, "count" if key == "valid_count" else unit)
                + "</li>"
                for key, item in value.items()
            )
            + "</ul>"
        )
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int | float):
        if unit == "fraction":
            return f"{value:z.2%}"
        if isinstance(value, int) and unit in ("", "count"):
            return str(value)
        return f"{value:z.2f}"
    if isinstance(value, list):
        return "、".join(_format_metric(item, unit) for item in value) or "無"
    return escape(value)


def _pairs_html(values: dict[str, Any]) -> str:
    """Render already-labelled report fields without changing shared raw helpers."""
    return (
        "<table><tbody>"
        + "".join(
            f"<tr><th>{escape(label)}</th><td>{_format_metric(value, '')}</td></tr>"
            for label, value in values.items()
        )
        + "</tbody></table>"
    )


def _percentage(value: Any) -> Any:
    return (
        f"{value:z.2%}"
        if isinstance(value, int | float) and not isinstance(value, bool)
        else value
    )


def _enum(field: str, value: Any) -> Any:
    return enum_label(field, value) if isinstance(value, str) else value


def _metrics(scope: dict[str, Any]) -> str:
    unavailable = _object(scope["metadata"].get("unavailable"))
    rows = []
    for key, value in scope["values"].items():
        definition = scope["definitions"][key]
        unit = definition["unit"]
        reason = (
            reason_label(str(unavailable[key]))
            if value is None and key in unavailable
            else "未記錄原因"
            if value is None
            else ""
        )
        rows.append(
            f"<tr><th>{escape(metric_label(key))}</th><td class='metric'>"
            f"{_format_metric(value, unit)}</td><td>{escape(unit_label(unit))}</td>"
            f"<td>{escape(reason)}</td><td>"
            + details_html("保存定義與指標代碼（原文）", definition)
            + "</td></tr>"
        )
    note = (
        "各折摘要呈現每個測試折的平均值、標準差、最小值、最大值與有效折數，"
        "不是串接資金、合併報酬率或可交易的單一權益曲線。"
        if scope["kind"] == "fold_summary"
        else "以下值與定義取自保存結果；比例顯示為百分比，原始值保留於報告資料。"
    )
    return (
        f"<h3>保存績效指標</h3><p>{note}</p><div class='scroll'><table>"
        "<thead><tr><th>指標</th><th>值</th><th>顯示單位</th>"
        "<th>不可用原因</th><th>保存定義</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div><p class='muted'>小數與百分比顯示至小數點後兩位；"
        "原始值及保存定義保留完整精度與原文。</p>"
    )


def _chart(
    values: list[float | None],
    timestamps: list[str] | None,
    title: str,
    unit: str,
    color: str,
) -> str:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return f"<p>{escape(title)}：沒有可繪製的有限數值。</p>"
    low, high = min(valid), max(valid)
    padding = (high - low) * 0.08 or max(abs(high) * 0.05, 1.0)
    low, high = low - padding, high + padding
    width, height, left, right, top, bottom = 1050, 285, 95, 25, 25, 42
    plot_width, plot_height = width - left - right, height - top - bottom
    lines = []
    for index in range(5):
        tick = low + (high - low) * index / 4
        y = top + plot_height * (1 - index / 4)
        lines.append(
            f'<line x1="{left}" x2="{width - right}" y1="{y}" '
            f'y2="{y}" stroke="#e3eaf2"/><text x="{left - 9}" '
            f'y="{y + 4}" text-anchor="end">{tick:z,.2f}</text>'
        )
    segments: list[str] = []
    current: list[str] = []
    for index, value in enumerate(values):
        if value is None:
            if current:
                segments.append(" ".join(current))
                current = []
            continue
        x = left + index / max(len(values) - 1, 1) * plot_width
        y = top + (high - value) / (high - low) * plot_height
        current.append(f"{x:.3f},{y:.3f}")
    if current:
        segments.append(" ".join(current))
    paths = "".join(
        f'<polyline points="{segment}" fill="none" '
        f'stroke="{color}" stroke-width="1.6"/>'
        for segment in segments
    )
    first = (
        _utc(timestamps[0]).strftime("%Y-%m-%d %H:%M UTC")
        if timestamps
        else "K 線索引 0"
    )
    last = (
        _utc(timestamps[-1]).strftime("%Y-%m-%d %H:%M UTC")
        if timestamps
        else f"K 線索引 {len(values) - 1}"
    )
    return (
        f"<h3>{escape(title)} <small>({escape(unit)})</small></h3>"
        f'<svg role="img" aria-label="{escape(title)}" '
        f'viewBox="0 0 {width} {height}"><title>{escape(title)}</title>'
        + "".join(lines)
        + paths
        + f'<text x="{left}" y="{height - 12}">{escape(first)}</text>'
        + f'<text x="{width - right}" y="{height - 12}" text-anchor="end">'
        + escape(last)
        + "</text></svg>"
    )


def _charts(scope: dict[str, Any]) -> str:
    obs = scope["observations"]
    values = obs["equity_curve"]
    drawdown: list[float | None] = []
    peak: float | None = None
    is_return = scope["mode"] == "signal" and obs["init_cash"] is not None
    for value in values:
        if value is None:
            drawdown.append(None)
            continue
        peak = value if peak is None else max(value, peak)
        drawdown.append(
            (value / peak - 1) * 100
            if is_return and peak > 0
            else None
            if is_return
            else value - peak
        )
    return (
        _chart(values, obs["timestamps"], "權益／保存序列", "報價貨幣金額", "#2367cc")
        + _chart(
            drawdown,
            obs["timestamps"],
            "觀測值回撤",
            "%" if is_return else "報價貨幣金額",
            "#ba4456",
        )
        + "<p class='muted'>保留全部觀測點、極值與缺值間隙，橫軸按保存 K 線等距排列；"
        "有時間戳時使用 UTC 日期標記。回撤圖僅由保存序列的歷史高點導出，"
        "不取代上方保存指標；沒有初始資金的名目金額回測使用金額回撤。"
        "座標軸數值顯示至小數點後兩位，圖形仍使用原始精度。</p>"
    )


_TRADE_COLUMNS = (
    ("#", "序號"),
    ("entry UTC", "進場時間（UTC）"),
    ("exit UTC", "平倉時間（UTC）"),
    ("mark UTC", "期末估值時間（UTC）"),
    ("direction", "交易方向"),
    ("status", "持倉狀態"),
    ("entry price", "進場價格"),
    ("exit price", "平倉價格"),
    ("mark price", "期末估值價格"),
    ("size", "數量（標的單位）"),
    ("fees", "手續費（報價貨幣）"),
    ("net P&L", "淨損益（報價貨幣）"),
    ("raw", "原始資料"),
)


def _trade_rows(scope: dict[str, Any]) -> list[dict[str, Any]]:
    observations = scope["observations"]
    timestamps = observations["timestamps"]
    rows = []
    for index, trade in enumerate(observations["trades"]):
        status = trade.get("status", "unknown")
        is_closed = status == "closed"

        def timestamp(key: str, record: dict[str, Any] = trade) -> Any:
            bar = record.get(key)
            return (
                _utc(timestamps[bar]).isoformat()
                if timestamps and isinstance(bar, int)
                else None
            )

        raw_direction = trade.get("direction")
        direction = (
            "做多" if raw_direction == 1 else "做空" if raw_direction == -1 else "未知"
        )
        rows.append(
            {
                "#": index + 1,
                "status": "已平倉"
                if is_closed
                else "未平倉"
                if status == "open"
                else "未知",
                "direction": direction,
                "entry UTC": timestamp("entry_idx"),
                "entry price": trade.get("entry_price"),
                "exit UTC": timestamp("exit_idx") if is_closed else None,
                "exit price": trade.get("exit_price") if is_closed else None,
                "mark UTC": timestamp("exit_idx") if status == "open" else None,
                "mark price": trade.get("exit_price") if status == "open" else None,
                "size": trade.get("size"),
                "net P&L": trade.get("net_pnl"),
                "fees": trade.get("fee"),
                "raw": trade,
            }
        )
    return rows


def _trade_cell(key: str, value: Any) -> str:
    if key == "raw":
        return "<td>" + details_html("原始欄位", value) + "</td>"
    if key != "#" and isinstance(value, int | float) and not isinstance(value, bool):
        return f"<td data-sort-value='{escape(value)}'>{value:z.2f}</td>"
    return "<td>" + value_html(value) + "</td>"


def _download_csv(
    records: list[dict[str, Any]],
    *,
    dataset: RecordDataset,
    run_index: int,
    scope_index: int,
) -> str:
    """Embed the exact export bytes; the browser never rebuilds evidence values."""
    encoded = base64.b64encode(render_records_csv(records, dataset=dataset)).decode(
        "ascii"
    )
    title = {
        "trades": "交易",
        "execution_records": "執行紀錄",
        "account_history": "帳戶狀態",
    }[dataset]
    return (
        f"<button data-download='{run_index}:{scope_index}' data-dataset='{dataset}' "
        f"data-csv='{encoded}'>下載全部{title} CSV（中文欄名）</button>"
    )


def _trades(scope: dict[str, Any], run_index: int, scope_index: int) -> str:
    rows = _trade_rows(scope)
    identifier = f"trades-{run_index}-{scope_index}"
    prefix = (
        "<h3>完整交易紀錄</h3><p>包含全部已平倉與未平倉交易。"
        "未平倉交易的期末估值不是實際平倉；估值不代表已支付平倉費。"
        "價格為每單位標的的報價貨幣金額；數量以標的單位表示，"
        "淨損益與手續費以報價貨幣表示。展開原始欄位可查看所有保存內容。</p>"
    )
    if not rows:
        return prefix + "<p>本範圍沒有保存交易紀錄。</p>"
    headers = "".join(
        f"<th><button data-column='{i}'>{escape(label)}</button></th>"
        for i, (_, label) in enumerate(_TRADE_COLUMNS)
    )
    body = "".join(
        "<tr>"
        + "".join(_trade_cell(key, row[key]) for key, _ in _TRADE_COLUMNS)
        + "</tr>"
        for row in rows
    )
    return (
        prefix + f"<p>共 {len(rows)} 筆，無分頁截斷。</p><div class='controls'>"
        f"<label>篩選 <input data-filter='{identifier}' "
        "placeholder='狀態 / 方向 / 日期 / 文字'>"
        "</label>"
        + _download_csv(
            scope["observations"]["trades"],
            dataset="trades",
            run_index=run_index,
            scope_index=scope_index,
        )
        + "</div><p class='muted'>CSV 使用中文欄名、單位及已知狀態說明，"
        "保留全部交易與完整數值精度，不受畫面篩選或排序影響。"
        "欄位依進出場 K 線索引、方向與狀態、價格與數量、成本與損益排列。"
        "未平倉交易的平倉／期末估值欄位表示估值；K 線索引從 0 起算。"
        "以公式字元開頭的文字加單引號，避免試算表執行公式。</p>"
        f"<div class='scroll'><table id='{identifier}' data-sortable>"
        f"<thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table></div>"
        "<p class='muted'>價格、數量、淨損益與手續費僅在畫面四捨五入至小數點後兩位，"
        "較小數值可能顯示為 0.00；排序、原始資料與 CSV 保留完整精度。</p>"
    )


_LEDGER_SEMANTICS = (
    "<p class='notice'>帳戶欄位採 VectorBT 一般資產帳本："
    "現金、可用現金、空單負債、總持倉與帳戶權益都是引擎保存值，"
    "不是 Binance 永續合約錢包、可用保證金或清算狀態。"
    "時間為 K 線時間，不是精確盤中成交時間；K 線索引與識別編號從 0 起算。"
    "價格以報價貨幣／標的單位表示。持倉數量為正代表多倉，為負代表空倉；"
    "持倉市值保留相同正負號。空單負債為未回補空單的成本基礎金額。</p>"
)

_RECORD_ENUMS = {
    "status",
    "status_info",
    "side",
    "requested_size_kind",
    "requested_size_type",
    "requested_direction",
}
_RECORD_RATES = {"requested_fee_rate", "requested_slippage"}


def _record_cell(key: str, value: Any, row: dict[str, Any]) -> str:
    if key == "raw":
        return "<td>" + details_html("原始欄位", value) + "</td>"
    if key in _RECORD_ENUMS:
        return "<td>" + _format_metric(_enum(key, value), "") + "</td>"
    if key in {"execution_id", "order_id", "bar_index"}:
        return "<td>" + _format_metric(value, "count") + "</td>"
    if isinstance(value, int | float) and not isinstance(value, bool):
        rate = key in _RECORD_RATES or (
            key == "requested_size"
            and row["requested_size_type"] in ("percent", "targetpercent")
        )
        return (
            f"<td data-sort-value='{escape(value)}'>"
            + _format_metric(value, "fraction" if rate else "amount")
            + "</td>"
        )
    return "<td>" + _format_metric(value, "") + "</td>"


def _saved_table(
    rows: list[dict[str, Any]],
    *,
    dataset: Literal["execution_records", "account_history"],
    run_index: int,
    scope_index: int,
) -> str:
    """Render every saved row with reusable search, sorting and dataset-bound CSV."""
    identifier = f"{dataset}-{run_index}-{scope_index}"
    columns = (*COLUMN_ORDER[dataset], "raw")
    headers = "".join(
        f"<th><button data-column='{index}'>{escape(RECORD_LABELS[key])}</button></th>"
        for index, key in enumerate(columns)
    )
    body = "".join(
        "<tr>" + "".join(_record_cell(key, row[key], row) for key in columns) + "</tr>"
        for row in rows
    )
    return (
        f"<p>已記錄 {len(rows)} 筆，全部顯示且無分頁截斷。</p>"
        "<div class='controls'><label>篩選 "
        f"<input data-filter='{identifier}' placeholder='狀態 / 日期 / 文字'></label>"
        + _download_csv(
            [row["raw"] for row in rows],
            dataset=dataset,
            run_index=run_index,
            scope_index=scope_index,
        )
        + "</div><p class='muted'>點欄名排序，水平捲動查看完整欄位。"
        "CSV 使用中文欄名、單位及已知狀態說明，保留全部紀錄與完整數值精度，"
        "不受畫面篩選或排序影響；費率保留原始比例，1 表示 100%。"
        "HTML 與 CSV 均以時間、索引或編號起頭，依相同欄位順序閱讀。"
        "以公式字元開頭的文字加單引號避免試算表執行公式。</p>"
        f"<div class='scroll'><table id='{identifier}' data-sortable>"
        f"<thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table></div>"
        "<p class='muted'>金額、數量與百分比顯示至小數點後兩位，"
        "較小數值可能顯示為 0.00；"
        "排序、原始資料與 CSV 保留完整精度。</p>"
    )


def _stream_unavailable(scope: dict[str, Any], dataset: str) -> str:
    metadata = _object(scope["metadata"].get("observations"))
    availability_key = (
        "execution_records_availability"
        if dataset == "execution_records"
        else "account_history_availability"
    )
    reason = metadata.get(availability_key)
    if scope["mode"] == "volume" or reason == "unsupported_volume_accounting":
        reason = "unsupported_volume_accounting"
        detail = "名目金額回測模式不提供這類資金帳戶／原生成交紀錄。"
    else:
        reason = "not_recorded" if reason in (None, "available") else reason
        detail = "此保存結果未記錄這個資料流（舊產物亦可能缺少欄位），不能解讀為零筆。"
    return (
        "<p class='notice' data-availability='unavailable'>不可用："
        + escape(reason_label(str(reason)))
        + "。"
        + detail
        + "報告不重跑策略，也不由交易或權益曲線回推帳戶狀態。</p>"
    )


def _executions(scope: dict[str, Any], run_index: int, scope_index: int) -> str:
    records = scope["observations"].get("execution_records")
    intro = (
        _LEDGER_SEMANTICS + "<p>每列為一次委託嘗試，包含已成交、已忽略與已拒絕。"
        "反手可能是同一筆委託同時平舊倉並開新倉；不拆成虛構成交。"
        "市場開高低收價格取自原始 K 線；委託價格、成交價格與權益估值價格分開顯示。"
        "執行前後權益使用保存的委託價格（滑價前）估值，不是當根收盤權益；"
        "成交手續費為該次實際費用。未成交的成交欄位保持未知，不當成零費用成交。"
        "委託規模需搭配規模類型與數值類別閱讀；正負無限值不代表已成交無限數量。</p>"
    )
    if records is None:
        return intro + _stream_unavailable(scope, "execution_records")
    if not records:
        return intro + "<p data-availability='available'>已記錄：0 筆委託嘗試。</p>"
    rows = []
    for record in records:
        row = {
            key: record.get(key)
            for key in (
                "execution_id",
                "order_id",
                "bar_index",
                "status",
                "status_info",
                "side",
            )
        }
        row["timestamp"] = (
            _utc(record["timestamp"]).isoformat() if record["timestamp"] else None
        )
        for key in ("open", "high", "low", "close"):
            row[f"market.{key}"] = record["market"][key]
        row.update(
            {
                key: record[key]
                for key in (
                    "requested_price",
                    "filled_price",
                    "filled_size",
                    "fees",
                    "valuation_price",
                    "requested_size",
                    "requested_size_kind",
                    "requested_size_type",
                    "requested_direction",
                    "requested_fee_rate",
                    "requested_fixed_fees",
                    "requested_slippage",
                )
            }
        )
        for key in ("cash", "position", "equity", "free_cash", "debt"):
            for phase in ("before", "after"):
                row[f"{phase}.{key}"] = record[phase][key]
        row["raw"] = record
        rows.append(row)
    return intro + _saved_table(
        rows, dataset="execution_records", run_index=run_index, scope_index=scope_index
    )


def _account_history(scope: dict[str, Any], run_index: int, scope_index: int) -> str:
    records = scope["observations"].get("account_history")
    intro = (
        _LEDGER_SEMANTICS + "<p>每列保存一根 K 線結束後的帳戶狀態，"
        "包括沒有委託的 K 線。收盤估值價格為當根收盤價；帳戶權益等於"
        "現金加持倉市值，與執行紀錄採用的委託價格估值分開。"
        "表內數值直接取自保存帳本，不由報告重建。</p>"
    )
    if records is None:
        return intro + _stream_unavailable(scope, "account_history")
    if not records:
        return (
            intro + "<p data-availability='available'>已記錄：0 根 K 線的帳戶狀態。</p>"
        )
    rows = []
    for record in records:
        row = {
            "bar_index": record["bar_index"],
            "timestamp": _utc(record["timestamp"]).isoformat()
            if record["timestamp"]
            else None,
        }
        row.update(
            {
                key: record[key]
                for key in (
                    "mark_price",
                    "cash",
                    "free_cash",
                    "position",
                    "asset_value",
                    "equity",
                )
            }
        )
        row["raw"] = record
        rows.append(row)
    return intro + _saved_table(
        rows, dataset="account_history", run_index=run_index, scope_index=scope_index
    )


def _scope_name(scope_id: str | None) -> str | None:
    if scope_id is None:
        return None
    names = {
        "full": "完整區間",
        "train": "訓練區間",
        "test": "測試區間",
        "test_summary": "各折測試摘要",
    }
    if scope_id in names:
        return names[scope_id]
    parts = scope_id.split("/")
    if (
        len(parts) == 3
        and parts[0] in ("fold", "trial")
        and parts[1].isascii()
        and parts[1].isdigit()
        and parts[2] in names
    ):
        try:
            index = str(int(parts[1]) + 1)
        except ValueError:
            pass  # Excessively long numeric identifiers remain custom scopes.
        else:
            group = "折" if parts[0] == "fold" else "次參數試驗"
            return f"第 {index} {group} · {names[parts[2]]}（{scope_id}）"
    return f"自訂結果範圍（{scope_id}）"


def _parameters_html(parameters: dict[str, Any]) -> str:
    if not parameters:
        return "<p>沒有保存策略參數。</p>"
    return (
        "<table><tbody>"
        + "".join(
            f"<tr><th>{escape('自訂參數（' + key + '）')}</th><td>"
            + (
                details_html("結構化參數原始值", value)
                if isinstance(value, dict | list)
                else _format_metric(value, "")
            )
            + "</td></tr>"
            for key, value in parameters.items()
        )
        + "</tbody></table>"
    )


def _scope_settings(run: dict[str, Any], scope: dict[str, Any]) -> str:
    context = _object(scope["metadata"].get("execution_context"))
    observations = scope["observations"]
    coverage: dict[str, Any] = {
        "結果範圍": _scope_name(scope["scope_id"]),
        "結果類型": _enum("kind", scope["kind"]),
        "回測模式": _enum("mode", scope["mode"]),
        "資料用途": _enum("split", scope["split"]),
        "折索引（從 0 起算）": scope["fold_index"],
        "參數試驗索引（從 0 起算）": scope["trial_index"],
    }
    if observations is not None:
        stamps = observations["timestamps"]
        coverage.update(
            {
                "實際 K 線數": len(observations["equity_curve"]),
                "實際開始時間（UTC）": _utc(stamps[0]).isoformat() if stamps else None,
                "實際結束時間（UTC）": _utc(stamps[-1]).isoformat() if stamps else None,
                "初始資金（報價貨幣）": observations["init_cash"],
            }
        )
    identity = run["identity"]
    controls = {
        "交易標的": context.get("symbol"),
        "K 線週期": context.get("timeframe"),
        "設定開始日期": context.get("start_date"),
        "設定結束日期": context.get("end_date"),
        "單邊手續費率": _percentage(context.get("fees")),
        "滑價率": _percentage(context.get("slippage")),
        "每次交易名目金額（報價貨幣）": context.get("position_size"),
        "設定初始資金（報價貨幣）": context.get("init_cash"),
        "隨機種子": context.get("random_seed", identity.get("random_seed")),
        "停損比例": _percentage(context.get("stop_loss")),
        "停利比例": _percentage(context.get("take_profit")),
        "訊號代表目標持倉": context.get("signal_as_position")
        if scope["mode"] == "volume"
        else "不適用（此設定僅用於名目金額回測）",
        "停損／停利後重新進場": context.get("re_entry_after_sl")
        if scope["mode"] == "volume"
        else "不適用（此設定僅用於名目金額回測）",
        "每年計算期數": context.get("periods_per_year"),
    }
    config = _object(run["execution_config"])
    data = config.get("data")
    return (
        "<div class='two'><div><h3>資料與實際覆蓋</h3>"
        + _pairs_html(coverage)
        + details_html("保存資料來源設定（原始欄位）", data)
        + "<p class='notice'>暖機／區間前資料：保存產物沒有標準化暖機證據，"
        "因此未知；不能由第一筆交易日期推定暖機長度。</p></div>"
        + "<div><h3>有效參數與執行設定</h3>"
        + _pairs_html(
            {
                "參數來源": _enum(
                    "parameter_provenance", scope["parameter_provenance"]
                ),
                "參數是否完整": scope["parameters_complete"],
            }
        )
        + _parameters_html(scope["parameters"])
        + "<p class='muted'>自訂參數保留策略原名；未保存用途說明時不推測其含義。"
        "小數與百分比顯示至小數點後兩位，完整精度見原始參數。</p>"
        + details_html("策略參數（原始名稱與完整精度）", scope["parameters"])
        + _pairs_html(controls)
        + "</div></div>"
        + details_html("全部保存範圍中繼資料（原始欄位）", scope["metadata"])
    )


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return (
        parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    )


def _overlap_warnings(payload: dict[str, Any]) -> list[str]:
    ranges = []
    for run in payload["runs"]:
        for scope in run["scopes"]:
            obs = scope["observations"]
            if obs and obs["timestamps"]:
                ranges.append(
                    (
                        run["identity"]["run_id"],
                        scope["scope_id"],
                        scope["split"],
                        _utc(obs["timestamps"][0]),
                        _utc(obs["timestamps"][-1]),
                    )
                )
    notes = set()
    for index, first in enumerate(ranges):
        for other in ranges[index + 1 :]:
            if first[0] == other[0] and {first[2], other[2]} != {"train", "test"}:
                continue
            if max(first[3], other[3]) <= min(first[4], other[4]):
                notes.add(
                    f"{first[0]} / {first[1]} 與 {other[0]} / {other[1]} "
                    "的觀測日期重疊。"
                    "這項比較本身不能視為獨立樣本外驗證；若用包含測試日或更晚日期的資料"
                    "選參數，會有選擇資訊滲入。"
                )
    return sorted(notes)


_LIMITATIONS = (
    "<ul><li>手續費、滑價、倉位與種子以各範圍保存值為準；null 不等於零，"
    "缺少設定或語意時明示未知，不推定 1x、全倉或成交時點。</li>"
    "<li>資金費率、槓桿／保證金、交易所精度與流動性限制：若未保存於"
    "執行設定，是否模擬及金額均未知。此報告不自行補入成本。</li>"
    "<li>實際觀測日期、K 線數與請求起迄分開顯示；沒有市場檔"
    "快照時，資料集識別碼只是保存識別資訊，不能證明行情完整無缺。</li>"
    "<li>未平倉交易為期末持倉估值，不能當成已平倉；報告不強制平倉或補收費用。</li>"
    "<li>滾動驗證摘要只描述各折結果的分佈；參數最佳化的選定訓練範圍"
    "用來標示選取結果，歷史最佳不代表未來績效。</li>"
    "<li>圖表回撤由保存權益繪製，績效表使用保存值及其原始定義。"
    "時間戳缺失時只顯示 K 線索引，不猜測日期。</li></ul>"
)


def _overview(payload: dict[str, Any]) -> str:
    comparison = payload["default_scope_comparison"]
    result = "".join(
        f"<p class='notice'>{escape(note)}</p>" for note in _overlap_warnings(payload)
    )
    for run in payload["runs"]:
        result += _pairs_html(
            {
                "執行識別碼": run["identity"]["run_id"],
                "預設結果範圍": _scope_name(run["default_scope"]),
                "選定的訓練範圍": _scope_name(run["selected_train_scope"]),
                "可用結果範圍": [_scope_name(s["scope_id"]) for s in run["scopes"]],
            }
        )
    if comparison is not None:
        result += (
            "<h3>保存預設範圍比較</h3><p>保留每次執行的預設結果範圍，"
            "不合併或重算績效。可比較性依指標定義與設定判斷，"
            "仍須核對日期及資料差異。</p>"
            + _pairs_html(
                {
                    "指標是否可比較": comparison["comparable"],
                    "設定與觀測差異項目數": len(comparison["context_differences"]),
                }
            )
            + details_html("完整比較與每指標相容性原因（原始資料）", comparison)
        )
    return result


_IDENTITY_LABELS = {
    "run_id": "執行識別碼",
    "job_id": "工作識別碼",
    "strategy_id": "策略識別碼",
    "revision_id": "策略修訂識別碼",
    "manifest_hash": "執行設定雜湊",
    "source_hash": "策略原始碼雜湊",
    "config_hash": "策略設定雜湊",
    "random_seed": "隨機種子",
    "dataset_id": "資料集識別碼",
    "created_at": "建立時間（保存值）",
}


def _identity_html(identity: dict[str, Any]) -> str:
    return _pairs_html(
        {_IDENTITY_LABELS[key]: value for key, value in identity.items()}
    )


def _provenance(payload: dict[str, Any]) -> str:
    result = ""
    for run in payload["runs"]:
        identity = run["identity"]
        result += (
            "<h3>"
            + escape(identity["run_id"])
            + "</h3>"
            + _identity_html(identity)
            + _pairs_html(
                {
                    "執行設定完整性": "已驗證"
                    if identity["manifest_hash"]
                    else "未知／舊版資料未保存",
                }
            )
            + details_html("來源產物 SHA-256 證據（原始欄位）", run["provenance"])
        )
    return result


def _scoped_section(payload: dict[str, Any], section: str) -> str:
    result = ""
    for ri, run in enumerate(payload["runs"]):
        for si, scope in enumerate(run["scopes"]):
            result += (
                f"<div class='scope' data-scope='scope-{ri}-{si}'"
                f"{' hidden' if ri or si else ''}>"
                f"<h3>{escape(run['identity']['run_id'])} · "
                f"{escape(_scope_name(scope['scope_id']))}</h3>"
            )
            if section == "settings":
                result += _scope_settings(run, scope)
            elif section == "metrics":
                result += _metrics(scope)
            elif scope["observations"] is None:
                result += (
                    "<p>折疊摘要沒有單一交易帳本或權益曲線；"
                    "請選擇原始訓練／測試折，不將各折串接或合併計算。</p>"
                )
            elif section == "equity":
                result += _charts(scope)
            elif section == "trades":
                result += _trades(scope, ri, si)
            elif section == "executions":
                result += _executions(scope, ri, si)
            elif section == "account_history":
                result += _account_history(scope, ri, si)
            result += "</div>"
    return result


def render_report(payload: dict[str, Any], report_id: str) -> str:
    """Render chosen built-ins and escaped commentary, with minimal fixed provenance."""
    options = []
    identities = []
    for ri, run in enumerate(payload["runs"]):
        identity = run["identity"]
        identities.append(
            {
                key: identity[key]
                for key in ("run_id", "strategy_id", "revision_id", "manifest_hash")
            }
        )
        for si, scope in enumerate(run["scopes"]):
            options.append(
                f"<option value='scope-{ri}-{si}'>"
                f"{escape(identity['run_id'])} / "
                f"{escape(_scope_name(scope['scope_id']))} "
                f"({escape(enum_label('kind', scope['kind']))})</option>"
            )
    sections = []
    for key in payload["sections"]:
        if key == "overview":
            content = _overview(payload)
        elif key == "provenance":
            content = _provenance(payload)
        elif key == "limitations":
            content = _LIMITATIONS
        else:
            content = _scoped_section(payload, key)
        sections.append(
            f"<section data-section='{key}'>"
            f"<h2>{escape(REPORT_SECTIONS[key].title)}</h2>" + content + "</section>"
        )
    commentary = ""
    if payload["commentary"]:
        commentary = (
            "<section data-section='commentary'><h2>LLM 評語（非計算結果）</h2>"
            "<p class='muted'>以下為呼叫端提供的純文字評語；後端不驗證其結論，"
            "不將評語視為回測計算證據。</p>"
            + "".join(
                "<article><h3>"
                + escape(note["title"])
                + "</h3><p style='white-space:pre-wrap'>"
                + escape(note["text"])
                + "</p></article>"
                for note in payload["commentary"]
            )
            + "</section>"
        )
    selector = ""
    if set(payload["sections"]) & {
        "settings",
        "metrics",
        "equity",
        "trades",
        "executions",
        "account_history",
    }:
        selector = (
            "<div class='panel'><label for='scope-select'>選擇執行／結果範圍</label>"
            "<select id='scope-select'>" + "".join(options) + "</select>"
            "<p class='muted'>切換選取章節的範圍；列印時展開全部範圍。</p></div>"
        )
    body = (
        "<header><span class='badge'>"
        "TRADINGDEV · SAVED EVIDENCE</span><h1>保存回測研究報告</h1>"
        "<p>資料取自已驗證 JSON 產物；不重跑策略、不載入 pickle，"
        "也不讀取目前策略設定。僅顯示呼叫端選取的章節；未選取章節不表示資料不存在。</p>"
        f"<p class='muted'>Report SHA-256 <code>{escape(report_id)}</code></p>"
        + _pairs_html(
            {
                "已選取章節": [
                    REPORT_SECTIONS[key].title for key in payload["sections"]
                ],
                "未選取章節": [
                    section.title
                    for key, section in REPORT_SECTIONS.items()
                    if key not in payload["sections"]
                ],
            }
        )
        + "<h3>來源執行</h3>"
        + "".join(_identity_html(identity) for identity in identities)
        + "</header>"
        + selector
        + "".join(sections)
        + commentary
        + "<footer><p class='muted'>獨立 reports 目錄的 JSON manifest "
        "保存 HTML SHA-256、"
        "章節與評語；原回測產物不改寫。原始資料保留於內嵌 JSON。</p></footer>"
    )
    return render_html_document(
        title="TradingDev · 保存回測報告",
        body=body,
        style=_STYLE,
        script=_SCRIPT,
        data=payload,
        data_id="report-data",
    )
