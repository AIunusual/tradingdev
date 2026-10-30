"""Deterministic, self-contained HTML/SVG rendering of saved research evidence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from tradingdev.adapters.presentation.html import (
    details_html,
    escape,
    render_html_document,
)
from tradingdev.adapters.reporting.bokeh_tables import (
    BOKEH_TABLE_SCRIPT,
    bokeh_resources_html,
)
from tradingdev.adapters.reporting.interpretation import render_interpretation
from tradingdev.adapters.reporting.labels import (
    enum_label,
    metric_label,
    reason_label,
    unit_label,
)
from tradingdev.adapters.reporting.labels import (
    scope_name as _scope_name,
)
from tradingdev.adapters.reporting.ml import (
    ML_REPORT_SCRIPT,
    ML_REPORT_STYLE,
    render_ml_report,
)
from tradingdev.adapters.reporting.research import (
    execution_assumptions_html,
    parameter_sources_html,
    render_research_documents,
)
from tradingdev.adapters.reporting.tables import render_table
from tradingdev.domain.presentation.reports import (
    REPORT_SECTIONS,
    SCOPED_REPORT_SECTIONS,
    report_navigation_instructions,
)

_STYLE = """
.metric{font-variant-numeric:tabular-nums}
svg{width:100%;height:auto;display:block}svg text{font-size:12px;fill:#52657a}
.two{display:grid;grid-template-columns:1fr 1fr;gap:20px}.scope[hidden]{display:none}
[hidden]{display:none!important}main{max-width:1600px}.bokeh-table{min-width:0}
.report-nav{display:flex;gap:18px;align-items:center;flex-wrap:wrap}
.precision-controls label{font-weight:600;cursor:pointer}
.table-navigation{padding:12px 16px;border-left:4px solid #2367cc;background:#edf4ff}
a{color:#2367cc}.bokeh-column-controls{border:1px solid #dce5ef;padding:12px}
.quantstats-frame{width:100%;height:1050px;border:1px solid #dce5ef;background:white}
@media(max-width:750px){.two{grid-template-columns:1fr}}
@media print{.scope[hidden],#result-view[hidden],#overview-view[hidden]{
display:block!important}
.bokeh-table,.report-nav,.precision-controls{display:none!important}
.static-table{display:block!important}}
"""
_SCRIPT = r"""
'use strict';
const reportData=JSON.parse(document.getElementById('report-data').textContent);
const select=document.getElementById('scope-select');
const tableHandles=new WeakMap();
function applyPrecision(container){
 const full=container.querySelector('[data-full-precision]').checked;
 container.querySelectorAll('td[data-full]').forEach(cell=>{
  if(cell.dataset.rounded===undefined){cell.dataset.rounded=cell.textContent;}
  cell.textContent=full?cell.dataset.full:cell.dataset.rounded;
 });
 tableHandles.get(container)?.setPrecision(full);
}
document.querySelectorAll('[data-full-precision]').forEach(checkbox=>{
 checkbox.addEventListener('change',()=>applyPrecision(checkbox.closest('.report-table')));
});
function materializeStatic(details){
 const source=details.querySelector('noscript[data-static-table]');
 if(!source||details.querySelector('table')){return;}
 // Server-rendered, escaped markup; no user-authored HTML is accepted here.
 const template=document.createElement('template');
 template.innerHTML=source.textContent;
 details.appendChild(template.content);
 applyPrecision(details.closest('.report-table'));
}
const staticTables=Array.from(document.querySelectorAll('.static-table'));
staticTables.forEach(details=>{
 details.open=false;
 details.addEventListener('toggle',()=>{if(details.open){materializeStatic(details);}});
});
let printOpenTables=[];
window.addEventListener('beforeprint',()=>{
 printOpenTables=staticTables.filter(details=>details.open);
 staticTables.forEach(details=>{materializeStatic(details);details.open=true;});
});
window.addEventListener('afterprint',()=>{
 staticTables.forEach(details=>{details.open=printOpenTables.includes(details);});
});
function tableSpec(key){
 if(key==='comparison'){return reportData.comparison_table;}
 const [ri,si,kind]=key.split(':');
 return reportData.runs[Number(ri)].scopes[Number(si)].tables[kind];
}
function downloadBlob(bytes,type,name){
 const url=URL.createObjectURL(new Blob([bytes],{type}));
 const link=document.createElement('a');link.href=url;link.download=name;link.click();
 setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function quantstatsDocument(key){
 const [ri,si]=key.split(':').map(Number);
 return reportData.runs[ri].scopes[si].quantstats.html;
}
function loadQuantstats(){
 document.querySelectorAll('iframe[data-quantstats]').forEach(frame=>{
  if(frame.closest('.scope').hidden||frame.dataset.loaded){return;}
  frame.srcdoc=quantstatsDocument(frame.dataset.quantstats);frame.dataset.loaded='true';
 });
}
async function loadTables(root){
 for(const target of root.querySelectorAll('[data-table-key]')){
  if(target.closest('.scope')?.hidden||target.dataset.loaded){continue;}
  target.dataset.loaded='true';
  try{
   const handle=await TradingDevTables.mount(target,tableSpec(target.dataset.tableKey));
   const container=target.closest('.report-table');
   tableHandles.set(container,handle);
   // The checkbox may have changed while the lazy Bokeh mount was pending.
   applyPrecision(container);
  }catch(error){
   target.textContent='互動表格無法載入；下方文字版與下載仍可使用。';
   const fallback=target.parentElement.querySelector('.static-table');
   materializeStatic(fallback);fallback.open=true;
  }
 }
}
function route(){
 const match=location.hash.match(/^#result-(\d+)-(\d+)$/);
 const anchor=match&&document.getElementById('result-'+match[1]+'-'+match[2]);
 const scope=anchor&&reportData.runs[Number(match[1])]?.scopes[Number(match[2])];
 const key=scope?'scope-'+match[1]+'-'+match[2]:null;
 const overview=document.getElementById('overview-view');
 const details=document.getElementById('result-view');
 overview.hidden=Boolean(scope);if(details){details.hidden=!scope;}
 document.querySelectorAll('.scope').forEach(el=>{el.hidden=el.dataset.scope!==key;});
 if(scope){
  if(select){select.value='result-'+match[1]+'-'+match[2];}
  document.getElementById('result-heading').textContent=
   '單組結果 · '+reportData.runs[Number(match[1])].identity.run_id+' / '+scope.scope_id;
  details.scrollIntoView({block:'start'});
  loadQuantstats();loadTables(details);
 }else{loadTables(overview);}
}
if(select){select.addEventListener('change',()=>{location.hash=select.value;});}
window.addEventListener('hashchange',route);
document.querySelectorAll('[data-quantstats-download]').forEach(button=>{
 button.addEventListener('click',()=>downloadBlob(
  quantstatsDocument(button.dataset.quantstatsDownload),'text/html;charset=utf-8',
  'quantstats_'+button.dataset.quantstatsDownload.replace(':','_')+'.html'));
});
document.querySelectorAll('[data-table-csv]').forEach(button=>{
 button.addEventListener('click',()=>downloadBlob(
  Uint8Array.from(atob(button.dataset.csv),char=>char.charCodeAt(0)),
  'text/csv;charset=utf-8',button.dataset.tableCsv+'.csv'));
});
document.querySelectorAll('[data-table-json]').forEach(button=>{
 button.addEventListener('click',()=>downloadBlob(
  Uint8Array.from(atob(button.dataset.json),char=>char.charCodeAt(0)),
  'application/json;charset=utf-8',button.dataset.tableJson.replaceAll(':','-')+'.json'));
});
route();
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


def _saved_metrics(scope: dict[str, Any], *, title: str = "保存績效指標") -> str:
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
        f"<h3>{escape(title)}</h3><p>{note}</p><div class='scroll'><table>"
        "<thead><tr><th>指標</th><th>值</th><th>顯示單位</th>"
        "<th>不可用原因</th><th>保存定義</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div><p class='muted'>小數與百分比顯示至小數點後兩位；"
        "原始值及保存定義保留完整精度與原文。</p>"
    )


def _metrics(scope: dict[str, Any], run_index: int, scope_index: int) -> str:
    saved = _saved_metrics(scope)
    analysis = _object(scope.get("quantstats"))
    if not analysis:
        return saved
    evidence = details_html(
        "QuantStats 分析來源與計算設定", analysis.get("metadata", {})
    )
    if not analysis.get("available"):
        return (
            "<p class='notice'>此範圍無法產生 QuantStats 完整分析："
            + escape(analysis.get("message", analysis.get("reason", "資料不足")))
            + "。以下仍顯示當次保存的績效結果。</p>"
            + evidence
            + saved
        )
    settings = _object(scope.get("metadata", {}).get("settings"))
    rates = (
        "QuantStats 原生表格的 Sharpe 與 Sortino 均採無風險利率 "
        + _format_metric(settings.get("risk_free_rate"), "fraction")
        + "；本專案保存的 Sortino 採目標報酬率 "
        + _format_metric(settings.get("required_return"), "fraction")
        + "。原生滾動 Sharpe／Sortino 圖使用零利率。"
    )
    return (
        "<h3>QuantStats 完整績效分析</h3>"
        "<p>由保存的報酬觀測值產生，包含績效詳表、月報酬、回撤期間、"
        "報酬分布與滾動分析；不重跑策略、不改寫原回測結果。</p>"
        "<p class='notice'>原生報告的勝率、連勝與盈虧比依報酬期間計算，"
        "不是逐筆交易統計。圖表與表格保留 QuantStats 原生名稱；"
        "已平倉交易統計請見下方保存指標。</p>"
        f"<p>{rates}</p>"
        f"<div class='controls'><button data-quantstats-download='{run_index}:"
        f"{scope_index}'>下載 QuantStats 完整績效 HTML</button></div>"
        "<p class='muted'>下方可捲動閱讀；完整列印請使用下載的績效 HTML。</p>"
        f"<iframe class='quantstats-frame' title='QuantStats 完整績效分析' "
        f"sandbox='' data-quantstats='{run_index}:{scope_index}'></iframe>"
        "<details><summary>QuantStats 績效詳表（文字版）</summary>"
        "<p>EOY 表的 Return 是當年度日報酬率加總；Cumulative 是當年度複利報酬率，"
        "不是跨年累積報酬率。Return、Cumulative 與 Drawdown 欄名標示百分比單位。</p>"
        "<div class='scroll'>"
        + (analysis.get("tables_html") or "")
        + "</div></details>"
        + evidence
        + "<details><summary>當次保存的績效與交易統計（含定義及不可用原因）</summary>"
        + saved
        + "</details>"
    )


_BENCHMARK_METRICS = (
    "total_return",
    "annual_return",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "annual_volatility",
    "calmar_ratio",
    "total_pnl",
)

# Full-length series are already embedded once for the chart. Repeating them in
# an inspectable <pre> would multiply the report size by the number of bars.
_BENCHMARK_SERIES_KEYS = ("timestamps", "equity_curve", "returns")


def _benchmark_evidence(benchmark: dict[str, Any]) -> dict[str, Any]:
    """Return the scalar benchmark fields; the chart carries the series."""
    return {
        key: value
        for key, value in benchmark.items()
        if key not in _BENCHMARK_SERIES_KEYS
    }


def _number(value: Any) -> float | None:
    """Return a finite scalar as a float; booleans and null are not numbers."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _benchmark_table(scope: dict[str, Any], benchmark: dict[str, Any]) -> str:
    """Put the strategy's own saved values beside the baseline, same units."""
    definitions = _object(scope.get("definitions"))
    values = _object(scope.get("values"))
    baseline = _object(benchmark.get("values"))
    comparable_amounts = benchmark.get("capital_basis") == "run_init_cash"
    comparable_period = _object(benchmark.get("coverage")).get("complete") is True
    rows = []
    for key in _BENCHMARK_METRICS:
        if key not in values and key not in baseline:
            continue
        unit = str(_object(definitions.get(key)).get("unit", ""))
        strategy = values.get(key)
        reference = baseline.get(key)
        strategy_number = _number(strategy)
        reference_number = _number(reference)
        difference = None
        comparable_unit = comparable_period and (
            unit in {"fraction", "ratio"} or (unit == "amount" and comparable_amounts)
        )
        if (
            strategy_number is not None
            and reference_number is not None
            and comparable_unit
        ):
            difference = strategy_number - reference_number
        # A normalized baseline is priced on one quote-currency unit, so its
        # amounts are not the same quantity as the strategy's realized amounts.
        show_reference = unit != "amount" or comparable_amounts
        reference_cell = _format_metric(reference, unit) if show_reference else "—"
        # "—" means the two columns are not comparable; a missing saved value
        # stays visibly unknown instead of masquerading as a withheld number.
        difference_cell = _format_metric(difference, unit) if comparable_unit else "—"
        rows.append(
            f"<tr><th>{escape(metric_label(key))}</th>"
            f"<td class='metric'>{_format_metric(strategy, unit)}</td>"
            f"<td class='metric'>{reference_cell}</td>"
            f"<td class='metric'>{difference_cell}</td>"
            f"<td>{escape(unit_label(unit))}</td></tr>"
        )
    if not rows:
        return "<p>策略與基準都沒有可比較的報酬或風險指標。</p>"
    note = (
        "差異為策略減基準；只有基準涵蓋全部策略 K 線、"
        "兩側都是有限數值且單位相同時才計算，"
        "期間或單位不同時顯示「—」，缺少數值時顯示「未知／未記錄」。"
        + (
            ""
            if comparable_amounts
            else "此結果沒有初始本金，基準以 1 個報價貨幣單位計算，"
            "因此金額類指標不與策略並列（策略金額仍為實際成交金額），"
            "也不計算差異。"
        )
    )
    return (
        "<h3>策略與基準對照</h3>"
        f"<p class='muted'>{note}</p><div class='scroll'><table>"
        "<thead><tr><th>指標</th><th>策略</th><th>基準</th>"
        "<th>差異（策略−基準）</th><th>顯示單位</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _benchmark_chart(
    scope: dict[str, Any], benchmark: dict[str, Any], label: str
) -> str:
    """Overlay both equity paths as growth multiples of the same initial capital."""
    observations = scope["observations"]
    if observations is None:
        return (
            "<p class='muted'>此結果範圍是各折摘要，沒有單一權益曲線，"
            "因此不繪製基準疊圖；請查看原始訓練／測試折。</p>"
        )
    init_cash = _number(observations.get("init_cash"))
    stamps = observations.get("timestamps")
    if init_cash is None or init_cash <= 0 or not stamps:
        return (
            "<p class='muted'>此結果沒有初始本金或時間戳，無法把策略與基準換算成"
            "相同的成長倍數；對照表仍為保存值。</p>"
        )
    baseline = dict(
        zip(
            benchmark.get("timestamps") or [],
            benchmark.get("equity_curve") or [],
            strict=True,
        )
    )
    capital = init_cash
    strategy = [
        None if value is None else value / capital
        for value in observations["equity_curve"]
    ]
    reference = [
        None if baseline.get(stamp) is None else baseline[stamp] / capital
        for stamp in stamps
    ]
    return _chart(
        strategy,
        stamps,
        "策略與基準的權益成長倍數",
        "初始資本＝1",
        "#2367cc",
        overlay=(reference, label, "#d9822b"),
    )


def _benchmark(scope: dict[str, Any]) -> str:
    """Show the fixed baseline, or exactly why this result has none."""
    benchmark = scope.get("benchmark")
    if not isinstance(benchmark, dict):
        if scope.get("kind") == "fold_summary":
            return (
                "<p class='notice'>此範圍是各折彙總，沒有單一權益曲線，"
                "依設計不提供基準；請查看各折的訓練／測試結果。</p>"
            )
        return (
            "<p class='notice'>此結果沒有保存 benchmark 證據。既有結果不會被重新計算，"
            "也不會借用其他市場的指數。一般回測重新執行後會依固定定義計算基準；"
            "參數搜尋的試驗範圍目前不產生基準。</p>"
        )
    definition = _object(benchmark.get("definition"))
    status = str(benchmark.get("status"))
    label = str(definition.get("label") or benchmark.get("asset_class") or "未定義基準")
    if status == "undefined":
        return (
            "<p class='notice'>此資產類別（"
            + escape(str(benchmark.get("asset_class") or "未分類"))
            + "）沒有定義 benchmark，因此沒有基準數值；不以其他市場的指數代替，"
            "也不把缺少基準視為零報酬。</p>"
            + details_html("基準狀態（原始欄位）", benchmark)
        )
    header = f"<h3>基準：{escape(label)}</h3>" + _pairs_html(
        {
            "基準識別碼": definition.get("id"),
            "基準標的": definition.get("symbol"),
            "資產類別": benchmark.get("asset_class"),
            "加權方式": "買進持有（單次全額買進，持有至期末）"
            if definition.get("weighting") == "buy_and_hold"
            else definition.get("weighting"),
            "基準狀態": _enum("benchmark_status", status),
            "請求資料源": benchmark.get("source"),
            "有效市場設定（未適用時為空）": benchmark.get("market_type"),
            "行情區間識別": benchmark.get("dataset_id"),
            "沿用本次回測行情": "是（基準標的即交易標的）"
            if benchmark.get("reused_run_frame")
            else "否（另行載入基準標的行情）",
            "本金基準": _enum("capital_basis", benchmark.get("capital_basis")),
        }
    )
    rationale = definition.get("rationale")
    if isinstance(rationale, str) and rationale:
        header += f"<p>{escape(rationale)}</p>"
    if benchmark.get("reused_run_frame"):
        header += (
            "<p class='notice'>此基準沿用策略的同一份行情；"
            "資料源與市場欄位描述請求設定，"
            "不代表已獨立核驗既有快取的實際來源。來源不明的行情亦會影響策略本身。</p>"
        )
    if status != "available":
        return (
            header
            + "<p class='notice'>基準無法建立："
            + escape(reason_label(str(benchmark.get("reason"))))
            + "。"
            + (escape(str(benchmark["detail"])) if benchmark.get("detail") else "")
            + " 沒有基準數值，也不以零報酬代替。</p>"
            + details_html("基準狀態（原始欄位）", benchmark)
        )
    coverage = _object(benchmark.get("coverage"))
    costs = (
        "<p>基準套用本次回測的手續費 "
        + _format_metric(benchmark.get("fees"), "fraction")
        + " 與滑價 "
        + _format_metric(benchmark.get("slippage"), "fraction")
        + "，在首根 K 線收盤一次買進；期末不強制平倉，以最後一根收盤估值，"
        "因此不計出場成本，與策略未平倉部位的估值方式一致。</p>"
    )
    missing = coverage.get("missing_bars")
    coverage_note = (
        f"<p class='notice'>基準行情缺少 {missing} 根回測 K 線（對齊 "
        f"{coverage.get('aligned_bars')}／{coverage.get('scope_bars')} 根），"
        "比較期間不完全相同，保留各自數值但不計算差異或超額報酬。</p>"
        if isinstance(missing, int) and missing > 0
        else ""
    )
    return (
        header
        + costs
        + coverage_note
        + _benchmark_table(scope, benchmark)
        + _benchmark_chart(scope, benchmark, label)
        + "<details><summary>基準全部保存指標、定義與不可用原因</summary>"
        + _saved_metrics(
            {
                "values": _object(benchmark.get("values")),
                "definitions": _object(scope.get("definitions")),
                "metadata": _object(benchmark.get("metadata")),
                "kind": "backtest",
            },
            title="基準全部保存指標",
        )
        + "</details>"
        + details_html(
            "基準來源、覆蓋與成本（原始欄位；權益序列見上圖）",
            _benchmark_evidence(benchmark),
        )
    )


def _chart(
    values: list[float | None],
    timestamps: list[str] | None,
    title: str,
    unit: str,
    color: str,
    *,
    overlay: tuple[list[float | None], str, str] | None = None,
) -> str:
    """Draw saved values, optionally beside a second index-aligned series."""
    series: list[tuple[list[float | None], str]] = [(values, color)]
    if overlay is not None:
        series.append((overlay[0], overlay[2]))
    valid = [
        float(value) for group, _ in series for value in group if value is not None
    ]
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
    paths = ""
    for group, stroke in series:
        segments: list[str] = []
        current: list[str] = []
        for index, value in enumerate(group):
            if value is None:
                if current:
                    segments.append(" ".join(current))
                    current = []
                continue
            # Both series are positioned by bar index, so a shorter baseline
            # never stretches across bars it did not observe.
            x = left + index / max(len(values) - 1, 1) * plot_width
            y = top + (high - value) / (high - low) * plot_height
            current.append(f"{x:.3f},{y:.3f}")
        if current:
            segments.append(" ".join(current))
        paths += "".join(
            f'<polyline points="{segment}" fill="none" '
            f'stroke="{stroke}" stroke-width="1.6"/>'
            for segment in segments
        )
    legend = ""
    if overlay is not None:
        legend = (
            "<p class='muted'>"
            + "".join(
                f"<span style='color:{escape(stroke)}'>■</span> {escape(name)}　"
                for name, stroke in (("策略權益", color), (overlay[1], overlay[2]))
            )
            + "</p>"
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
        + legend
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


def _trades(scope: dict[str, Any], run_index: int, scope_index: int) -> str:
    return (
        "<p>此表按進出場配對呈現每筆交易的損益，供檢查獲利來源；"
        "逐次委託與帳戶變動請見交易與帳戶核對表。"
        "未平倉交易的期末估值不是實際平倉；期末指此結果範圍的最後觀測時點，"
        "估值不代表已支付平倉費。</p>"
        + render_table(
            scope["tables"]["trades"],
            identifier=f"trades-{run_index}-{scope_index}",
            key=f"{run_index}:{scope_index}:trades",
        )
    )


def _ledger(scope: dict[str, Any], run_index: int, scope_index: int) -> str:
    spec = scope["tables"]["ledger"]
    availability = []
    for key, label in (
        ("execution_availability", "委託處理紀錄"),
        ("account_availability", "收盤帳戶紀錄"),
    ):
        value = spec[key]
        availability.append(
            f"<p data-availability='{escape(value)}'>{label}："
            + ("已保存" if value == "available" else escape(reason_label(value)))
            + "</p>"
        )
    return (
        "<p>依時間核對：市場參考價 → 委託規模與單位 → 成交數量與價格 → 費用 → "
        "現金與持倉變化。"
        "使用事件篩選查看成交、未成交或收盤估值；展開「勾選顯示欄位」調整核對內容。</p>"
        + "".join(availability)
        + "<details><summary>欄位解釋、核對算式與資料限制</summary><ul>"
        + "".join(f"<li>{escape(note)}</li>" for note in spec["notes"])
        + "<li>預期現金變動＝−帶正負號成交量 × 成交價 − 手續費；"
        "買入數量為正，賣出數量為負。預期持倉變動＝帶正負號成交量。</li>"
        "<li>預期手續費＝成交數量 × 成交價格 × 單邊費率 ＋ 固定費用；"
        "預期成交價格＝委託基準價 ×（1 ＋ 買賣方向 × 滑價率）。</li>"
        "</ul></details>"
        + render_table(
            spec,
            identifier=f"ledger-{run_index}-{scope_index}",
            key=f"{run_index}:{scope_index}:ledger",
        )
    )


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


def _scope_settings(
    run: dict[str, Any], scope: dict[str, Any], research: dict[str, Any] | None = None
) -> str:
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
        + (
            "<h3>此參數組的選擇依據</h3>"
            + parameter_sources_html(
                scope["parameter_sources"],
                research["draft"]["presentation"]["parameter_descriptions"],
            )
            if research is not None and scope.get("parameter_sources")
            else "<p class='notice'>此範圍沒有可直接對應的參數選擇說明；"
            "彙總結果應選擇原始訓練／測試範圍查看，不由模型推測來源。</p>"
        )
        + execution_assumptions_html(scope.get("execution_assumptions"))
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
    result = (
        "<p>每列是一個保存結果。先核對訓練／測試用途、日期與成本，再比較績效。</p>"
        + (
            "<p class='notice'>沒有成功候選可提供單組詳情。</p>"
            if "fixed_comparison" in payload and not payload["runs"]
            else "<p class='notice'>本報告未選取單組詳情章節。</p>"
            if not payload["comparison_table"]["navigate_field"]
            else ""
        )
        + "<p class='muted'>下列績效採當次保存值，保留其定義；"
        "不同日期或定義的數值不應直接排名。"
        "各折摘要不是合併資金曲線，詳情保留其描述統計。</p>"
        + (
            render_table(
                payload["comparison_table"], identifier="comparison", key="comparison"
            )
            if "fixed_comparison" not in payload
            else ""
        )
        + "".join(
            f"<p class='notice'>{escape(note)}</p>"
            for note in _overlap_warnings(payload)
        )
        + "<details><summary>執行與結果範圍清單</summary>"
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
    result += "</details>"
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
    result = (
        details_html("固定比較與共用行情來源證據", payload["fixed_comparison"])
        if "fixed_comparison" in payload
        else ""
    )
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
    if not payload["runs"]:
        return (
            "<p>沒有成功候選的保存結果；此章節沒有可用績效、交易或資金曲線。"
            "共用設定與每組原因見固定比較紀錄。</p>"
        )
    result = ""
    first = next(key for key in payload["sections"] if key in SCOPED_REPORT_SECTIONS)
    for ri, run in enumerate(payload["runs"]):
        for si, scope in enumerate(run["scopes"]):
            anchor = f" id='result-{ri}-{si}'" if section == first else ""
            result += (
                f"<div class='scope'{anchor} data-scope='scope-{ri}-{si}'"
                f"{' hidden' if ri or si else ''}>"
                f"<h3>{escape(run['identity']['run_id'])} · "
                f"{escape(_scope_name(scope['scope_id']))}</h3>"
            )
            if section == "settings":
                result += _scope_settings(
                    run,
                    scope,
                    payload.get("research_documents", {}).get(
                        run.get("research_evidence_hash")
                    ),
                )
            elif section == "metrics":
                result += _metrics(scope, ri, si)
            elif section == "benchmark":
                result += _benchmark(scope)
            elif scope["observations"] is None:
                result += (
                    "<p>折疊摘要沒有單一交易帳本或權益曲線；"
                    "請選擇原始訓練／測試折，不將各折串接或合併計算。</p>"
                )
            elif section == "equity":
                result += _charts(scope)
            elif section == "trades":
                result += _trades(scope, ri, si)
            elif section == "ledger":
                result += _ledger(scope, ri, si)
            result += "</div>"
    return result


def _fixed_comparison(payload: dict[str, Any]) -> str:
    """Always show the accepted candidates, including zero-success outcomes."""
    fixed = payload["fixed_comparison"]
    config = fixed["shared_config"]
    backtest = config["backtest"]
    receipt = fixed["receipt"]
    candidates = []
    for index, candidate in enumerate(fixed["candidates"], 1):
        candidates.append(
            f"<article><h3>{index}. {escape(candidate['label'])}</h3>"
            "<p style='white-space:pre-wrap'>比較理由："
            f"{escape(candidate['rationale'])}</p>"
            + _parameters_html(candidate["parameters"])
            + (
                "<p class='notice' style='white-space:pre-wrap'>無結果："
                + escape(candidate["reason"])
                + "</p>"
                if candidate["status"] != "succeeded"
                else ""
            )
            + "</article>"
        )
    return (
        "<section data-section='comparison_candidates'>"
        "<h2>固定參數比較：完整候選紀錄</h2>"
        "<p>依執行前固定的順序列出全部候選，不選出最佳組合。失敗、取消與未執行"
        "不代表零報酬；只有成功列有保存績效及可選取的明細，不合併各組資金曲線。</p>"
        f"<p>比較工作：<code>{escape(fixed['comparison_job_id'])}</code>；"
        f"成功 {fixed['counts']['succeeded']} 組，失敗 {fixed['counts']['failed']} 組，"
        f"取消 {fixed['counts']['cancelled']} 組，"
        f"未執行 {fixed['counts']['not_run']} 組。</p>"
        + render_table(
            payload["comparison_table"], identifier="comparison", key="comparison"
        )
        + "".join(candidates)
        + "<h3>共用設定與行情來源</h3>"
        + _pairs_html(
            {
                "比較工作識別碼": fixed["comparison_job_id"],
                "工作狀態": {
                    "done": "全部候選已處理（不代表全部成功）",
                    "failed": "工作失敗",
                    "cancelled": "已取消",
                }[fixed["status"]],
                "成功候選數": fixed["counts"]["succeeded"],
                "失敗候選數": fixed["counts"]["failed"],
                "取消候選數": fixed["counts"]["cancelled"],
                "未執行候選數": fixed["counts"]["not_run"],
                "比較設定雜湊": fixed["manifest"]["manifest_hash"],
                "標的": backtest["symbol"],
                "K 線週期": backtest["timeframe"],
                "請求開始日期": backtest["start_date"],
                "請求結束日期": backtest["end_date"],
                "初始本金": backtest.get("init_cash"),
                "單邊手續費率": _percentage(backtest.get("fees")),
                "滑價率": _percentage(backtest.get("slippage")),
                "共用行情快照": "已核對實際內容與 SHA-256"
                if receipt is not None
                else fixed["data_unavailable_reason"],
                "共用快照 SHA-256": fixed["snapshot_sha256"],
            }
        )
        + details_html("固定共用設定與資料來源（完整保存值）", config)
        + details_html("比較清單、狀態與行情證據（完整保存值）", fixed)
        + "</section>"
    )


def render_report(payload: dict[str, Any], report_id: str) -> str:
    """Two navigable reading levels in one portable, fully inspectable artifact."""
    has_ml = isinstance(payload.get("ml"), dict)
    show_ml = has_ml and "ml" in payload["sections"]
    has_details = bool(
        payload["runs"] and SCOPED_REPORT_SECTIONS.intersection(payload["sections"])
    )
    options = []
    links = []
    candidate_labels = {
        candidate["run_id"]: candidate["label"]
        for candidate in payload.get("fixed_comparison", {}).get("candidates", [])
    }
    for ri, run in enumerate(payload["runs"]):
        for si, scope in enumerate(run["scopes"]):
            name = candidate_labels.get(
                run["identity"]["run_id"], run["identity"]["strategy_id"]
            )
            label = f"{name} · {_scope_name(scope['scope_id'])}"
            target = f"result-{ri}-{si}"
            options.append(f"<option value='{target}'>{escape(label)}</option>")
            links.append(f"<li><a href='#{target}'>{escape(label)}</a></li>")
    overview: list[str] = []
    sections: list[str] = []
    for key in payload["sections"]:
        if key == "overview":
            content = _overview(payload)
        elif key == "ml":
            content = (
                render_ml_report(payload["ml"], has_runs=bool(payload["runs"]))
                if has_ml
                else "<p>沒有選取 ML 訊號證據；未提供不代表零值。</p>"
            )
        elif key == "research":
            content = render_research_documents(payload)
        elif key == "provenance":
            content = _provenance(payload)
        elif key == "limitations":
            content = _LIMITATIONS
        else:
            content = _scoped_section(payload, key)
        rendered = (
            f"<section data-section='{key}'{" id='ml'" if key == 'ml' else ''}>"
            f"<h2>{escape(REPORT_SECTIONS[key].title)}</h2>" + content + "</section>"
        )
        if key == "overview":
            overview.insert(0, rendered)
        elif key in {"research", "ml"}:
            overview.append(rendered)
        else:
            (sections if has_details else overview).append(rendered)
    if "fixed_comparison" in payload:
        overview.insert(0, _fixed_comparison(payload))
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
    navigation = (
        "<nav class='report-nav panel' id='scope-controls'>"
        + (
            "<a href='#overview'>← 返回參數比較總覽</a>"
            if "overview" in payload["sections"]
            else "<a href='#overview'>← 返回結果清單</a>"
        )
        + "<label for='scope-select'>切換結果</label><select id='scope-select'>"
        + "".join(options)
        + "</select></nav>"
    )
    introduction = report_navigation_instructions(
        payload["sections"],
        has_results=bool(payload["runs"]),
        comparison="fixed_comparison" in payload,
        has_ml=has_ml,
    )
    heading = (
        "ML 訊號研究與回測結果"
        if has_ml and payload["runs"]
        else "ML 訊號研究與候選比較"
        if has_ml
        else "回測結果與交易核帳"
    )
    body = (
        "<header><span class='badge'>TRADINGDEV · "
        + ("訊號研究" if has_ml else "回測研究")
        + f"</span><h1>{heading}</h1>"
        f"<p>{introduction}</p>"
        "<details><summary>報告內容與來源</summary>"
        "<p>資料取自已驗證 JSON 產物；不重跑策略、不載入 pickle，"
        "也不讀取目前策略設定。未選取章節不表示資料不存在。</p>"
        f"<p class='muted'>Report SHA-256 <code>{escape(report_id)}</code></p>"
        + _pairs_html(
            {
                "已選取章節": [
                    REPORT_SECTIONS[key].title for key in payload["sections"]
                ],
                "未選取章節": [
                    section.title
                    for key, section in REPORT_SECTIONS.items()
                    if key not in payload["sections"] and (key != "ml" or has_ml)
                ],
            }
        )
        + "".join(_identity_html(run["identity"]) for run in payload["runs"])
        + "</details></header>"
        + "<noscript><style>.scope[hidden],#result-view[hidden],#overview-view[hidden]{"
        "display:block!important}"
        "#scope-controls,.bokeh-table,.quantstats-frame,[data-quantstats-download],"
        ".controls,.precision-controls{display:none}"
        "</style><p>未啟用 JavaScript：以下展開全部結果範圍，"
        "可直接閱讀完整文字表格與績效詳表；數值亦保存在內嵌 JSON。</p></noscript>"
        + "<div id='overview-view'><span id='overview'></span>"
        + render_interpretation(payload)
        + "".join(overview)
        + (
            "<section><h2>選擇結果</h2><ul>" + "".join(links) + "</ul></section>"
            if has_details
            and "overview" not in payload["sections"]
            and "fixed_comparison" not in payload
            else ""
        )
        + commentary
        + "</div>"
        + (
            "<div id='result-view' hidden>"
            + navigation
            + "<h2 id='result-heading'>單組結果</h2>"
            + "".join(sections)
            + "</div>"
            if has_details
            else ""
        )
        + "<footer><p class='muted'>JSON manifest 保存 HTML SHA-256、章節與評語；"
        + ("來源產物不改寫。" if has_ml else "原回測產物不改寫。")
        + "原始資料與表格資料保留於內嵌 JSON。</p></footer>"
    )
    has_tables = "fixed_comparison" in payload or bool(
        set(payload["sections"]) & {"overview", "ledger", "trades"}
    )
    resources = bokeh_resources_html() if has_tables else ""
    return render_html_document(
        title="TradingDev · " + heading
        if has_ml
        else "TradingDev · 回測比較與交易核帳",
        body=body + resources,
        style=_STYLE + (ML_REPORT_STYLE if show_ml else ""),
        script=(BOKEH_TABLE_SCRIPT if has_tables else "")
        + _SCRIPT
        + (ML_REPORT_SCRIPT if show_ml else ""),
        data=payload,
        data_id="report-data",
        allow_embedded_frames=True,
    )
