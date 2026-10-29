"""Offline presentation of saved signal evidence, separate from trading runs."""

from __future__ import annotations

import base64
from typing import Any

from tradingdev.adapters.presentation.html import details_html, escape, json_text
from tradingdev.domain.ml.report import ML_SAMPLE_PROVENANCE_NOTE

ML_REPORT_STYLE = """
#ml .ml-candidate{border-top:1px solid var(--line);padding:12px 0}
#ml .ml-candidate>summary{font-weight:600}
#ml .ml-metric{font-variant-numeric:tabular-nums;white-space:nowrap}
#ml .ml-state{color:var(--muted);white-space:normal}
#ml .ml-id{display:block;color:var(--muted);font-weight:normal}
#ml .ml-definitions td{min-width:130px}
#ml .ml-evidence{margin-top:20px}
"""

ML_REPORT_SCRIPT = """
document.querySelectorAll('[data-ml-download]').forEach(button=>{
 button.addEventListener('click',()=>downloadBlob(
  Uint8Array.from(atob(button.dataset.json),char=>char.charCodeAt(0)),
  'application/json;charset=utf-8','ml-evidence.json'));
});
"""

_METRIC_LABELS = {
    "ic_mean": "平均 IC",
    "ic_std": "IC 標準差",
    "icir": "IC 資訊比率",
    "ic_t_stat": "IC t 統計量",
    "ic_positive_rate": "IC 正值比例",
    "rank_ic_mean": "平均 Rank IC",
    "rank_ic_std": "Rank IC 標準差",
    "rank_icir": "Rank IC 資訊比率",
    "rank_ic_t_stat": "Rank IC t 統計量",
    "rank_ic_positive_rate": "Rank IC 正值比例",
    "hit_rate": "方向命中率",
    "signal_turnover": "訊號變動幅度",
    "strata_monotonicity": "分層單調性",
    "long_short_spread": "多空分層報酬差",
    "annual_return": "模擬年化報酬",
    "max_drawdown": "模擬最大回撤",
    "sharpe": "模擬 Sharpe 比率",
    "alpha": "模擬 Alpha",
    "beta": "模擬 Beta 曝險",
    "periods_per_year": "年化持有區間數",
    "n_return_periods": "模擬報酬區間數",
    "n_periods": "有效 IC 滾動視窗數",
    "n_effective_periods": "非重疊有效視窗數",
}

_CONTEXT_LABELS = {
    "model_id": "模型識別碼",
    "dataset_id": "資料集識別碼",
    "signal_fingerprint": "訊號指紋",
    "symbol": "標的",
    "timeframe": "K 棒時間頻率",
    "start_date": "開始日期",
    "end_date": "結束日期",
    "feature_set": "特徵集",
    "code_fingerprint": "特徵程式指紋",
    "data_fingerprint": "資料內容指紋",
    "n_observations": "觀測數",
    "horizon": "預測期距（horizon）",
    "window": "評估視窗（window）",
    "n_quantiles": "分位組數（n_quantiles）",
}


def _value_html(value: Any) -> str:
    if value is None:
        return "未知／未記錄（null）"
    if value == "":
        return "未記錄（空字串）"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, dict):
        return (
            "<ul>"
            + "".join(
                f"<li>{escape(key)}：{_value_html(item)}</li>"
                for key, item in value.items()
            )
            + "</ul>"
            if value
            else "無"
        )
    if isinstance(value, list | tuple):
        return "、".join(_value_html(item) for item in value) or "無"
    return escape(value)


def _pairs(values: dict[str, Any]) -> str:
    return (
        "<table><tbody>"
        + "".join(
            f"<tr><th>{escape(key)}</th><td>{_value_html(value)}</td></tr>"
            for key, value in values.items()
        )
        + "</tbody></table>"
    )


def _metric_name(metric: str) -> str:
    return (
        escape(_METRIC_LABELS.get(metric, metric)) + f" <code>({escape(metric)})</code>"
    )


def _direction(value: Any) -> str:
    return {
        "maximize": "較大優先",
        "minimize": "較小優先",
        None: "無選優方向（僅描述，不排名）",
    }.get(value, "未識別方向")


def _metric_value(scope: dict[str, Any], metric: str) -> str:
    values = scope["values"]
    if metric not in values:
        return "<span class='ml-state' data-ml-state='absent'>未記錄（absent）</span>"
    if values[metric] is None:
        return (
            "<span class='ml-state' data-ml-state='null'>"
            "不可用（null；含非有限值）</span>"
        )
    return (
        "<span class='ml-metric' data-ml-state='value'>"
        + _value_html(values[metric])
        + "</span>"
    )


def _candidate_name(scope: dict[str, Any], index: int) -> str:
    return (
        f"<a href='#ml-candidate-{index}'>"
        + escape(scope["metadata"]["label"] or scope["scope_id"])
        + "</a><code class='ml-id'>"
        + escape(scope["scope_id"])
        + "</code>"
    )


def _metrics_table(block: dict[str, Any], metrics: list[str]) -> str:
    if not metrics:
        return "<p>此群組沒有保存指標。</p>"
    definitions = block["definitions"]
    return (
        "<div class='scroll'><table><thead><tr><th>候選／Entry ID</th>"
        + "".join(
            "<th>"
            + _metric_name(metric)
            + "<br><small>"
            + escape(_direction(definitions[metric]["optimization_direction"]))
            + "</small></th>"
            for metric in metrics
        )
        + "</tr></thead><tbody>"
        + "".join(
            "<tr><th>"
            + _candidate_name(scope, index)
            + "</th>"
            + "".join(f"<td>{_metric_value(scope, metric)}</td>" for metric in metrics)
            + "</tr>"
            for index, scope in enumerate(block["scopes"])
        )
        + "</tbody></table></div>"
    )


def _ranking(block: dict[str, Any]) -> str:
    ranking = block["ranking"]
    metric = ranking["metric"]
    positions = {
        scope_id: index for index, scope_id in enumerate(ranking["ranked_scope_ids"], 1)
    }
    skipped = {item["scope_id"]: item["reason"] for item in ranking["skipped"]}
    ordered = sorted(
        enumerate(block["scopes"]),
        key=lambda item: positions.get(item[1]["scope_id"], len(positions) + 1),
    )
    rows = []
    for index, scope in ordered:
        metadata = scope["metadata"]
        scope_id = scope["scope_id"]
        rows.append(
            "<tr><td>"
            + str(positions.get(scope_id, "未排名"))
            + "</td><th>"
            + _candidate_name(scope, index)
            + "</th>"
            + "".join(
                f"<td>{_value_html(metadata.get(key))}</td>"
                for key in ("horizon", "window", "n_quantiles")
            )
            + f"<td>{_metric_value(scope, metric)}</td><td>"
            + escape(skipped.get(scope_id, "已納入排序"))
            + "</td></tr>"
        )
    return (
        "<h3>候選排名與未排名原因</h3><p>排序指標："
        + _metric_name(metric)
        + "；"
        + escape(_direction(ranking["direction"]))
        + f"。共 {len(block['scopes'])} 個候選，{len(positions)} 個有有效排序值。</p>"
        + (
            "<p class='notice'>沒有可排名的候選；以下保留全部候選及其參數座標，"
            "不可用或未記錄均不視為零分。</p>"
            if not positions
            else ""
        )
        + "<p class='muted'>排序位置依保存的排序順序；同分先後不代表顯著差異。"
        "未排名列仍保留原座標，不能將其兩側候選視為相鄰設定。</p>"
        "<div class='scroll'><table><thead><tr><th>排序位置</th>"
        "<th>候選／Entry ID</th><th>horizon</th><th>window</th><th>n_quantiles</th>"
        "<th>排序值</th><th>狀態／保存原因</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _definitions(block: dict[str, Any]) -> str:
    rows = []
    for metric, definition in block["definitions"].items():
        rows.append(
            "<tr><th>"
            + _metric_name(metric)
            + "</th><td>"
            + escape(definition["description"])
            + "</td><td>"
            + escape(_direction(definition["optimization_direction"]))
            + "</td><td>"
            + _pairs(
                {
                    "單位（原始值）": definition["unit"],
                    "類別": definition["category"],
                    "計算來源": definition["provider"],
                    "摘要指標": definition["summary"],
                    "使用模式": definition["modes"],
                    "需要年化假設": definition["requires_annualization"],
                }
            )
            + "</td></tr>"
        )
    return (
        "<details class='ml-definitions'><summary>展開全部保存指標定義與計算單位"
        "</summary><p>定義取自當次證據；fraction 顯示原始比例，"
        "例如 0.20 代表 20%，IC 則依原定義解讀為相關係數。"
        "值保留完整精度，沒有另行年化或重算。</p><div class='scroll'><table>"
        "<thead><tr><th>指標</th><th>保存定義（原文）</th><th>選優方向</th>"
        "<th>計算屬性</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></details>"
    )


def render_ml_report(block: dict[str, Any], *, has_runs: bool) -> str:
    """Render every saved candidate and metric without inventing trading evidence."""
    # Keep the export as bytes: parsing the payload in JavaScript would round
    # valid integer model parameters outside Number's safe-integer range.
    json_encoded = base64.b64encode(json_text(block).encode("utf-8")).decode("ascii")
    candidates = []
    for index, scope in enumerate(block["scopes"]):
        metadata = scope["metadata"]
        candidates.append(
            f"<details class='ml-candidate' id='ml-candidate-{index}'><summary>"
            + escape(metadata["label"] or scope["scope_id"])
            + f" · Entry ID {escape(scope['scope_id'])}</summary>"
            + _pairs(
                {label: metadata.get(key) for key, label in _CONTEXT_LABELS.items()}
            )
            + "<h4>模型擬合參數</h4>"
            + (
                _pairs(scope["parameters"])
                if scope["parameters"]
                else "<p>沒有保存擬合參數；不由模型識別碼推測。</p>"
            )
            + "<p class='muted'>模型擬合參數與上方評估座標各自保存，"
            "同名參數不會互相覆寫。</p></details>"
        )
    axes = block["parameter_axes"]
    return (
        "<p class='notice'>這是尚未計入交易成本的 ML 訊號研究："
        "指標來自預測與未來報酬，模擬多空統計沒有成交紀錄或可交付的權益曲線；"
        "不代表可實現的交易績效。</p>"
        f"<p class='notice'>{escape(ML_SAMPLE_PROVENANCE_NOTE)}</p>"
        + (
            "<p class='notice'>本節與報告中的回測執行為各自獨立的證據；"
            "並列不代表這些模型產生了任何回測執行的訊號或交易。</p>"
            if has_runs
            else ""
        )
        + "<h3>研究範圍與參數座標</h3>"
        + _pairs({_CONTEXT_LABELS.get(key, key): value for key, value in axes.items()})
        + (
            "<p class='notice'>跨 horizon 比較：來源與原始行情等價性未經驗證。"
            "不同期距改變標籤及持有區間，須核對各資料集 manifest 與原始 K 棒；"
            "此參數曲面的排名不是單一同條件測量。</p>"
            if len(axes.get("horizon", [])) > 1
            else ""
        )
        + _ranking(block)
        + "<h3>選優限制與保存警語</h3><p>從多個候選選出第一名會產生選擇偏誤；"
        "排名本身不證明樣本外有效性，單一候選也不能證明優於其他選擇。"
        "顯著性門檻及有效樣本限制依下列保存警語解讀。</p><ul>"
        + "".join(f"<li>{escape(note)}</li>" for note in block["caveats"])
        + "</ul><h3>預設比較指標</h3>"
        "<p>依保存的可見性政策顯示具選優方向的摘要指標；"
        "數值是保存原始值，比例未乘以 100。零值是有效測量，"
        "與未記錄（absent）及不可用（null）分開呈現。</p>"
        + _metrics_table(block, block["visible_metrics"])
        + "<details data-ml-hidden-metrics><summary>展開其他保存指標（含曝險與樣本數）"
        "</summary><p>無選優方向的指標僅描述狀態；"
        "Beta 表示市場曝險，不代表越大越好，也不作為排名依據。</p>"
        + _metrics_table(block, block["hidden_metrics"])
        + "</details>"
        + _definitions(block)
        + "<h3>候選身分、資料與模型參數</h3>"
        + "".join(candidates)
        + "<div class='ml-evidence'><h3>完整證據</h3>"
        + _pairs(
            {
                "來源 SHA-256": block.get("source_sha256"),
                "證據 schema 版本": block["schema_version"],
                "ML 呈現版本": block.get("template_version"),
            }
        )
        + "<div class='controls'><button type='button' data-ml-download "
        f"data-json='{json_encoded}'>"
        "下載完整 ML 證據 JSON</button></div>"
        + details_html("展開完整 ML 證據（原始欄位與完整精度）", block)
        + "</div>"
    )
