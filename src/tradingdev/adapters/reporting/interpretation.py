"""Readable client interpretation alongside independently computed diagnostics."""

from typing import Any

from tradingdev.adapters.presentation.html import details_html, escape


def render_interpretation(payload: dict[str, Any]) -> str:
    analysis = payload.get("analysis")
    body = "<section id='research-analysis'><h2>策略觀察與後續研究</h2>"
    if analysis is None:
        body += "<p>尚未提供 LLM 研究解讀；以下診斷提示不能取代策略分析。</p>"
    else:
        body += (
            "<p class='muted'>以下由 LLM 整理；"
            "觀察依據、原因假說與未執行的實驗分開呈現。</p>"
        )
        for key, title in (
            ("summary", "研究結論"),
            ("observations", "觀察與依據"),
            ("hypotheses_and_limits", "可能原因與限制"),
            ("next_steps", "下一步研究建議"),
        ):
            body += (
                f"<h3>{title}</h3><p style='white-space:pre-wrap'>"
                f"{escape(analysis[key])}</p>"
            )
        body += (
            "<p>解讀對象："
            + escape(
                "、".join(
                    f"{r['run_id']} / {r['scope']}" for r in analysis["references"]
                )
                or "無回測範圍引用：ML 研究或沒有成功回測"
            )
            + "</p>"
        )
    body += "<details><summary>診斷來源、指標依據與適用限制</summary>"
    for run in payload["runs"]:
        for scope in run["scopes"]:
            body += details_html(
                f"{run['identity']['run_id']} / {scope['scope_id']}",
                scope["diagnostics"],
            )
    return body + "</details></section>"
