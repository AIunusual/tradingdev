"""Text and offline HTML views of one authoritative confirmation document."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import quote

from tradingdev.adapters.presentation.html import escape, render_html_document

if TYPE_CHECKING:
    from typing import Literal

    from tradingdev.domain.presentation.confirmation import (
        ConfirmationDocument,
        ConfirmationField,
    )


def _value(field: ConfirmationField) -> str:
    return field.value_text + (f" {field.unit}" if field.unit else "")


def _field_text(field: ConfirmationField) -> str:
    text = f"- {field.label}：{_value(field)}"
    if field.description:
        text += "。" + field.description
    if field.explanation_source == "model":
        text += "（名稱、說明與單位由模型提供）"
    return text


def render_confirmation_text(document: ConfirmationDocument) -> str:
    """Keep all settings and evidence in text while omitting technical identity."""
    lines = ["回測前確認", document.title, "策略說明（模型提供）：" + document.summary]
    for section in document.sections:
        lines.extend(("", section.title))
        lines.extend(section.paragraphs)
        lines.extend(_field_text(field) for field in section.fields)
    return "\n".join(lines) + "\n"


def render_confirmation_link(html_path: str) -> str:
    """Provide a directly reusable Markdown link to the exact local HTML path."""
    return f"[閱讀確認單](<{quote(html_path, safe='/')}>)"


def render_confirmation_summary(
    document: ConfirmationDocument, *, mode: Literal["signal", "volume"]
) -> str:
    """Show major settings and the saved concise backend execution rules."""
    labels = {
        "strategy": {"執行方式", "參數依據"},
        "market": {"回測商品", "回測 K 線週期", "回測開始", "回測結束", "時間邊界"},
        "execution": {
            "計算方式",
            "單次投入金額",
            "手續費率",
            "滑價率",
            "本金與成本來源",
            "年化計算的每年日數",
            "年化統計取樣",
            *({"初始資金"} if mode == "signal" else set()),
        },
        "optimization": {
            "訓練開始日",
            "訓練結束日",
            "保留測試開始日",
            "保留測試結束日",
            "參數組合總數",
        },
    }
    lines = ["回測前確認", "策略說明：" + document.summary]
    for section in document.sections:
        selected = labels.get(section.id, set())
        values = [
            f"{field.label}：{_value(field)}"
            for field in section.fields
            if field.label in selected
        ]
        if values:
            lines.append("；".join(values))
        if section.id == "execution":
            lines.extend(section.summary_paragraphs)
    lines.append("確認前僅檢查程式，未取得真實行情、訓練或執行回測。")
    lines.append(
        "請先閱讀完整 HTML 確認單中的設定與檢查限制，再於對話回覆同意、拒絕或修改要求。"
    )
    return "\n".join(lines)


def _field_html(field: ConfirmationField) -> str:
    description = escape(field.description)
    if field.explanation_source == "model":
        description += "<small>（名稱、說明與單位由模型提供）</small>"
    return (
        f"<tr><th>{escape(field.label)}</th><td>{escape(_value(field))}</td>"
        f"<td>{description}</td></tr>"
    )


def _table(fields: tuple[ConfirmationField, ...]) -> str:
    return (
        "<table><thead><tr><th>設定</th><th>內容</th><th>說明</th></tr></thead>"
        "<tbody>" + "".join(_field_html(field) for field in fields) + "</tbody></table>"
    )


def render_confirmation_html(document: ConfirmationDocument) -> str:
    """Render the same complete template; detailed execution settings stay visible."""
    body = (
        "<header><span class='badge'>回測前確認</span>"
        f"<h1>{escape(document.title)}</h1>"
        "<p class='muted'>頁面標題與下方說明由模型提供。</p>"
        f"<p>{escape(document.summary)}</p></header>"
        "<section><h2>閱讀後回到對話確認</h2>"
        "<p>請閱讀下方完整設定與檢查限制。確認後，在原對話回覆同意執行、拒絕，"
        "或說明要修改的內容。閱讀本頁不會啟動回測；修改設定後會產生新的確認單。</p></section>"
    )
    for section in document.sections:
        body += (
            f"<section><h2>{escape(section.title)}</h2>"
            + "".join(f"<p>{escape(text)}</p>" for text in section.paragraphs)
            + (_table(section.fields) if section.fields else "")
            + "</section>"
        )
    identity = document.identity
    body += (
        "<details class='panel'><summary>技術識別與檔案資訊</summary>"
        "<p>這些資訊用於確認執行版本，不需要修改檔案或設定。</p>"
        f"<p>執行內容摘要：<code>{escape(identity.manifest_hash)}</code></p>"
        + (
            f"<p>確認計畫：<code>{escape(identity.plan_id)}</code></p>"
            if identity.plan_id
            else ""
        )
        + _table(identity.details)
        + "</details>"
    )
    return render_html_document(title="TradingDev · 回測前確認", body=body)
