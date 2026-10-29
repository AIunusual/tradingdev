"""Offline reading view of a bounded, client-authored research notebook."""

from typing import TYPE_CHECKING

from tradingdev.adapters.presentation.html import escape, render_html_document

if TYPE_CHECKING:
    from tradingdev.domain.research_notes import ResearchNoteRecord


def render_notebook(record: ResearchNoteRecord, links: dict[str, str]) -> str:
    draft = record.draft
    body = (
        f"<header><p>策略研究筆記 · 版本 {record.version}</p>"
        f"<h1>{escape(draft.title)}</h1><p>更新：{escape(record.updated_at)}</p>"
        "<p>這份筆記由對話端整理，保留需求、階段摘要與待辦；並非逐字對話紀錄。"
        "建議與追加需求不代表已執行或已授權實驗。</p>"
        "<nav><a href='#brief'>目前需求</a> · <a href='#next'>後續研究</a>"
        " · <a href='research.json'>完整筆記 JSON</a></nav></header>"
        f"<section id='brief'><h2>目前需求</h2><p class='prose'>"
        f"{escape(draft.current_brief)}</p></section>"
    )
    for index, entry in enumerate(draft.entries, 1):
        body += (
            f"<section id='stage-{index}'><h2>{index}. {escape(entry.title)}</h2>"
            f"<p class='prose'>{escape(entry.summary)}</p>"
        )
        for artifact_id in entry.artifact_ids:
            body += (
                f"<p><a href='{escape(links[artifact_id])}'>閱讀對應報告</a>"
                f" <small>{escape(artifact_id)}</small></p>"
            )
        body += "</section>"
    body += (
        f"<section id='next'><h2>後續研究</h2><p class='prose'>"
        f"{escape(draft.next_steps)}</p></section>"
        "<footer><p>報告為匯出時驗證的保存版本；其中的研究解讀與限制請連同閱讀。"
        "<a href='manifest.json'>封存內容與檔案校驗</a></p></footer>"
    )
    return render_html_document(
        title=draft.title, body=body, style=".prose{white-space:pre-wrap}"
    )
