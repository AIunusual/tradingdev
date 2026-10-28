"""Offline views of verified, saved research explanations and source claims."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from tradingdev.adapters.presentation.html import escape

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tradingdev.domain.approved_research import ApprovedResearch

_PROVENANCE = {
    "user": "使用者指定",
    "project_default": "專案基準設定",
    "reference": "參考來源支持的設定",
    "model_proposal": "模型提案（未經來源查證）",
}


def research_document_payload(snapshot: ApprovedResearch) -> dict[str, Any]:
    """Keep one complete explanation without repeating execution manifests."""
    snapshot.verify()
    research = snapshot.research
    record = research.record
    return {
        "content_hash": snapshot.content_hash,
        "evidence_id": record.evidence_id,
        "record_hash": record.content_hash,
        "manifest_hash": research.manifest_hash,
        "created_at": record.created_at.isoformat(),
        "draft": record.draft.model_dump(mode="json"),
        "parameters": [item.model_dump(mode="json") for item in research.parameters],
        "execution_assumptions": [
            item.model_dump(mode="json") for item in snapshot.execution_assumptions
        ],
    }


def _json(value: object) -> str:
    return escape(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    )


def _table(headers: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> str:
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{escape(header)}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table>"
    )


def _source_link(source: dict[str, Any]) -> str:
    url = source["url"]
    parsed = urlsplit(url)
    label = escape(source["title"])
    if (
        parsed.scheme in {"http", "https"}
        and parsed.hostname
        and parsed.username is None
        and parsed.password is None
    ):
        return f"<a href='{escape(url)}' rel='noreferrer noopener'>{label}</a>"
    return label + "（來源網址不可作為連結）"


def _target_label(target_id: str, candidates: dict[str, str]) -> str:
    if target_id == "base":
        return "基礎建構設定"
    if target_id in candidates:
        return candidates[target_id]
    if target_id.startswith("search_"):
        parts = target_id.split("_")
        if len(parts) == 3 and all(part.isdigit() for part in parts[1:]):
            return f"搜尋設定第 {parts[1]} 組／選項 {parts[2]}"
    return target_id


def parameter_sources_html(
    parameters: list[dict[str, Any]],
    descriptions: dict[str, Any],
    *,
    candidates: dict[str, str] | None = None,
) -> str:
    """Display effective saved values and their individual proposal provenance."""
    rows = []
    for parameter in parameters:
        path = parameter["parameter_path"]
        description = descriptions.get(path, {})
        citations = ", ".join(
            f"{claim['source_id']} / {claim['claim_id']}"
            for claim in parameter["source_claims"]
        )
        rows.append(
            (
                escape(_target_label(parameter["target_id"], candidates or {})),
                escape(description.get("label", path)),
                _json(parameter["value"]),
                escape(description.get("unit") or "未指定單位"),
                escape(_PROVENANCE[parameter["provenance"]]),
                escape(parameter["reason"]),
                escape(citations or "未引用參數值來源"),
            )
        )
    return _table(
        ("設定組", "參數", "保存值", "單位", "來源類別", "選擇理由", "引用"), rows
    )


def execution_assumptions_html(snapshot: dict[str, Any] | None) -> str:
    """Use the saved policy text, never today's execution resolver."""
    if snapshot is None:
        return (
            "<p class='notice'>此執行未保存執行規則說明，不能以目前引擎規則補寫。</p>"
        )
    rows = [
        (escape(item["label"]), escape(item["value"])) for item in snapshot["items"]
    ]
    return (
        "<h4>本次保存的執行假設</h4>"
        + _table(("項目", "保存的處理方式"), rows)
        + "<p class='muted'>執行政策版本："
        + escape(snapshot["policy_version"])
        + "</p>"
    )


def render_research_documents(payload: dict[str, Any]) -> str:
    """Render shared explanations once, including comparisons with no successes."""
    documents = payload.get("research_documents", {})
    if not documents:
        return (
            "<p class='notice'>這些執行未保存交易規則、公式與參數來源說明。"
            "本報告不從目前策略程式或設定推測、補寫當時的研究意圖。</p>"
        )
    candidates = {
        row["candidate_id"]: row["label"]
        for row in payload.get("fixed_comparison", {}).get("candidates", [])
    }
    rendered = [
        "<p>下列研究意圖、公式與參考來源由呼叫端在執行前提供；"
        "雜湊與參數綁定用於核對保存內容，不證明來源真實、瀏覽已發生，"
        "也不證明公式完整實作了 Python 或機器學習策略。"
        "可點擊來源標題閱讀原文；外部來源需要網路。</p>"
    ]
    for index, (content_hash, document) in enumerate(sorted(documents.items())):
        draft = document["draft"]
        presentation = draft["presentation"]
        pieces = [
            f"<article id='research-document-{index}'>"
            f"<h3>{escape(presentation['title'])}</h3>",
            f"<p>{escape(presentation['summary'])}</p><h4>交易規則與公式</h4>",
        ]
        for rule in draft["rules"]:
            pieces.append(
                "<div class='panel'><p><strong>條件：</strong>"
                + escape(rule["condition"])
                + "</p><p><strong>動作：</strong>"
                + escape(rule["action"])
                + "</p><pre style='white-space:pre-wrap'><code>"
                + escape(rule["formula"])
                + "</code></pre>"
            )
            if rule["variables"]:
                pieces.append(
                    _table(
                        ("變數", "定義", "單位"),
                        [
                            (
                                escape(v["symbol"]),
                                escape(v["meaning"]),
                                escape(v["unit"]),
                            )
                            for v in rule["variables"]
                        ],
                    )
                )
            if rule["source_ids"]:
                pieces.append(
                    "<p>規則來源：" + escape(", ".join(rule["source_ids"])) + "</p>"
                )
            pieces.append("</div>")
        pieces.append("<h4>完整候選設定與參數依據</h4>")
        pieces.append(
            parameter_sources_html(
                document["parameters"],
                presentation["parameter_descriptions"],
                candidates=candidates,
            )
        )
        pieces.append("<h4>參考來源與明示主張</h4>")
        if not draft["sources"]:
            pieces.append(
                "<p>沒有保存外部參考來源；模型提案不代表已查證的市場常用設定。</p>"
            )
        for source in draft["sources"]:
            retrieval = (
                "客戶端回報已透過網路讀取"
                if source["retrieval_kind"] == "host_web"
                else "使用者提供"
            )
            pieces.append(
                "<div class='panel'><h5>"
                + _source_link(source)
                + "</h5><p>"
                + escape(source["source_id"])
                + " · "
                + retrieval
                + " · "
                + escape(source["accessed_at"])
                + "</p><p>"
                + escape(source["support"])
                + "</p>"
                + _table(
                    ("主張 ID", "參數說明", "來源主張值"),
                    [
                        (
                            escape(claim["claim_id"]),
                            escape(claim["parameter"]),
                            _json(claim["value"]),
                        )
                        for claim in source["claims"]
                    ],
                )
                + "</div>"
            )
        pieces.append(
            "<h4>執行前聲明的限制</h4><ul>"
            + "".join("<li>" + escape(limit) + "</li>" for limit in draft["limits"])
            + "</ul>"
        )
        for assumptions in document["execution_assumptions"]:
            pieces.append(execution_assumptions_html(assumptions))
        pieces.append(
            "<details><summary>研究說明識別</summary><p>"
            + escape(document["evidence_id"])
            + "</p><p>"
            + escape(content_hash)
            + "</p></details></article>"
        )
        rendered.extend(pieces)
    missing = [
        run["identity"]["run_id"]
        for run in payload["runs"]
        if not run.get("research_evidence_hash")
    ]
    if missing:
        rendered.append(
            "<p class='notice'>以下執行未保存研究說明："
            + escape(", ".join(missing))
            + "。</p>"
        )
    return "".join(rendered)
