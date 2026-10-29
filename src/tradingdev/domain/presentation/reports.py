"""Canonical report sections and suggested recipes for saved research evidence."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

REPORT_SCHEMA_VERSION = 2
REPORT_TEMPLATE_VERSION = "22"
SCOPED_REPORT_SECTIONS = frozenset(
    {"settings", "metrics", "equity", "ledger", "trades"}
)


@dataclass(frozen=True)
class ReportSectionDefinition:
    """A selectable section with the same human-facing title in every adapter."""

    id: str
    title: str
    description: str


REPORT_SECTIONS = MappingProxyType(
    {
        section.id: section
        for section in (
            ReportSectionDefinition(
                "overview",
                "參數與結果比較總覽",
                "All run/scope parameters, dates, roles and saved performance, "
                "linked to individual result details when detail sections "
                "are selected.",
            ),
            ReportSectionDefinition(
                "research",
                "交易規則、公式與參數依據",
                "Saved pre-execution rules, formulas, source claims, parameter "
                "provenance and execution assumptions; absence stays explicit.",
            ),
            ReportSectionDefinition(
                "settings",
                "設定與資料覆蓋",
                "Recorded data, effective parameters and execution assumptions.",
            ),
            ReportSectionDefinition(
                "metrics",
                "績效分析與保存指標",
                "QuantStats HTML analysis derived from saved returns, plus original "
                "values, definitions and explicit unavailable reasons.",
            ),
            ReportSectionDefinition(
                "equity",
                "權益與回撤",
                "All saved equity points and a derived drawdown visualization.",
            ),
            ReportSectionDefinition(
                "ledger",
                "委託、成交與帳戶核對",
                "One chronological ledger of order attempts and bar-close "
                "valuations, with Bokeh status filters and selectable columns, "
                "static HTML and full-precision CSV/JSON exports. Static tables "
                "remain readable without JavaScript; interactive filters and "
                "download buttons require it. Generic balances are not exchange "
                "margin, funding or liquidation wallets.",
            ),
            ReportSectionDefinition(
                "trades",
                "配對交易損益",
                "Supplementary paired position/trade PnL; not individual "
                "order attempts or the account reconciliation ledger.",
            ),
            ReportSectionDefinition(
                "limitations",
                "假設與限制",
                "Known limits and unknown assumptions; no invented conclusions.",
            ),
            ReportSectionDefinition(
                "provenance",
                "版本、資料與來源證據",
                "Strategy/run versions, data IDs and registered source hashes.",
            ),
        )
    }
)

REPORT_RECIPES = MappingProxyType(
    {
        "standard": tuple(REPORT_SECTIONS),
        "comparison": ("overview", "research", "metrics", "equity", "provenance"),
        "trades": ("overview", "research", "ledger", "trades", "provenance"),
    }
)


def report_navigation_instructions(
    sections: list[str], *, has_results: bool, comparison: bool
) -> str:
    """Describe only navigation and detail sections present in this report."""
    details = [
        REPORT_SECTIONS[key].title for key in sections if key in SCOPED_REPORT_SECTIONS
    ]
    if has_results and details:
        return (
            "點選總覽表格或結果清單中的結果列，可進入該組參數的詳情；"
            "再以「返回」或「切換結果」查看其他組。已選取的詳情章節："
            + "、".join(details)
            + "。所有內容保存在同一份離線 HTML。"
        )
    if comparison and not has_results:
        return (
            "此比較沒有成功候選的績效結果。以下保留全部候選、無結果原因、"
            "共用設定與行情證據；沒有推算報酬或資金曲線，也沒有單組交易明細入口。"
        )
    return (
        "本報告僅呈現已選取的總覽或共用說明，未選取單組詳情章節。"
        "所有內容保存在同一份離線 HTML。"
    )
