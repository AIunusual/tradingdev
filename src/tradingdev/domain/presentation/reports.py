"""Canonical report sections and suggested recipes for saved research evidence."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

REPORT_SCHEMA_VERSION = 3
REPORT_TEMPLATE_VERSION = "24"
SCOPED_REPORT_SECTIONS = frozenset(
    {"settings", "metrics", "benchmark", "equity", "ledger", "trades"}
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
                "benchmark",
                "基準比較（Benchmark）",
                "The asset class's fixed buy-and-hold baseline over the same bars "
                "and costs, shown beside the strategy's saved values and equity. "
                "An undefined or unusable baseline states its reason instead of "
                "implying a zero return.",
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
            ReportSectionDefinition(
                "ml",
                "ML 訊號研究",
                "Selected saved leaderboard candidates, full metrics/definitions, "
                "evaluation coordinates, unavailable scores and ranking caveats. "
                "Select their entry IDs with generate_report(ml_entry_ids=...). "
                "Supports standalone ML reports or independent evidence alongside "
                "runs; this is cost-free signal research, not executed backtests. "
                "Series and cross-horizon source equivalence remain unavailable.",
            ),
        )
    }
)

REPORT_RECIPES = MappingProxyType(
    {
        "standard": tuple(key for key in REPORT_SECTIONS if key != "ml"),
        "comparison": (
            "overview",
            "research",
            "metrics",
            "benchmark",
            "equity",
            "provenance",
        ),
        "trades": ("overview", "research", "ledger", "trades", "provenance"),
        "ml_research": ("ml",),
    }
)


def report_navigation_instructions(
    sections: list[str], *, has_results: bool, comparison: bool, has_ml: bool = False
) -> str:
    """Describe only navigation and detail sections present in this report."""
    run_navigation = _run_navigation_instructions(
        sections, has_results=has_results, comparison=comparison
    )
    if not has_ml:
        return run_navigation
    ml_navigation = (
        "在「ML 訊號研究」查看候選排序與缺分數原因，"
        "展開候選身分、其他指標及保存定義，或下載完整 ML 證據 JSON。"
        "ML 證據與回測各自獨立，不代表模型已用於該回測。"
        if "ml" in sections
        else "未選取 ML 顯示章節；所選 ML 候選的完整證據仍保存在離線 HTML。"
    )
    if has_results or comparison:
        return ml_navigation + " " + run_navigation
    return ml_navigation + " 所有內容保存在同一份離線 HTML。"


def _run_navigation_instructions(
    sections: list[str], *, has_results: bool, comparison: bool
) -> str:
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
