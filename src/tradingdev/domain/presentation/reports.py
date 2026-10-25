"""Canonical report sections and suggested recipes for saved research evidence."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

REPORT_SCHEMA_VERSION = 2
REPORT_TEMPLATE_VERSION = "17"
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
        "comparison": ("overview", "metrics", "equity", "provenance"),
        "trades": ("overview", "ledger", "trades", "provenance"),
    }
)
