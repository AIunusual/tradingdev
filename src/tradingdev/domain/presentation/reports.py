"""Canonical report sections and suggested recipes for saved research evidence."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

REPORT_SCHEMA_VERSION = 1
REPORT_TEMPLATE_VERSION = "12"


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
                "總覽",
                "Run identity, recorded coverage and default-scope comparison.",
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
                "trades",
                "完整交易紀錄",
                "All trades with raw fields, sort, filter and offline CSV export.",
            ),
            ReportSectionDefinition(
                "executions",
                "原生執行紀錄",
                "Native order attempts, prices, fees and before/after "
                "account transitions.",
            ),
            ReportSectionDefinition(
                "account_history",
                "逐根帳戶狀態",
                "Saved end-of-bar VectorBT cash, position and equity; "
                "not exchange margin.",
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
        "trades": ("overview", "trades", "provenance"),
    }
)
