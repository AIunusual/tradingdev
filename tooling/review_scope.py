"""Deterministic review ownership and lossless Markdown context selection.

These scopes organize evidence; they never exempt a path from review. Unknown
paths have the explicit ``general`` owner. Unknown documents retain all text;
unknown sections and their descendants appear in every scope so a stale
responsibility registry cannot hide new content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from markdown_it import MarkdownIt

if TYPE_CHECKING:
    from collections.abc import Iterable

    from markdown_it.token import Token

SCOPES: dict[str, str] = {
    "tooling": "Development policy, quality checks, Git workflow and dependencies",
    "strategy": "Strategy creation, validation, revisions and bundled strategies",
    "execution": "Execution plans, backtests, optimization, jobs and workers",
    "results": "Stored results, history, performance, reports and dashboards",
    "data": "Market data, providers, sampling and data preparation",
    "ml": "Machine learning datasets, features, training and evaluation",
    "presentation": "Registered slide decks and their build and publishing tools",
    "interfaces": "MCP, CLI and shared public interface contracts",
    "general": "Unclassified or shared changes requiring conservative context",
}


@dataclass(frozen=True)
class MarkdownSection:
    """An exact, inclusive, one-based line slice with its heading ancestry."""

    path: str
    start_line: int
    end_line: int
    headings: tuple[str, ...]
    text: str


_ALL = frozenset(SCOPES)
_PRODUCT = frozenset({"strategy", "execution", "results", "data", "ml", "interfaces"})


def _headings(scopes: Iterable[str], *titles: str) -> dict[str, frozenset[str]]:
    roles = frozenset(scopes)
    return dict.fromkeys(titles, roles)


# Heading titles are evidence selectors, not prose summaries or instructions.
# A section can be omitted only when its full heading ancestry is recognized.
# New headings widen context by retaining their entire subtree in every scope.
_DOC_HEADINGS: dict[str, dict[str, frozenset[str]]] = {
    "README.md": {
        **_headings(_ALL, "TradingDev", "常見問題", "相關文件"),
        **_headings(
            {"tooling", "interfaces"},
            "安裝與連接助手",
            "1. 準備環境並安裝",
            "2. 啟動 MCP server 並連接助手",
            "由客戶端啟動（stdio）",
            "手動啟動並使用 HTTP 連接",
            "3. 確認助手能使用工具",
        ),
        **_headings(
            {"strategy", "execution", "interfaces"},
            "完成第一次回測",
            "建立與研究自己的策略",
            "把交易規則交給助手",
            "比較參數",
            "使用樣本外驗證與最佳化",
        ),
        **_headings({"ml", "strategy", "interfaces"}, "研究機器學習訊號"),
        **_headings(
            {"results", "interfaces"},
            "查看結果與圖表",
            "在對話中查詢",
            "交易紀錄與 HTML 報告",
            "開啟 Dashboard",
            "從已保存的結果產生報告",
        ),
        **_headings(_PRODUCT, "資料與工作區"),
        **_headings({"execution", "interfaces"}, "終端機用法", "直接執行設定檔"),
    },
    "ARCHITECTURE.md": {
        **_headings(
            _ALL,
            "系統架構",
            "系統邊界與模組責任",
            "核心概念與身分",
            "從策略建立到結果查詢",
            "一致性、程序管理與保證範圍",
        ),
        **_headings({"tooling", "presentation"}, "Repository 開發工具與研究原型"),
        **_headings({"strategy", "interfaces"}, "建立與驗證策略"),
        **_headings(_PRODUCT, "研究需求與有效設定"),
        **_headings(
            {"strategy", "execution", "interfaces"},
            "準備、確認與執行回測",
            "最佳化執行",
            "可重現性與策略執行安全",
        ),
        **_headings(
            {"results", "interfaces"},
            "保存與查詢結果",
            "績效計算與對話摘要",
            "歷史查詢與報告組裝",
        ),
        **_headings({"data", "interfaces"}, "獨立行情取樣"),
        **_headings({"ml", "strategy", "interfaces"}, "ML 訊號研究"),
        **_headings(_PRODUCT, "設定與儲存一致性", "MCP 回覆契約"),
        **_headings({"execution", "tooling", "interfaces"}, "程序控制與清理"),
    },
    "AGENTS.md": {
        **_headings(_ALL, "AGENTS.md", "政策與溝通", "設計原則", "完成回報"),
        **_headings(_PRODUCT | {"tooling"}, "模型上下文與提示詞", "產品邊界與架構"),
        **_headings(
            {"tooling", "presentation"},
            "環境與品質工具",
            "測試與清理",
            "執行時機",
            "開發階段的程式碼審查",
            "Issue 接手與指派",
            "工作樹與分階段提交",
            "Git hooks 與本機 PR 審查",
            "PR 合併後的本機分支清理",
        ),
    },
    "CLAUDE.md": _headings(_ALL, "Claude Code"),
    "docs/development.md": {
        **_headings(_ALL, "開發指南", "更新文件"),
        **_headings(
            {"tooling", "presentation"},
            "環境與常用命令",
            "依賴相容性限制",
            "測試設計與精簡",
            "開發階段的程式碼審查",
            "接手 issue",
            "從開發到合併",
            "1. 分階段提交並推送",
            "2. 在 GitHub 建立 PR，交由審閱者檢查",
            "3. 在 GitHub 審閱並合併",
            "4. 合併後更新 main、清理本機分支",
            "尚未建立 PR 的本機檢查",
            "注意事項",
        ),
        **_headings(
            {"presentation"},
            "簡報的編輯與輸出",
            "在 VS Code 儲存時自動編譯",
            "手動輸出與預覽",
        ),
        **_headings({"tooling", "results"}, "MAE 研究原型"),
        **_headings({"tooling", "interfaces"}, "選擇性 LLM 測試", "量測模型上下文"),
    },
    "docs/strategy_contract.md": {
        **_headings(_ALL, "Strategy Contract"),
        **_headings(
            _PRODUCT, "Contract References", "Research Inputs and Simulation Policy"
        ),
        **_headings(
            {"strategy", "execution", "interfaces"},
            "Python Contract",
            "YAML Contract",
            "Strategy Walk-Forward Settings",
            "Lifecycle",
            "Revision Identity and Eligibility",
            "Execution and Parameter Experiments",
            "Legacy Recovery",
            "Security Model",
        ),
        **_headings(
            {"strategy", "execution", "results", "interfaces"},
            "Signal and Execution Semantics",
            "Constructor Binding",
            "Draft Cleanup",
        ),
        **_headings({"strategy", "interfaces"}, "Indicators and Code Quality"),
        **_headings(
            {"strategy", "execution", "ml", "interfaces"},
            "Randomness",
            "Training Before Scoring",
        ),
        **_headings(
            {"ml", "strategy", "interfaces"},
            "ML Research Contract",
            "Dataset and Model Inputs",
            "Fold Validation and Signal Interpretation",
        ),
    },
    "docs/run_artifacts.md": {
        **_headings(_ALL, "Run Artifacts"),
        **_headings(
            _PRODUCT,
            "Artifact References",
            "Research Evidence Records",
            "Research Settings and Provenance",
            "Fixed Comparison Evidence",
        ),
        **_headings(_PRODUCT, "Workspace Layout", "SQLite Tables", "JSON Values"),
        **_headings(
            {"ml", "results", "interfaces"},
            "ML Research Artifacts",
            "Immutable Datasets",
            "Feature Matrix Cache",
            "Trained Models",
            "Recorded Signal Evaluations",
        ),
        **_headings(
            {"execution", "results", "interfaces"},
            "Run Publication",
            "Job State and Worker Identity",
            "Execution Plans and Confirmation Documents",
        ),
        **_headings(
            {"strategy", "execution", "results", "interfaces"}, "Execution Manifest"
        ),
        **_headings(
            {"data", "results", "interfaces"},
            "Market Data Cache",
            "Automatic DVOL Queries",
            "Bounded Historical Sampling",
        ),
        **_headings(
            {"results", "interfaces"},
            "MCP Lookup",
            "Discovery Pages",
            "Reading Artifact Text",
            "Completed Run Files",
            "Performance Scopes and Provenance",
            "Historical Observation Queries",
            "Research Notes and Archives",
            "Composable Offline Reports",
        ),
    },
}


# Relocated contracts keep their original section responsibilities. Unknown
# headings still widen coverage; moving a file must not hide new contracts.
_DOC_HEADINGS.update(
    {
        "docs/ml_research.md": {
            **_headings(_ALL, "ML Research"),
            **{
                title: _DOC_HEADINGS["docs/strategy_contract.md"][title]
                for title in (
                    "ML Research Contract",
                    "Dataset and Model Inputs",
                    "Fold Validation and Signal Interpretation",
                )
            },
            **{
                title: _DOC_HEADINGS["docs/run_artifacts.md"][title]
                for title in (
                    "ML Research Artifacts",
                    "Immutable Datasets",
                    "Feature Matrix Cache",
                    "Trained Models",
                    "Recorded Signal Evaluations",
                )
            },
            "Method families": _DOC_HEADINGS["docs/run_artifacts.md"][
                "Recorded Signal Evaluations"
            ],
        },
        **{
            path: {
                **_headings(_ALL, title),
                section: _DOC_HEADINGS["docs/run_artifacts.md"][section],
            }
            for path, title, section in (
                (
                    "docs/execution_plans.md",
                    "Execution Plans",
                    "Execution Plans and Confirmation Documents",
                ),
                (
                    "docs/performance_artifacts.md",
                    "Performance Artifacts",
                    "Performance Scopes and Provenance",
                ),
                ("docs/reports.md", "Reports", "Composable Offline Reports"),
            )
        },
    }
)

# The baseline contract sits in its own section of the same results-owned
# document, so it is reviewed by the same scopes as the scope/provenance section.
_DOC_HEADINGS["docs/performance_artifacts.md"].update(
    {
        **_headings({"results", "interfaces"}, "Benchmarks"),
        **_headings(
            {"strategy", "execution", "results", "interfaces"}, "Metric Interpretation"
        ),
    }
)

# Strategy lifecycle now delegates preparation/approval details to their owner.
_DOC_HEADINGS["docs/execution_plans.md"].update(
    _headings(
        {"strategy", "execution", "results", "interfaces"},
        "Execution Plans and Confirmation Documents",
    )
)


# These narrowly focused documents are selected as complete files. New paths
# are deliberately not matched by a wildcard: an unfamiliar document widens
# context until its responsibility has been reviewed and registered.
_WHOLE_DOC_SCOPES: dict[str, frozenset[str]] = {
    **dict.fromkeys(
        (
            "docs/strategies/README.md",
            "docs/strategies/kd_strategy.md",
            "docs/strategies/glft_strategy.md",
        ),
        frozenset({"strategy", "execution"}),
    ),
    **dict.fromkeys(
        (
            "docs/strategies/glft_ml_strategy.md",
            "docs/strategies/safety_volume_strategy.md",
            "docs/strategies/xgboost_strategy.md",
        ),
        frozenset({"strategy", "execution", "ml"}),
    ),
    # Quantile scoring details moved here from the shared authoring contract.
    "docs/strategies/quantile_strategy.md": frozenset(
        {"strategy", "execution", "ml", "interfaces"}
    ),
}


_PREFIX_SCOPES: tuple[tuple[str, str], ...] = (
    ("docs/talks/", "presentation"),
    ("tooling/marp/", "presentation"),
    ("tooling/", "tooling"),
    ("scripts/", "tooling"),
    (".githooks/", "tooling"),
    (".Codex/hooks/", "tooling"),
    (".github/", "tooling"),
    (".vscode/", "tooling"),
    ("tests/tooling/", "tooling"),
    ("docs/strategies/", "strategy"),
    ("strategies/", "strategy"),
    ("configs/strategies/", "strategy"),
)
_EXACT_SCOPES = {
    "tooling/presentations.py": "presentation",
    "scripts/presentations.sh": "presentation",
    "scripts/check-presentations.sh": "presentation",
    "tests/tooling/test_presentations.py": "presentation",
    ".vscode/tasks.json": "presentation",
    "docs/.nojekyll": "presentation",
    "AGENTS.md": "tooling",
    "CLAUDE.md": "tooling",
    "pyproject.toml": "tooling",
    "uv.lock": "tooling",
    "docs/development.md": "tooling",
    "docs/strategy_contract.md": "strategy",
    "docs/run_artifacts.md": "results",
    "docs/ml_research.md": "ml",
    "docs/execution_plans.md": "execution",
    "docs/performance_artifacts.md": "results",
    "docs/reports.md": "results",
}
# A directory describes a layer, not necessarily the behavior of every module
# inside it. These exceptions keep calculated results and execution consent
# paired with the contracts that explain them, including their regression tests.
_MODULE_EXACT_SCOPES = {
    "domain/backtest/metrics.py": "results",
    "domain/backtest/test_performance_metrics.py": "results",
    "domain/presentation/confirmation.py": "execution",
    "adapters/presentation/confirmation.py": "execution",
}
_MODULE_PREFIX_SCOPES = (
    ("domain/strategies/", "strategy"),
    ("domain/indicators/", "strategy"),
    ("domain/ml/", "ml"),
    ("domain/data/", "data"),
    ("domain/backtest/", "execution"),
    ("domain/optimization/", "execution"),
    ("domain/validation/", "execution"),
    ("domain/performance/", "results"),
    ("domain/presentation/", "results"),
    ("adapters/execution/", "execution"),
    ("adapters/reporting/", "results"),
    ("adapters/dashboard/", "results"),
    ("adapters/presentation/", "results"),
    ("adapters/cli/", "interfaces"),
    ("mcp/workers/", "execution"),
    ("mcp/", "interfaces"),
    ("e2e/", "interfaces"),
    ("integration/", "interfaces"),
)
_STEM_SCOPES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ml", ("ml", "feature_request")),
    ("strategy", ("strategy", "generated_strategy", "capability", "quality_policy")),
    (
        "execution",
        (
            "execution",
            "backtest",
            "optimization",
            "job",
            "jobs",
            "plans",
            "preflight",
            "confirmation",
            "worker",
        ),
    ),
    (
        "data",
        (
            "data",
            "sample",
            "market_sample",
            "market_samples",
            "parquet_samples",
            "feature_samples",
        ),
    ),
    (
        "results",
        (
            "run",
            "report",
            "reports",
            "history",
            "trade_history",
            "artifact",
            "performance",
            "pagination",
        ),
    ),
)


def scope_for_path(path: str) -> str:
    """Return one owner; classification never filters a path out of evidence."""
    normalized = PurePosixPath(path)
    if normalized.is_absolute() or ".." in normalized.parts:
        return "general"
    name = str(normalized)
    if name in _EXACT_SCOPES:
        return _EXACT_SCOPES[name]
    for prefix, scope in _PREFIX_SCOPES:
        if name.startswith(prefix):
            return scope
    module = name.removeprefix("src/tradingdev/").removeprefix("tests/")
    if module == name:
        return "general"
    if module in _MODULE_EXACT_SCOPES:
        return _MODULE_EXACT_SCOPES[module]
    for prefix, scope in _MODULE_PREFIX_SCOPES:
        if module.startswith(prefix):
            return scope
    if module.startswith(("app/", "adapters/", "domain/")):
        stem = PurePosixPath(module).stem.removeprefix("test_")
        for scope, prefixes in _STEM_SCOPES:
            if any(
                stem == prefix or stem.startswith(prefix + "_") for prefix in prefixes
            ):
                return scope
    return "general"


def _heading_text(tokens: Iterable[Token]) -> str:
    return "".join(
        token.content
        if token.type in {"text", "code_inline"}
        else _heading_text(token.children or [])
        if token.type == "image"
        else " "
        if token.type in {"softbreak", "hardbreak"}
        else ""
        for token in tokens
    )


def markdown_lines(text: str) -> list[str]:
    """Split only CommonMark line endings, preserving every original character."""
    lines = re.findall(r"[^\r\n]*(?:\r\n|\r|\n|$)", text)
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _parse_sections(
    path: str, text: str
) -> tuple[tuple[MarkdownSection, ...], frozenset[int]]:
    if not text:
        return (), frozenset()
    # Match CommonMark's CR/LF line normalization without changing the slices.
    # str.splitlines() would incorrectly count Unicode paragraph separators.
    lines = markdown_lines(text)
    environment: dict[str, Any] = {}
    tokens = MarkdownIt("commonmark").parse(text, environment)
    boundaries: list[tuple[int, tuple[str, ...]]] = [(0, ())]
    ancestors: list[tuple[int, str]] = []
    for index, token in enumerate(tokens):
        # Container headings are part of the enclosing document section. They
        # must not change the responsibility of prose after a quote or list.
        if token.type != "heading_open" or token.map is None or token.level != 0:
            continue
        level = int(token.tag[1:])
        while ancestors and ancestors[-1][0] >= level:
            ancestors.pop()
        ancestors.append((level, _heading_text(tokens[index + 1].children or [])))
        boundary = token.map[0], tuple(title for _, title in ancestors)
        if boundaries[-1][0] == boundary[0]:
            boundaries[-1] = boundary
        else:
            boundaries.append(boundary)
    sections = tuple(
        MarkdownSection(
            path=path,
            start_line=start + 1,
            end_line=end,
            headings=headings,
            text="".join(lines[start:end]),
        )
        for (start, headings), end in zip(
            boundaries,
            [point for point, _ in boundaries[1:]] + [len(lines)],
            strict=True,
        )
    )
    references = [
        *environment.get("references", {}).values(),
        *environment.get("duplicate_refs", []),
    ]
    reference_lines = frozenset(
        line + 1 for reference in references for line in range(*reference["map"])
    )
    return sections, reference_lines


def markdown_sections(path: str, text: str) -> tuple[MarkdownSection, ...]:
    """Partition original text at document-level headings, retaining the prelude."""
    return _parse_sections(path, text)[0]


def _selection_reason(
    scope: str, path: str, sections: tuple[MarkdownSection, ...]
) -> str:
    if scope not in SCOPES:
        raise ValueError(f"Unknown documentation review scope: {scope!r}")
    if scope == "general":
        return "general-scope-full-document"
    if path in _WHOLE_DOC_SCOPES:
        return "registered-document-responsibility"
    registry = _DOC_HEADINGS.get(path)
    if registry is None:
        return "unfamiliar-document-full-document"
    if any(title not in registry for section in sections for title in section.headings):
        return "unknown-sections-preserved"
    return "registered-heading-responsibility"


def selection_reason(scope: str, path: str, text: str) -> str:
    """Explain registered selection or conservative unknown-content retention."""
    return _selection_reason(scope, path, markdown_sections(path, text))


def sections_for_scope(scope: str, path: str, text: str) -> tuple[MarkdownSection, ...]:
    """Select original context, preserving ancestor prose and link definitions.

    Changed documents must separately receive full-content review. This function
    only selects context for other scopes; it never grants a coverage exemption.
    Unrecognized documents return every section. Within registered documents,
    unknown headings retain their complete subtree in every scope, together
    with ancestor prose and reference definitions. Only recognized, unrelated
    sections can be omitted, as reported by :func:`selection_reason`.
    """
    sections, reference_lines = _parse_sections(path, text)
    reason = _selection_reason(scope, path, sections)
    if reason == "registered-document-responsibility":
        return sections if scope in _WHOLE_DOC_SCOPES[path] else ()
    if reason not in {
        "registered-heading-responsibility",
        "unknown-sections-preserved",
    }:
        return sections
    registry = _DOC_HEADINGS[path]
    selected = {
        index
        for index, section in enumerate(sections)
        if not section.headings
        or any(title not in registry for title in section.headings)
        or scope in registry[section.headings[-1]]
        or any(
            section.start_line <= line <= section.end_line for line in reference_lines
        )
    }
    # Find each actual nearest ancestor, rather than matching all equal heading
    # names: repeated headings can belong to different parts of the document.
    for index in tuple(selected):
        headings = sections[index].headings
        for length in range(1, len(headings)):
            for previous in range(index - 1, -1, -1):
                if sections[previous].headings == headings[:length]:
                    selected.add(previous)
                    break
    return tuple(section for index, section in enumerate(sections) if index in selected)
