"""Evidence selection retains exact text and conservatively handles new content."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from tooling.review_scope import (
    SCOPES,
    markdown_sections,
    scope_for_path,
    sections_for_scope,
    selection_reason,
)


def test_sections_partition_unicode_text_without_rewriting() -> None:
    text = (
        "前言\u2028仍是同一行\n\n"
        "# 文件\n標題內容\n\n"
        "## 第一節\n資料與 `code`。\n\n"
        "### 細節\n末行沒有換行"
    )
    sections = markdown_sections("docs/example.md", text)

    assert [section.headings for section in sections] == [
        (),
        ("文件",),
        ("文件", "第一節"),
        ("文件", "第一節", "細節"),
    ]
    assert [(section.start_line, section.end_line) for section in sections] == [
        (1, 2),
        (3, 5),
        (6, 8),
        (9, 10),
    ]
    assert "".join(section.text for section in sections) == text
    lines = text.split("\n")
    for section in sections:
        assert section.path == "docs/example.md"
        expected = "\n".join(lines[section.start_line - 1 : section.end_line])
        if section.end_line < len(lines):
            expected += "\n"
        assert section.text == expected
    with pytest.raises(FrozenInstanceError):
        sections[0].text = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_sections_preserve_line_endings(newline: str) -> None:
    text = newline.join(["# Title", "", "## Child", "body", ""])
    sections = markdown_sections("README.md", text)
    assert [(section.start_line, section.end_line) for section in sections] == [
        (1, 2),
        (3, 4),
    ]
    assert "".join(section.text for section in sections) == text


@pytest.mark.parametrize("text", ["", "plain text", "\n", "\n\n", "text\n"])
def test_heading_free_text_is_preserved(text: str) -> None:
    sections = markdown_sections("notes.md", text)
    assert "".join(section.text for section in sections) == text
    assert len(sections) == (1 if text else 0)
    assert all(section.headings == () for section in sections)


def test_fenced_and_indented_code_do_not_create_sections() -> None:
    text = (
        "# Title\n\n```markdown\n## Fake\nFake setext\n---\n```\n\n"
        "    # Indented fake\n\n~~~\n# Also fake\n~~~\n\n## Actual\nBody\n"
    )
    sections = markdown_sections("README.md", text)
    assert [section.headings for section in sections] == [
        ("Title",),
        ("Title", "Actual"),
    ]
    assert "".join(section.text for section in sections) == text


@pytest.mark.parametrize(
    "container",
    [
        "> ## 研究機器學習訊號\n> Quoted example.\n",
        "- ## 研究機器學習訊號\n\n  List example.\n",
        "> 研究機器學習訊號\n> ----------------\n> Quoted setext example.\n",
        "> # Unregistered example heading\n> Example.\n",
    ],
)
def test_container_heading_does_not_reassign_following_document_prose(
    container: str,
) -> None:
    installation = (
        "## 安裝與連接助手\nInstallation before example.\n\n"
        + container
        + "\nInstallation after container.\n\n"
    )
    text = "# TradingDev\n" + installation + "## 查看結果與圖表\nUnrelated reports.\n"

    sections = markdown_sections("README.md", text)
    assert [section.headings for section in sections] == [
        ("TradingDev",),
        ("TradingDev", "安裝與連接助手"),
        ("TradingDev", "查看結果與圖表"),
    ]
    assert "".join(section.text for section in sections) == text
    assert sections[1].text == installation
    assert sections[1].end_line + 1 == sections[2].start_line
    selected = sections_for_scope("tooling", "README.md", text)
    assert (
        "".join(section.text for section in selected) == "# TradingDev\n" + installation
    )
    assert selection_reason("tooling", "README.md", text) == (
        "registered-heading-responsibility"
    )


def test_setext_and_formatted_headings_keep_exact_slices() -> None:
    text = "**Title**\n=========\n\nSub `heading`\n------------\nBody\n"
    sections = markdown_sections("README.md", text)
    assert [section.headings for section in sections] == [
        ("Title",),
        ("Title", "Sub heading"),
    ]
    assert sections[0].text == "**Title**\n=========\n\n"
    assert sections[1].text == "Sub `heading`\n------------\nBody\n"
    assert sections[1].start_line == 4


def test_repeated_headings_remain_distinct_with_correct_ancestors() -> None:
    text = (
        "# Root\n## A\n### Repeated\nfirst\n"
        "## B\n### Repeated\nsecond\n## A\n### Repeated\nthird\n"
    )
    sections = markdown_sections("README.md", text)
    repeated = [section for section in sections if section.headings[-1] == "Repeated"]
    assert [section.headings for section in repeated] == [
        ("Root", "A", "Repeated"),
        ("Root", "B", "Repeated"),
        ("Root", "A", "Repeated"),
    ]
    assert [section.start_line for section in repeated] == [3, 6, 9]
    assert "".join(section.text for section in sections) == text


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("tooling/review_docs.py", "tooling"),
        ("tests/tooling/test_review_docs.py", "tooling"),
        (".githooks/pre-commit", "tooling"),
        (".Codex/hooks/finalize-python", "tooling"),
        ("pyproject.toml", "tooling"),
        ("AGENTS.md", "tooling"),
        ("src/tradingdev/domain/strategies/loader.py", "strategy"),
        ("tests/domain/strategies/test_loader_contract.py", "strategy"),
        ("src/tradingdev/app/strategy_service.py", "strategy"),
        ("tests/app/test_strategy_cleanup_service.py", "strategy"),
        ("src/tradingdev/domain/backtest/engines.py", "execution"),
        ("tests/app/test_execution_plan_service.py", "execution"),
        ("src/tradingdev/app/contracts/plans.py", "execution"),
        ("src/tradingdev/app/contracts/jobs.py", "execution"),
        ("src/tradingdev/mcp/workers/backtest.py", "execution"),
        ("tests/adapters/execution/test_worker_supervisor.py", "execution"),
        ("src/tradingdev/domain/performance/catalog.py", "results"),
        ("tests/app/test_trade_history.py", "results"),
        ("src/tradingdev/app/contracts/reports.py", "results"),
        ("src/tradingdev/app/contracts/history.py", "results"),
        ("src/tradingdev/adapters/reporting/html.py", "results"),
        ("src/tradingdev/domain/data/crawlers/registry.py", "data"),
        ("tests/domain/data/test_processor.py", "data"),
        ("src/tradingdev/adapters/storage/market_samples.py", "data"),
        ("tests/app/test_sample_data_boundaries.py", "data"),
        ("src/tradingdev/domain/ml/features/features.py", "ml"),
        ("src/tradingdev/app/ml_training.py", "ml"),
        ("tests/app/test_ml_dataset_training.py", "ml"),
        ("tests/adapters/test_ml_datasets.py", "ml"),
        ("src/tradingdev/mcp/server.py", "interfaces"),
        ("src/tradingdev/mcp/tools/strategy.py", "interfaces"),
        ("tests/mcp/test_execution_plans.py", "interfaces"),
        ("tests/integration/test_mcp_protocol.py", "interfaces"),
        ("src/tradingdev/adapters/cli/main.py", "interfaces"),
        ("docs/talks/tradingdev-introduction/slides.md", "presentation"),
        ("tooling/presentations.py", "presentation"),
        ("tooling/marp/package-lock.json", "presentation"),
        ("tests/tooling/test_presentations.py", "presentation"),
        ("scripts/check-presentations.sh", "presentation"),
        (".vscode/tasks.json", "presentation"),
        ("docs/.nojekyll", "presentation"),
        ("docs/strategy_contract.md", "strategy"),
        ("docs/run_artifacts.md", "results"),
        ("docs/development.md", "tooling"),
        ("new/component.unknown", "general"),
        ("src/tradingdev/new_component.py", "general"),
        ("tests/new_component/test_unknown.py", "general"),
        ("not-tooling/review_docs.py", "general"),
        ("../tooling/review_docs.py", "general"),
        ("/tooling/review_docs.py", "general"),
    ],
)
def test_path_ownership_is_explicit_and_deterministic(path: str, expected: str) -> None:
    assert scope_for_path(path) == expected
    assert expected in SCOPES
    assert SCOPES[expected]


@pytest.mark.parametrize(
    ("path", "expected_scope", "document", "heading"),
    [
        (
            path,
            "results",
            document,
            heading,
        )
        for path in (
            "src/tradingdev/domain/backtest/metrics.py",
            "tests/domain/backtest/test_performance_metrics.py",
        )
        for document, heading in (
            ("ARCHITECTURE.md", "績效計算與對話摘要"),
            ("docs/run_artifacts.md", "Performance Scopes and Provenance"),
        )
    ]
    + [
        (
            path,
            "execution",
            "ARCHITECTURE.md",
            "準備、確認與執行回測",
        )
        for path in (
            "src/tradingdev/adapters/presentation/confirmation.py",
            "src/tradingdev/domain/presentation/confirmation.py",
            "tests/domain/test_confirmation_presentation.py",
        )
    ],
)
def test_actual_module_owner_receives_its_documented_behavior(
    path: str, expected_scope: str, document: str, heading: str
) -> None:
    root = Path(__file__).resolve().parents[2]
    assert (root / path).is_file()
    scope = scope_for_path(path)
    assert scope == expected_scope
    text = (root / document).read_text(encoding="utf-8")
    required = tuple(
        section
        for section in markdown_sections(document, text)
        if section.headings and section.headings[-1] == heading
    )
    assert required, f"Expected the actual documented behavior in {document}"
    assert set(required) <= set(sections_for_scope(scope, document, text))


def test_known_doc_selects_role_and_ancestor_context() -> None:
    text = (
        "前言\n# 開發指南\nGeneral introduction.\n"
        "## 更新文件\nDocument ownership.\n"
        "### 簡報的編輯與輸出\nSlide workflow.\n"
        "#### 手動輸出與預覽\nPresentation command.\n"
        "## 選擇性 LLM 測試\nModel commands.\n"
    )
    selected = sections_for_scope("presentation", "docs/development.md", text)
    joined = "".join(section.text for section in selected)
    assert "前言" in joined
    assert "General introduction." in joined
    assert "Document ownership." in joined
    assert "Slide workflow." in joined
    assert "Presentation command." in joined
    assert "Model commands." not in joined
    assert selection_reason("presentation", "docs/development.md", text) == (
        "registered-heading-responsibility"
    )
    assert set(selected) <= set(markdown_sections("docs/development.md", text))


def test_selected_child_keeps_unselected_parent_prose() -> None:
    text = (
        "# 開發指南\n## 選擇性 LLM 測試\nNecessary parent context.\n"
        "### 簡報的編輯與輸出\nSelected slide content.\n"
        "## MAE 研究原型\nUnrelated research.\n"
    )
    selected = sections_for_scope("presentation", "docs/development.md", text)
    joined = "".join(section.text for section in selected)
    assert "Necessary parent context." in joined
    assert "Selected slide content." in joined
    assert "Unrelated research." not in joined


def test_reference_definitions_keep_their_complete_original_sections() -> None:
    text = (
        "# TradingDev\n## 安裝與連接助手\nUse [manual] and [multi].\n"
        "## 研究機器學習訊號\n"
        "[manual]: ../README.md\n[multi]:\n  ../ARCHITECTURE.md\n  'Title'\n"
        "## 查看結果與圖表\n[manual]: ../docs/development.md\n"
    )
    sections = sections_for_scope("tooling", "README.md", text)
    assert "".join(section.text for section in sections) == text
    assert [section.start_line for section in sections] == [1, 2, 4, 9]


@pytest.mark.parametrize(
    ("scope", "path", "text", "reason"),
    [
        (
            "tooling",
            "docs/unknown.md",
            "# New document\n## Contract\nAll content.\n",
            "unfamiliar-document-full-document",
        ),
        (
            "data",
            "README.md",
            "# TradingDev\n## New behavior\nMust not disappear.\n",
            "unknown-sections-preserved",
        ),
        (
            "tooling",
            "docs/development.md",
            "# 開發指南\n## 更新文件\n### New nested heading\nNew rules.\n",
            "unknown-sections-preserved",
        ),
        (
            "general",
            "README.md",
            "# TradingDev\n## 研究機器學習訊號\nAll context.\n",
            "general-scope-full-document",
        ),
    ],
)
def test_unfamiliar_content_is_preserved_with_explicit_reason(
    scope: str, path: str, text: str, reason: str
) -> None:
    assert sections_for_scope(scope, path, text) == markdown_sections(path, text)
    assert selection_reason(scope, path, text) == reason


@pytest.mark.parametrize("scope", SCOPES)
def test_unknown_section_and_entire_subtree_are_preserved_in_every_scope(
    scope: str,
) -> None:
    parent = "## 查看結果與圖表\nRequired ancestor context.\n"
    unknown = "### New behavior\nNew evidence [manual].\n"
    child = "#### 開啟 Dashboard\nKnown title inside the unknown subtree.\n"
    grandchild = "##### New details\nDeeper unknown evidence.\n"
    unrelated = "## 研究機器學習訊號\n" + "Unrelated known ML details.\n" * 1_000
    definitions = "## 終端機用法\n[manual]: ../README.md\n"
    text = "前言\n# TradingDev\n" + parent + unknown + child + grandchild + unrelated
    text += definitions

    all_sections = markdown_sections("README.md", text)
    selected = sections_for_scope(scope, "README.md", text)
    assert "".join(section.text for section in all_sections) == text
    expected = {parent, unknown, child, grandchild, definitions}
    assert expected <= {section.text for section in selected}
    assert set(selected) <= set(all_sections)
    assert selected[0].text == "前言\n"
    assert selection_reason(scope, "README.md", text) == (
        "general-scope-full-document"
        if scope == "general"
        else "unknown-sections-preserved"
    )
    if scope == "tooling":
        # A new heading must not repeat a long, recognized unrelated section.
        assert unrelated not in {section.text for section in selected}
        assert sum(len(section.text) for section in selected) < len(text) // 10
    if scope == "general":
        assert selected == all_sections


def test_unknown_ancestor_retains_known_descendants_until_its_subtree_ends() -> None:
    text = (
        "# TradingDev\n"
        "## New workflow\nNew contract.\n"
        "### 研究機器學習訊號\nNested known title still belongs to new contract.\n"
        "## 研究機器學習訊號\nSeparate known ML section.\n"
        "## 常見問題\nCommon context.\n"
    )
    sections = markdown_sections("README.md", text)
    selected = sections_for_scope("tooling", "README.md", text)
    assert selected == (sections[0], sections[1], sections[2], sections[4])
    assert selected[2].headings == ("TradingDev", "New workflow", "研究機器學習訊號")
    assert "Separate known ML section." not in "".join(
        section.text for section in selected
    )


def test_registered_strategy_docs_follow_whole_document_responsibility() -> None:
    text = "# Bundled strategy\n## Arbitrary details\nComplete contract.\n"
    path = "docs/strategies/quantile_strategy.md"
    for scope in ("strategy", "execution", "ml", "general"):
        assert sections_for_scope(scope, path, text) == markdown_sections(path, text)
    assert sections_for_scope("tooling", path, text) == ()
    assert (
        selection_reason("tooling", path, text) == "registered-document-responsibility"
    )
    unknown = "docs/strategies/new_strategy.md"
    assert sections_for_scope("tooling", unknown, text) == markdown_sections(
        unknown, text
    )


def test_invalid_scope_is_not_treated_as_empty_context() -> None:
    with pytest.raises(ValueError, match="Unknown documentation review scope"):
        sections_for_scope("typo", "README.md", "")
    with pytest.raises(ValueError, match="Unknown documentation review scope"):
        selection_reason("typo", "README.md", "# TradingDev\n")


@pytest.mark.parametrize(
    ("path", "owner", "heading", "scopes", "evidence"),
    [
        (
            "docs/development.md",
            "tooling",
            "測試設計與精簡",
            {"tooling", "presentation"},
            "mutation testing",
        ),
        (
            "docs/ml_research.md",
            "ml",
            "Dataset and Model Inputs",
            {"ml", "strategy", "interfaces"},
            "in_sample_accuracy",
        ),
        (
            "docs/ml_research.md",
            "ml",
            "Recorded Signal Evaluations",
            {"ml", "results", "interfaces"},
            "signal_fingerprint",
        ),
        (
            "docs/execution_plans.md",
            "execution",
            "Execution Plans and Confirmation Documents",
            {"execution", "results", "interfaces"},
            "confirmation_adjustment_requested",
        ),
        (
            "docs/performance_artifacts.md",
            "results",
            "Performance Scopes and Provenance",
            {"results", "interfaces"},
            "display_metrics",
        ),
        (
            "docs/reports.md",
            "results",
            "Composable Offline Reports",
            {"results", "interfaces"},
            "navigation_instructions",
        ),
    ],
)
def test_relocated_contracts_retain_complete_evidence_for_responsible_scopes(
    path: str, owner: str, heading: str, scopes: set[str], evidence: str
) -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / path).read_text(encoding="utf-8")
    contract = next(
        section
        for section in markdown_sections(path, text)
        if section.headings[-1] == heading
    )
    assert evidence in contract.text
    assert scope_for_path(path) == owner
    for scope in SCOPES:
        selected = sections_for_scope(scope, path, text)
        assert (contract in selected) == (scope in scopes or scope == "general")
        if scope == "general":
            assert "".join(section.text for section in selected) == text
    # New contracts must not silently inherit an old section's narrow ownership.
    expanded = text + "\n## New contract\nDo not hide new behavior.\n"
    for scope in SCOPES:
        assert "Do not hide new behavior." in "".join(
            section.text for section in sections_for_scope(scope, path, expanded)
        )


@pytest.mark.parametrize(
    ("path", "heading"),
    [
        ("ARCHITECTURE.md", "研究需求與有效設定"),
        ("docs/strategy_contract.md", "Research Inputs and Simulation Policy"),
        ("docs/run_artifacts.md", "Research Evidence Records"),
        ("docs/run_artifacts.md", "Research Settings and Provenance"),
        ("docs/run_artifacts.md", "Fixed Comparison Evidence"),
    ],
)
def test_research_contracts_reach_all_product_reviewers_without_tooling_duplication(
    path: str, heading: str
) -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / path).read_text(encoding="utf-8")
    contract = next(
        section
        for section in markdown_sections(path, text)
        if section.headings[-1] == heading
    )
    for scope in ("strategy", "execution", "results", "data", "ml", "interfaces"):
        assert contract in sections_for_scope(scope, path, text)
    for scope in ("tooling", "presentation"):
        assert contract not in sections_for_scope(scope, path, text)
