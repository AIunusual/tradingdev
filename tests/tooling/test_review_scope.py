"""Evidence selection retains exact text and conservatively handles new content."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from tooling.review_scope import (
    SCOPES,
    BehaviorContext,
    behavior_context_for_path,
    markdown_sections,
    scope_for_path,
    sections_for_reference,
    sections_for_scope,
    selection_reason,
    shared_document_context_ranges,
    unknown_document_sections,
)


@pytest.mark.parametrize(
    ("path", "headings"),
    [
        ("AGENTS.md", "# AGENTS.md\n\n## Issue 接手與指派\n"),
        ("AGENTS.md", "# AGENTS.md\n\n## Issue 撰寫\n"),
        (
            "docs/development.md",
            "# 開發指南\n\n## 接手 issue\n",
        ),
        (
            "docs/development.md",
            "# 開發指南\n\n## 撰寫 issue\n",
        ),
        (
            "docs/development.md",
            "# 開發指南\n\n## 環境與常用命令\n\n### 型別檢查與契約保護\n",
        ),
        (
            "docs/development.md",
            "# 開發指南\n\n## 環境與常用命令\n\n### 執行離線測試\n",
        ),
    ],
)
def test_development_guidance_is_retained_in_its_review_scope(
    path: str, headings: str
) -> None:
    text = headings + "Claim policy evidence.\n"
    assert "Claim policy evidence." in "".join(
        section.text for section in sections_for_scope("tooling", path, text)
    )
    assert "Claim policy evidence." not in "".join(
        section.text for section in sections_for_scope("results", path, text)
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
        ("tests/llm_plugin.py", "tooling"),
        ("tests/conftest.py", "general"),
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
        ("src/tradingdev/domain/presentation/confirmation_contract.py", "execution"),
        ("src/tradingdev/app/contracts/jobs.py", "execution"),
        ("src/tradingdev/mcp/workers/backtest.py", "execution"),
        ("tests/adapters/execution/test_worker_supervisor.py", "execution"),
        ("src/tradingdev/domain/performance/catalog.py", "results"),
        ("src/tradingdev/domain/statistics/periods.py", "results"),
        ("tests/domain/statistics/test_statistics_periods.py", "results"),
        ("research/mae_analysis.py", "tooling"),
        ("tests/test_mae_analysis.py", "tooling"),
        ("research/new_analysis.py", "general"),
        ("tests/test_new_analysis.py", "general"),
        ("src/tradingdev/shared/utils/json_values.py", "general"),
        ("src/tradingdev/adapters/storage/filesystem.py", "general"),
        ("src/tradingdev/adapters/storage/sqlite.py", "general"),
        ("src/tradingdev/domain/research_evidence.py", "general"),
        ("tests/conftest.py", "general"),
        ("tests/research_fixtures.py", "general"),
        ("tests/parallel_plugin.py", "general"),
        (".agents/skills/new-skill/SKILL.md", "general"),
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
        ("tooling/presentation_errors.py", "presentation"),
        ("tooling/presentation_html.py", "presentation"),
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
    ("path", "owner", "requirements", "document_scopes"),
    [
        (
            "src/tradingdev/mcp/tools/ml.py",
            "interfaces",
            {
                "FR-ML-001",
                "FR-ML-002",
                "FR-ML-003",
                "FR-ML-004",
                "FR-RES-001",
                "FR-RES-002",
                "FR-RES-003",
                "FR-RES-004",
                "NFR-DATA-001",
                "NFR-UX-001",
            },
            {"ml", "results"},
        ),
        (
            "src/tradingdev/app/contracts/ml.py",
            "ml",
            {
                "FR-ML-001",
                "FR-ML-002",
                "FR-ML-003",
                "FR-ML-004",
                "FR-RES-001",
                "FR-RES-002",
                "FR-RES-003",
                "FR-RES-004",
                "NFR-DATA-001",
                "NFR-UX-001",
            },
            {"ml", "results"},
        ),
        (
            "tests/app/test_ml_training_binding.py",
            "ml",
            {"FR-ML-002", "FR-ML-003", "FR-ML-004", "NFR-DATA-001"},
            {"ml"},
        ),
        (
            "src/tradingdev/domain/ml/registry/leaderboard_store.py",
            "ml",
            {"FR-ML-003", "FR-ML-004", "NFR-DATA-001"},
            {"ml"},
        ),
        (
            "src/tradingdev/adapters/reporting/ml.py",
            "results",
            {"FR-ML-004", "FR-RES-003", "FR-RES-004", "NFR-DATA-001", "NFR-UX-001"},
            {"ml", "results"},
        ),
        (
            "tests/e2e/test_llm_ml.py",
            "interfaces",
            {
                "FR-EXEC-005",
                "FR-ML-001",
                "FR-ML-002",
                "FR-ML-003",
                "FR-ML-004",
                "FR-RES-001",
                "FR-RES-002",
                "FR-RES-003",
                "FR-RES-004",
                "NFR-DATA-001",
                "NFR-UX-001",
            },
            {"ml", "results"},
        ),
    ],
)
def test_behavior_context_selects_obligations_without_changing_owner(
    path: str, owner: str, requirements: set[str], document_scopes: set[str]
) -> None:
    context = behavior_context_for_path(path)
    assert context is not None
    assert context == BehaviorContext(
        frozenset(requirements), frozenset(document_scopes)
    )
    assert scope_for_path(path) == owner
    assert "interfaces" not in context.document_scopes
    # These are SRS identities, not a second copy of their contract URLs.
    root = Path(__file__).resolve().parents[2]
    recorded_ids = {
        section.headings[-1].split()[0]
        for source in (root / "docs/requirements").glob("*.md")
        for section in markdown_sections(
            source.name, source.read_text(encoding="utf-8")
        )
        if section.headings
    }
    assert context.requirement_ids <= recorded_ids


@pytest.mark.parametrize(
    ("paths", "additional_requirements"),
    [
        (
            (
                "src/tradingdev/app/ml_service.py",
                "src/tradingdev/app/contracts/ml.py",
                "src/tradingdev/mcp/tools/ml.py",
                "tests/app/test_ml_service.py",
                "tests/mcp/test_ml_tools.py",
            ),
            {"FR-RES-003", "FR-RES-004"},
        ),
        (("tests/mcp/test_ml_leaderboard_runs.py",), set()),
        (("tests/adapters/test_ml_report_html.py",), {"FR-RES-003", "NFR-UX-001"}),
        (
            (
                "tests/adapters/test_report_cli.py",
                "tests/app/test_ml_report_service.py",
                "tests/mcp/test_ml_report_tools.py",
            ),
            {"FR-EXEC-004", "FR-RES-003"},
        ),
        (("tests/e2e/test_llm_ml.py",), {"FR-EXEC-005", "FR-RES-004"}),
    ],
    ids=["discovery", "run-ranking", "mixed-html", "comparison-report", "rolling"],
)
def test_mixed_ml_entrypoints_retain_saved_run_obligations(
    paths: tuple[str, ...], additional_requirements: set[str]
) -> None:
    # These modules also own run schemas, guidance, reads or report selections.
    # An edit to that side must retain its obligations despite the ML filename.
    required = {"FR-ML-004", "FR-RES-001", "FR-RES-002", "NFR-DATA-001"}
    for path in paths:
        context = behavior_context_for_path(path)
        assert context is not None, path
        assert required | additional_requirements <= context.requirement_ids, path
        assert context.document_scopes == {"ml", "results"}, path


@pytest.mark.parametrize(
    "path",
    [
        "src/tradingdev/mcp/tools/new_ml.py",
        "src/tradingdev/domain/ml/new_behavior.py",
        "tests/e2e/test_ml_new_behavior.py",
        "src/tradingdev/domain/research_extension.py",
        "src/tradingdev/app/research_preparation.py",
        "tests/app/test_research_preparation.py",
        "src/tradingdev/app/strategy_service.py",
        "src/tradingdev/app/contracts/research.py",
        "src/tradingdev/app/artifact_service.py",
        "src/tradingdev/app/conversation_extension.py",
        "src/tradingdev/mcp/tools/new_history.py",
        "tests/mcp/test_new_data_contracts.py",
        "docs/ml_research.md",
        "docs/reports.md",
        "docs/requirements/ml.md",
        "src/tradingdev/mcp/tools/../tools/ml.py",
        "src//tradingdev/mcp/tools/ml.py",
        "/src/tradingdev/mcp/tools/ml.py",
    ],
)
def test_unregistered_or_noncanonical_behavior_keeps_conservative_fallback(
    path: str,
) -> None:
    assert behavior_context_for_path(path) is None


def test_submitted_research_test_retains_its_additional_workflow_obligations() -> None:
    storage = behavior_context_for_path(
        "src/tradingdev/adapters/storage/submitted_research.py"
    )
    workflow = behavior_context_for_path("tests/app/test_submitted_research.py")
    assert storage is not None and workflow is not None
    assert storage.requirement_ids < workflow.requirement_ids
    assert {"FR-EXEC-007", "FR-RES-003"} <= workflow.requirement_ids
    assert "FR-EXEC-007" not in storage.requirement_ids
    assert scope_for_path("tests/app/test_submitted_research.py") == "general"


def test_unknown_context_preserves_original_subtree_and_actual_ancestors() -> None:
    path = "README.md"
    text = (
        "Prelude.\r\n# TradingDev\r\nReader context.\r\n"
        "## 研究機器學習訊號\r\nFirst occurrence.\r\n"
        "## 使用 Docker 部署\r\nUnrelated deployment.\r\n"
        "## 研究機器學習訊號\r\nActual parent.\r\n"
        "### New provenance contract\r\nNew [evidence][source].\r\n"
        "#### New child\r\nChild evidence.\r\n"
        "## 常見問題\r\n[source]: ./ml_research.md\r\n"
    )
    selected = unknown_document_sections(path, text)
    original = "".join(section.text for section in selected)
    assert "Prelude.\r\n# TradingDev\r\nReader context." in original
    assert "Actual parent.\r\n### New provenance contract" in original
    assert "#### New child\r\nChild evidence." in original
    assert "[source]: ./ml_research.md" in original
    assert "First occurrence." not in original
    assert "Unrelated deployment." not in original
    assert set(selected) <= set(markdown_sections(path, text))


@pytest.mark.parametrize("path", ["docs/new_contract.md", "README.md"])
def test_unknown_or_unstructured_document_retains_complete_original(path: str) -> None:
    text = "Unstructured original\r\nwith unchanged bytes.\r\n"
    assert "".join(s.text for s in unknown_document_sections(path, text)) == text


def test_known_document_does_not_expand_unrelated_registered_sections() -> None:
    text = "# TradingDev\nIntro.\n## 使用 Docker 部署\nKnown unrelated content.\n"
    assert unknown_document_sections("README.md", text) == ()
    # Whole-document responsibilities are handled by their own selectors. SRS
    # requirements must be parsed as records instead of all headings being unknown.
    assert unknown_document_sections("docs/requirements/ml.md", "# New SRS\n") == ()


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
    ]
    + [
        (path, "tooling", document, heading)
        for path in ("research/mae_analysis.py", "tests/test_mae_analysis.py")
        for document, heading in (
            ("docs/development.md", "MAE 研究原型"),
            ("ARCHITECTURE.md", "Repository 開發工具與研究原型"),
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


@pytest.mark.parametrize(
    ("scope", "document", "heading", "paths"),
    [
        (
            "interfaces",
            "README.md",
            "終端機用法",
            (
                "src/tradingdev/__main__.py",
                "src/tradingdev/main.py",
                "tests/adapters/test_cli.py",
                "tests/adapters/test_cli_execution_guidance.py",
                "tests/adapters/test_cli_report.py",
            ),
        ),
        (
            "interfaces",
            "docs/reports.md",
            "Codex CLI recording adapter",
            (
                "src/tradingdev/adapters/clients/codex_history.py",
                "src/tradingdev/adapters/clients/codex_protocol.py",
                "src/tradingdev/adapters/clients/codex_session.py",
                "src/tradingdev/adapters/clients/codex_transport.py",
                "tests/adapters/test_codex_recording.py",
            ),
        ),
        (
            "data",
            "ARCHITECTURE.md",
            "行情快取",
            (
                "src/tradingdev/app/dataset_inspection.py",
                "src/tradingdev/app/dvol_data.py",
                "src/tradingdev/app/feature_data.py",
                "tests/app/test_dvol_cache.py",
            ),
        ),
        (
            "execution",
            "docs/execution_plans.md",
            "Execution Plans and Confirmation Documents",
            (
                "src/tradingdev/app/comparison_service.py",
                "src/tradingdev/app/prepared_execution.py",
                "src/tradingdev/app/research_preparation.py",
                "src/tradingdev/domain/comparison.py",
                "src/tradingdev/domain/presentation/comparison_confirmation.py",
                "src/tradingdev/domain/presentation/research_confirmation.py",
                "tests/app/test_comparison_lifecycle.py",
                "tests/app/test_comparison_plans.py",
                "tests/app/test_comparison_preflight.py",
                "tests/app/test_research_preparation.py",
                "tests/app/test_rolling_preflight.py",
                "tests/app/test_signal_boundaries.py",
                "tests/domain/test_comparison.py",
                "tests/domain/test_comparison_confirmation.py",
                "tests/domain/test_research_confirmation.py",
            ),
        ),
        (
            "execution",
            "ARCHITECTURE.md",
            "容器執行 adapter",
            (
                "src/tradingdev/domain/runtime.py",
                "tests/adapters/test_container_runtime.py",
                "tests/adapters/test_frame_codec.py",
            ),
        ),
        (
            "execution",
            "docs/strategy_contract.md",
            "Randomness",
            (
                "src/tradingdev/domain/randomness.py",
                "tests/domain/test_randomness.py",
            ),
        ),
        (
            "results",
            "docs/performance_artifacts.md",
            "Benchmarks",
            (
                "src/tradingdev/app/benchmark_service.py",
                "src/tradingdev/domain/benchmark.py",
                "tests/app/test_benchmark_service.py",
                "tests/domain/test_benchmark.py",
            ),
        ),
        (
            "results",
            "docs/run_artifacts.md",
            "Ranking Saved Run Scopes",
            (
                "src/tradingdev/app/ranking_service.py",
                "src/tradingdev/domain/ranking.py",
                "tests/domain/test_ranking.py",
            ),
        ),
        (
            "results",
            "docs/reports.md",
            "Research interpretation and diagnostic providers",
            (
                "src/tradingdev/app/diagnostic_service.py",
                "src/tradingdev/domain/diagnostics.py",
                "tests/app/test_diagnostics.py",
            ),
        ),
        (
            "results",
            "docs/reports.md",
            "Research notebooks and archives",
            (
                "src/tradingdev/adapters/storage/research_archives.py",
                "src/tradingdev/adapters/storage/research_notes.py",
                "src/tradingdev/app/contracts/research_notes.py",
                "src/tradingdev/app/research_note_service.py",
                "src/tradingdev/domain/research_notes.py",
                "tests/app/test_research_notes.py",
            ),
        ),
        (
            "results",
            "docs/reports.md",
            "Recorded conversations",
            (
                "src/tradingdev/adapters/storage/conversation_exports.py",
                "src/tradingdev/adapters/storage/conversations.py",
                "src/tradingdev/app/contracts/conversations.py",
                "src/tradingdev/app/conversation_service.py",
                "src/tradingdev/domain/conversations.py",
                "tests/app/test_conversations.py",
            ),
        ),
        (
            "results",
            "docs/run_artifacts.md",
            "Fixed Comparison Evidence",
            (
                "src/tradingdev/adapters/storage/comparisons.py",
                "src/tradingdev/domain/comparison_results.py",
                "tests/adapters/test_comparisons.py",
                "tests/app/test_comparison_report.py",
            ),
        ),
        (
            "results",
            "docs/performance_artifacts.md",
            "Performance Scopes and Provenance",
            (
                "src/tradingdev/app/contracts/research.py",
                "src/tradingdev/app/rolling_trace_summary.py",
                "src/tradingdev/domain/research_scope.py",
                "tests/app/test_metric_json_contract.py",
                "tests/app/test_saved_walk_forward.py",
                "tests/app/test_volume_monthly_loss_artifacts.py",
                "tests/domain/test_research_walk_forward.py",
            ),
        ),
        (
            "results",
            "docs/reports.md",
            "Composable Offline Reports",
            (
                "tests/adapters/test_bokeh_tables.py",
                "tests/adapters/test_dashboard_analysis.py",
                "tests/adapters/test_dashboard_render.py",
                "tests/adapters/test_dashboard_report.py",
                "tests/adapters/test_presentation_html.py",
                "tests/adapters/test_quantstats_report.py",
                "tests/adapters/test_research_report.py",
                "tests/app/test_research_report_service.py",
                "tests/domain/test_ledger_presentation.py",
            ),
        ),
        (
            "presentation",
            "docs/development.md",
            "簡報網站發布設定",
            (
                "tooling/presentation_bundle.py",
                "scripts/stage-presentations.sh",
                "scripts/build-presentations-site.sh",
                "wrangler.jsonc",
                "tests/tooling/test_presentation_bundle.py",
            ),
        ),
        (
            "tooling",
            "docs/development.md",
            "研究成果工程回饋",
            (".agents/skills/research-feedback/SKILL.md",),
        ),
        (
            "tooling",
            "docs/development.md",
            "2. 在 GitHub 建立 PR，交由審閱者檢查",
            (".agents/skills/wait-for-check/SKILL.md",),
        ),
    ],
)
def test_known_components_receive_their_actual_contracts(
    scope: str, document: str, heading: str, paths: tuple[str, ...]
) -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / document).read_text(encoding="utf-8")
    required = next(
        section
        for section in markdown_sections(document, text)
        if section.headings and section.headings[-1] == heading
    )
    for path in paths:
        assert (root / path).is_file()
        owner = scope_for_path(path)
        assert owner == scope, path
        assert required in sections_for_scope(owner, document, text), path


@pytest.mark.parametrize(
    ("path", "headings", "scopes"),
    [
        (
            "README.md",
            (
                "使用 Docker 部署",
                "準備檔案與資料夾",
                "設定連線入口",
                "啟動、更新與保存資料",
            ),
            {"tooling", "execution", "interfaces"},
        ),
        (
            "README.md",
            (
                "保存研究筆記與打包分享",
                "自動保存 Codex CLI 對話",
                "閱讀保存的對話",
                "從終端機操作研究筆記",
            ),
            {"results", "interfaces"},
        ),
        (
            "README.md",
            (
                "行情來源與下載",
                "選擇與保存工作區",
                "升級舊工作區",
            ),
            {"strategy", "execution", "results", "data", "ml", "interfaces"},
        ),
        (
            "README.md",
            ("沒有 Homebrew 時設定 OpenMP（macOS）",),
            {"tooling", "interfaces", "ml"},
        ),
        (
            "ARCHITECTURE.md",
            ("行情快取",),
            {"data", "execution", "ml", "interfaces"},
        ),
        (
            "ARCHITECTURE.md",
            (
                "診斷擴充與替換",
                "研究筆記與可攜封存",
                "可見對話紀錄",
            ),
            {"results", "interfaces"},
        ),
        (
            "docs/development.md",
            (
                "平行執行離線測試",
                "SRS 維護與檢核",
                "測試證據與文件修正",
                "檢查期間的主線更新",
            ),
            {"tooling", "presentation"},
        ),
        (
            "docs/development.md",
            (
                "簡報網站發布設定",
                "Cloudflare 自動發布",
                "共用網域與專案路徑",
            ),
            {"presentation"},
        ),
        (
            "docs/development.md",
            ("研究成果工程回饋",),
            {"tooling"},
        ),
        (
            "docs/ml_research.md",
            ("Rolling Retraining Schemes",),
            {"ml", "strategy", "execution", "interfaces"},
        ),
        (
            "docs/reports.md",
            (
                "Research interpretation and diagnostic providers",
                "Research notebooks and archives",
                "Recorded conversations",
                "Codex CLI recording adapter",
            ),
            {"results", "interfaces"},
        ),
    ],
)
def test_registered_operational_sections_reach_their_responsible_reviewers(
    path: str, headings: tuple[str, ...], scopes: set[str]
) -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / path).read_text(encoding="utf-8")
    sections = markdown_sections(path, text)
    for heading in headings:
        required = next(s for s in sections if s.headings[-1:] == (heading,))
        for scope in SCOPES:
            selected = sections_for_scope(scope, path, text)
            assert (required in selected) == (scope in scopes or scope == "general")
    expanded = text + "\n## Unregistered behavior\nRetain complete new evidence.\n"
    for scope in SCOPES:
        assert "Retain complete new evidence." in "".join(
            s.text for s in sections_for_scope(scope, path, expanded)
        )


def test_editor_guide_is_scoped_but_product_overviews_remain_shared() -> None:
    root = Path(__file__).resolve().parents[2]
    editor = "docs/talks/tradingdev-introduction/README.md"
    for path in (
        editor,
        "docs/talks/tradingdev-introduction/slides.md",
        "docs/requirements.md",
    ):
        text = (root / path).read_text(encoding="utf-8")
        for scope in SCOPES:
            selected = "".join(s.text for s in sections_for_scope(scope, path, text))
            expected = (
                text
                if path != editor or scope in {"presentation", "tooling", "general"}
                else ""
            )
            assert selected == expected


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


def test_shared_document_context_uses_complete_commonmark_sections() -> None:
    prelude = "Shared prelude.\n\n"
    fenced = "# Guide\n```md\n[fake]: https://example.com/fake\n```\n"
    definitions = (
        "## References\nDefinitions apply to every section.\n\n"
        "[mode]:\n  https://example.com/simulation\n  'Simulation'\n"
    )
    duplicate = "## Duplicate\n[MODE]: https://example.com/duplicate\n"
    text = prelude + fenced + definitions + duplicate + "## Final\nOrdinary prose.\n"
    sections = markdown_sections("context.md", text)
    ranges = shared_document_context_ranges("context.md", text)
    assert [
        section.text
        for section in sections
        if (section.start_line, section.end_line) in ranges
    ] == [prelude, definitions, duplicate]


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
            "ARCHITECTURE.md",
            "general",
            "容器執行 adapter",
            {"strategy", "execution", "interfaces", "tooling"},
            "runtime.recover_abandoned()",
        ),
        (
            "docs/development.md",
            "tooling",
            "容器建置與驗證",
            {"strategy", "execution", "interfaces", "tooling"},
            "TRADINGDEV_RUNTIME_LIMITS",
        ),
        (
            "docs/strategy_contract.md",
            "strategy",
            "Security Model",
            {"strategy", "execution", "interfaces", "tooling"},
            "UID/GID",
        ),
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
            {"strategy", "execution", "results", "interfaces"},
            "confirmation_adjustment_requested",
        ),
        (
            "docs/performance_artifacts.md",
            "results",
            "Metric Interpretation",
            {"strategy", "execution", "results", "interfaces"},
            "periods_per_year",
        ),
        (
            "docs/run_artifacts.md",
            "results",
            "Execution Manifest",
            {"strategy", "execution", "results", "interfaces"},
            "Empty\nobjects preserve existing mappings",
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
def test_registered_contracts_retain_complete_evidence_for_responsible_scopes(
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


@pytest.mark.parametrize(
    ("path", "heading", "scopes", "evidence"),
    [
        (
            "docs/strategy_contract.md",
            "Signal and Execution Semantics",
            {"strategy", "execution", "results", "interfaces"},
            "Stop-loss takes priority",
        ),
        (
            "docs/strategy_contract.md",
            "Draft Cleanup",
            {"strategy", "execution", "results", "interfaces"},
            "Ambiguous history",  # Matched case-insensitively below.
        ),
        (
            "docs/run_artifacts.md",
            "Job State and Worker Identity",
            {"execution", "results", "interfaces"},
            "potentially reused PID",
        ),
        (
            "docs/run_artifacts.md",
            "Market Data Cache",
            {"data", "results", "interfaces"},
            "successful empty ones",
        ),
        (
            "docs/run_artifacts.md",
            "Automatic DVOL Queries",
            {"data", "results", "interfaces"},
            "unqueried interval",
        ),
        (
            "docs/run_artifacts.md",
            "Research Notes and Archives",
            {"results", "interfaces"},
            "do not modify runs",
        ),
        (
            "docs/run_artifacts.md",
            "Reading Artifact Delivery",
            {"execution", "results", "interfaces"},
            "anyone holding one can read those bytes until expiry",
        ),
    ],
)
def test_focused_contract_sections_retain_semantics_without_unrelated_context(
    path: str, heading: str, scopes: set[str], evidence: str
) -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / path).read_text(encoding="utf-8")
    section = next(
        part for part in markdown_sections(path, text) if part.headings[-1] == heading
    )
    assert evidence.lower() in " ".join(section.text.lower().split())
    for scope in SCOPES:
        selected = sections_for_scope(scope, path, text)
        assert (section in selected) == (scope in scopes or scope == "general")


def test_quantile_scoring_references_retain_reader_specific_contracts() -> None:
    root = Path(__file__).resolve().parents[2]
    guide_path = "docs/strategies/quantile_strategy.md"
    guide_text = (root / guide_path).read_text(encoding="utf-8")
    contract_path = "docs/strategy_contract.md"
    contract_text = (root / contract_path).read_text(encoding="utf-8")
    for scope in ("strategy", "execution", "ml", "interfaces"):
        guide = "".join(
            s.text for s in sections_for_scope(scope, guide_path, guide_text)
        )
        contract = "".join(
            s.text for s in sections_for_scope(scope, contract_path, contract_text)
        )
        assert "malformed output fails" in contract
        assert "feature gaps and chunks" in guide
    assert not sections_for_scope("tooling", guide_path, guide_text)


@pytest.mark.parametrize("path", ["docs/strategy_contract.md", "docs/run_artifacts.md"])
def test_contract_headings_have_reviewed_ownership(path: str) -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / path).read_text(encoding="utf-8")
    for scope in SCOPES:
        if scope != "general":
            assert (
                selection_reason(scope, path, text)
                == "registered-heading-responsibility"
            )


def test_explicit_contract_subtree_and_duplicate_anchor() -> None:
    text = (
        "Intro\r\n# Contract\r\nContext.\r\n## Repeat\r\nFirst.\r\n"
        "### Child\r\nFirst child.\r\n## Repeat\r\nSecond.\r\n"
        "### Child\r\nSecond child.\r\n## Tail\r\nUnrelated.\r\n"
    )
    selected = sections_for_reference("docs/contract.md", text, "repeat-1")
    original = "".join(section.text for section in selected)
    assert "Intro\r\n# Contract\r\nContext." in original
    assert "Second.\r\n### Child\r\nSecond child." in original
    assert "First." not in original and "Unrelated." not in original
    assert (
        "".join(section.text for section in sections_for_reference("x", text, ""))
        == text
    )
    with pytest.raises(ValueError, match="Missing SRS contract heading"):
        sections_for_reference("docs/contract.md", text, "deleted")


def test_srs_registry_drives_change_ownership_and_context() -> None:
    from tooling.requirements import DOCUMENT_SCOPES

    for path, scopes in DOCUMENT_SCOPES.items():
        assert set(scopes) <= SCOPES.keys()
        assert scope_for_path(path) == scopes[0]
        for scope in SCOPES:
            parts = sections_for_scope(scope, path, "# Requirements\nContent.\n")
            assert bool(parts) == (scope == "general" or scope in scopes)


@pytest.mark.parametrize("container", [">", "-"])
@pytest.mark.parametrize("fragment", ["repeat", "repeat-1"])
def test_contract_reference_uses_link_checker_container_anchors(
    tmp_path: Path, container: str, fragment: str
) -> None:
    from tooling.documentation import check_local_links

    text = (
        f"# Contract\n\n{container} ## Repeat\n\n"
        "## Repeat\nactual second\n\n## Tail\nunrelated\n"
    )
    contract = tmp_path / "contract.md"
    contract.write_text(text)
    source = tmp_path / "README.md"
    source.write_text(f"[contract](contract.md#{fragment})\n")
    check_local_links(tmp_path, [source])
    parts = sections_for_reference("contract.md", text, fragment)
    selected = "".join(part.text for part in parts)
    assert "unrelated" not in selected
    if fragment == "repeat-1":
        assert "actual second" in selected
    else:
        assert f"{container} ## Repeat" in selected
        assert "actual second" not in selected


def test_first_line_container_reference_does_not_expand_the_document() -> None:
    prelude = "> ## Quote\n> cited evidence\n\n"
    text = prelude + "# Actual\nunrelated actual\n\n## Tail\nunrelated tail\n"
    parts = sections_for_reference("contract.md", text, "quote")
    assert "".join(part.text for part in parts) == prelude
