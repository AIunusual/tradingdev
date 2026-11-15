"""Scope selection must preserve immutable changes and enforce input budgets."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from tooling import review_docs, review_plan
from tooling.presentations import DECK, PRESENTATION_PATHS
from tooling.requirements import check_requirements
from tooling.review_evidence import ReviewError, build_evidence
from tooling.review_plan import build_review_plan
from tooling.review_scope import markdown_sections

STRATEGY = "src/tradingdev/app/strategy_service.py"
# Synthetic ownership tests deliberately use an unregistered consumer. Reviewed
# services have real SRS obligations, which the real-document tests provide.
REPORT = "src/tradingdev/app/report_consumer.py"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _write(repo: Path, path: str, text: str) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _tree(repo: Path) -> str:
    _git(repo, "add", ".")
    return _git(repo, "write-tree")


@pytest.fixture
def change(tmp_path: Path) -> tuple[Path, str, str]:
    _git(tmp_path, "init", "-q")
    _write(tmp_path, "README.md", "# TradingDev\nUser-facing outcomes.\n")
    _write(tmp_path, STRATEGY, "VALUE = 'old strategy'\n")
    _write(tmp_path, REPORT, "VALUE = 'old report'\n")
    _write(tmp_path, "src/context.py", "CONTRACT = 'saved context'\n")
    base = _tree(tmp_path)
    _write(tmp_path, STRATEGY, "VALUE = 'new strategy'\n")
    _write(tmp_path, REPORT, "VALUE = 'new report'\n")
    return tmp_path, base, _tree(tmp_path)


def _payload(prompt: str) -> dict[str, Any]:
    return dict(json.loads(prompt.rsplit("\n\n", 1)[-1]))


def test_model_options_review_preserves_source_and_development_guidance(
    tmp_path: Path,
) -> None:
    _git(tmp_path, "init", "-q")
    path = "tests/llm_plugin.py"
    _write(tmp_path, path, "MODEL = 'old'\n")
    _write(
        tmp_path,
        "docs/development.md",
        "# 開發指南\n\n## 選擇性 LLM 測試\nModel selection contract.\n",
    )
    _write(
        tmp_path,
        "README.md",
        "# TradingDev\n\n## 查看結果與圖表\nReport-only instructions.\n",
    )
    base = _tree(tmp_path)
    _write(tmp_path, path, "MODEL = 'new'\n")
    plan = review_docs.prepare_review(base, _tree(tmp_path), repo=tmp_path)
    assert plan.manifest["selected_scopes"] == ["tooling"]
    batch = _payload(plan.prompts[0])
    assert batch["assigned_paths"] == [path]
    assert batch["candidate_files"][path] == "MODEL = 'new'\n"
    assert "+MODEL = 'new'" in batch["path_diffs"][path]
    assert "Model selection contract." in plan.prompts[0]
    assert "Report-only instructions." not in plan.prompts[0]


@pytest.mark.parametrize(
    ("path", "scope", "required", "excluded"),
    [
        (
            "src/tradingdev/domain/statistics/periods.py",
            "results",
            "Saved numeric semantics.",
            "Developer model setup.",
        ),
        (
            "tests/domain/statistics/test_statistics_periods.py",
            "results",
            "Saved numeric semantics.",
            "Developer model setup.",
        ),
        (
            "research/mae_analysis.py",
            "tooling",
            "MAE date and complete-window semantics.",
            "Saved numeric semantics.",
        ),
        (
            "tests/test_mae_analysis.py",
            "tooling",
            "MAE date and complete-window semantics.",
            "Saved numeric semantics.",
        ),
    ],
)
def test_numeric_analysis_review_preserves_its_source_and_contract(
    tmp_path: Path, path: str, scope: str, required: str, excluded: str
) -> None:
    _git(tmp_path, "init", "-q")
    _write(tmp_path, path, "VALUE = 'old'\n")
    _write(
        tmp_path,
        "docs/performance_artifacts.md",
        "# Performance Artifacts\n\n## Metric Interpretation\n"
        "Saved numeric semantics.\n",
    )
    _write(
        tmp_path,
        "docs/development.md",
        "# 開發指南\n\n## MAE 研究原型\n"
        "MAE date and complete-window semantics.\n\n"
        "## 選擇性 LLM 測試\nDeveloper model setup.\n",
    )
    base = _tree(tmp_path)
    source = "VALUE = 'new'\n"
    _write(tmp_path, path, source)
    plan = review_docs.prepare_review(base, _tree(tmp_path), repo=tmp_path)
    assert plan.manifest["selected_scopes"] == [scope]
    batch = _payload(plan.prompts[0])
    assert batch["assigned_paths"] == [path]
    assert batch["candidate_files"][path] == source
    assert "+VALUE = 'new'" in batch["path_diffs"][path]
    assert required in plan.prompts[0]
    assert excluded not in plan.prompts[0]


@pytest.mark.parametrize(
    "path",
    [
        "src/tradingdev/app/execution_plan_service.py",
        "src/tradingdev/app/artifact_service.py",
        "src/tradingdev/mcp/artifact_downloads.py",
    ],
)
def test_artifact_delivery_review_retains_contract_and_complete_source(
    tmp_path: Path, path: str
) -> None:
    _git(tmp_path, "init", "-q")
    _write(tmp_path, path, "VALUE = 'old'\n")
    contract = (
        "# Run Artifacts\n\n## MCP Lookup\n\n### Reading Artifact Delivery\n"
        "Hash-bound bearer URLs expire without changing saved artifacts.\n"
    )
    _write(tmp_path, "docs/run_artifacts.md", contract)
    base = _tree(tmp_path)
    source = "VALUE = 'new'\n"
    _write(tmp_path, path, source)
    plan = review_docs.prepare_review(base, _tree(tmp_path), repo=tmp_path)
    batch = _payload(plan.prompts[0])
    assert batch["candidate_files"][path] == source
    assert "+VALUE = 'new'" in batch["path_diffs"][path]
    selected = "".join(
        section["text"]
        for section in batch["document_sections"]
        if section["path"] == "docs/run_artifacts.md"
    )
    assert "### Reading Artifact Delivery\n" in selected
    assert contract.split("### Reading Artifact Delivery\n", 1)[1] in selected


def test_each_changed_file_has_one_complete_owner_and_one_cross_scope_audit(
    change: tuple[Path, str, str],
) -> None:
    repo, base, tree = change
    raw = build_evidence(repo, base, tree)
    original = json.loads(raw)
    plan = build_review_plan(raw)
    payloads = [_payload(p) for p in plan.prompts]
    assigned = [p for b in payloads for p in b["assigned_paths"]]
    assert sorted(assigned) == [REPORT, STRATEGY]
    assert len(assigned) == len(set(assigned))
    for batch in payloads:
        for path in batch["assigned_paths"]:
            assert batch["candidate_files"][path] == original["candidate_files"][path]
            assert batch["path_diffs"][path] == original["path_diffs"][path]
            assert batch["candidate_file_blobs"][path] == _git(
                repo, "rev-parse", f"{tree}:{path}"
            )
        for section in batch.get("document_sections", []):
            lines = original["candidate_files"][section["path"]].splitlines(True)
            assert section["text"] == "".join(
                lines[section["start_line"] - 1 : section["end_line"]]
            )
    audit = [b for b in payloads if b["phase"] == "cross_scope"]
    assert len(audit) == 1
    assert audit[0]["path_diffs"] == original["path_diffs"]
    assert audit[0]["candidate_files"] == {}
    assert "README.md" in audit[0]["document_inventory"]
    assert all(b["candidate_tree"] == tree for b in payloads)
    assert plan.manifest["status"] == "planned_not_reviewed"


@pytest.mark.parametrize("change_deck", [False, True])
def test_required_deck_context_is_distinguished_from_actual_changes(
    tmp_path: Path, change_deck: bool
) -> None:
    _git(tmp_path, "init", "-q")
    for path in PRESENTATION_PATHS:
        _write(tmp_path, path, "Existing presentation content.\n")
    compiler = "tooling/presentations.py"
    bundle = "tooling/presentation_bundle.py"
    quality = "tooling/quality.py"
    for path in (compiler, bundle, quality):
        _write(tmp_path, path, "VALUE = 'old'\n")
    base = _tree(tmp_path)
    for path in (compiler, bundle, quality):
        _write(tmp_path, path, "VALUE = 'new'\n")
    changed = {compiler, bundle, quality}
    if change_deck:
        _write(tmp_path, f"{DECK}/slides.md", "Changed slide.\n")
        changed.add(f"{DECK}/slides.md")
    tree = _tree(tmp_path)
    _write(tmp_path, f"{DECK}/theme.css", "UNSTAGED_STYLE\n")
    plan = review_docs.prepare_review(base, tree, repo=tmp_path)
    for prompt in plan.prompts:
        payload = _payload(prompt)
        assert set(payload["changed_paths"]) == changed
        assert set(payload["unchanged_presentation_context_paths"]) == (
            PRESENTATION_PATHS - changed
        )
        assert "UNSTAGED_STYLE" not in prompt
        if payload["scope"] == "presentation":
            assert set(payload["assigned_paths"]) == PRESENTATION_PATHS | {
                compiler,
                bundle,
            }
            assert set(payload["path_diffs"]) == changed - {quality}
            for path in PRESENTATION_PATHS:
                assert (
                    payload["candidate_files"][path]
                    == _git(tmp_path, "show", f"{tree}:{path}") + "\n"
                )


def test_operation_batches_split_without_truncating_changed_files(
    tmp_path: Path,
) -> None:
    _git(tmp_path, "init", "-q")
    for path in PRESENTATION_PATHS:
        _write(tmp_path, path, "Original deck context\n")
    operation = "tooling/presentations.py"
    test = "tests/tooling/test_presentations.py"
    before = {
        operation: "def setup(root):\n    pass\n"
        + "# stable operation context\n" * 500,
        test: "EXPECTED = 'old'\n" + "# stable test context\n" * 600,
    }
    for path, text in before.items():
        _write(tmp_path, path, text)
    base = _tree(tmp_path)
    after = {
        operation: before[operation].replace("pass", "install(root)"),
        test: before[test].replace("'old'", "'new'"),
    }
    for path, text in after.items():
        _write(tmp_path, path, text)
    evidence = build_evidence(tmp_path, base, _tree(tmp_path))
    plan = build_review_plan(evidence, max_prompt_bytes=30_000)
    batches = [_payload(prompt) for prompt in plan.prompts]
    owners = [batch for batch in batches if batch["scope"] == "presentation"]
    assert len(owners) > 1
    for batch in batches:
        assert batch["candidate_files"][f"{DECK}/README.md"] == (
            "Original deck context\n"
        )
    for path, content in after.items():
        assigned = [batch for batch in owners if path in batch["assigned_paths"]]
        assert len(assigned) == 1
        assert assigned[0]["candidate_files"][path] == content
        assert (
            assigned[0]["path_diffs"][path] == json.loads(evidence)["path_diffs"][path]
        )
    assert all(len(prompt.encode()) <= 30_000 for prompt in plan.prompts)
    assert not any(
        "Original deck context" in section["text"]
        for batch in batches
        for section in batch["document_sections"]
        if section["path"] == f"{DECK}/slides.md"
    )
    audit = next(batch for batch in batches if batch["scope"] == "cross_scope")
    assert audit["path_diffs"] == json.loads(evidence)["path_diffs"]


@pytest.fixture
def context_heavy_change(tmp_path: Path) -> tuple[Path, str, str, str]:
    _git(tmp_path, "init", "-q")
    source = "tests/tooling/test_example.py"
    _write(tmp_path, source, "VALUE = 'old'\n" + "# Stable source context\n" * 180)
    _write(tmp_path, "src/shared_context.py", "SHARED = 'original context'\n")
    _write(tmp_path, "src/scoped_context.py", "SCOPED = 'original context'\n")
    _write(
        tmp_path,
        "docs/unregistered_contract.md",
        "# Unregistered contract\n\n"
        + "".join(f"## Section {i}\n" + str(i) * 6_000 + "\n\n" for i in range(3)),
    )
    base = _tree(tmp_path)
    _write(tmp_path, source, "VALUE = 'new'\n" + "# Stable source context\n" * 180)
    return tmp_path, base, _tree(tmp_path), source


def test_context_partitions_repeat_complete_sources_and_cover_original_sections(
    context_heavy_change: tuple[Path, str, str, str],
) -> None:
    repo, base, tree, source = context_heavy_change
    evidence = build_evidence(
        repo,
        base,
        tree,
        review_context=("src/shared_context.py",),
        review_context_for=(("tooling", "src/scoped_context.py"),),
    )
    original = json.loads(evidence)
    whole = build_review_plan(evidence)
    expected_sections = _payload(whole.prompts[0])["document_sections"]
    assert (
        "".join(section["text"] for section in expected_sections)
        == (original["candidate_files"]["docs/unregistered_contract.md"])
    )
    plan = build_review_plan(evidence, max_prompt_bytes=20_000)
    batches = [_payload(prompt) for prompt in plan.prompts]
    parts = [batch for batch in batches if batch["phase"] == "scope"]
    assert len(parts) > 1
    assert plan.manifest["path_owners"] == {source: "tooling"}
    assert len(plan.manifest["source_units"]) == 1
    assert plan.manifest["source_units"][0]["document_sections"] == [
        batch["document_sections"][index]
        for batch in plan.manifest["batches"]
        if batch["phase"] == "scope"
        for index in batch["assigned_document_sections"]
    ]
    assert [
        part["document_sections"][index]
        for part in parts
        for index in part["assigned_document_sections"]
    ] == expected_sections
    for index, part in enumerate(parts, start=1):
        assert part["context_partition"] == {"index": index, "count": len(parts)}
        assert part["assigned_paths"] == [source]
        assert part["path_diffs"] == original["path_diffs"]
        for path in (source, "src/shared_context.py", "src/scoped_context.py"):
            assert part["candidate_files"][path] == original["candidate_files"][path]
        assert part["candidate_tree_inventory"] == original["candidate_tree_inventory"]
    audit = [batch for batch in batches if batch["phase"] == "cross_scope"]
    assert len(audit) == 1
    assert audit[0]["path_diffs"] == original["path_diffs"]
    assert "docs/unregistered_contract.md" in audit[0]["document_inventory"]
    assert all(len(prompt.encode()) <= 20_000 for prompt in plan.prompts)
    assert plan.manifest["evidence_utf8_bytes"] == whole.manifest["evidence_utf8_bytes"]
    source_originals = (
        "".join(
            original["candidate_files"][path]
            for path in (source, "src/shared_context.py", "src/scoped_context.py")
        )
        + original["path_diffs"][source]
    )
    unique_originals = (
        source_originals + original["candidate_files"]["docs/unregistered_contract.md"]
    )
    # Each extra partition repeats its complete sources and the contract prelude;
    # the cross-scope audit repeats the full diff and global explicit context.
    repeated_originals = (
        (source_originals + "# Unregistered contract\n\n") * (len(parts) - 1)
        + original["candidate_files"]["src/shared_context.py"]
        + original["path_diffs"][source]
    )
    assert audit[0]["document_sections"] == []
    for suffix in ("utf8_bytes", "characters"):

        def measure(text: str, unit: str = suffix) -> int:
            return len(text.encode("utf-8")) if unit == "utf8_bytes" else len(text)

        assert plan.manifest[f"evidence_{suffix}"] == measure(unique_originals)
        assert plan.manifest[f"repeated_original_{suffix}"] == measure(
            repeated_originals
        )
        assert plan.manifest[f"delivered_original_{suffix}"] == measure(
            unique_originals + repeated_originals
        )
        overheads = []
        for batch, entry, prompt in zip(
            batches, plan.manifest["batches"], plan.prompts, strict=True
        ):
            originals = (
                source_originals
                + "".join(section["text"] for section in batch["document_sections"])
                if batch["phase"] == "scope"
                else original["candidate_files"]["src/shared_context.py"]
                + original["path_diffs"][source]
            )
            assert entry[f"original_{suffix}"] == measure(originals)
            overhead = measure(prompt) - measure(originals)
            assert entry[f"prompt_overhead_{suffix}"] == overhead > 0
            overheads.append(overhead)
        assert plan.manifest[f"prompt_overhead_{suffix}"] == sum(overheads)
        assert plan.manifest[f"total_prompt_{suffix}"] == sum(
            measure(prompt) for prompt in plan.prompts
        )
    assert "another partition" in plan.prompts[0]
    assert "missing original context" in plan.prompts[0]
    plan.verify()
    with pytest.raises(ReviewError, match="Total review input"):
        build_review_plan(
            evidence,
            max_prompt_bytes=20_000,
            max_total_prompt_bytes=plan.manifest["total_prompt_utf8_bytes"] - 1,
        )


@pytest.mark.parametrize("oversized", ["source", "section", "explicit_context"])
def test_context_partitions_reject_indivisible_originals(
    context_heavy_change: tuple[Path, str, str, str], oversized: str
) -> None:
    repo, base, _, source = context_heavy_change
    current_source = (repo / source).read_text(encoding="utf-8")
    if oversized == "source":
        _write(repo, source, "VALUE = 'new'\n" + "# Long complete source\n" * 1_000)
    elif oversized == "section":
        _write(repo, "docs/unregistered_contract.md", "# One section\n" + "X" * 22_000)
    else:
        _write(repo, "src/shared_context.py", "# Complete explicit context\n" * 1_000)
    if oversized != "source":
        # Keep the oversized contract unchanged so failure tests context
        # indivisibility, rather than a newly assigned changed-document source.
        _write(repo, source, _git(repo, "show", f"{base}:{source}") + "\n")
        base = _tree(repo)
        _write(repo, source, current_source)
    evidence = build_evidence(
        repo, base, _tree(repo), review_context=("src/shared_context.py",)
    )
    with pytest.raises(ReviewError, match="cannot fit.*review budget"):
        build_review_plan(evidence, max_prompt_bytes=20_000)


def test_context_partitions_keep_the_complete_mandatory_deck_group(
    context_heavy_change: tuple[Path, str, str, str],
) -> None:
    repo, _, _, _ = context_heavy_change
    for path in PRESENTATION_PATHS:
        _write(repo, path, "Original complete deck content.\n")
    base = _tree(repo)
    _write(repo, f"{DECK}/theme.css", "/* Updated complete theme */\n")
    evidence = build_evidence(repo, base, _tree(repo))
    original = json.loads(evidence)
    plan = build_review_plan(evidence, max_prompt_bytes=20_000)
    parts = [
        _payload(prompt)
        for prompt in plan.prompts
        if _payload(prompt)["phase"] == "scope"
    ]
    assert len(parts) > 1
    assert len(plan.manifest["source_units"]) == 1
    for part in parts:
        assert set(part["assigned_paths"]) == PRESENTATION_PATHS
        assert part["path_diffs"] == original["path_diffs"]
        assert all(
            part["candidate_files"][path] == original["candidate_files"][path]
            for path in PRESENTATION_PATHS
        )


@pytest.mark.parametrize("parent_size", [800, 5_000])
def test_context_partitions_keep_selected_ancestor_prose_with_each_child(
    context_heavy_change: tuple[Path, str, str, str], parent_size: int
) -> None:
    repo, base, _, source = context_heavy_change
    current_source = (repo / source).read_text(encoding="utf-8")
    _write(repo, source, _git(repo, "show", f"{base}:{source}") + "\n")
    _write(
        repo,
        "docs/unregistered_contract.md",
        "# Contract\nROOT_PREMISE\n\n"
        + "".join(
            f"## Repeated parent\nPARENT_{i}_PREMISE\n"
            + "P" * parent_size
            + f"\n\n### Child {i}\nCHILD_{i}_OBLIGATION\n"
            + "C" * 6_000
            + "\n\n"
            for i in range(2)
        ),
    )
    base = _tree(repo)
    _write(repo, source, current_source)
    evidence = build_evidence(repo, base, _tree(repo))
    if parent_size == 5_000:
        with pytest.raises(ReviewError, match="original document section"):
            build_review_plan(evidence, max_prompt_bytes=20_000)
        return
    plan = build_review_plan(evidence, max_prompt_bytes=20_000)
    parts = [
        _payload(prompt)
        for prompt in plan.prompts
        if _payload(prompt)["phase"] == "scope"
    ]
    assert len(parts) > 1
    for child in range(2):
        owners = [
            part
            for part in parts
            if any(
                f"CHILD_{child}_OBLIGATION" in section["text"]
                for section in part["document_sections"]
            )
        ]
        assert owners
        for part in owners:
            text = "".join(section["text"] for section in part["document_sections"])
            assert "ROOT_PREMISE" in text
            assert f"PARENT_{child}_PREMISE" in text


@pytest.mark.parametrize(
    ("source_lines", "section_size", "budget", "merged"),
    [
        pytest.param(80, 6_000, 26_000, True, id="shared-context-saves-input"),
        pytest.param(180, 6_000, 20_000, False, id="joint-originals-cannot-fit"),
        pytest.param(860, 12_000, 63_700, False, id="fewer-requests-cost-more-bytes"),
    ],
)
def test_partitioned_sources_share_context_only_when_serialized_input_decreases(
    tmp_path: Path, source_lines: int, section_size: int, budget: int, merged: bool
) -> None:
    _git(tmp_path, "init", "-q")
    paths = ("tests/tooling/test_one.py", "tooling/two.py")
    for path in paths:
        _write(
            tmp_path,
            path,
            "VALUE = 'old'\n" + "# Stable source context\n" * source_lines,
        )
    document = "# Contract\n\n" + "".join(
        f"## Section {index}\n" + str(index) * section_size + "\n\n"
        for index in range(3)
    )
    _write(tmp_path, "docs/unregistered.md", document)
    base = _tree(tmp_path)
    for path in paths:
        _write(
            tmp_path,
            path,
            "VALUE = 'new'\n" + "# Stable source context\n" * source_lines,
        )
    evidence = build_evidence(tmp_path, base, _tree(tmp_path))
    original = json.loads(evidence)
    plan = build_review_plan(evidence, max_prompt_bytes=budget)
    assert len(plan.manifest["source_units"]) == (1 if merged else 2)
    for unit in plan.manifest["source_units"]:
        parts = [
            _payload(prompt)
            for prompt in plan.prompts
            if _payload(prompt).get("source_unit") == unit["id"]
        ]
        assert len(parts) > 1
        assigned_text = "".join(
            part["document_sections"][index]["text"]
            for part in parts
            for index in part["assigned_document_sections"]
        )
        assert assigned_text == document
        for part in parts:
            assert part["assigned_paths"] == (
                list(paths) if merged else unit["assigned_paths"]
            )
            for path in part["assigned_paths"]:
                assert (
                    part["candidate_files"][path] == original["candidate_files"][path]
                )
                assert part["path_diffs"][path] == original["path_diffs"][path]
    assert all(len(prompt.encode()) <= budget for prompt in plan.prompts)
    audit = _payload(plan.prompts[-1])
    assert audit["phase"] == "cross_scope"
    assert audit["path_diffs"] == original["path_diffs"]


@pytest.mark.parametrize("definition_parent_size", [0, 5_000])
def test_context_partitions_keep_prelude_and_later_reference_dependencies(
    context_heavy_change: tuple[Path, str, str, str],
    monkeypatch: pytest.MonkeyPatch,
    definition_parent_size: int,
) -> None:
    repo, base, _, source = context_heavy_change
    current_source = (repo / source).read_text(encoding="utf-8")
    _write(repo, source, _git(repo, "show", f"{base}:{source}") + "\n")
    path = "docs/unregistered_contract.md"
    _write(
        repo,
        path,
        "ALL GUIDANCE IS SIMULATION ONLY\n\n"
        + "".join(
            f"## Section {index}\nUse [policy][mode].\n" + str(index) * 6_000 + "\n\n"
            for index in range(3)
        )
        + "# Reference appendix\nDEFINITION_PARENT_PREMISE\n"
        + "P" * definition_parent_size
        + "\n\n### Mode definition\n[mode]: https://example.com/simulation\n",
    )
    base = _tree(repo)
    _write(repo, source, current_source)
    evidence = build_evidence(repo, base, _tree(repo))
    if definition_parent_size:
        with pytest.raises(ReviewError, match="original document section"):
            build_review_plan(evidence, max_prompt_bytes=20_000)
        return
    plan = build_review_plan(evidence, max_prompt_bytes=20_000)
    parts = [
        _payload(prompt)
        for prompt in plan.prompts
        if _payload(prompt)["phase"] == "scope"
    ]
    assert len(parts) > 1
    for part in parts:
        text = "".join(section["text"] for section in part["document_sections"])
        assert "ALL GUIDANCE IS SIMULATION ONLY" in text
        assert "[mode]: https://example.com/simulation" in text
        assert "DEFINITION_PARENT_PREMISE" in text
    assert (
        "".join(
            part["document_sections"][index]["text"]
            for part in parts
            for index in part["assigned_document_sections"]
        )
        == json.loads(evidence)["candidate_files"][path]
    )
    plan.verify()
    manifest_part = next(
        part for part in plan.manifest["batches"] if part["phase"] == "scope"
    )
    definition_index = next(
        index
        for index, section in enumerate(manifest_part["document_sections"])
        if section["headings"][-1:] == ("Mode definition",)
    )
    assert definition_index not in manifest_part["assigned_document_sections"]
    manifest_part["document_sections"].pop(definition_index)
    manifest_part["assigned_document_sections"] = [
        index - (index > definition_index)
        for index in manifest_part["assigned_document_sections"]
    ]
    monkeypatch.setattr(
        review_plan, "_digest", lambda _value: plan.manifest["plan_hash"]
    )
    with pytest.raises(ReviewError, match="required original"):
        plan.verify()


@pytest.mark.parametrize("nested_definition", [False, True])
def test_requirement_partitions_keep_original_hierarchy_and_selected_parent(
    tmp_path: Path,
    nested_definition: bool,
) -> None:
    _git(tmp_path, "init", "-q")
    source = "src/tradingdev/app/strategy_new.py"
    source_before = "VALUE = 'old'\n" + "# Stable context\n" * (
        500 if nested_definition else 700
    )
    _write(tmp_path, source, source_before)
    _write(tmp_path, "docs/contract.md", "# Contract\nORIGINAL_CONTRACT\n")
    requirement_path = "docs/requirements/strategy.md"
    _write(
        tmp_path,
        requirement_path,
        "# Strategy requirements\nSRS_PARENT_PREMISE\n"
        + "P" * 1_800
        + "\n\n"
        + "".join(
            f"## FR-STR-{index:03d} Requirement\n\n"
            f"- **需求**：Keep obligation {index}.\n- **驗收**："
            + str(index)
            * 6_000
            + "\n- **契約**：[contract](../contract.md)\n"
            "- **驗證**：人工驗收：inspect the original.\n\n"
            + (
                "### Link definitions\n[mode]: https://example.com/simulation\n"
                if nested_definition and index == 3
                else ""
            )
            for index in range(1, 4)
        ),
    )
    base = _tree(tmp_path)
    _write(tmp_path, source, source_before.replace("'old'", "'new'"))
    evidence = build_evidence(tmp_path, base, _tree(tmp_path))
    plan = build_review_plan(
        evidence, max_prompt_bytes=35_000 if nested_definition else 30_000
    )
    parts = [
        _payload(prompt)
        for prompt in plan.prompts
        if _payload(prompt)["phase"] == "scope"
    ]
    assert len(parts) > 1
    observed = set()
    for part in parts:
        sections = [
            section
            for section in part["document_sections"]
            if section["path"] == requirement_path
        ]
        requirements = [
            section
            for section in sections
            if section["selection_reason"] == "affected-requirement"
        ]
        if requirements:
            assert any("SRS_PARENT_PREMISE" in section["text"] for section in sections)
        if nested_definition:
            shared_records = [
                section for section in requirements if section.get("shared_context")
            ]
            assert shared_records
            assert all(
                section["headings"][-1] == "FR-STR-003 Requirement"
                and "### Link definitions" in section["text"]
                and "[mode]: https://example.com/simulation" in section["text"]
                for section in shared_records
            )
        for section in requirements:
            observed.add(tuple(section["headings"]))
    assert observed == {
        ("Strategy requirements", f"FR-STR-{index:03d} Requirement")
        for index in range(1, 4)
    }
    plan.verify()


@pytest.mark.parametrize("missing", ["ownership", "section", "partition"])
def test_context_partition_manifest_rejects_incomplete_coverage(
    context_heavy_change: tuple[Path, str, str, str],
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    repo, base, tree, _ = context_heavy_change
    plan = build_review_plan(build_evidence(repo, base, tree), max_prompt_bytes=20_000)
    parts = [batch for batch in plan.manifest["batches"] if batch["phase"] == "scope"]
    assert len(parts) > 1
    if missing == "ownership":
        plan.manifest["source_units"][0]["assigned_paths"] = []
    elif missing == "section":
        assert parts[0]["document_sections"]
        parts[0]["document_sections"].pop()
    else:
        plan.manifest["batches"].remove(parts[0])
    # Isolate semantic coverage validation from the separately tested hash guard.
    monkeypatch.setattr(
        review_plan, "_digest", lambda _value: plan.manifest["plan_hash"]
    )
    with pytest.raises(ReviewError, match="ownership|original sections|partition"):
        plan.verify()


def test_explicit_scope_cannot_hide_a_changed_responsibility(
    change: tuple[Path, str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, base, tree = change
    monkeypatch.setattr(
        review_docs, "_run", lambda *_a, **_kw: pytest.fail("model ran")
    )
    with pytest.raises(ReviewError, match="uncovered.*report_consumer"):
        review_docs.review(base, tree, repo=repo, review_scopes=("strategy",))
    plan = review_docs.prepare_review(
        base, tree, repo=repo, review_scopes=("strategy", "results", "data")
    )
    assert plan.manifest["selected_scopes"] == ["data", "results", "strategy"]
    assert any(b["scope"] == "data" for b in plan.manifest["batches"])


def test_additional_context_uses_git_content_not_dirty_working_file(
    change: tuple[Path, str, str],
) -> None:
    repo, base, tree = change
    _write(repo, "src/context.py", "UNSTAGED_SECRET = True\n")
    plan = review_docs.prepare_review(
        base, tree, repo=repo, review_context=("src/context.py",)
    )
    for prompt in plan.prompts:
        assert "UNSTAGED_SECRET" not in prompt
        assert _payload(prompt)["candidate_files"]["src/context.py"] == (
            "CONTRACT = 'saved context'\n"
        )


@pytest.mark.parametrize("path", ["../outside.py", "/tmp/secret.py", "missing.py"])
def test_unavailable_context_fails_before_any_model(
    change: tuple[Path, str, str], monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    repo, base, tree = change
    monkeypatch.setattr(
        review_docs, "_run", lambda *_a, **_kw: pytest.fail("model ran")
    )
    with pytest.raises(ReviewError, match="candidate Git tree"):
        review_docs.review(base, tree, repo=repo, review_context=(path,))


def test_deleted_and_renamed_paths_keep_exact_old_and_new_evidence(
    tmp_path: Path,
) -> None:
    _git(tmp_path, "init", "-q")
    _write(tmp_path, "scripts/old.sh", "echo old-source\n")
    base = _tree(tmp_path)
    (tmp_path / "scripts/old.sh").rename(tmp_path / "scripts/new.sh")
    tree = _tree(tmp_path)
    plan = review_docs.prepare_review(base, tree, repo=tmp_path)
    payloads = [_payload(p) for p in plan.prompts]
    owners = {p: b for b in payloads for p in b["assigned_paths"]}
    assert set(owners) == {"scripts/old.sh", "scripts/new.sh"}
    assert (
        "-echo old-source" in owners["scripts/old.sh"]["path_diffs"]["scripts/old.sh"]
    )
    assert "scripts/old.sh" not in owners["scripts/old.sh"]["candidate_files"]
    assert owners["scripts/old.sh"]["base_file_blobs"]["scripts/old.sh"] == _git(
        tmp_path, "rev-parse", f"{base}:scripts/old.sh"
    )
    assert owners["scripts/new.sh"]["candidate_files"]["scripts/new.sh"] == (
        "echo old-source\n"
    )


def test_every_scope_can_identify_deletions_from_the_candidate_tree(
    tmp_path: Path,
) -> None:
    _git(tmp_path, "init", "-q")
    _write(tmp_path, "web/index.html", "<p>Generated preview</p>\n")
    _write(tmp_path, "README.md", "# Preview\nGenerated HTML is tracked.\n")
    base = _tree(tmp_path)
    (tmp_path / "web/index.html").unlink()
    _write(tmp_path, ".gitignore", "/web/index.html\n")
    _write(tmp_path, "README.md", "# Preview\nGenerated HTML is not tracked.\n")
    _write(tmp_path, STRATEGY, "VALUE = 'separate review scope'\n")
    tree = _tree(tmp_path)
    plan = review_docs.prepare_review(base, tree, repo=tmp_path)
    payloads = [_payload(prompt) for prompt in plan.prompts]
    assert len({batch["scope"] for batch in payloads}) > 1
    expected = _git(tmp_path, "ls-tree", "-r", "--name-only", tree).splitlines()
    for batch in payloads:
        assert batch["candidate_tree_inventory"] == expected
        assert "web/index.html" in batch["changed_paths"]
        assert "web/index.html" not in batch["candidate_tree_inventory"]
        assert ".gitignore" in batch["candidate_tree_inventory"]


def test_path_diffs_treat_git_pathspec_characters_literally(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _write(tmp_path, "src/a[1].py", "OLD_BRACKETS\n")
    _write(tmp_path, "src/a1.py", "OLD_PLAIN\n")
    base = _tree(tmp_path)
    _write(tmp_path, "src/a[1].py", "NEW_BRACKETS\n")
    _write(tmp_path, "src/a1.py", "NEW_PLAIN\n")
    tree = _tree(tmp_path)
    evidence = json.loads(build_evidence(tmp_path, base, tree))
    assert "NEW_BRACKETS" in evidence["path_diffs"]["src/a[1].py"]
    assert "NEW_PLAIN" not in evidence["path_diffs"]["src/a[1].py"]
    assert "NEW_PLAIN" in evidence["path_diffs"]["src/a1.py"]


@pytest.mark.parametrize("option", ["max_prompt_bytes", "max_total_prompt_bytes"])
def test_context_budget_overflow_calls_no_model(
    change: tuple[Path, str, str], monkeypatch: pytest.MonkeyPatch, option: str
) -> None:
    repo, base, tree = change
    monkeypatch.setattr(
        review_docs, "_run", lambda *_a, **_kw: pytest.fail("model ran")
    )
    with pytest.raises(ReviewError, match="budget|Total review input"):
        review_docs.review(
            base,
            tree,
            repo=repo,
            max_prompt_bytes=100 if option == "max_prompt_bytes" else None,
            max_total_prompt_bytes=100 if option == "max_total_prompt_bytes" else None,
        )


def test_original_sizes_distinguish_versions_overlaps_and_repeated_delivery() -> None:
    changed = "docs/changed.md"
    contract = "docs/contract.md"
    raw = {
        "candidate_files": {changed: '新\n"\\\n', contract: "甲\n乙\n丙\n"},
        "base_document_files": {changed: "舊\n", contract: "甲\n乙\n丙\n"},
    }

    def section(path: str, version: str, start: int, end: int) -> dict[str, Any]:
        files = raw[
            "candidate_files" if version == "candidate" else "base_document_files"
        ]
        return {
            "path": path,
            "version": version,
            "blob_oid": "same-contract-blob" if path == contract else version,
            "start_line": start,
            "end_line": end,
            "text": "".join(files[path].splitlines(keepends=True)[start - 1 : end]),
        }

    first = {
        "candidate_files": {changed: raw["candidate_files"][changed]},
        "path_diffs": {changed: "-舊\n+新\n"},
        "document_sections": [
            section(changed, "candidate", 1, 1),
            section(changed, "base", 1, 1),
            section(contract, "candidate", 1, 2),
            section(contract, "base", 1, 1),
        ],
    }
    second = first | {
        "document_sections": [
            section(contract, "candidate", 2, 3),
            section(contract, "candidate", 2, 2),
        ]
    }
    # Unique originals: changed candidate 7/5, diff 10/6, old changed 4/2,
    # candidate contract line union 12/6, and old contract 4/2 (bytes/chars).
    unique = review_plan._selected_evidence_size([first, second], raw)
    assert (unique.utf8_bytes, unique.characters) == (37, 21)
    # Actual deliveries also count the full-file/slice and slice/slice overlaps.
    delivered = [
        review_plan._text_size(review_plan._original_texts(batch))
        for batch in (first, second)
    ]
    assert [(size.utf8_bytes, size.characters) for size in delivered] == [
        (37, 21),
        (29, 17),
    ]


def test_manifest_is_deterministic_bound_to_prompts_and_reports_real_sizes(
    change: tuple[Path, str, str],
) -> None:
    repo, base, tree = change
    first = review_docs.prepare_review(base, tree, repo=repo)
    second = review_docs.prepare_review(base, tree, repo=repo)
    assert first == second
    assert first.manifest["total_prompt_utf8_bytes"] == sum(
        len(prompt.encode("utf-8")) for prompt in first.prompts
    )
    first.verify()
    first.manifest["candidate_tree"] = "modified"
    with pytest.raises(ReviewError, match="changed after preparation"):
        first.verify()


def test_missing_path_diff_is_rejected_instead_of_reporting_complete_coverage(
    change: tuple[Path, str, str],
) -> None:
    repo, base, tree = change
    evidence = json.loads(build_evidence(repo, base, tree))
    del evidence["path_diffs"][REPORT]
    with pytest.raises(ReviewError, match="every changed path"):
        build_review_plan(json.dumps(evidence))


def test_plan_only_cli_works_without_codex_and_does_not_claim_review_passed(
    change: tuple[Path, str, str], tmp_path: Path
) -> None:
    repo, base, tree = change
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tooling.review_docs",
            "--repo",
            str(repo),
            "--base",
            base,
            "--tree",
            tree,
            "--plan-only",
            "--review-context-for=strategy:src/context.py",
            "--review-context-for=cross_scope:README.md",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env=os.environ
        | {
            "TRADINGDEV_CODEX_BIN": "/not/installed/codex",
            "PYTHONDONTWRITEBYTECODE": "1",
            "XDG_CACHE_HOME": str(tmp_path / "cache"),
        },
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)
    assert manifest["status"] == "planned_not_reviewed"
    assert manifest["candidate_tree"] == tree
    assert manifest["scoped_context_paths"] == {
        "strategy": ["src/context.py"],
        "cross_scope": ["README.md"],
    }
    expected = review_docs.prepare_review(
        base,
        tree,
        repo=repo,
        review_context_for=(
            ("strategy", "src/context.py"),
            ("cross_scope", "README.md"),
        ),
    ).manifest
    for prefix in (
        "evidence",
        "delivered_original",
        "repeated_original",
        "prompt_overhead",
        "total_prompt",
    ):
        for suffix in ("utf8_bytes", "characters"):
            key = f"{prefix}_{suffix}"
            assert manifest[key] == expected[key] > 0
    for batch, expected_batch in zip(
        manifest["batches"], expected["batches"], strict=True
    ):
        for prefix in ("original", "prompt_overhead"):
            for suffix in ("utf8_bytes", "characters"):
                key = f"{prefix}_{suffix}"
                assert batch[key] == expected_batch[key] > 0
    assert "Documentation review passed" not in result.stdout


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize("separator", ["\u2028", "\u0085"])
def test_git_document_slices_preserve_commonmark_lines(
    change: tuple[Path, str, str], newline: str, separator: str
) -> None:
    repo, _base, _tree_id = change
    prelude = newline.join(["# TradingDev", f"前言{separator}仍是同一行", "", ""])
    faq = newline.join(["## 常見問題", "其他內容", ""])
    _write(repo, "README.md", prelude + faq)
    base = _tree(repo)
    _write(repo, STRATEGY, "VALUE = 'another strategy'\n")
    plan = review_docs.prepare_review(base, _tree(repo), repo=repo)
    sections = [
        section
        for section in _payload(plan.prompts[0])["document_sections"]
        if section["path"] == "README.md"
    ]
    assert [(s["start_line"], s["end_line"], s["text"]) for s in sections] == [
        (1, 3, prelude),
        (4, 5, faq),
    ]


@pytest.mark.parametrize("path", ["contract.txt", "uv.lock", "schema.custom"])
def test_explicit_context_can_add_text_outside_automatic_selection(
    change: tuple[Path, str, str], path: str
) -> None:
    repo, base, _tree_id = change
    _write(repo, path, "version = 1\n")
    tree = _tree(repo)
    automatic = review_docs.prepare_review(base, tree, repo=repo)
    assert path in automatic.manifest["excluded_from_content"]
    plan = review_docs.prepare_review(base, tree, repo=repo, review_context=(path,))
    assert path not in plan.manifest["excluded_from_content"]
    owners = []
    for prompt in plan.prompts:
        payload = _payload(prompt)
        assert payload["candidate_files"][path] == "version = 1\n"
        if path in payload["assigned_paths"]:
            owners.append(payload)
            assert "+version = 1" in payload["path_diffs"][path]
    assert len(owners) == 1


@pytest.mark.parametrize(
    ("path", "content"),
    [("invalid.txt", b"\xff"), ("nul.txt", b"before\0after")],
)
def test_explicit_context_rejects_nontext_before_model(
    change: tuple[Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    content: bytes,
) -> None:
    repo, base, _tree_id = change
    (repo / path).write_bytes(content)
    tree = _tree(repo)
    monkeypatch.setattr(
        review_docs, "_run", lambda *_a, **_kw: pytest.fail("model ran")
    )
    with pytest.raises(ReviewError, match="non-UTF-8|binary data"):
        review_docs.review(base, tree, repo=repo, review_context=(path,))


@pytest.mark.parametrize(
    "path", ["workspace/contract.txt", ".venv/contract.txt", "link.txt"]
)
def test_explicit_context_keeps_runtime_and_symlink_restrictions(
    change: tuple[Path, str, str],
    path: str,
) -> None:
    repo, base, _tree_id = change
    if path == "link.txt":
        (repo / path).symlink_to("src/context.py")
    else:
        _write(repo, path, "private context\n")
    with pytest.raises(ReviewError, match="regular non-runtime"):
        review_docs.prepare_review(base, _tree(repo), repo=repo, review_context=(path,))


def test_scoped_originals_are_isolated_and_preserve_other_document_sections(
    change: tuple[Path, str, str],
) -> None:
    repo, base, tree = change
    original = json.loads(
        build_evidence(repo, base, tree, review_context=("src/context.py",))
    )
    plan = review_docs.prepare_review(
        base,
        tree,
        repo=repo,
        review_context=("src/context.py",),
        review_context_for=(("strategy", "README.md"), ("cross_scope", "README.md")),
    )
    payloads = [_payload(prompt) for prompt in plan.prompts]
    for batch, entry in zip(payloads, plan.manifest["batches"], strict=True):
        assert (
            batch["candidate_files"]["src/context.py"]
            == original["candidate_files"]["src/context.py"]
        )
        assert entry["candidate_file_blobs"] == batch["candidate_file_blobs"]
        if batch["scope"] in {"strategy", "cross_scope"}:
            assert (
                batch["candidate_files"]["README.md"]
                == original["candidate_files"]["README.md"]
            )
            assert not any(
                s["path"] == "README.md" for s in batch.get("document_sections", [])
            )
        else:
            assert "README.md" not in batch["candidate_files"]
            assert any(
                s["path"] == "README.md"
                and s["text"] == original["candidate_files"]["README.md"]
                for s in batch["document_sections"]
            )
    assert payloads[-1]["path_diffs"] == original["path_diffs"]
    assert plan.manifest["scoped_context_paths"] == {
        "cross_scope": ["README.md"],
        "strategy": ["README.md"],
    }
    other = review_docs.prepare_review(
        base,
        tree,
        repo=repo,
        review_context=("src/context.py",),
        review_context_for=(("results", "README.md"), ("cross_scope", "README.md")),
    )
    assert plan.manifest["evidence_hash"] != other.manifest["evidence_hash"]
    assert plan.manifest["plan_hash"] != other.manifest["plan_hash"]
    plan.verify()


def test_cross_scope_context_forces_audit_for_single_scope(
    change: tuple[Path, str, str],
) -> None:
    repo, base, _ = change
    _write(repo, REPORT, "VALUE = 'old report'\n")
    tree = _tree(repo)
    plan = review_docs.prepare_review(
        base, tree, repo=repo, review_context_for=(("cross_scope", "src/context.py"),)
    )
    assert [batch["scope"] for batch in plan.manifest["batches"]] == [
        "strategy",
        "cross_scope",
    ]
    assert (
        _payload(plan.prompts[-1])["path_diffs"]
        == json.loads(build_evidence(repo, base, tree))["path_diffs"]
    )


@pytest.mark.parametrize("scope", ["unknown", "data"])
def test_invalid_or_unselected_scoped_context_fails_before_model(
    change: tuple[Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
) -> None:
    repo, base, tree = change
    monkeypatch.setattr(
        review_docs, "_run", lambda *_a, **_kw: pytest.fail("model ran")
    )
    with pytest.raises(ReviewError, match="known scope|unselected scopes"):
        review_docs.review(
            base, tree, repo=repo, review_context_for=((scope, "src/context.py"),)
        )
    if scope == "data":
        plan = review_docs.prepare_review(
            base,
            tree,
            repo=repo,
            review_scopes=("strategy", "results", "data"),
            review_context_for=((scope, "src/context.py"),),
        )
        assert any(batch["scope"] == scope for batch in plan.manifest["batches"])


@pytest.mark.parametrize(
    "path", ["../escape.py", "/tmp/secret.py", "missing.py", "src/context-link.py"]
)
def test_scoped_context_rejects_unsafe_or_unavailable_git_paths(
    change: tuple[Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    repo, base, _ = change
    (repo / "src/context-link.py").symlink_to("context.py")
    tree = _tree(repo)
    monkeypatch.setattr(
        review_docs, "_run", lambda *_a, **_kw: pytest.fail("model ran")
    )
    with pytest.raises(ReviewError, match="candidate Git tree"):
        review_docs.review(
            base, tree, repo=repo, review_context_for=(("strategy", path),)
        )


def test_scoped_missing_blob_and_capacity_fail_before_model(
    change: tuple[Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base, tree = change
    monkeypatch.setattr(
        review_docs, "_run", lambda *_a, **_kw: pytest.fail("model ran")
    )
    raw = json.loads(
        build_evidence(
            repo, base, tree, review_context_for=(("strategy", "src/context.py"),)
        )
    )
    del raw["candidate_files"]["src/context.py"]
    with pytest.raises(ReviewError, match="context is missing"):
        build_review_plan(json.dumps(raw))
    with pytest.raises(ReviewError, match="cannot fit"):
        review_docs.review(
            base,
            tree,
            repo=repo,
            review_context_for=(("strategy", "src/context.py"),),
            max_prompt_bytes=1,
        )


@pytest.mark.parametrize("missing", ["candidate_file_blobs", "candidate_file_modes"])
def test_scoped_context_requires_blob_identity_and_mode(
    change: tuple[Path, str, str],
    missing: str,
) -> None:
    repo, base, tree = change
    raw = json.loads(
        build_evidence(
            repo, base, tree, review_context_for=(("strategy", "src/context.py"),)
        )
    )
    del raw[missing]["src/context.py"]
    with pytest.raises(ReviewError, match="blob identity or mode"):
        build_review_plan(json.dumps(raw))


def test_scoped_changed_extra_retains_owner_and_cross_scope_diff(
    change: tuple[Path, str, str],
) -> None:
    repo, base, _ = change
    _write(repo, "plain.txt", "new explicit context\n")
    tree = _tree(repo)
    plan = review_docs.prepare_review(
        base, tree, repo=repo, review_context_for=(("strategy", "plain.txt"),)
    )
    payloads = [_payload(prompt) for prompt in plan.prompts]
    owners = [batch for batch in payloads if "plain.txt" in batch["assigned_paths"]]
    assert len(owners) == 1
    assert owners[0]["scope"] == "general"
    assert owners[0]["candidate_files"]["plain.txt"] == "new explicit context\n"
    assert "plain.txt" in payloads[-1]["path_diffs"]


@pytest.mark.parametrize("target", ["strategy", "cross_scope"])
@pytest.mark.parametrize("global_context", [False, True])
@pytest.mark.parametrize("changed_document", [False, True])
def test_extra_markdown_does_not_escape_its_scope_unless_independently_selected(
    change: tuple[Path, str, str],
    target: str,
    global_context: bool,
    changed_document: bool,
) -> None:
    repo, _, _ = change
    path = "notes.md"
    _write(repo, path, "# Notes\nSCOPED_ONLY_ORIGINAL_MARKER\n")
    base = _tree(repo)
    _write(repo, STRATEGY, "VALUE = 'later strategy'\n")
    _write(repo, REPORT, "VALUE = 'later report'\n")
    if changed_document:
        _write(repo, path, "# Notes\nSCOPED_ONLY_ORIGINAL_MARKER\nChanged notes.\n")
    tree = _tree(repo)
    plan = review_docs.prepare_review(
        base,
        tree,
        repo=repo,
        review_scopes=("strategy", "results", "tooling", "general"),
        review_context=(path,) if global_context else (),
        review_context_for=((target, path),),
    )
    original = _git(repo, "show", f"{tree}:{path}") + "\n"
    for prompt in plan.prompts:
        batch = _payload(prompt)
        if (
            global_context
            or batch["scope"] == target
            or path in batch["assigned_paths"]
        ):
            assert batch["candidate_files"][path] == original
        elif not changed_document:
            assert path not in batch["candidate_files"]
            assert "SCOPED_ONLY_ORIGINAL_MARKER" not in prompt
        if not changed_document and not global_context:
            assert path not in batch.get("document_inventory", {})
            assert not any(
                section["path"] == path
                for section in batch.get("document_sections", [])
            )
    audit = _payload(plan.prompts[-1])
    raw = json.loads(build_evidence(repo, base, tree))
    assert audit["path_diffs"] == raw["path_diffs"]
    assert (path in audit["path_diffs"]) == changed_document
    assert path not in plan.manifest["automatic_document_paths"]


@pytest.mark.parametrize("corruption", ["missing", "omitted", "extra", "duplicate"])
def test_automatic_document_provenance_fails_closed(
    change: tuple[Path, str, str],
    corruption: str,
) -> None:
    repo, base, tree = change
    raw = json.loads(build_evidence(repo, base, tree))
    if corruption == "missing":
        del raw["automatic_document_paths"]
    elif corruption == "omitted":
        raw["automatic_document_paths"] = []
    elif corruption == "extra":
        raw["automatic_document_paths"].append("src/context.py")
    else:
        raw["automatic_document_paths"].append("README.md")
    with pytest.raises(ReviewError, match="document provenance"):
        build_review_plan(json.dumps(raw))


@pytest.mark.parametrize("scope", ["strategy", "cross_scope"])
def test_srs_explicit_contract_sections_come_from_candidate_git(
    change: tuple[Path, str, str],
    scope: str,
) -> None:
    repo, base, _ = change
    contract = "docs/run_artifacts.md"
    original = "# Run Artifacts\n\n## Market Data Cache\nOriginal data contract.\n"
    _write(
        repo, contract, original + "\n## Discovery Pages\nUnrelated contract text.\n"
    )
    base = _tree(repo)
    _write(repo, REPORT, "VALUE = 'cross-scope report change'\n")
    _write(
        repo,
        "docs/requirements/strategy.md",
        "# Strategy\n\n## FR-STR-001 Inputs\n\n"
        "- **需求**：Use cached input.\n- **驗收**：Preserve its identity.\n"
        "- **契約**：[cache](../run_artifacts.md#market-data-cache)\n"
        "- **驗證**：人工驗收：inspect the preserved evidence.\n",
    )
    tree = _tree(repo)
    # Working tree changes must not replace immutable contract evidence.
    _write(repo, contract, "# Locally changed\n")
    raw = build_evidence(repo, base, tree)
    plan = build_review_plan(raw)
    batch = next(
        _payload(prompt)
        for prompt in plan.prompts
        if _payload(prompt).get("scope") == scope
    )
    sections = [
        section for section in batch["document_sections"] if section["path"] == contract
    ]
    assert "".join(section["text"] for section in sections) == original + "\n"
    assert all(
        section["blob_oid"] == _git(repo, "rev-parse", f"{tree}:{contract}")
        for section in sections
    )
    assert contract in batch["requirement_context_paths"][scope]
    assert len({section["start_line"] for section in sections}) == len(sections)
    evidence = json.loads(raw)
    del evidence["candidate_files"][contract]
    with pytest.raises(ReviewError, match="context is missing"):
        build_review_plan(json.dumps(evidence))


@pytest.mark.parametrize("mode", ["missing", "symlink"])
def test_missing_or_nonregular_srs_contract_fails_before_review(
    change: tuple[Path, str, str], mode: str
) -> None:
    repo, base, _ = change
    _write(
        repo,
        "docs/requirements/strategy.md",
        "# Strategy\n\n## FR-STR-001 Inputs\n\n"
        "- **需求**：Use cached input.\n- **驗收**：Preserve its identity.\n"
        "- **契約**：[missing](../missing.md)\n"
        "- **驗證**：人工驗收：inspect the preserved evidence.\n",
    )
    if mode == "symlink":
        (repo / "docs/missing.md").symlink_to("../README.md")
    tree = _tree(repo)
    with pytest.raises(ReviewError, match="Missing regular SRS contract evidence"):
        build_review_plan(build_evidence(repo, base, tree))


@pytest.mark.parametrize("explicit_context", [False, True])
def test_cross_scope_keeps_srs_contract_when_presentation_owns_its_file(
    change: tuple[Path, str, str], explicit_context: bool
) -> None:
    repo, _, _ = change
    for path in PRESENTATION_PATHS:
        _write(repo, path, "# Deck\n" if path.endswith(".md") else "asset\n")
    contract = f"{DECK}/README.md"
    original = "# Deck\n\n## Contract\nORIGINAL_CONTRACT_EVIDENCE\n"
    _write(repo, contract, original)
    _write(repo, "tests/test_example.py", "# Evidence target, never executed.\n")
    base = _tree(repo)
    _write(repo, f"{DECK}/theme.css", "/* changed asset */\n")
    _write(
        repo,
        "docs/requirements.md",
        "# SRS\n\n[Interface](requirements/interfaces.md)\n",
    )
    _write(
        repo,
        "docs/requirements/interfaces.md",
        "# Interface\n\n## NFR-API-001 Contract\n\n"
        "- **需求**：Keep the contract.\n- **驗收**：Deliver its original evidence.\n"
        "- **契約**：[contract](../talks/tradingdev-introduction/README.md#contract)\n"
        "- **驗證**：[test](../../tests/test_example.py)\n",
    )
    assert check_requirements(repo) == 1
    tree = _tree(repo)
    raw = build_evidence(
        repo,
        base,
        tree,
        review_context_for=(("cross_scope", contract),) if explicit_context else (),
    )
    plan = build_review_plan(raw)
    batch = next(
        _payload(p) for p in plan.prompts if _payload(p)["scope"] == "cross_scope"
    )
    sections = [s for s in batch["document_sections"] if s["path"] == contract]
    assert contract in batch["requirement_context_paths"]["cross_scope"]
    if explicit_context:
        assert not sections
        assert batch["candidate_files"][contract] == original
    else:
        assert contract not in batch["candidate_files"]
        assert "".join(s["text"] for s in sections) == original
        assert all(
            s["blob_oid"] == _git(repo, "rev-parse", f"{tree}:{contract}")
            for s in sections
        )


def _behavior_requirement(identifier: str, href: str, obligation: str) -> str:
    return (
        f"## {identifier} Saved evidence\n\n"
        f"- **需求**：{obligation}\n"
        "- **驗收**：Keep the original source and identity.\n"
        f"- **契約**：[original contract]({href})\n"
        "- **驗證**：人工驗收：inspect the saved evidence.\n\n"
    )


@pytest.fixture
def behavior_requirement_change(tmp_path: Path) -> tuple[Path, str, str, str]:
    _git(tmp_path, "init", "-q")
    _write(tmp_path, "README.md", "# TradingDev\nReader context.\n")
    _write(
        tmp_path,
        "docs/ml_research.md",
        "# ML Research\nML reader context.\n\n## ML Research Artifacts\n"
        "Preserved identities.\n\n### Recorded Signal Evaluations\n"
        "FIRST_CONTRACT_ORIGINAL\n",
    )
    _write(
        tmp_path,
        "docs/reports.md",
        "# Reports\nReport reader context.\n\n## Composable Offline Reports\n"
        "SECOND_CONTRACT_ORIGINAL\n",
    )
    first = _behavior_requirement(
        "FR-ML-003",
        "../ml_research.md#recorded-signal-evaluations",
        "Keep the original signal obligation.",
    )
    second = _behavior_requirement(
        "FR-ML-004",
        "../reports.md#composable-offline-reports",
        "Keep the original report obligation.",
    )
    _write(
        tmp_path, "docs/requirements/ml.md", "# ML requirements\n\n" + first + second
    )
    return tmp_path, _tree(tmp_path), first, second


@pytest.mark.parametrize("change_second", [False, True])
def test_changed_requirement_selects_its_contract_without_expanding_its_book(
    behavior_requirement_change: tuple[Path, str, str, str], change_second: bool
) -> None:
    repo, base, first, second = behavior_requirement_change
    first = first.replace("original signal obligation", "revised signal obligation")
    if change_second:
        second = second.replace(
            "original report obligation", "revised report obligation"
        )
    path = "docs/requirements/ml.md"
    current = "# ML requirements\n\n" + first + second
    _write(repo, path, current)
    tree = _tree(repo)
    evidence = build_evidence(repo, base, tree)
    original = json.loads(evidence)
    plan = build_review_plan(evidence)
    batches = [_payload(prompt) for prompt in plan.prompts]
    owner = next(batch for batch in batches if path in batch["assigned_paths"])
    assert owner["candidate_files"][path] == current
    assert owner["path_diffs"][path] == original["path_diffs"][path]
    assert "FR-ML-004" in owner["candidate_files"][path]
    context = "".join(
        section["text"] for batch in batches for section in batch["document_sections"]
    )
    assert "FIRST_CONTRACT_ORIGINAL" in context
    assert ("SECOND_CONTRACT_ORIGINAL" in context) is change_second
    assert owner["behavior_contexts"][path]["requirement_ids"] == (
        ["FR-ML-003", "FR-ML-004"] if change_second else ["FR-ML-003"]
    )


@pytest.mark.parametrize("ending", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize(
    "common_edit", ["modify-preface", "remove-preface", "appendix"]
)
def test_shared_srs_edits_keep_other_requirements_contracts(
    behavior_requirement_change: tuple[Path, str, str, str],
    common_edit: str,
    ending: str,
) -> None:
    repo, _, first, second = behavior_requirement_change
    path = "docs/requirements/ml.md"
    heading = "# ML requirements\n\n"
    premise = (
        "All requirements apply to saved evaluations and reports."
        + "\u2028 Shared premise." * 6
        + "\n\n"
    )
    _write(repo, path, (heading + premise + first + second).replace("\n", ending))
    base = _tree(repo)
    revised = first.replace("original signal obligation", "revised signal obligation")
    if common_edit == "modify-preface":
        premise = premise.replace(
            "saved evaluations and reports", "evaluations only; reports are excluded"
        )
    elif common_edit == "remove-preface":
        premise = ""
    appendix = (
        "## Shared applicability\nReports no longer follow this chapter.\n"
        if common_edit == "appendix"
        else ""
    )
    current = (heading + premise + revised + second + appendix).replace("\n", ending)
    _write(repo, path, current)
    evidence = build_evidence(repo, base, _tree(repo))
    original = json.loads(evidence)
    plan = build_review_plan(evidence)
    batches = [_payload(prompt) for prompt in plan.prompts]
    owner = next(batch for batch in batches if path in batch["assigned_paths"])
    assert owner["candidate_files"][path] == current
    assert owner["path_diffs"][path] == original["path_diffs"][path]
    context = "".join(
        section["text"] for batch in batches for section in batch["document_sections"]
    )
    assert "FIRST_CONTRACT_ORIGINAL" in context
    assert "SECOND_CONTRACT_ORIGINAL" in context
    selected = owner["behavior_contexts"][path]
    assert selected["fallback_scope"] == "ml"
    assert selected["requirement_ids"] == ["FR-ML-003"]


def test_reference_definition_change_keeps_old_and_new_contracts_with_body_edit(
    behavior_requirement_change: tuple[Path, str, str, str],
) -> None:
    repo, _, first, second = behavior_requirement_change
    _write(
        repo,
        "docs/run_artifacts.md",
        "# Run Artifacts\n\n## Discovery Pages\nNEW_REFERENCE_CONTRACT\n",
    )
    second = second.replace(
        "[original contract](../reports.md#composable-offline-reports)",
        "[original contract][target]",
    )
    requirement = "docs/requirements/ml.md"
    definition = "## Appendix\n\n[target]: ../reports.md#composable-offline-reports\n"
    _write(repo, requirement, "# ML requirements\n\n" + first + second + definition)
    base = _tree(repo)
    _write(
        repo,
        requirement,
        "# ML requirements\n\n"
        + first.replace("original signal obligation", "revised signal obligation")
        + second
        + definition.replace(
            "../reports.md#composable-offline-reports",
            "../run_artifacts.md#discovery-pages",
        ),
    )
    plan = build_review_plan(build_evidence(repo, base, _tree(repo)))
    selected = plan.manifest["behavior_contexts"][requirement]
    assert selected["fallback_scope"] == "ml"
    assert selected["requirement_ids"] == ["FR-ML-003", "FR-ML-004"]
    text = "".join(
        s["text"] for p in plan.prompts for s in _payload(p)["document_sections"]
    )
    assert "SECOND_CONTRACT_ORIGINAL" in text
    assert "NEW_REFERENCE_CONTRACT" in text


@pytest.mark.parametrize("ending", ["\n", "\r\n", "\r"])
def test_contract_edits_map_git_lines_to_markdown_without_unicode_line_splitting(
    behavior_requirement_change: tuple[Path, str, str, str],
    ending: str,
) -> None:
    repo, _, first, second = behavior_requirement_change
    contract = "docs/ml_research.md"
    lines = [
        "# ML Research",
        "",
        "## ML Research Artifacts",
        "",
        "### Recorded Signal Evaluations",
        "prefix" + "\u2028 continuation" * 6,
        "FIRST_OLD",
        "",
        "### Trained Models",
        "SECOND_UNCHANGED",
        "kept line",
        "kept line",
        "last line",
        "",
    ]
    _write(repo, contract, ending.join(lines))
    second = second.replace(
        "../reports.md#composable-offline-reports", "../ml_research.md#trained-models"
    )
    _write(repo, "docs/requirements/ml.md", "# ML requirements\n\n" + first + second)
    base = _tree(repo)
    _write(repo, contract, ending.join(lines).replace("FIRST_OLD", "FIRST_NEW"))
    evidence = build_evidence(repo, base, _tree(repo))
    plan = build_review_plan(evidence)
    identifiers = plan.manifest["behavior_contexts"][contract]["requirement_ids"]
    assert "FR-ML-003" in identifiers
    if ending != "\r":
        assert identifiers == ["FR-ML-003"]
    text = "".join(
        s["text"] for p in plan.prompts for s in _payload(p)["document_sections"]
    )
    assert "Keep the original signal obligation." in text


@pytest.mark.parametrize(
    "operation", ["remove_requirement", "remove_contract", "rename_contract"]
)
def test_removed_or_renamed_contract_retains_base_obligation_and_complete_diff(
    behavior_requirement_change: tuple[Path, str, str, str], operation: str
) -> None:
    repo, _, first, second = behavior_requirement_change
    caller = REPORT
    _write(repo, caller, "USE_RETIRED_SCORE = True\n")
    base = _tree(repo)
    requirement = "docs/requirements/ml.md"
    contract = "docs/ml_research.md"
    contract_before = (repo / contract).read_text(encoding="utf-8")
    if operation == "rename_contract":
        renamed = "docs/renamed_ml_contract.md"
        (repo / contract).rename(repo / renamed)
        current = (
            "# ML requirements\n\n"
            + first.replace("ml_research.md", "renamed_ml_contract.md")
            + second
        )
    else:
        current = "# ML requirements\n\n" + second
        if operation == "remove_contract":
            (repo / contract).unlink()
    _write(repo, requirement, current)
    _write(repo, caller, "USE_RETIRED_SCORE = False\n")
    tree = _tree(repo)
    evidence = build_evidence(repo, base, tree)
    original = json.loads(evidence)
    batches = [_payload(prompt) for prompt in build_review_plan(evidence).prompts]
    audit = next(batch for batch in batches if batch["phase"] == "cross_scope")
    assert audit["path_diffs"] == original["path_diffs"]
    assert audit["path_diffs"][requirement]
    old_obligation = [
        section
        for section in audit["document_sections"]
        if section["path"] == requirement and section["version"] == "base"
    ]
    assert old_obligation and "".join(s["text"] for s in old_obligation) == first
    assert all(
        s["blob_oid"] == _git(repo, "rev-parse", f"{base}:{requirement}")
        for s in old_obligation
    )
    contract_sections = [
        section for section in audit["document_sections"] if section["path"] == contract
    ]
    assert "".join(s["text"] for s in contract_sections) == contract_before
    expected_version = "candidate" if operation == "remove_requirement" else "base"
    assert {s["version"] for s in contract_sections} == {expected_version}
    assert all(
        s["blob_oid"] == _git(repo, "rev-parse", f"{base}:{contract}")
        for s in contract_sections
    )
    if operation != "remove_requirement":
        assert contract not in audit["candidate_tree_inventory"]
        assert audit["path_diffs"][contract] == original["path_diffs"][contract]
    for batch in batches:
        for path in batch["assigned_paths"]:
            assert batch["path_diffs"][path] == original["path_diffs"][path]
            if path in original["candidate_files"]:
                assert (
                    batch["candidate_files"][path] == original["candidate_files"][path]
                )


@pytest.fixture
def real_document_repository(tmp_path: Path) -> Path:
    root = Path(__file__).resolve().parents[2]
    _git(tmp_path, "init", "-q")
    documents = [
        root / name
        for name in ("README.md", "ARCHITECTURE.md", "AGENTS.md", "CLAUDE.md")
    ]
    documents.extend((root / "docs").rglob("*.md"))
    for document in documents:
        _write(
            tmp_path,
            document.relative_to(root).as_posix(),
            document.read_text(encoding="utf-8"),
        )
    return tmp_path


@pytest.mark.parametrize(
    ("sources", "operations", "required", "excluded"),
    [
        (
            (
                "tooling/git/claim_issue.py",
                "scripts/claim-issue.sh",
                "tests/tooling/test_claim_issue.py",
            ),
            {"issue_claim"},
            (("AGENTS.md", "Issue 接手與指派"), ("docs/development.md", "接手 issue")),
            ("撰寫 issue", "4. 合併後更新 main、清理本機分支", "檢查期間的主線更新"),
        ),
        (
            (
                "tooling/git/cleanup_pr.py",
                "scripts/cleanup-pr.sh",
                "tests/tooling/test_pr_cleanup.py",
            ),
            {"pr_cleanup"},
            (
                ("AGENTS.md", "PR 合併後的本機分支清理"),
                ("docs/development.md", "4. 合併後更新 main、清理本機分支"),
            ),
            ("接手 issue", "檢查期間的主線更新", "測試證據與文件修正"),
        ),
        (
            ("tooling/git/pr_context.py",),
            {"candidate_review", "pr_cleanup"},
            (
                ("docs/development.md", "4. 合併後更新 main、清理本機分支"),
                ("docs/development.md", "檢查期間的主線更新"),
            ),
            ("接手 issue", "容器建置與驗證", "選擇性 LLM 測試"),
        ),
        (
            ("tooling/review_options.py",),
            {"candidate_review"},
            (
                ("AGENTS.md", "模型上下文與提示詞"),
                ("AGENTS.md", "Git hooks 與本機 PR 審查"),
                ("docs/development.md", "測試證據與文件修正"),
                ("docs/development.md", "檢查期間的主線更新"),
                ("docs/development.md", "SRS 維護與檢核"),
            ),
            ("接手 issue", "4. 合併後更新 main、清理本機分支", "選擇性 LLM 測試"),
        ),
        (
            ("scripts/check-fast.sh", "tooling/quality.py"),
            {"environment_quality"},
            (
                ("docs/development.md", "型別檢查與契約保護"),
                ("docs/development.md", "量測模型上下文"),
                ("docs/development.md", "SRS 維護與檢核"),
            ),
            ("接手 issue", "4. 合併後更新 main、清理本機分支", "檢查期間的主線更新"),
        ),
        (
            ("tooling/environment.py", "tooling/git/receipts.py"),
            {"environment_quality", "candidate_review", "presentation"},
            (
                ("docs/development.md", "容器建置與驗證"),
                ("docs/development.md", "選擇性 LLM 測試"),
                ("docs/development.md", "測試證據與文件修正"),
                ("docs/development.md", "在 VS Code 儲存時自動編譯"),
                ("ARCHITECTURE.md", "容器執行 adapter"),
                ("README.md", "1. 準備環境並安裝"),
            ),
            ("接手 issue", "4. 合併後更新 main、清理本機分支"),
        ),
        (
            ("scripts/presentations.sh",),
            {"presentation"},
            (
                ("docs/development.md", "在 VS Code 儲存時自動編譯"),
                ("docs/development.md", "Cloudflare 自動發布"),
            ),
            ("接手 issue", "4. 合併後更新 main、清理本機分支", "檢查期間的主線更新"),
        ),
    ],
)
def test_development_operations_select_exact_real_guidance_and_complete_sources(
    real_document_repository: Path,
    sources: tuple[str, ...],
    operations: set[str],
    required: tuple[tuple[str, str], ...],
    excluded: tuple[str, ...],
) -> None:
    repo = real_document_repository
    root = Path(__file__).resolve().parents[2]
    for path in sources:
        _write(repo, path, (root / path).read_text(encoding="utf-8"))
    base = _tree(repo)
    for path in sources:
        with (repo / path).open("a", encoding="utf-8") as stream:
            stream.write("\n# Operation context regression.\n")
    raw = build_evidence(repo, base, _tree(repo))
    original = json.loads(raw)
    plan = build_review_plan(raw)
    batches = [_payload(prompt) for prompt in plan.prompts]
    owned = [batch for batch in batches if batch["phase"] == "scope"]
    for path in sources:
        owner = next(batch for batch in owned if path in batch["assigned_paths"])
        assert set(owner["development_contexts"][path]["operations"]) == operations
        assert path not in owner["behavior_contexts"]
        assert owner["candidate_files"][path] == original["candidate_files"][path]
        assert owner["path_diffs"][path] == original["path_diffs"][path]
        assert owner["requirement_context_paths"][owner["scope"]] == []
    sections = [section for batch in owned for section in batch["document_sections"]]
    required += (
        ("AGENTS.md", "工作樹與分階段提交"),
        ("AGENTS.md", "測試與清理"),
        ("docs/development.md", "注意事項"),
        ("ARCHITECTURE.md", "Repository 開發工具與研究原型"),
    )
    for path, heading in required:
        expected = next(
            section
            for section in markdown_sections(path, original["candidate_files"][path])
            if section.headings[-1:] == (heading,)
        )
        assert any(
            section["path"] == path
            and section["text"] == expected.text
            and section["start_line"] == expected.start_line
            and section["end_line"] == expected.end_line
            and section["blob_oid"] == original["candidate_file_blobs"][path]
            for section in sections
        ), (sources, path, heading)
    assert not {
        heading for section in sections for heading in section["headings"][-1:]
    } & set(excluded)
    assert not any(
        section["path"].startswith("docs/requirements/") for section in sections
    )
    selected = {(s["path"], s["headings"][-1]) for s in sections if s["headings"]}
    assert ("README.md", "手動啟動並使用 HTTP 連接") not in selected
    assert ("README.md", "使用 Docker 部署") not in selected
    assert ("ARCHITECTURE.md", "核心概念與身分") not in selected
    assert ("docs/development.md", "MAE 研究原型") not in selected
    if "presentation" in operations:
        guide = f"{DECK}/README.md"
        assert any(
            batch["candidate_files"].get(guide) == original["candidate_files"][guide]
            or "".join(
                s["text"] for s in batch["document_sections"] if s["path"] == guide
            )
            == original["candidate_files"][guide]
            for batch in batches
        )


def test_development_and_product_changes_keep_their_union_in_cross_scope_audit(
    real_document_repository: Path,
) -> None:
    repo = real_document_repository
    paths = (
        "tooling/git/claim_issue.py",
        "tooling/git/cleanup_pr.py",
        "src/tradingdev/mcp/tools/data.py",
    )
    for path in paths:
        _write(repo, path, "VALUE = 'before'\n")
    base = _tree(repo)
    for path in paths:
        _write(repo, path, "VALUE = 'after'\n")
    raw = build_evidence(repo, base, _tree(repo))
    original = json.loads(raw)
    plan = build_review_plan(raw)
    batches = [_payload(prompt) for prompt in plan.prompts]
    audit = next(batch for batch in batches if batch["phase"] == "cross_scope")
    assert audit["path_diffs"] == original["path_diffs"]
    assert set(audit["behavior_contexts"][paths[-1]]["requirement_ids"]) >= {
        "FR-DATA-001",
        "FR-DATA-002",
        "NFR-API-001",
    }
    assert set(audit["development_contexts"]) == set(paths[:2])
    product = next(batch for batch in batches if paths[-1] in batch["assigned_paths"])
    tooling = next(batch for batch in batches if paths[0] in batch["assigned_paths"])
    assert set(tooling["assigned_paths"]) == set(paths[:2])
    for batch in (tooling, audit):
        selected = {
            (s["path"], s["headings"][-1])
            for s in batch["document_sections"]
            if s["headings"]
        }
        assert ("docs/development.md", "接手 issue") in selected
        assert ("docs/development.md", "4. 合併後更新 main、清理本機分支") in selected
    for batch in (product, audit):
        assert any(
            s["path"] == "docs/requirements/data.md" and "FR-DATA-001" in s["text"]
            for s in batch["document_sections"]
        )
        expected = next(
            s.text
            for s in markdown_sections(
                "docs/run_artifacts.md",
                original["candidate_files"]["docs/run_artifacts.md"],
            )
            if s.headings[-1:] == ("Market Data Cache",)
        )
        assert any(s["text"] == expected for s in batch["document_sections"])
    for path in paths:
        owner = next(batch for batch in batches if path in batch["assigned_paths"])
        assert owner["candidate_files"][path] == original["candidate_files"][path]
        assert owner["path_diffs"][path] == original["path_diffs"][path]


@pytest.mark.parametrize("change_kind", ["modify", "delete", "rename"])
def test_unknown_tooling_path_keeps_fallback_and_old_operation_obligations(
    real_document_repository: Path, change_kind: str
) -> None:
    repo = real_document_repository
    known = "tooling/git/claim_issue.py"
    unknown = "tooling/git/claim_extension.py"
    _write(repo, known, "VALUE = 'original claim implementation'\n")
    if change_kind != "rename":
        _write(repo, unknown, "VALUE = 'old unknown consumer'\n")
    base = _tree(repo)
    if change_kind == "rename":
        (repo / known).rename(repo / unknown)
    else:
        _write(repo, unknown, "VALUE = 'changed unknown consumer'\n")
        if change_kind == "delete":
            (repo / known).unlink()
        else:
            _write(repo, known, "VALUE = 'updated claim implementation'\n")
    raw = build_evidence(repo, base, _tree(repo))
    original = json.loads(raw)
    plan = build_review_plan(raw)
    batches = [_payload(prompt) for prompt in plan.prompts]
    batch = next(batch for batch in batches if known in batch["assigned_paths"])
    assert set(batch["assigned_paths"]) == {known, unknown}
    assert batch["behavior_contexts"][unknown] == {"fallback_scope": "tooling"}
    assert batch["development_contexts"][known] == {
        "operations": ["issue_claim"],
        "selection_basis": "registered-path",
    }
    assert batch["path_diffs"] == original["path_diffs"]
    assert batch["base_file_blobs"][known] == original["base_file_blobs"][known]
    if change_kind != "modify":
        assert known not in batch["candidate_files"]
        assert "-VALUE = 'original claim implementation'" in batch["path_diffs"][known]
    for path in (known, unknown):
        if path in original["candidate_files"]:
            assert batch["candidate_files"][path] == original["candidate_files"][path]
    for path, heading in (
        ("docs/development.md", "接手 issue"),
        ("docs/development.md", "4. 合併後更新 main、清理本機分支"),
        ("docs/development.md", "MAE 研究原型"),
        ("README.md", "手動啟動並使用 HTTP 連接"),
    ):
        expected = next(
            s.text
            for s in markdown_sections(path, original["candidate_files"][path])
            if s.headings[-1:] == (heading,)
        )
        assert any(
            s["path"] == path and s["text"] == expected
            for s in batch["document_sections"]
        )


def test_development_unknown_heading_is_preserved_and_default_capacity_fails_closed(
    real_document_repository: Path,
) -> None:
    repo = real_document_repository
    source = "tooling/git/claim_issue.py"
    unknown = "## New operational contract\n" + "Necessary unknown original.\n" * 10_000
    readme = (repo / "README.md").read_text(encoding="utf-8")
    _write(repo, "README.md", readme + "\n" + unknown)
    _write(repo, source, "VALUE = 'before'\n")
    base = _tree(repo)
    _write(repo, source, "VALUE = 'after'\n")
    # The new section is one indivisible original. No implicit budget increase,
    # truncation or successful plan is allowed just because the path is known.
    with pytest.raises(ReviewError, match="original document section.*README.md"):
        build_review_plan(build_evidence(repo, base, _tree(repo)))


@pytest.mark.parametrize(
    ("document", "heading", "edit", "precise"),
    [
        ("docs/development.md", "接手 issue", "modify", True),
        ("AGENTS.md", "Issue 接手與指派", "modify", True),
        ("AGENTS.md", "模型上下文與提示詞", "modify", False),
        ("docs/development.md", "接手 issue", "delete", False),
        ("docs/development.md", "接手 issue", "rename", False),
    ],
)
def test_tool_and_operational_document_edits_narrow_only_reviewed_responsibilities(
    real_document_repository: Path,
    document: str,
    heading: str,
    edit: str,
    precise: bool,
) -> None:
    repo = real_document_repository
    source = "tooling/git/claim_issue.py"
    _write(repo, source, "VALUE = 'before'\n")
    base = _tree(repo)
    _write(repo, source, "VALUE = 'after'\n")
    original_text = (repo / document).read_text(encoding="utf-8")
    section = next(
        part
        for part in markdown_sections(document, original_text)
        if part.headings[-1:] == (heading,)
    )
    if edit == "modify":
        replacement = section.text.rstrip() + "\n\n操作說明修訂。\n\n"
    elif edit == "delete":
        replacement = ""
    else:
        replacement = section.text.replace(heading, "新操作標題", 1)
    _write(repo, document, original_text.replace(section.text, replacement, 1))
    raw = build_evidence(repo, base, _tree(repo))
    original = json.loads(raw)
    plan = build_review_plan(raw)
    batches = [_payload(prompt) for prompt in plan.prompts]
    owner = next(batch for batch in batches if document in batch["assigned_paths"])
    assert owner["candidate_files"][document] == original["candidate_files"][document]
    assert owner["path_diffs"][document] == original["path_diffs"][document]
    if precise:
        assert owner["development_contexts"][document] == {
            "operations": ["issue_claim"],
            "selection_basis": "changed-operational-sections",
        }
        assert document not in owner["behavior_contexts"]
    else:
        assert document not in owner["development_contexts"]
        assert owner["behavior_contexts"][document] == {"fallback_scope": "tooling"}
    # Shared product policy, removed sections and unfamiliar headings cannot
    # borrow the neighboring claim tool's narrower context.
    readme_headings = {
        s["headings"][-1]
        for s in owner["document_sections"]
        if s["path"] == "README.md" and s["headings"]
    }
    assert ("手動啟動並使用 HTTP 連接" in readme_headings) is not precise


def test_unknown_path_keeps_registered_obligations_in_the_same_owner_batch(
    real_document_repository: Path,
) -> None:
    repo = real_document_repository
    known = "tests/app/test_ml_report_service.py"
    unknown = "tests/app/test_ml_new_consumer.py"
    for path in (known, unknown):
        _write(repo, path, "VALUE = 'old'\n")
    base = _tree(repo)
    for path in (known, unknown):
        _write(repo, path, "VALUE = 'new'\n")
    plan = build_review_plan(
        build_evidence(repo, base, _tree(repo)), max_prompt_bytes=450_000
    )
    batch = next(
        _payload(p) for p in plan.prompts if known in _payload(p)["assigned_paths"]
    )
    assert unknown in batch["assigned_paths"]
    assert batch["scope"] == "ml"
    sections = batch["document_sections"]
    assert any(
        s["path"] == "docs/requirements/execution.md" and "FR-EXEC-004" in s["text"]
        for s in sections
    ), "Unknown paths must not erase a registered shared comparison obligation"
    assert any(s["path"] == "docs/requirements/quality.md" for s in sections)


@pytest.mark.parametrize(
    ("source", "budget", "shared_runs"),
    [
        ("tests/mcp/test_ml_leaderboard_tool.py", 240_000, False),
        ("src/tradingdev/mcp/tools/ml.py", 450_000, True),
    ],
)
def test_real_document_corpus_keeps_ml_contracts_without_unrelated_interface_context(
    real_document_repository: Path,
    source: str,
    budget: int,
    shared_runs: bool,
) -> None:
    tmp_path = real_document_repository
    _write(tmp_path, source, "PRODUCER_SOURCE = 'unknown'\n")
    base = _tree(tmp_path)
    current = "PRODUCER_SOURCE = 'backend_model'\n"
    _write(tmp_path, source, current)
    tree = _tree(tmp_path)
    evidence = build_evidence(tmp_path, base, tree)
    original = json.loads(evidence)
    # The shared tool also owns run ranking and discovery. Preserve those
    # contracts even when they require an explicit larger operational budget.
    plan = (
        build_review_plan(evidence)
        if budget == 240_000
        else build_review_plan(evidence, max_prompt_bytes=budget)
    )
    batches = [_payload(prompt) for prompt in plan.prompts]
    owner = next(batch for batch in batches if source in batch["assigned_paths"])
    assert owner["scope"] == "interfaces"
    assert owner["candidate_files"][source] == current
    assert owner["path_diffs"][source] == original["path_diffs"][source]
    sections = owner["document_sections"]
    selected = {
        (section["path"], heading)
        for section in sections
        for heading in section["headings"]
    }
    assert {
        ("docs/ml_research.md", heading)
        for heading in (
            "Fold Validation and Signal Interpretation",
            "Recorded Signal Evaluations",
        )
    } <= selected
    if shared_runs:
        assert ("docs/ml_research.md", "Dataset and Model Inputs") in selected
        assert ("docs/run_artifacts.md", "Ranking Saved Run Scopes") in selected
        assert (
            "docs/performance_artifacts.md",
            "Performance Scopes and Provenance",
        ) in selected
    assert ("README.md", "使用 Docker 部署") not in selected
    assert (
        not {
            ("docs/reports.md", heading)
            for heading in (
                "Research notebooks and archives",
                "Recorded conversations",
                "Codex CLI recording adapter",
            )
        }
        & selected
    )
    for section in sections:
        lines = original["candidate_files"][section["path"]].splitlines(keepends=True)
        assert section["text"] == "".join(
            lines[section["start_line"] - 1 : section["end_line"]]
        )
    assert all(
        batch["prompt_utf8_bytes"] <= budget for batch in plan.manifest["batches"]
    )


@pytest.mark.parametrize(
    ("sources", "requirements", "contracts"),
    [
        (
            (
                "src/tradingdev/app/research.py",
                "src/tradingdev/domain/research.py",
                "tests/app/test_research_resolution.py",
            ),
            {"FR-STR-001"},
            {
                ("docs/strategy_contract.md", "Research Inputs and Simulation Policy"),
                ("docs/run_artifacts.md", "Research Settings and Provenance"),
            },
        ),
        (
            (
                "src/tradingdev/domain/research_policy.py",
                "tests/domain/test_research.py",
            ),
            {
                "FR-STR-001",
                "FR-EXEC-003",
                "FR-RES-002",
                "FR-ML-003",
                "NFR-REPRO-001",
            },
            {
                ("docs/strategy_contract.md", "Research Inputs and Simulation Policy"),
                ("docs/strategy_contract.md", "Signal and Execution Semantics"),
                ("docs/ml_research.md", "Recorded Signal Evaluations"),
            },
        ),
        (
            (
                "src/tradingdev/app/research_evidence_service.py",
                "src/tradingdev/app/contracts/research_evidence.py",
                "src/tradingdev/adapters/storage/research_evidence.py",
                "tests/app/test_research_evidence_service.py",
                "tests/adapters/test_research_evidence_storage.py",
            ),
            {"FR-EXEC-001", "NFR-DATA-001", "NFR-OPS-002"},
            {("docs/run_artifacts.md", "Research Evidence Records")},
        ),
        (
            (
                "src/tradingdev/domain/research_evidence.py",
                "tests/domain/test_research_evidence.py",
            ),
            {
                "FR-EXEC-001",
                "FR-EXEC-002",
                "FR-EXEC-004",
                "FR-EXEC-006",
                "NFR-DATA-001",
                "NFR-OPS-002",
            },
            {
                ("docs/run_artifacts.md", "Research Evidence Records"),
                ("docs/run_artifacts.md", "Fixed Comparison Evidence"),
                ("docs/run_artifacts.md", "Execution Manifest"),
            },
        ),
        (
            ("src/tradingdev/domain/research_evidence_values.py",),
            {"FR-EXEC-001", "FR-EXEC-002", "NFR-DATA-001", "NFR-OPS-002"},
            {("docs/run_artifacts.md", "Research Evidence Records")},
        ),
        (
            ("src/tradingdev/domain/parameter_values.py",),
            {
                "FR-EXEC-001",
                "FR-EXEC-004",
                "FR-EXEC-006",
                "FR-RES-001",
                "NFR-DATA-001",
                "NFR-REPRO-001",
            },
            {
                ("docs/run_artifacts.md", "Research Evidence Records"),
                ("docs/run_artifacts.md", "Execution Manifest"),
                ("docs/run_artifacts.md", "Historical Observation Queries"),
            },
        ),
        (
            (
                "src/tradingdev/domain/approved_research.py",
                "tests/domain/test_approved_research.py",
            ),
            {
                "FR-EXEC-001",
                "FR-EXEC-002",
                "FR-EXEC-004",
                "FR-RES-001",
                "NFR-DATA-001",
                "NFR-REPRO-001",
            },
            {
                ("docs/run_artifacts.md", "Research Evidence Records"),
                ("docs/run_artifacts.md", "Fixed Comparison Evidence"),
                ("docs/run_artifacts.md", "Historical Observation Queries"),
            },
        ),
        (
            (
                "src/tradingdev/adapters/storage/submitted_research.py",
                "tests/app/test_submitted_research.py",
            ),
            {
                "FR-EXEC-001",
                "FR-EXEC-002",
                "FR-EXEC-004",
                "FR-RES-001",
                "NFR-DATA-001",
                "NFR-OPS-002",
                "NFR-REPRO-001",
            },
            {
                ("docs/run_artifacts.md", "Research Evidence Records"),
                ("docs/run_artifacts.md", "Historical Observation Queries"),
            },
        ),
    ],
    ids=[
        "intake",
        "shared-policy",
        "saved-records",
        "bindings",
        "snapshot-values",
        "parameter-values",
        "approved-research",
        "submitted-research",
    ],
)
def test_research_review_keeps_original_obligations_and_contracts(
    real_document_repository: Path,
    sources: tuple[str, ...],
    requirements: set[str],
    contracts: set[tuple[str, str]],
) -> None:
    repo = real_document_repository
    for source in sources:
        _write(repo, source, "VALUE = 'old'\n")
    base = _tree(repo)
    for source in sources:
        _write(repo, source, "VALUE = 'fixed research'\n")
    raw = build_evidence(repo, base, _tree(repo))
    original = json.loads(raw)
    # Shared policy and integrity requirements intentionally retain their full
    # cross-domain contracts. This is an explicit planner budget, not a promise
    # that every research responsibility fits the default model capacity.
    plan = build_review_plan(raw, max_prompt_bytes=450_000)
    batches = [_payload(prompt) for prompt in plan.prompts]
    for source in sources:
        batch = next(part for part in batches if source in part["assigned_paths"])
        assert batch["scope"] == "general"
        assert batch["candidate_files"][source] == original["candidate_files"][source]
        assert batch["path_diffs"][source] == original["path_diffs"][source]
        assert requirements <= set(
            batch["behavior_contexts"][source]["requirement_ids"]
        )
        sections = batch["document_sections"]
        selected = {
            (s["path"], s["headings"][-1]): s for s in sections if s["headings"]
        }
        assert requirements <= {heading.split()[0] for _, heading in selected}
        assert contracts <= selected.keys()
        for section in sections:
            lines = original["candidate_files"][section["path"]].splitlines(True)
            assert section["text"] == "".join(
                lines[section["start_line"] - 1 : section["end_line"]]
            )
        for path, text in original["candidate_files"].items():
            if not path.endswith(".md"):
                continue
            for section in markdown_sections(path, text):
                if section.headings and (
                    section.headings[-1].split()[0] in requirements
                    or (path, section.headings[-1]) in contracts
                ):
                    assert selected[(path, section.headings[-1])]["text"] == (
                        section.text
                    )
        assert ("README.md", "安裝與連接助手") not in selected
        assert ("README.md", "手動啟動並使用 HTTP 連接") not in selected
        assert ("docs/reports.md", "Recorded conversations") not in selected
        assert ("docs/reports.md", "Codex CLI recording adapter") not in selected
        assert not any(
            heading.startswith(("NFR-SAFE-002", "NFR-OPS-001"))
            for _, heading in selected
        )


def test_unknown_research_module_preserves_known_and_conservative_context(
    real_document_repository: Path,
) -> None:
    repo = real_document_repository
    known = "src/tradingdev/domain/approved_research.py"
    unknown = "src/tradingdev/domain/research_extension.py"
    for path in (known, unknown):
        _write(repo, path, "VALUE = 'old'\n")
    base = _tree(repo)
    for path in (known, unknown):
        _write(repo, path, "VALUE = 'new'\n")
    raw = build_evidence(repo, base, _tree(repo))
    original = json.loads(raw)
    # General fallback retains the entire document corpus. It must fail an
    # insufficient budget rather than silently discard the unknown obligation.
    with pytest.raises(ReviewError, match="budget|capacity"):
        build_review_plan(raw, max_prompt_bytes=1_000)
    plan = build_review_plan(raw, max_prompt_bytes=1_500_000)
    batch = next(
        _payload(p) for p in plan.prompts if known in _payload(p)["assigned_paths"]
    )
    assert set(batch["assigned_paths"]) == {known, unknown}
    assert batch["behavior_contexts"][unknown] == {"fallback_scope": "general"}
    assert {"FR-EXEC-002", "FR-RES-001", "NFR-REPRO-001"} <= set(
        batch["behavior_contexts"][known]["requirement_ids"]
    )
    assert batch["path_diffs"] == original["path_diffs"]
    for path in (known, unknown):
        assert batch["candidate_files"][path] == original["candidate_files"][path]
    sections = batch["document_sections"]
    for path in ("README.md", "docs/run_artifacts.md", "docs/requirements/quality.md"):
        assert (
            "".join(s["text"] for s in sections if s["path"] == path)
            == (original["candidate_files"][path])
        )


@pytest.mark.parametrize(
    ("sources", "requirements", "contracts"),
    [
        (
            (
                "src/tradingdev/mcp/tools/data.py",
                "src/tradingdev/app/contracts/data.py",
                "tests/mcp/test_data_contracts.py",
            ),
            {"FR-DATA-001", "FR-DATA-002", "NFR-API-001", "NFR-OPS-002"},
            {
                ("docs/run_artifacts.md", "Market Data Cache"),
                ("docs/run_artifacts.md", "Automatic DVOL Queries"),
                ("ARCHITECTURE.md", "MCP 回覆契約"),
            },
        ),
        (
            (
                "src/tradingdev/mcp/tools/strategy_cleanup.py",
                "src/tradingdev/app/strategy_cleanup_service.py",
                "src/tradingdev/app/contracts/strategy_cleanup.py",
                "tests/app/test_strategy_cleanup_service.py",
            ),
            {"FR-STR-005"},
            {("docs/strategy_contract.md", "Draft Cleanup")},
        ),
        (
            (
                "src/tradingdev/mcp/tools/research_notes.py",
                "src/tradingdev/app/research_note_service.py",
                "tests/app/test_research_notes.py",
                "tests/mcp/test_mcp_research_notes.py",
            ),
            {"FR-RES-005", "NFR-OPS-002"},
            {
                ("docs/reports.md", "Research notebooks and archives"),
                ("docs/run_artifacts.md", "Reading Artifact Delivery"),
            },
        ),
        (
            (
                "src/tradingdev/app/conversation_service.py",
                "src/tradingdev/app/contracts/conversations.py",
                "tests/app/test_conversations.py",
            ),
            {"FR-RES-006", "NFR-OPS-002"},
            {
                ("docs/reports.md", "Recorded conversations"),
                ("docs/reports.md", "Codex CLI recording adapter"),
            },
        ),
        (
            (
                "src/tradingdev/app/benchmark_service.py",
                "tests/app/test_benchmark_service.py",
            ),
            {"FR-RES-002"},
            {("docs/performance_artifacts.md", "Benchmarks")},
        ),
        (
            (
                "src/tradingdev/mcp/tools/runs.py",
                "src/tradingdev/app/run_service.py",
                "src/tradingdev/app/rolling_trace_summary.py",
                "src/tradingdev/app/trade_history_service.py",
                "tests/app/test_run_metrics.py",
                "tests/app/test_trade_history.py",
                "tests/app/test_execution_history.py",
            ),
            {"FR-RES-001", "FR-RES-002", "NFR-DATA-001", "NFR-OPS-002"},
            {
                ("docs/performance_artifacts.md", "Performance Scopes and Provenance"),
                ("docs/run_artifacts.md", "Historical Observation Queries"),
                ("docs/run_artifacts.md", "Research Evidence Records"),
                ("docs/run_artifacts.md", "Research Settings and Provenance"),
                ("docs/run_artifacts.md", "Execution Manifest"),
            },
        ),
        (
            (
                "src/tradingdev/app/report_comparison.py",
                "tests/app/test_comparison_report.py",
                "tests/app/test_research_report_service.py",
            ),
            {"FR-EXEC-004", "FR-RES-001", "FR-RES-003", "FR-RES-004"},
            {
                ("docs/run_artifacts.md", "Fixed Comparison Evidence"),
                ("docs/run_artifacts.md", "Research Evidence Records"),
                ("docs/reports.md", "Composable Offline Reports"),
            },
        ),
        (
            (
                "src/tradingdev/app/report_service.py",
                "src/tradingdev/mcp/tools/history.py",
                "tests/mcp/test_history_contracts.py",
            ),
            {"FR-ML-004", "FR-EXEC-004", "FR-RES-001", "FR-RES-003", "FR-RES-004"},
            {
                ("docs/ml_research.md", "Recorded Signal Evaluations"),
                ("docs/run_artifacts.md", "Fixed Comparison Evidence"),
                ("docs/reports.md", "Research interpretation and diagnostic providers"),
                ("docs/performance_artifacts.md", "Performance Scopes and Provenance"),
            },
        ),
        (
            (
                "src/tradingdev/app/ranking_service.py",
                "tests/app/test_run_ranking.py",
            ),
            {"FR-ML-003", "FR-ML-004", "FR-RES-001", "FR-RES-002"},
            {
                ("docs/run_artifacts.md", "Ranking Saved Run Scopes"),
                ("docs/ml_research.md", "Recorded Signal Evaluations"),
                ("docs/performance_artifacts.md", "Performance Scopes and Provenance"),
            },
        ),
        (
            ("src/tradingdev/mcp/tools/artifacts.py",),
            {"FR-EXEC-001", "FR-RES-001", "FR-RES-004", "FR-RES-005", "NFR-OPS-002"},
            {
                ("docs/run_artifacts.md", "Reading Artifact Text"),
                ("docs/run_artifacts.md", "Reading Artifact Delivery"),
                (
                    "docs/execution_plans.md",
                    "Execution Plans and Confirmation Documents",
                ),
            },
        ),
        (
            ("tests/mcp/test_artifact_downloads.py",),
            {"FR-EXEC-001", "FR-RES-004", "NFR-DATA-001", "NFR-API-001"},
            {("docs/run_artifacts.md", "Reading Artifact Delivery")},
        ),
        (
            ("tests/app/test_artifact_text_pages.py",),
            {"FR-RES-001", "NFR-DATA-001", "NFR-OPS-002", "NFR-API-001"},
            {("docs/run_artifacts.md", "Reading Artifact Text")},
        ),
    ],
    ids=[
        "data-boundary",
        "draft-cleanup",
        "notebooks",
        "conversations",
        "benchmark",
        "saved-run-scopes",
        "comparison-report",
        "shared-report-entrypoints",
        "mixed-ranking",
        "artifact-delivery",
        "artifact-http-consumer",
        "artifact-text-consumer",
    ],
)
def test_non_ml_entrypoints_keep_original_contracts_and_complete_changes(
    real_document_repository: Path,
    sources: tuple[str, ...],
    requirements: set[str],
    contracts: set[tuple[str, str]],
) -> None:
    repo = real_document_repository
    for source in sources:
        _write(repo, source, "VALUE = 'before'\n")
    base = _tree(repo)
    for source in sources:
        _write(repo, source, "VALUE = 'after'\n")
    raw = build_evidence(repo, base, _tree(repo))
    original = json.loads(raw)
    # Shared integrity/report requirements still carry cross-domain contracts.
    # This explicit test budget does not change the default reviewer capacity.
    plan = build_review_plan(raw, max_prompt_bytes=500_000)
    batches = [_payload(prompt) for prompt in plan.prompts]
    for source in sources:
        batch = next(part for part in batches if source in part["assigned_paths"])
        assert requirements <= set(
            batch["behavior_contexts"][source]["requirement_ids"]
        )
        assert batch["candidate_files"][source] == original["candidate_files"][source]
        assert batch["path_diffs"][source] == original["path_diffs"][source]
        sections = batch["document_sections"]
        selected = {
            (s["path"], s["headings"][-1]): s for s in sections if s["headings"]
        }
        assert contracts <= selected.keys()
        assert requirements <= {heading.split()[0] for _, heading in selected}
        # Verify complete original records and contracts, not just selected IDs.
        for path, text in original["candidate_files"].items():
            if not path.endswith(".md"):
                continue
            for section in markdown_sections(path, text):
                if section.headings and (
                    section.headings[-1].split()[0] in requirements
                    or (path, section.headings[-1]) in contracts
                ):
                    assert (
                        selected[(path, section.headings[-1])]["text"] == section.text
                    )
        for section in sections:
            lines = original["candidate_files"][section["path"]].splitlines(True)
            assert section["text"] == "".join(
                lines[section["start_line"] - 1 : section["end_line"]]
            )
        assert ("README.md", "使用 Docker 部署") not in selected
        assert ("README.md", "手動啟動並使用 HTTP 連接") not in selected
        for requirement, heading in (
            ("FR-RES-005", "Research notebooks and archives"),
            ("FR-RES-006", "Recorded conversations"),
        ):
            if requirement not in requirements:
                assert ("docs/reports.md", heading) not in selected
        assert not any(
            heading.startswith(("NFR-SAFE-002", "NFR-OPS-001"))
            for _, heading in selected
        )
    audit = next((b for b in batches if b["phase"] == "cross_scope"), batches[0])
    assert audit["path_diffs"] == original["path_diffs"]


@pytest.mark.parametrize(
    ("known", "unknown", "owner"),
    [
        (
            "src/tradingdev/mcp/tools/data.py",
            "src/tradingdev/mcp/tools/new_history.py",
            "interfaces",
        ),
        (
            "src/tradingdev/app/benchmark_service.py",
            "src/tradingdev/app/report_extension.py",
            "results",
        ),
    ],
)
def test_unknown_non_ml_consumer_keeps_owner_fallback_and_registered_obligations(
    real_document_repository: Path, known: str, unknown: str, owner: str
) -> None:
    repo = real_document_repository
    for path in (known, unknown):
        _write(repo, path, "VALUE = 'before'\n")
    base = _tree(repo)
    for path in (known, unknown):
        _write(repo, path, "VALUE = 'after'\n")
    raw = build_evidence(repo, base, _tree(repo))
    original = json.loads(raw)
    with pytest.raises(ReviewError, match="budget|capacity"):
        build_review_plan(raw, max_prompt_bytes=1_000)
    plan = build_review_plan(raw, max_prompt_bytes=1_500_000)
    batch = next(
        _payload(p) for p in plan.prompts if known in _payload(p)["assigned_paths"]
    )
    assert batch["scope"] == owner
    assert set(batch["assigned_paths"]) == {known, unknown}
    assert batch["behavior_contexts"][unknown] == {"fallback_scope": owner}
    assert batch["behavior_contexts"][known]["requirement_ids"]
    assert batch["path_diffs"] == original["path_diffs"]
    for path in (known, unknown):
        assert batch["candidate_files"][path] == original["candidate_files"][path]
    # These unrelated known sections must return when an unreviewed consumer
    # shares the owner; a precise neighbor must never narrow the whole batch.
    sections = batch["document_sections"]
    for heading in ("Research notebooks and archives", "Recorded conversations"):
        original_section = next(
            section
            for section in markdown_sections(
                "docs/reports.md", original["candidate_files"]["docs/reports.md"]
            )
            if section.headings[-1:] == (heading,)
        )
        assert any(section["text"] == original_section.text for section in sections)


def _edit_performance_architecture(repo: Path) -> str:
    path = repo / "ARCHITECTURE.md"
    original = path.read_text(encoding="utf-8")
    changed = original.replace(
        "## 績效計算與對話摘要\n",
        "## 績效計算與對話摘要\n\nPerformance evidence change.\n",
    )
    assert changed != original
    path.write_text(changed, encoding="utf-8")
    return changed


def test_architecture_section_review_preserves_sources_and_selects_its_obligations(
    real_document_repository: Path,
) -> None:
    repo = real_document_repository
    caller = "src/tradingdev/app/benchmark_service.py"
    _write(repo, caller, "VALUE = 'before'\n")
    base = _tree(repo)
    source = _edit_performance_architecture(repo)
    _write(repo, caller, "VALUE = 'after'\n")
    raw = build_evidence(repo, base, _tree(repo))
    original = json.loads(raw)
    plan = build_review_plan(raw)
    assert plan.manifest["path_owners"]["ARCHITECTURE.md"] == "general"
    obligations = plan.manifest["behavior_contexts"]["ARCHITECTURE.md"]
    assert "FR-RES-002" in obligations["requirement_ids"]
    assert "FR-ML-001" not in obligations["requirement_ids"]
    batches = [_payload(prompt) for prompt in plan.prompts]
    owner = next(b for b in batches if "ARCHITECTURE.md" in b["assigned_paths"])
    assert owner["candidate_files"]["ARCHITECTURE.md"] == source
    assert (
        owner["path_diffs"]["ARCHITECTURE.md"]
        == original["path_diffs"]["ARCHITECTURE.md"]
    )
    audit = next(b for b in batches if b["phase"] == "cross_scope")
    assert audit["path_diffs"] == original["path_diffs"]
    for batch in (owner, audit):
        sections = batch["document_sections"]
        assert any(s["path"] == "docs/performance_artifacts.md" for s in sections)
        assert not any(s["path"] == "docs/requirements/ml.md" for s in sections)
    plan.verify()


@pytest.mark.parametrize(
    "extra_change",
    ["unknown", "preamble", "reference", "rename", "delete", "duplicate"],
)
def test_architecture_section_review_keeps_uncertain_edits_conservative(
    real_document_repository: Path, extra_change: str
) -> None:
    repo = real_document_repository
    base = _tree(repo)
    source = _edit_performance_architecture(repo)
    if extra_change == "unknown":
        source += "\n## Unregistered responsibility\nNew contract.\n"
    elif extra_change == "preamble":
        source = "Shared premise changed.\n\n" + source
    elif extra_change == "reference":
        source = source.replace(
            "Performance evidence change.",
            "Performance evidence change.\n\n[shared]: docs/requirements/ml.md",
        )
    elif extra_change == "rename":
        source = source.replace("## 績效計算與對話摘要", "## Renamed performance")
    elif extra_change == "delete":
        start = source.index("## 績效計算與對話摘要")
        end = source.index("### 診斷擴充與替換", start)
        source = source[:start] + source[end:]
    else:
        source += "\n## 績效計算與對話摘要\nAmbiguous duplicate.\n"
    _write(repo, "ARCHITECTURE.md", source)
    raw = build_evidence(repo, base, _tree(repo))
    original = json.loads(raw)
    plan = build_review_plan(raw, max_prompt_bytes=1_500_000)
    owners = [
        _payload(p)
        for p in plan.prompts
        if "ARCHITECTURE.md" in _payload(p)["assigned_paths"]
    ]
    assert owners
    for owner in owners:
        assert owner["candidate_files"]["ARCHITECTURE.md"] == source
        assert (
            owner["path_diffs"]["ARCHITECTURE.md"]
            == original["path_diffs"]["ARCHITECTURE.md"]
        )
    assert any(
        s["path"] == "docs/requirements/ml.md"
        for owner in owners
        for s in owner["document_sections"]
    )
    plan.verify()


def test_architecture_section_review_unions_explicit_cross_domain_requirements(
    real_document_repository: Path,
) -> None:
    repo = real_document_repository
    requirement = repo / "docs/requirements/ml.md"
    text = requirement.read_text(encoding="utf-8")
    start = text.index("## FR-ML-003")
    contract = text.index("- **契約**：", start)
    end = text.index("\n", contract)
    text = (
        text[:end] + " [架構](../../ARCHITECTURE.md#績效計算與對話摘要)。" + text[end:]
    )
    requirement.write_text(text, encoding="utf-8")
    base = _tree(repo)
    _edit_performance_architecture(repo)
    plan = build_review_plan(
        build_evidence(repo, base, _tree(repo)), max_prompt_bytes=1_500_000
    )
    obligations = plan.manifest["behavior_contexts"]["ARCHITECTURE.md"]
    assert {"FR-RES-002", "FR-ML-003"} <= set(obligations["requirement_ids"])
    sections = [s for p in plan.prompts for s in _payload(p)["document_sections"]]
    assert any(s["path"] == "docs/performance_artifacts.md" for s in sections)
    assert any(s["path"] == "docs/requirements/ml.md" for s in sections)
