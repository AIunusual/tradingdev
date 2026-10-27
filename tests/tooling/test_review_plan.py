"""Scope selection must preserve immutable changes and enforce input budgets."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest
from tooling import review_docs
from tooling.review_evidence import ReviewError, build_evidence
from tooling.review_plan import build_review_plan

if TYPE_CHECKING:
    from pathlib import Path

STRATEGY = "src/tradingdev/app/strategy_service.py"
REPORT = "src/tradingdev/app/report_service.py"


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


def test_explicit_scope_cannot_hide_a_changed_responsibility(
    change: tuple[Path, str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, base, tree = change
    monkeypatch.setattr(
        review_docs, "_run", lambda *_a, **_kw: pytest.fail("model ran")
    )
    with pytest.raises(ReviewError, match="uncovered.*report_service"):
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
