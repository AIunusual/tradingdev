"""Check routing is based on both sides of immutable Git changes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import pytest
from tooling.git import change_scope
from tooling.git.change_scope import ChangeScope, classify_changes

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest import CaptureFixture, MonkeyPatch


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments], text=True
    ).strip()


@pytest.fixture
def repository() -> Iterator[tuple[Path, str]]:
    with TemporaryDirectory(prefix="tradingdev-change-scope-") as temporary:
        root = Path(temporary)
        _git(root, "init", "--quiet", "--template=")
        (root / "docs").mkdir()
        (root / "README.md").write_text("# Guide\n")
        (root / "docs/guide.md").write_text("# Details\n")
        (root / "app.py").write_text("VALUE = 1\n")
        _git(root, "add", ".")
        yield root, _git(root, "write-tree")


@pytest.mark.parametrize(
    "name, expected",
    [
        ("README.md", "docs"),
        ("ARCHITECTURE.md", "docs"),
        ("docs/guide.md", "docs"),
        ("docs/nested/中文 guide\nnotes.md", "docs"),
        ("AGENTS.md", "tooling"),
        ("CLAUDE.md", "tooling"),
        ("OTHER.md", "code"),
        (".agents/policy.md", "code"),
        ("docs/config.yaml", "code"),
        ("docs/check.sh", "code"),
        ("src/README.md", "code"),
        ("pyproject.toml", "code"),
        ("uv.lock", "code"),
        (".githooks/pre-commit", "tooling"),
        (".github/workflows/pr-agent.yml", "tooling"),
        (".github/workflows/deploy.yml", "code"),
        ("docs/pr_review.md", "tooling"),
        ("tests/test_app.py", "code"),
        ("tooling/git/claim_issue.py", "tooling"),
        ("tooling/git/change_scope.py", "tooling"),
        ("tooling/review_scope.py", "tooling"),
        ("tooling/presentation_html.py", "tooling"),
        ("tooling/presentation_errors.py", "tooling"),
        ("tooling/requirements.py", "tooling"),
        ("tests/tooling/test_new_tool.py", "tooling"),
        ("tests/tooling/new_helper.py", "code"),
        (".agents/skills/new-skill/SKILL.md", "code"),
        (".agents/skills/wait-for-check/helper.py", "code"),
        ("scripts/check-tooling.sh", "tooling"),
        ("tests/conftest.py", "code"),
        ("tests/tooling/conftest.py", "code"),
        ("tests/llm_plugin.py", "code"),
        ("tooling/prompt_contracts.py", "code"),
        ("tooling/context_budget.py", "code"),
        ("tooling/codex_binary.py", "code"),
        ("tooling/process_guard.py", "code"),
        ("tooling/new_tool.py", "code"),
        ("scripts/unknown.sh", "code"),
        ("scripts/check-full.sh", "code"),
        ("scripts/check-llm.sh", "code"),
        ("scripts/rebuild-env.sh", "code"),
    ],
)
def test_conservative_path_routing(
    repository: tuple[Path, str], name: str, expected: str
) -> None:
    root, base = repository
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Changed\n")
    _git(root, "add", name)
    assert classify_changes(root, base, _git(root, "write-tree")).kind == expected


@pytest.mark.parametrize(
    "name",
    [
        "tests/tooling/git_check_scenarios.py",
        ".agents/skills/wait-for-check/SKILL.md",
    ],
)
@pytest.mark.parametrize("change", ["add", "modify", "delete"])
def test_registered_support_files_use_tooling_checks(
    repository: tuple[Path, str], name: str, change: str
) -> None:
    root, base = repository
    path = root / name
    path.parent.mkdir(parents=True)
    if change != "add":
        path.write_text("original support file\n")
        _git(root, "add", name)
        base = _git(root, "write-tree")
    if change == "delete":
        path.unlink()
    else:
        path.write_text("changed support file\n")
    _git(root, "add", name)
    scope = classify_changes(root, base, _git(root, "write-tree"))
    assert scope.payload() == {
        "scope": "tooling",
        "commit_profile": "fast",
        "delivery_profile": "tooling",
        "candidate_checks": ["documentation_review", "tooling"],
    }


@pytest.mark.parametrize(
    "name, change",
    [
        ("tests/tooling/git_check_scenarios.py", "executable"),
        (".agents/skills/wait-for-check/SKILL.md", "remove_executable"),
        (".agents/skills/wait-for-check/SKILL.md", "symlink"),
        ("tests/tooling/git_check_scenarios.py", "rename"),
        (".agents/skills/wait-for-check/SKILL.md", "rename"),
        ("tests/tooling/git_check_scenarios.py", "mixed"),
        (".agents/skills/wait-for-check/SKILL.md", "mixed"),
        (".github/workflows/pr-agent.yml", "mixed"),
    ],
)
def test_support_registration_preserves_full_check_boundaries(
    repository: tuple[Path, str], name: str, change: str
) -> None:
    root, _ = repository
    path = root / name
    path.parent.mkdir(parents=True)
    path.write_text("original support file\n")
    _git(root, "add", name)
    if change == "remove_executable":
        _git(root, "update-index", "--chmod=+x", name)
    base = _git(root, "write-tree")
    if change == "symlink":
        path.unlink()
        path.symlink_to(root / "app.py")
    elif change == "rename":
        _git(root, "mv", name, str(path.with_name("unknown" + path.suffix)))
    elif change in {"executable", "remove_executable"}:
        mode = "+x" if change == "executable" else "-x"
        _git(root, "update-index", f"--chmod={mode}", name)
    else:
        path.write_text("changed support file\n")
        (root / "app.py").write_text("VALUE = 2\n")
    if change in {"symlink", "mixed"}:
        _git(root, "add", ".")
    scope = classify_changes(root, base, _git(root, "write-tree"))
    assert scope.delivery_profile == "full"
    assert scope.candidate_checks == ("documentation_review", "full")


@pytest.mark.parametrize("change", ["delete", "rename", "code_rename", "mixed"])
def test_deletions_renames_and_mixed_changes(
    repository: tuple[Path, str], change: str
) -> None:
    root, base = repository
    if change == "delete":
        (root / "docs/guide.md").unlink()
        _git(root, "add", "--update", "docs/guide.md")
    elif change == "rename":
        _git(root, "mv", "docs/guide.md", "docs/new guide.md")
    elif change == "code_rename":
        _git(root, "mv", "app.py", "docs/code.md")
    else:
        (root / "README.md").write_text("# Changed\n")
        (root / "app.py").write_text("VALUE = 2\n")
        _git(root, "add", ".")
    assert classify_changes(root, base, _git(root, "write-tree")).kind == (
        "docs" if change in {"delete", "rename"} else "code"
    )


@pytest.mark.parametrize("change", ["executable", "symlink", "remove_executable"])
def test_file_modes_never_bypass_code_checks(
    repository: tuple[Path, str], change: str
) -> None:
    root, base = repository
    if change == "symlink":
        (root / "README.md").unlink()
        (root / "README.md").symlink_to("app.py")
        _git(root, "add", "README.md")
    else:
        _git(root, "update-index", "--chmod=+x", "README.md")
        if change == "remove_executable":
            base = _git(root, "write-tree")
            _git(root, "update-index", "--chmod=-x", "README.md")
    assert classify_changes(root, base, _git(root, "write-tree")) == ChangeScope("code")


def test_empty_changes_and_working_tree_edits_do_not_select_docs(
    repository: tuple[Path, str],
) -> None:
    root, base = repository
    (root / "README.md").write_text("# Unstaged\n")
    assert classify_changes(root, base, _git(root, "write-tree")) == ChangeScope("code")
    _git(root, "add", "README.md")
    tree = _git(root, "write-tree")
    (root / "app.py").write_text("VALUE = 2\n")
    assert classify_changes(root, base, tree) == ChangeScope("docs")


def test_invalid_git_identity_fails_instead_of_selecting_docs(
    repository: tuple[Path, str],
) -> None:
    root, base = repository
    with pytest.raises(subprocess.CalledProcessError):
        classify_changes(root, base, "missing-tree")


@pytest.mark.parametrize(
    "paths, commit, delivery, candidate",
    [
        (["README.md"], "docs", "docs", ["docs"]),
        (["docs/requirements.md"], "docs", "docs", ["docs", "documentation_review"]),
        (
            [
                "docs/requirements/execution.md",
                "docs/talks/tradingdev-introduction/slides.md",
            ],
            "presentations",
            "presentations",
            ["presentations", "documentation_review"],
        ),
        (
            ["docs/talks/tradingdev-introduction/slides.md"],
            "presentations",
            "presentations",
            ["presentations"],
        ),
        (["AGENTS.md"], "fast", "tooling", ["documentation_review", "tooling"]),
        (
            [
                "tests/tooling/git_check_scenarios.py",
                ".agents/skills/wait-for-check/SKILL.md",
            ],
            "fast",
            "tooling",
            ["documentation_review", "tooling"],
        ),
        (["app.py"], "fast", "full", ["documentation_review", "full"]),
        (
            ["app.py", "docs/talks/tradingdev-introduction/slides.md"],
            "fast",
            "full",
            ["presentations", "documentation_review", "full"],
        ),
    ],
)
def test_plan_cli_reports_required_checks_for_the_staged_tree(
    repository: tuple[Path, str],
    capsys: CaptureFixture[str],
    paths: list[str],
    commit: str,
    delivery: str,
    candidate: list[str],
) -> None:
    root, base = repository
    for name in paths:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Changed\n")
        _git(root, "add", name)
    staged = _git(root, "write-tree")
    (root / "app.py").write_text("Unstaged edits must not affect this plan\n")
    assert change_scope.main(["--repo", str(root), "--base", base, "--staged"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["code"] == "verification_planned"
    assert plan["source"] == "staged"
    assert (plan["base_tree"], plan["candidate_tree"]) == (base, staged)
    assert plan["commit_profile"] == commit
    assert plan["delivery_profile"] == delivery
    assert plan["candidate_checks"] == candidate
    assert _git(root, "write-tree") == staged
    assert (root / "app.py").read_text() == "Unstaged edits must not affect this plan\n"


def test_plan_cli_uses_explicit_trees_and_rejects_missing_identity(
    repository: tuple[Path, str], capfd: CaptureFixture[str]
) -> None:
    root, base = repository
    (root / "README.md").write_text("# Changed\n")
    _git(root, "add", "README.md")
    candidate = _git(root, "write-tree")
    (root / "app.py").write_text("VALUE = 2\n")
    _git(root, "add", "app.py")
    arguments = ["--repo", str(root), "--base", base, "--tree", candidate]
    assert change_scope.main(arguments) == 0
    plan = json.loads(capfd.readouterr().out)
    assert plan["source"] == "tree"
    assert plan["candidate_tree"] == candidate
    assert plan["scope"] == "docs"
    assert change_scope.main([*arguments[:-1], "missing-tree"]) == 1
    output = capfd.readouterr()
    assert output.out == ""
    assert json.loads(output.err)["code"] == "verification_plan_failed"


def test_plan_cli_keeps_git_blob_read_failures_in_json(
    repository: tuple[Path, str], capfd: CaptureFixture[str]
) -> None:
    root, base = repository
    path = root / "docs/.nojekyll"
    path.write_text("nonempty marker\n")
    _git(root, "add", "docs/.nojekyll")
    candidate = _git(root, "write-tree")
    blob = _git(root, "rev-parse", f"{candidate}:docs/.nojekyll")
    (root / ".git/objects" / blob[:2] / blob[2:]).unlink()
    assert (
        change_scope.main(["--repo", str(root), "--base", base, "--tree", candidate])
        == 1
    )
    output = capfd.readouterr()
    assert output.out == ""
    failure = json.loads(output.err)
    assert failure["code"] == "verification_plan_failed"
    assert "fatal:" in failure["error"]


def test_plan_cli_rejects_index_movement_without_discarding_it(
    repository: tuple[Path, str],
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    root, base = repository

    def racing_plan(*args: Path | str) -> ChangeScope:
        result = classify_changes(root, base, base)
        (root / "README.md").write_text("# Concurrent staged change\n")
        _git(root, "add", "README.md")
        return result

    monkeypatch.setattr(change_scope, "classify_changes", racing_plan)
    assert change_scope.main(["--repo", str(root), "--base", base, "--staged"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err)["code"] == "verification_plan_failed"
    assert _git(root, "write-tree") != base


@pytest.mark.parametrize(
    "name, content, expected",
    [
        ("docs/talks/tradingdev-introduction/slides.md", "# Slides\n", "presentations"),
        ("docs/talks/tradingdev-introduction/theme.css", "body {}\n", "presentations"),
        ("docs/talks/tradingdev-introduction/index.html", "<html>\n", "code"),
        ("docs/.nojekyll", "", "presentations"),
        ("docs/.nojekyll", "unexpected config\n", "code"),
        ("docs/unknown/index.html", "<html>\n", "code"),
        ("docs/unknown/theme.css", "body {}\n", "code"),
        ("docs/unknown/slides.md", "# Ordinary Markdown\n", "docs"),
    ],
)
def test_presentation_routing_requires_registered_paths_and_content(
    repository: tuple[Path, str], name: str, content: str, expected: str
) -> None:
    root, base = repository
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    _git(root, "add", name)
    assert classify_changes(root, base, _git(root, "write-tree")).kind == expected


@pytest.mark.parametrize("other", ["README.md", "app.py", "AGENTS.md"])
def test_mixed_changes_retain_presentation_validation(
    repository: tuple[Path, str], other: str
) -> None:
    root, base = repository
    slides = root / "docs/talks/tradingdev-introduction/slides.md"
    slides.parent.mkdir(parents=True)
    slides.write_text("# Slides\n")
    (root / other).write_text("changed\n")
    _git(root, "add", ".")
    scope = classify_changes(root, base, _git(root, "write-tree"))
    assert (
        scope.kind
        == {"README.md": "presentations", "AGENTS.md": "tooling", "app.py": "code"}[
            other
        ]
    )
    assert scope.presentations_changed


@pytest.mark.parametrize("change", ["add", "modify", "delete"])
def test_saved_report_changes_use_presentation_checks(
    repository: tuple[Path, str], change: str
) -> None:
    root, base = repository
    name = "docs/talks/tradingdev-introduction/evidence/demo/cases/example/report.html"
    target = root / name
    target.parent.mkdir(parents=True)
    if change != "add":
        target.write_text('<a href="index.html">Report</a>\n')
        _git(root, "add", name)
        base = _git(root, "write-tree")
    if change == "delete":
        target.unlink()
    else:
        target.write_text('<a href="missing.html">Broken report</a>\n')
    _git(root, "add", name)
    scope = classify_changes(root, base, _git(root, "write-tree"))
    assert scope.kind == "presentations"
    assert scope.candidate_checks == ("presentations",)


@pytest.mark.parametrize(
    "change",
    [
        "executable",
        "symlink",
        "remove_executable",
        "product_rename",
        "mixed",
        "unknown",
    ],
)
def test_evidence_registration_does_not_bypass_code_checks(
    repository: tuple[Path, str], change: str
) -> None:
    root, base = repository
    name = "docs/talks/tradingdev-introduction/evidence/demo/cases/example/strategy.py"
    target = root / name
    target.parent.mkdir(parents=True)
    if change == "product_rename":
        _git(root, "mv", "app.py", name)
    elif change == "symlink":
        target.symlink_to(root / "app.py")
        _git(root, "add", name)
    else:
        target.write_text("# Saved source, never imported by the product\n")
        _git(root, "add", name)
        if change in {"executable", "remove_executable"}:
            _git(root, "update-index", "--chmod=+x", name)
            if change == "remove_executable":
                base = _git(root, "write-tree")
                _git(root, "update-index", "--chmod=-x", name)
        elif change == "mixed":
            (root / "app.py").write_text("VALUE = 2\n")
            _git(root, "add", "app.py")
        else:
            _git(root, "mv", name, str(target.with_name("run.py")))
    scope = classify_changes(root, base, _git(root, "write-tree"))
    assert scope.kind == "code"
    assert scope.candidate_checks == ("presentations", "documentation_review", "full")


def test_evidence_formats_use_both_git_blobs_not_unstaged_repairs(
    repository: tuple[Path, str],
) -> None:
    root, base = repository
    name = "docs/talks/tradingdev-introduction/evidence/demo/case.json"
    target = root / name
    target.parent.mkdir(parents=True)
    target.write_bytes(b"\xff")
    _git(root, "add", name)
    invalid = _git(root, "write-tree")
    target.write_text("{}")
    assert classify_changes(root, base, invalid).kind == "code"
    _git(root, "add", name)
    valid = _git(root, "write-tree")
    assert classify_changes(root, base, valid).kind == "presentations"
    assert classify_changes(root, invalid, valid).kind == "code"


def test_native_conversation_export_uses_presentation_checks(
    repository: tuple[Path, str],
) -> None:
    root, base = repository
    directory = (
        root
        / "docs/talks/tradingdev-introduction/evidence/demo/cases/example"
        / "conversation"
    )
    directory.mkdir(parents=True)
    files = {
        "index.html": '<a href="conversation.json">Original conversation</a>',
        "conversation.json": '{"turns": []}',
        "manifest.json": '{"files": {"index.html": "saved digest"}}',
    }
    for name, content in files.items():
        (directory / name).write_text(content)
    _git(root, "add", ".")
    saved = _git(root, "write-tree")
    assert classify_changes(root, base, saved).candidate_checks == ("presentations",)
    for name in files:
        (directory / name).unlink()
    _git(root, "add", ".")
    assert (
        classify_changes(root, saved, _git(root, "write-tree")).kind == "presentations"
    )


@pytest.mark.parametrize(
    "change", ["executable", "symlink", "delete", "product_rename"]
)
def test_registered_tool_modes_and_rename_source_are_checked(
    repository: tuple[Path, str], change: str
) -> None:
    root, base = repository
    name = "scripts/claim-issue.sh"
    target = root / name
    target.parent.mkdir()
    if change == "product_rename":
        _git(root, "mv", "app.py", name)
    elif change == "symlink":
        target.symlink_to("../app.py")
        _git(root, "add", name)
    else:
        target.write_text("#!/usr/bin/env bash\nexit 0\n")
        target.chmod(0o755)
        _git(root, "add", name)
        if change == "delete":
            base = _git(root, "write-tree")
            target.unlink()
            _git(root, "add", "--update", name)
    assert classify_changes(root, base, _git(root, "write-tree")).kind == (
        "tooling" if change in {"executable", "delete"} else "code"
    )


@pytest.mark.parametrize("other", ["README.md", "src/app.py", "uv.lock"])
def test_tooling_mixed_with_product_or_dependencies_uses_full_scope(
    repository: tuple[Path, str], other: str
) -> None:
    root, base = repository
    (root / "AGENTS.md").write_text("# Changed policy\n")
    target = root / other
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("changed\n")
    _git(root, "add", ".")
    tree = _git(root, "write-tree")
    target.write_text("unstaged content must not affect routing\n")
    assert classify_changes(root, base, tree).kind == (
        "tooling" if other == "README.md" else "code"
    )


@pytest.mark.parametrize("change", ["executable", "symlink", "remove_executable"])
def test_presentation_modes_retain_code_checks(
    repository: tuple[Path, str], change: str
) -> None:
    root, _ = repository
    name = "docs/talks/tradingdev-introduction/index.html"
    target = root / name
    target.parent.mkdir(parents=True)
    target.write_text("<html>\n")
    _git(root, "add", name)
    base = _git(root, "write-tree")
    if change == "symlink":
        target.unlink()
        target.symlink_to("../../../app.py")
        _git(root, "add", name)
    else:
        _git(root, "update-index", "--chmod=+x", name)
        if change == "remove_executable":
            base = _git(root, "write-tree")
            _git(root, "update-index", "--chmod=-x", name)
    assert classify_changes(root, base, _git(root, "write-tree")) == ChangeScope(
        "code", presentations_changed=True
    )


@pytest.mark.parametrize("direction", ["into_presentation", "out_of_presentation"])
def test_presentation_rename_checks_both_paths(
    repository: tuple[Path, str], direction: str
) -> None:
    root, base = repository
    name = "docs/talks/tradingdev-introduction/index.html"
    (root / name).parent.mkdir(parents=True)
    if direction == "into_presentation":
        _git(root, "mv", "app.py", name)
    else:
        (root / name).write_text("<html>\n")
        _git(root, "add", name)
        base = _git(root, "write-tree")
        _git(root, "mv", name, "unknown.html")
    assert classify_changes(root, base, _git(root, "write-tree")) == ChangeScope(
        "code", presentations_changed=True
    )


def test_special_presentation_contents_use_git_blobs_not_worktree(
    repository: tuple[Path, str],
) -> None:
    root, base = repository
    path = root / "docs/.nojekyll"
    path.write_text("unsafe config\n")
    _git(root, "add", "docs/.nojekyll")
    invalid_tree = _git(root, "write-tree")
    path.write_text("")
    assert classify_changes(root, base, invalid_tree) == ChangeScope("code", True)
    _git(root, "add", "docs/.nojekyll")
    valid_tree = _git(root, "write-tree")
    path.write_text("unstaged config\n")
    assert classify_changes(root, base, valid_tree) == ChangeScope(
        "presentations", True
    )
    assert classify_changes(root, invalid_tree, valid_tree) == ChangeScope("code", True)


@pytest.mark.parametrize("change", ["known", "command", "extra_task", "invalid_json"])
def test_vscode_tasks_only_allow_known_presentation_tasks(
    repository: tuple[Path, str], change: str
) -> None:
    root, base = repository
    path = root / ".vscode/tasks.json"
    path.parent.mkdir()
    known = Path(__file__).resolve().parents[2] / ".vscode/tasks.json"
    content = known.read_text()
    if change == "invalid_json":
        content = "{broken"
    elif change != "known":
        document = json.loads(content)
        if change == "command":
            document["tasks"][0]["command"] = "arbitrary-program"
        else:
            document["tasks"].append(
                {"label": "unrelated", "type": "shell", "command": "execute-code"}
            )
        content = json.dumps(document)
    path.write_text(content)
    _git(root, "add", ".vscode/tasks.json")
    scope = classify_changes(root, base, _git(root, "write-tree"))
    assert scope.kind == ("presentations" if change == "known" else "code")
    assert scope.presentations_changed


@pytest.mark.parametrize("change", ["delete", "rename", "edit", "unsafe_mode", "mixed"])
def test_srs_review_tracks_both_git_sides_and_keeps_stronger_checks(
    repository: tuple[Path, str], change: str
) -> None:
    root, _ = repository
    source = root / "docs/requirements.md"
    source.write_text("# SRS\n")
    _git(root, "add", ".")
    base = _git(root, "write-tree")
    if change == "delete":
        source.unlink()
        _git(root, "add", "--update", "docs/requirements.md")
    elif change == "rename":
        _git(root, "mv", "docs/requirements.md", "docs/other.md")
    elif change == "unsafe_mode":
        _git(root, "update-index", "--chmod=+x", "docs/requirements.md")
    else:
        source.write_text("# Revised SRS\n")
        if change == "mixed":
            (root / "app.py").write_text("VALUE = 2\n")
        _git(root, "add", ".")
    scope = classify_changes(root, base, _git(root, "write-tree"))
    assert scope.requirements_changed
    assert "documentation_review" in scope.candidate_checks
    assert scope.commit_profile == (
        "fast" if change in {"mixed", "unsafe_mode"} else "docs"
    )
    assert scope.delivery_profile == (
        "full" if change in {"mixed", "unsafe_mode"} else "docs"
    )


@pytest.mark.parametrize(
    "name",
    [
        "tooling/marp/package.json",
        "tooling/marp/package-lock.json",
        "tooling/marp/render-diagrams.mjs",
        "tooling/marp/presentation-url.mjs",
        "tooling/marp/build.mjs",
        "scripts/build-presentations-site.sh",
        "wrangler.jsonc",
    ],
)
def test_presentation_compilers_keep_tooling_and_presentation_checks(
    repository: tuple[Path, str],
    name: str,
) -> None:
    root, base = repository
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("changed compiler")
    _git(root, "add", name)
    scope = classify_changes(root, base, _git(root, "write-tree"))
    assert scope.delivery_profile == "tooling"
    assert scope.candidate_checks == (
        "presentations",
        "documentation_review",
        "tooling",
    )


@pytest.mark.parametrize(
    "other", ["tooling/marp/unknown.mjs", "src/app.py", "pyproject.toml"]
)
def test_presentation_toolchain_does_not_allow_unregistered_or_product_changes(
    repository: tuple[Path, str],
    other: str,
) -> None:
    root, base = repository
    for name in ("tooling/marp/package.json", other):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("changed")
    _git(root, "add", ".")
    assert (
        classify_changes(root, base, _git(root, "write-tree")).delivery_profile
        == "full"
    )


@pytest.mark.parametrize("extra", ["", "\nsrc/\n", "\n #pattern\n", "\n \n"])
def test_only_generated_preview_ignore_rule_uses_presentation_scope(
    repository: tuple[Path, str],
    extra: str,
) -> None:
    root, base = repository
    (root / ".gitignore").write_text(
        "/docs/talks/tradingdev-introduction/index.html\n" + extra
    )
    _git(root, "add", ".gitignore")
    assert classify_changes(root, base, _git(root, "write-tree")).kind == (
        "code" if extra else "presentations"
    )


def test_removing_generated_preview_retains_presentation_validation(
    repository: tuple[Path, str],
) -> None:
    root, _ = repository
    name = "docs/talks/tradingdev-introduction/index.html"
    path = root / name
    path.parent.mkdir(parents=True)
    path.write_text("<html>saved output</html>")
    _git(root, "add", name)
    base = _git(root, "write-tree")
    _git(root, "rm", "--cached", name)
    assert classify_changes(root, base, _git(root, "write-tree")).candidate_checks == (
        "presentations",
    )
