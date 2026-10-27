"""Check routing is based on both sides of immutable Git changes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import pytest
from tooling.git.change_scope import ChangeScope, classify_changes

if TYPE_CHECKING:
    from collections.abc import Iterator


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
        ("AGENTS.md", "code"),
        ("CLAUDE.md", "code"),
        ("OTHER.md", "code"),
        (".agents/policy.md", "code"),
        ("docs/config.yaml", "code"),
        ("docs/check.sh", "code"),
        ("src/README.md", "code"),
        ("pyproject.toml", "code"),
        ("uv.lock", "code"),
        (".githooks/pre-commit", "code"),
        ("tests/test_app.py", "code"),
    ],
)
def test_documentation_allowlist(
    repository: tuple[Path, str], name: str, expected: str
) -> None:
    root, base = repository
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Changed\n")
    _git(root, "add", name)
    assert classify_changes(root, base, _git(root, "write-tree")).kind == expected


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
    "name, content, expected",
    [
        ("docs/talks/tradingdev-introduction/slides.md", "# Slides\n", "presentations"),
        ("docs/talks/tradingdev-introduction/theme.css", "body {}\n", "presentations"),
        ("docs/talks/tradingdev-introduction/index.html", "<html>\n", "presentations"),
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
    assert scope.kind == ("presentations" if other == "README.md" else "code")
    assert scope.presentations_changed


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
