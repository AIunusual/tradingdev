"""Documentation routing is based on both sides of immutable Git changes."""

from __future__ import annotations

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import pytest
from tooling.git.change_scope import documentation_only

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
        ("README.md", True),
        ("ARCHITECTURE.md", True),
        ("docs/guide.md", True),
        ("docs/nested/中文 guide\nnotes.md", True),
        ("AGENTS.md", False),
        ("CLAUDE.md", False),
        ("OTHER.md", False),
        (".agents/policy.md", False),
        ("docs/config.yaml", False),
        ("docs/check.sh", False),
        ("src/README.md", False),
        ("pyproject.toml", False),
        ("uv.lock", False),
        (".githooks/pre-commit", False),
        ("tests/test_app.py", False),
    ],
)
def test_documentation_allowlist(
    repository: tuple[Path, str], name: str, expected: bool
) -> None:
    root, base = repository
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Changed\n")
    _git(root, "add", name)
    assert documentation_only(root, base, _git(root, "write-tree")) is expected


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
    assert documentation_only(root, base, _git(root, "write-tree")) is (
        change in {"delete", "rename"}
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
    assert not documentation_only(root, base, _git(root, "write-tree"))


def test_empty_changes_and_working_tree_edits_do_not_select_docs(
    repository: tuple[Path, str],
) -> None:
    root, base = repository
    (root / "README.md").write_text("# Unstaged\n")
    assert not documentation_only(root, base, _git(root, "write-tree"))
    _git(root, "add", "README.md")
    tree = _git(root, "write-tree")
    (root / "app.py").write_text("VALUE = 2\n")
    assert documentation_only(root, base, tree)


def test_invalid_git_identity_fails_instead_of_selecting_docs(
    repository: tuple[Path, str],
) -> None:
    root, base = repository
    with pytest.raises(subprocess.CalledProcessError):
        documentation_only(root, base, "missing-tree")
