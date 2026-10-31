"""Classify immutable changes by their required verification boundary."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from tooling.presentations import is_presentation_path, presentation_change_allowed

# These tools do not implement product behavior or shared product-test facilities.
# New/unknown tools remain on the full path until their consumers are reviewed.
_TOOLING_FILES = frozenset(
    {
        "AGENTS.md",
        "CLAUDE.md",
        "tooling/documentation.py",
        "tooling/quality.py",
        "tooling/presentations.py",
        "tooling/review_docs.py",
        "tooling/review_evidence.py",
        "tooling/review_options.py",
        "tooling/review_plan.py",
        "tooling/review_scope.py",
        "tooling/git/__init__.py",
        "tooling/git/candidates.py",
        "tooling/git/change_scope.py",
        "tooling/git/check_pr.py",
        "tooling/git/claim_issue.py",
        "tooling/git/cleanup_pr.py",
        "tooling/git/gate.py",
        "tooling/git/pr_context.py",
    }
)
_TOOLING_SCRIPTS = frozenset(
    {
        ".githooks/commit-msg",
        ".githooks/pre-commit",
        ".githooks/pre-push",
        "scripts/check-tooling.sh",
        "scripts/check-docs.sh",
        "scripts/check-fast.sh",
        "scripts/check-pr.sh",
        "scripts/check-presentations.sh",
        "scripts/claim-issue.sh",
        "scripts/cleanup-pr.sh",
        "scripts/detect-primary-branch.sh",
        "scripts/format.sh",
        "scripts/git-flow-commit.sh",
        "scripts/install-git-hooks.sh",
        "scripts/presentations.sh",
    }
)


@dataclass(frozen=True)
class ChangeScope:
    kind: Literal["docs", "presentations", "tooling", "code"]
    presentations_changed: bool = False


def classify_changes(repository: Path, base: str, tree: str) -> ChangeScope:
    """Inspect both Git sides; unknown changes keep code checks and related checks."""
    output = subprocess.check_output(
        [
            "git",
            "-C",
            str(repository),
            "diff-tree",
            "--no-commit-id",
            "--raw",
            "--no-abbrev",
            "--no-renames",
            "--no-ext-diff",
            "--no-textconv",
            "-r",
            "-z",
            base,
            tree,
            "--",
        ],
        text=True,
    )
    if not output:
        return ChangeScope("code")
    fields = output.split("\0")
    if fields.pop() != "" or len(fields) % 2:
        raise RuntimeError("Cannot classify the Git candidate changes.")
    code_changed = presentations_changed = tooling_changed = False
    for metadata, name in zip(fields[::2], fields[1::2], strict=True):
        parts = metadata.split()
        if len(parts) != 5 or not parts[0].startswith(":"):
            raise RuntimeError("Cannot classify the Git candidate changes.")
        old_mode, new_mode, old_blob, new_blob, status = parts
        presentation = is_presentation_path(name)
        presentations_changed |= presentation
        modes = {"000000", "100644"}
        if name in _TOOLING_SCRIPTS:
            modes.add("100755")
        if (
            old_mode[1:] not in modes
            or new_mode not in modes
            or status not in {"A", "M", "D"}
        ):
            code_changed = True
            continue
        if _tooling_path(name):
            tooling_changed = True
        elif presentation:
            before = _read_blob(repository, old_blob) if old_mode != ":000000" else None
            after = _read_blob(repository, new_blob) if new_mode != "000000" else None
            if not presentation_change_allowed(name, before, after):
                code_changed = True
        elif not _documentation_path(name):
            code_changed = True
    if code_changed:
        return ChangeScope("code", presentations_changed)
    if tooling_changed:
        return ChangeScope("tooling", presentations_changed)
    return ChangeScope(
        "presentations" if presentations_changed else "docs", presentations_changed
    )


def _read_blob(repository: Path, oid: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repository), "cat-file", "blob", oid]
    )


def _tooling_path(name: str) -> bool:
    path = PurePosixPath(name)
    return name in _TOOLING_FILES | _TOOLING_SCRIPTS or (
        path.parent == PurePosixPath("tests/tooling")
        and path.name.startswith("test_")
        and path.suffix == ".py"
    )


def _documentation_path(name: str) -> bool:
    path = PurePosixPath(name)
    return name in {"README.md", "ARCHITECTURE.md"} or (
        len(path.parts) > 1 and path.parts[0] == "docs" and path.suffix == ".md"
    )
