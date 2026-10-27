"""Classify immutable Git changes for documentation, presentations or code checks."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from tooling.presentations import is_presentation_path, presentation_change_allowed


@dataclass(frozen=True)
class ChangeScope:
    kind: Literal["docs", "presentations", "code"]
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
    code_changed = presentations_changed = False
    for metadata, name in zip(fields[::2], fields[1::2], strict=True):
        parts = metadata.split()
        if len(parts) != 5 or not parts[0].startswith(":"):
            raise RuntimeError("Cannot classify the Git candidate changes.")
        old_mode, new_mode, old_blob, new_blob, status = parts
        presentation = is_presentation_path(name)
        presentations_changed |= presentation
        if (
            old_mode[1:] not in {"000000", "100644"}
            or new_mode not in {"000000", "100644"}
            or status not in {"A", "M", "D"}
        ):
            code_changed = True
            continue
        if presentation:
            before = _read_blob(repository, old_blob) if old_mode != ":000000" else None
            after = _read_blob(repository, new_blob) if new_mode != "000000" else None
            if not presentation_change_allowed(name, before, after):
                code_changed = True
        elif not _documentation_path(name):
            code_changed = True
    if code_changed:
        return ChangeScope("code", presentations_changed)
    return ChangeScope(
        "presentations" if presentations_changed else "docs", presentations_changed
    )


def _read_blob(repository: Path, oid: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repository), "cat-file", "blob", oid]
    )


def _documentation_path(name: str) -> bool:
    path = PurePosixPath(name)
    return name in {"README.md", "ARCHITECTURE.md"} or (
        len(path.parts) > 1 and path.parts[0] == "docs" and path.suffix == ".md"
    )
