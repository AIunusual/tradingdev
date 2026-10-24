"""Classify immutable Git changes conservatively for documentation checks."""

from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath


def documentation_only(repository: Path, base: str, tree: str) -> bool:
    """Allow only ordinary documentation files; all other changes use code checks."""
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
        return False
    fields = output.split("\0")
    if fields.pop() != "" or len(fields) % 2:
        raise RuntimeError("Cannot classify the Git candidate changes.")
    for metadata, name in zip(fields[::2], fields[1::2], strict=True):
        parts = metadata.split()
        if len(parts) != 5 or not parts[0].startswith(":"):
            raise RuntimeError("Cannot classify the Git candidate changes.")
        old_mode, new_mode, _, _, status = parts
        if (
            old_mode[1:] not in {"000000", "100644"}
            or new_mode not in {"000000", "100644"}
            or status not in {"A", "M", "D"}
        ):
            return False
        path = PurePosixPath(name)
        if name not in {"README.md", "ARCHITECTURE.md"} and not (
            len(path.parts) > 1 and path.parts[0] == "docs" and path.suffix == ".md"
        ):
            return False
    return True
