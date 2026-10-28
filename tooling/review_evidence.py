"""Read complete review evidence from fixed Git trees, never working files."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tooling.presentations import PRESENTATION_PATHS, is_presentation_path
from tooling.review_options import validate_review_options

MAX_EVIDENCE_BYTES = 600_000


class ReviewError(RuntimeError):
    """Documentation review cannot approve incomplete or inconsistent evidence."""


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "--literal-pathspecs", *args],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ReviewError(result.stderr.decode("utf-8", errors="replace").strip())
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReviewError("Review evidence contains a non-UTF-8 file.") from exc


def _relevant(path: str) -> bool:
    candidate = Path(path)
    if candidate.parts[0] in {"workspace", ".git", ".venv"}:
        return False
    return (
        path.startswith(
            ("scripts/", "tooling/", "research/", ".githooks/", ".Codex/hooks/")
        )
        or candidate.suffix
        in {
            ".py",
            ".sh",
            ".toml",
            ".ini",
            ".yaml",
            ".yml",
            ".json",
            ".md",
            ".html",
            ".htm",
            ".css",
        }
        or path
        in {
            ".gitignore",
            ".gitattributes",
            "docs/.nojekyll",
        }
    )


def _tree_entries(repo: Path, tree: str) -> dict[str, tuple[str, str, str]]:
    entries: dict[str, tuple[str, str, str]] = {}
    for entry in _git(repo, "ls-tree", "-r", "-z", tree).split("\0"):
        if entry:
            metadata, path = entry.split("\t", 1)
            mode, kind, oid = metadata.split()
            entries[path] = mode, kind, oid
    return entries


def _text_blob(repo: Path, tree: str, path: str, entry: tuple[str, str, str]) -> str:
    if entry[1] != "blob":
        raise ReviewError(f"Required review evidence {path!r} is not a Git blob.")
    try:
        content = _git(repo, "show", f"{tree}:{path}")
    except ReviewError as exc:
        raise ReviewError(
            f"Cannot read required review evidence {path!r}: {exc}"
        ) from exc
    if "\0" in content:
        raise ReviewError(f"Required review evidence {path!r} contains binary data.")
    return content


def _presentation_context(
    changed: set[str], entries: dict[str, tuple[str, str, str]]
) -> set[str]:
    present = PRESENTATION_PATHS & entries.keys()
    if not (changed & PRESENTATION_PATHS) and not (
        present and any(is_presentation_path(path) for path in changed)
    ):
        return set()
    missing = PRESENTATION_PATHS - entries.keys()
    if missing:
        raise ReviewError(
            "Missing required registered presentation review evidence: "
            + ", ".join(sorted(missing))
        )
    for path in sorted(PRESENTATION_PATHS):
        if entries[path][:2] != ("100644", "blob"):
            raise ReviewError(
                f"Required presentation review evidence {path!r} "
                "is not an ordinary non-executable Git blob."
            )
    return set(PRESENTATION_PATHS)


def _exclusion_reason(path: str) -> str:
    if Path(path).parts[0] in {"workspace", ".git", ".venv"}:
        return "Runtime workspace or local environment content is excluded."
    return (
        "Outside the supported text evidence selection, "
        "including binary assets and lockfiles."
    )


def automatic_document_paths(paths: set[str]) -> set[str]:
    """Return the document set selected independently of explicit context."""
    return {
        path
        for path in paths
        if path in {"README.md", "ARCHITECTURE.md", "AGENTS.md", "CLAUDE.md"}
        or (path.startswith("docs/") and path.endswith(".md"))
    }


def build_evidence(
    repo: Path,
    base: str,
    tree: str,
    *,
    max_bytes: int = MAX_EVIDENCE_BYTES,
    review_context: tuple[str, ...] = (),
    review_context_for: tuple[tuple[str, str], ...] = (),
) -> str:
    """Read candidate blobs, never index or working-tree file contents."""
    base_tree = _git(repo, "rev-parse", "--verify", f"{base}^{{tree}}").strip()
    candidate_tree = _git(repo, "rev-parse", "--verify", f"{tree}^{{tree}}").strip()
    changed = _git(
        repo, "diff", "--name-only", "--no-renames", "-z", base_tree, candidate_tree
    ).split("\0")
    changed = [path for path in changed if path]
    relevant = sorted(path for path in changed if _relevant(path))
    entries = _tree_entries(repo, candidate_tree)
    files = set(entries)
    docs = automatic_document_paths(files)
    context = _presentation_context(set(changed), entries)
    try:
        validate_review_options(
            review_context=review_context, review_context_for=review_context_for
        )
    except RuntimeError as exc:
        raise ReviewError(str(exc)) from exc
    all_context = set(review_context) | {path for _, path in review_context_for}
    for path in all_context:
        relative = Path(path)
        if (
            not path
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != path
            or path not in entries
            or relative.parts[0] in {"workspace", ".git", ".venv"}
            or entries[path][:2] not in {("100644", "blob"), ("100755", "blob")}
        ):
            raise ReviewError(
                f"Additional review context must name a regular non-runtime file "
                f"in the candidate Git tree: {path!r}"
            )
    # Explicit context can supply text outside automatic selection. If changed,
    # it also needs an owner and its exact diff; do not still label it excluded.
    relevant = sorted(set(relevant) | (all_context & set(changed)))
    candidate_files = {
        path: _text_blob(repo, candidate_tree, path, entries[path])
        for path in sorted(docs | context | all_context | (set(relevant) & files))
    }
    # Force textual diffs only after validating both sides. Git attributes must
    # not hide selected source changes behind a "Binary files differ" marker.
    base_entries = _tree_entries(repo, base_tree)
    for path in relevant:
        if path in base_entries:
            _text_blob(repo, base_tree, path, base_entries[path])
    path_diffs = {
        path: _git(
            repo,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "--text",
            "--unified=5",
            base_tree,
            candidate_tree,
            "--",
            path,
        )
        for path in relevant
    }
    excluded = sorted(set(changed) - set(relevant))
    evidence = {
        "base_tree": base_tree,
        "candidate_tree": candidate_tree,
        "changed_paths": changed,
        "excluded_from_content": excluded,
        "exclusion_reasons": {path: _exclusion_reason(path) for path in excluded},
        "selection": (
            "All changed scripts (including extensionless Git hooks), source, "
            "configuration, Markdown, HTML and CSS text files; docs/.nojekyll; "
            "all candidate README, ARCHITECTURE, agent policies and docs/**/*.md. "
            "Registered presentation changes and related build/deployment changes "
            "include the complete candidate presentation source, theme, HTML and "
            "README as context. Selected content must be UTF-8 without NUL bytes. "
            "Other paths, including binary assets and unsupported lockfiles, and "
            "runtime workspace contents are excluded as individually listed. "
            "Explicit context can add regular UTF-8 text outside automatic "
            "selection, but never runtime content; changed additions get diffs. "
            "The tree inventory lists every candidate path; the file inventory "
            "lists complete files available to the scope planner. Changed files "
            "and required presentation context are assigned in full; unchanged "
            "Markdown may be supplied as exact, identified responsibility sections."
        ),
        "schema_version": 2,
        "path_diffs": path_diffs,
        "candidate_tree_inventory": sorted(files),
        "candidate_file_inventory": sorted(candidate_files),
        "candidate_file_modes": {path: entries[path][0] for path in candidate_files},
        "candidate_file_blobs": {path: entries[path][2] for path in candidate_files},
        "base_file_blobs": {
            path: base_entries[path][2] for path in relevant if path in base_entries
        },
        "base_file_modes": {
            path: base_entries[path][0] for path in relevant if path in base_entries
        },
        "presentation_context_paths": sorted(context),
        "additional_context_paths": sorted(set(review_context)),
        "automatic_document_paths": sorted(docs),
        "scoped_context_paths": {
            scope: sorted(
                {path for target, path in review_context_for if target == scope}
            )
            for scope in sorted({scope for scope, _ in review_context_for})
        },
        "candidate_files": candidate_files,
    }
    content = json.dumps(evidence, ensure_ascii=False)
    size = len(content.encode("utf-8"))
    if size > max_bytes:
        raise ReviewError(
            f"Documentation evidence is {size} bytes, exceeding the {max_bytes}-byte "
            "budget. Split the change or explicitly raise --max-evidence-bytes; "
            "no evidence was truncated and no model request was made."
        )
    return content
