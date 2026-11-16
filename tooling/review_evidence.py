"""Read complete review evidence from fixed Git trees, never working files."""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

from tooling.presentations import DECK, PRESENTATION_PATHS, is_presentation_path
from tooling.review_options import validate_review_options
from tooling.review_scope import source_context_for_path

MAX_EVIDENCE_BYTES = 600_000
PRESENTATION_GUIDE = f"{DECK}/README.md"
PRESENTATION_ARTIFACT_PATHS = PRESENTATION_PATHS - {PRESENTATION_GUIDE}
# These entrypoints manage editing/setup/deployment, not rendered slide content.
# Other registered presentation paths retain the complete deck as evidence.
_PRESENTATION_OPERATION_PATHS = frozenset(
    {
        PRESENTATION_GUIDE,
        ".vscode/tasks.json",
        ".gitignore",
        "docs/.nojekyll",
        "scripts/presentations.sh",
        "scripts/check-presentations.sh",
        "scripts/stage-presentations.sh",
        "scripts/build-presentations-site.sh",
        "wrangler.jsonc",
        "tooling/presentation_errors.py",
    }
)
# presentations.py also owns rendering. Only changes confined to these existing
# operation responsibilities may omit unchanged artwork; unknown nodes fail closed.
_PRESENTATION_OPERATION_DEFINITIONS = frozenset(
    {
        "presentation_task",
        "_toolchain",
        "setup",
        "_install_toolchain",
        "main",
        "_InstallationRequiredError",
    }
)


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
        # Registered presentation operations include nonstandard text formats,
        # such as wrangler.jsonc. Keep this aligned with their context selector
        # without treating unknown JSONC files or binary deck assets as text.
        path in _PRESENTATION_OPERATION_PATHS
        or path.startswith(
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


def _operation_definitions_only(before: str, after: str) -> bool:
    """Classify a mixed module using both immutable blobs, without executing it."""

    def statements(content: str) -> dict[str, str]:
        body = ast.parse(content).body
        result: dict[str, str] = {}
        other = 0

        def lock_import(node: ast.AST) -> bool:
            return (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module == "filelock"
                and len(node.names) == 1
                and node.names[0].name == "FileLock"
                and node.names[0].asname is None
            )

        # A new installer import must not alter a renderer's existing binding.
        lock_is_operation_only = not any(
            isinstance(child, ast.Name)
            and child.id == "FileLock"
            or isinstance(child, ast.FunctionDef | ast.ClassDef)
            and child.name == "FileLock"
            or isinstance(child, ast.alias)
            and (child.asname or child.name) == "FileLock"
            for node in body
            if not lock_import(node)
            and not (
                isinstance(node, ast.FunctionDef | ast.ClassDef)
                and node.name in _PRESENTATION_OPERATION_DEFINITIONS
            )
            for child in ast.walk(node)
        )
        for node in body:
            if lock_import(node) and lock_is_operation_only:
                continue
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == "TYPE_CHECKING"
                and not node.orelse
                and all(
                    isinstance(statement, ast.ImportFrom)
                    and statement.level == 0
                    and statement.module == "collections.abc"
                    and all(
                        alias.name in {"Callable", "Generator"} and alias.asname is None
                        for alias in statement.names
                    )
                    for statement in node.body
                )
            ):
                continue
            if isinstance(node, ast.FunctionDef | ast.ClassDef):
                key = f"<decorated:{node.name}>" if node.decorator_list else node.name
            else:
                key = f"<statement:{other}>"
                other += 1
            if key in result:
                raise ValueError("Ambiguous module definition")
            result[key] = ast.dump(node)
        return result

    try:
        old, new = statements(before), statements(after)
    except SyntaxError, ValueError, RecursionError:
        return False
    return {
        key for key in old.keys() | new.keys() if old.get(key) != new.get(key)
    } <= _PRESENTATION_OPERATION_DEFINITIONS


def _presentation_context(
    repo: Path,
    base: str,
    tree: str,
    changed: set[str],
    entries: dict[str, tuple[str, str, str]],
    base_entries: dict[str, tuple[str, str, str]],
) -> set[str]:
    present = PRESENTATION_PATHS & entries.keys()
    if not (changed & PRESENTATION_PATHS) and not (
        present and any(is_presentation_path(path) for path in changed)
    ):
        return set()
    related = {path for path in changed if is_presentation_path(path)}
    artwork_changes = related - _PRESENTATION_OPERATION_PATHS
    mixed_module = "tooling/presentations.py"
    if (
        mixed_module in artwork_changes
        and mixed_module in base_entries
        and mixed_module in entries
        and base_entries[mixed_module][:2]
        == entries[mixed_module][:2]
        == ("100644", "blob")
        and _operation_definitions_only(
            _text_blob(repo, base, mixed_module, base_entries[mixed_module]),
            _text_blob(repo, tree, mixed_module, entries[mixed_module]),
        )
    ):
        artwork_changes.remove(mixed_module)
    required = set(PRESENTATION_PATHS) if artwork_changes else {PRESENTATION_GUIDE}
    missing = required - entries.keys()
    if missing:
        raise ReviewError(
            "Missing required registered presentation review evidence: "
            + ", ".join(sorted(missing))
        )
    for path in sorted(required):
        if entries[path][:2] != ("100644", "blob"):
            raise ReviewError(
                f"Required presentation review evidence {path!r} "
                "is not an ordinary non-executable Git blob."
            )
    return required


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
        if path not in PRESENTATION_ARTIFACT_PATHS
        and (
            path in {"README.md", "ARCHITECTURE.md", "AGENTS.md", "CLAUDE.md"}
            or (path.startswith("docs/") and path.endswith(".md"))
        )
    }


def build_evidence(
    repo: Path,
    base: str,
    tree: str,
    *,
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
    base_entries = _tree_entries(repo, base_tree)
    files = set(entries)
    docs = automatic_document_paths(files)
    context = _presentation_context(
        repo, base_tree, candidate_tree, set(changed), entries, base_entries
    )
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
    source_contexts: dict[str, dict[str, str | list[str]]] = {}
    source_paths: dict[str, set[str]] = {"candidate": set(), "base": set()}
    for anchor in relevant:
        profile = source_context_for_path(anchor)
        if profile is None:
            continue
        version = "candidate" if anchor in entries else "base"
        source_entries = entries if version == "candidate" else base_entries
        if anchor not in source_entries:
            raise ReviewError(f"Registered changed source has no Git entry: {anchor}")
        paths = sorted(set(profile.required_paths))
        for path in paths:
            relative = Path(path)
            if (
                not path
                or relative.is_absolute()
                or ".." in relative.parts
                or relative.as_posix() != path
                or relative.parts[0] in {"workspace", ".git", ".venv"}
                or path not in source_entries
                or source_entries[path][:2]
                not in {("100644", "blob"), ("100755", "blob")}
            ):
                raise ReviewError(
                    f"Required source context for {anchor!r} must name a regular "
                    f"non-runtime file in the {version} Git tree: {path!r}"
                )
        source_contexts[anchor] = {"version": version, "paths": paths}
        source_paths[version].update(paths)
    candidate_files = {
        path: _text_blob(repo, candidate_tree, path, entries[path])
        for path in sorted(
            docs
            | context
            | all_context
            | source_paths["candidate"]
            | (set(relevant) & files)
        )
    }
    base_source_files = {
        path: _text_blob(repo, base_tree, path, base_entries[path])
        for path in sorted(source_paths["base"])
    }
    base_evidence_paths = (set(relevant) & base_entries.keys()) | source_paths["base"]
    # Force textual diffs only after validating both sides. Git attributes must
    # not hide selected source changes behind a "Binary files differ" marker.
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
            "configuration, Markdown, HTML and CSS text files; registered "
            "presentation operation files, including wrangler.jsonc; "
            "all candidate README, ARCHITECTURE, agent policies and docs/**/*.md. "
            "Registered presentation operation changes include the editing guide. "
            "Artwork, renderer and unclassified presentation changes additionally "
            "include the complete candidate slide source and theme. Unchanged "
            "slide source is an artifact, not automatic documentation context. "
            "Selected content must be UTF-8 without NUL bytes. "
            "Other paths, including binary assets and unsupported lockfiles, and "
            "runtime workspace contents are excluded as individually listed. "
            "Explicit context can add regular UTF-8 text outside automatic "
            "selection, but never runtime content; changed additions get diffs. "
            "Registered changed paths require their complete supporting source "
            "blobs from the candidate tree, or the base tree for deleted paths. "
            "Their source_contexts mapping assigns those originals to consumers "
            "without adding supporting sources to changed-path ownership. "
            "The tree inventory lists every candidate path; the file inventory "
            "lists complete files available to the scope planner. Changed files "
            "and required presentation context are assigned in full; unchanged "
            "Markdown may be supplied as exact, identified responsibility sections."
        ),
        "schema_version": 3,
        "path_diffs": path_diffs,
        "candidate_tree_inventory": sorted(files),
        "candidate_file_inventory": sorted(candidate_files),
        "candidate_file_modes": {path: entries[path][0] for path in candidate_files},
        "candidate_file_blobs": {path: entries[path][2] for path in candidate_files},
        "base_file_blobs": {
            path: base_entries[path][2] for path in sorted(base_evidence_paths)
        },
        "base_file_modes": {
            path: base_entries[path][0] for path in sorted(base_evidence_paths)
        },
        "base_document_files": {
            path: _text_blob(repo, base_tree, path, base_entries[path])
            for path in relevant
            if path.endswith(".md") and path in base_entries
        },
        "presentation_context_paths": sorted(context),
        "source_contexts": source_contexts,
        "base_source_files": base_source_files,
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
    return json.dumps(evidence, ensure_ascii=False)
