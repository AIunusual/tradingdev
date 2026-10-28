"""Discover persisted references that prevent removal of strategy drafts."""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import yaml

from tradingdev.adapters.storage.comparisons import ComparisonStore
from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.strategy_revisions import (
    StrategyRevisionIntegrityError,
)
from tradingdev.domain.comparison import ComparisonManifest
from tradingdev.domain.comparison_results import canonical_json_object
from tradingdev.domain.execution import ExecutionManifest

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tradingdev.adapters.storage.filesystem import WorkspacePaths
    from tradingdev.adapters.storage.sqlite import SQLiteStore


def referenced_revisions(
    workspace: WorkspacePaths, store: SQLiteStore, strategy_id: str
) -> dict[str, list[str]]:
    """Include all jobs/runs and orphaned manifests, regardless of job outcome.

    Invalid historical data fails the scan closed. It must be repaired explicitly
    before callers can prove that a draft is unreferenced.
    """
    references: dict[str, list[str]] = {}

    def retain(
        revision_id: object, reason: str, *, legacy_config: object = None
    ) -> None:
        if revision_id is None:
            if _is_legacy_reference(workspace, strategy_id, legacy_config):
                return
            raise StrategyRevisionIntegrityError(
                f"Cannot establish the revision referenced by {reason}; "
                "repair its history before cleaning this strategy"
            )
        if not isinstance(revision_id, str) or not revision_id:
            raise StrategyRevisionIntegrityError("Invalid stored revision reference")
        try:
            parsed = UUID(revision_id)
        except ValueError as exc:
            raise StrategyRevisionIntegrityError(
                f"Invalid stored revision reference: {reason}"
            ) from exc
        if parsed.version != 4 or parsed.hex != revision_id:
            raise StrategyRevisionIntegrityError(
                f"Invalid stored revision reference: {reason}"
            )
        references.setdefault(revision_id, []).append(reason)

    # list_jobs is a user-facing projection that omits non-object payloads.
    # Cleanup must instead validate every raw row against its indexed identity.
    with closing(store.connect()) as connection:
        jobs = connection.execute(
            "select job_id, job_type, strategy_name, config_path, payload from jobs"
        ).fetchall()
    for row in jobs:
        payload = row["payload"]
        if not isinstance(payload, str):
            raise StrategyRevisionIntegrityError("Stored job payload is not JSON text")
        job = json.loads(payload)
        if not isinstance(job, dict) or any(
            job.get(key, "backtest" if key == "job_type" else None) != row[key]
            for key in ("job_id", "job_type", "strategy_name", "config_path")
        ):
            raise StrategyRevisionIntegrityError(
                f"Invalid job payload or indexed identity mismatch: {row['job_id']}"
            )
        if row["strategy_name"] is None and row["job_type"] in {
            "backtest",
            "walk_forward",
            "optimization",
            "comparison",
        }:
            raise StrategyRevisionIntegrityError(
                f"Unknown strategy reference for job: {row['job_id']}"
            )
        if row["strategy_name"] is not None and (
            not isinstance(row["strategy_name"], str) or not row["strategy_name"]
        ):
            raise StrategyRevisionIntegrityError(
                f"Invalid strategy identity for job: {row['job_id']}"
            )
        if row["strategy_name"] == strategy_id:
            retain(
                job.get("revision_id"),
                f"job:{row['job_id']}",
                legacy_config=job.get("config_path")
                if job.get("manifest_hash") is None
                else None,
            )
    for run in store.list_runs():
        stored_strategy_id = run.get("strategy_id")
        if not isinstance(stored_strategy_id, str) or not stored_strategy_id:
            raise StrategyRevisionIntegrityError("Unknown strategy reference for run")
        if stored_strategy_id == strategy_id:
            artifact_dir = run.get("artifact_dir")
            retain(
                run.get("revision_id"),
                f"run:{run['run_id']}",
                legacy_config=str(Path(artifact_dir) / "config.yaml")
                if isinstance(artifact_dir, str) and run.get("manifest_hash") is None
                else None,
            )
    manifests = ExecutionManifestStore(workspace)
    if workspace.runs.is_symlink():
        raise StrategyRevisionIntegrityError("Run storage is a symlink")
    for directory in sorted(workspace.runs.iterdir()):
        if directory.is_symlink():
            raise StrategyRevisionIntegrityError("Run storage contains a symlink")
        if not directory.is_dir():
            continue
        path = manifests.path(directory.name)
        if not path.exists():
            continue
        manifest = ExecutionManifest.model_validate_json(path.read_bytes())
        manifest.verify()
        strategy = manifest.config_copy().get("strategy")
        if (
            not isinstance(strategy, dict)
            or not isinstance(strategy.get("id"), str)
            or not strategy["id"]
        ):
            raise StrategyRevisionIntegrityError(
                f"Unknown strategy reference for manifest: {directory.name}"
            )
        if strategy["id"] == strategy_id:
            retain(strategy.get("revision_id"), f"manifest:{directory.name}")
    for job_id, comparison in _comparison_manifests(workspace):
        for candidate in comparison.candidates:
            strategy = candidate.manifest.config_copy().get("strategy")
            reason = f"comparison:{job_id}:{candidate.candidate_id}"
            if (
                not isinstance(strategy, dict)
                or not isinstance(strategy.get("id"), str)
                or not strategy["id"]
            ):
                raise StrategyRevisionIntegrityError(
                    f"Unknown strategy reference for {reason}"
                )
            if strategy["id"] == strategy_id:
                retain(strategy.get("revision_id"), reason)
    return references


def _comparison_manifests(
    workspace: WorkspacePaths,
) -> Iterator[tuple[str, ComparisonManifest]]:
    """Read all orphaned and registered comparisons without relying on outcomes."""
    root = workspace.root / "comparisons"
    if root.is_symlink():
        raise StrategyRevisionIntegrityError("Comparison storage is a symlink")
    if not root.exists():
        return
    if not root.is_dir():
        raise StrategyRevisionIntegrityError("Comparison storage is not a directory")
    comparisons = ComparisonStore(workspace)
    for directory in sorted(root.iterdir()):
        if directory.is_symlink() or not directory.is_dir():
            raise StrategyRevisionIntegrityError(
                "Comparison storage contains an invalid directory"
            )
        path = comparisons.path(directory.name)
        entries = tuple(directory.iterdir())
        if any(entry.is_symlink() for entry in entries):
            raise StrategyRevisionIntegrityError(
                "Comparison storage contains a symlink"
            )
        if not path.exists():
            if any(
                entry.name != ".comparison.lock" or not entry.is_file()
                for entry in entries
            ):
                raise StrategyRevisionIntegrityError(
                    f"Comparison artifacts are missing their manifest: {directory.name}"
                )
            # Empty/lock-only directories contain no published research evidence.
            continue
        if not path.is_file():
            raise StrategyRevisionIntegrityError(
                f"Comparison manifest is not a regular file: {directory.name}"
            )
        try:
            manifest = ComparisonManifest.model_validate_json(
                canonical_json_object(path.read_text(encoding="utf-8"))
            )
            manifest.verify()
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise StrategyRevisionIntegrityError(
                f"Invalid comparison manifest: {directory.name}"
            ) from exc
        yield directory.name, manifest


def _is_legacy_reference(
    workspace: WorkspacePaths, strategy_id: str, config_path: object
) -> bool:
    """Recognize explicit flat-source history without inferring a revision.

    Missing or ambiguous legacy evidence protects the whole target strategy.
    This reads existing config only; it never repairs or migrates old history.
    """
    if not isinstance(config_path, str) or not config_path:
        return False
    path = Path(config_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_relative_to(workspace.root):
        return False
    for candidate in (path, *path.parents):
        if candidate == workspace.root:
            break
        if candidate.is_symlink():
            return False
    if not path.resolve().is_relative_to(workspace.root) or not path.is_file():
        return False
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise StrategyRevisionIntegrityError(
            "Cannot read legacy reference config"
        ) from exc
    strategy = config.get("strategy") if isinstance(config, dict) else None
    if (
        not isinstance(strategy, dict)
        or strategy.get("id") != strategy_id
        or strategy.get("revision_id") is not None
        or not isinstance(strategy.get("source_path"), str)
    ):
        return False
    source = Path(strategy["source_path"])
    expected = workspace.generated_strategies / f"{strategy_id}.py"
    return source.absolute() == expected and not source.is_symlink()
