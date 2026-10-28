"""Cleanup safeguards use isolated, immediately removed workspace fixtures."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from tradingdev.adapters.storage.comparisons import ComparisonStore
from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.adapters.storage.strategy_revisions import StrategyRevisionStore
from tradingdev.app.contracts.common import ErrorResponse
from tradingdev.app.contracts.strategy_cleanup import StrategyCleanupResult
from tradingdev.app.strategy_cleanup_service import StrategyCleanupService
from tradingdev.app.strategy_service import StrategyService
from tradingdev.domain.comparison import ComparisonCandidate, ComparisonManifest
from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.strategies.execution import StrategyExecution
from tradingdev.domain.strategies.schemas import StrategyMetadata, StrategyStatus

if TYPE_CHECKING:
    from collections.abc import Generator


def _setup(
    tmp_path: Path,
) -> tuple[WorkspacePaths, SQLiteStore, StrategyRevisionStore, StrategyCleanupService]:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    revisions = StrategyRevisionStore(workspace)
    return workspace, store, revisions, StrategyCleanupService(workspace, store)


def _draft(revisions: StrategyRevisionStore) -> StrategyMetadata:
    current = revisions.load("fixture")
    return revisions.create(
        "fixture",
        "class Fixture: pass\n",
        {"strategy": {"class_name": "Fixture"}},
        expected_revision_id=None if current is None else current.revision_id,
    )


def _manifest(revision_id: str) -> ExecutionManifest:
    return ExecutionManifest.create(
        strategy_execution=StrategyExecution(kind="generated", constructor_kwargs={}),
        kind="backtest",
        config={
            "strategy": {"id": "fixture", "revision_id": revision_id},
            "backtest": {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-02-01",
                "init_cash": 10000,
            },
        },
    )


def _comparison(
    revision_id: str | None, *, strategy_id: str = "fixture"
) -> ComparisonManifest:
    candidates = []
    for index in (1, 2):
        config = _manifest(
            revision_id or "00000000000040008000000000000000"
        ).config_copy()
        config["strategy"]["id"] = strategy_id
        config["strategy"]["revision_id"] = revision_id
        config["strategy"]["parameters"] = {"period": index}
        candidates.append(
            ComparisonCandidate(
                candidate_id=f"candidate_{index}",
                label=f"Period {index}",
                rationale="Fixed before execution",
                manifest=ExecutionManifest.create(
                    kind="backtest",
                    config=config,
                    strategy_execution=StrategyExecution(
                        kind="generated", constructor_kwargs={"period": index}
                    ),
                ),
            )
        )
    return ComparisonManifest.create(candidates)


def test_preview_preserves_every_revision_and_apply_removes_only_selected_drafts(
    tmp_path: Path,
) -> None:
    workspace, _, revisions, service = _setup(tmp_path)
    first, second, current = [_draft(revisions) for _ in range(3)]
    cache = Path(first.source_path).parent / "__pycache__"
    cache.mkdir()
    (cache / "strategy.cpython-313.pyc").write_bytes(b"cache")
    before = {
        path: path.read_bytes() for path in workspace.root.rglob("*") if path.is_file()
    }

    preview = service.cleanup("fixture")
    assert isinstance(preview, StrategyCleanupResult)
    assert preview.success and not preview.applied
    outcomes = {item.revision_id: item.outcome for item in preview.revisions}
    assert outcomes == {
        first.revision_id: "eligible",
        second.revision_id: "eligible",
        current.revision_id: "protected",
    }
    assert {path: path.read_bytes() for path in before} == before
    new_current = _draft(revisions)
    removed = service.cleanup("fixture", [first.revision_id], apply=True)
    assert isinstance(removed, StrategyCleanupResult)
    assert removed.success and removed.applied
    assert removed.revisions[0].outcome == "deleted"
    assert not Path(first.source_path).parent.exists()
    assert revisions.load("fixture", second.revision_id) == second
    assert revisions.load("fixture", current.revision_id) == current
    assert revisions.load("fixture") == new_current
    repeated = service.cleanup("fixture", [first.revision_id], apply=True)
    assert isinstance(repeated, StrategyCleanupResult)
    assert repeated.success and repeated.revisions[0].outcome == "missing"
    assert not list(workspace.root.rglob(".pending-*"))


@pytest.mark.parametrize("ids", [None, []])
def test_apply_requires_explicit_nonempty_revision_ids(
    tmp_path: Path, ids: list[str] | None
) -> None:
    _, _, revisions, service = _setup(tmp_path)
    first = _draft(revisions)
    _draft(revisions)
    result = service.cleanup("fixture", ids, apply=True)
    assert isinstance(result, ErrorResponse)
    assert result.code == "cleanup_revision_ids_required"
    assert revisions.load("fixture", first.revision_id) == first


@pytest.mark.parametrize(
    ("apply", "revision_ids"),
    [
        (False, None),
        (False, []),
        (False, ["00000000000040008000000000000000"]),
        (True, ["00000000000040008000000000000000"]),
    ],
)
@pytest.mark.parametrize("populated", [False, True])
def test_unknown_strategies_are_rejected_without_creating_lock_files(
    tmp_path: Path, apply: bool, revision_ids: list[str] | None, populated: bool
) -> None:
    workspace, _, revisions, service = _setup(tmp_path)
    current = _draft(revisions) if populated else None
    before = {
        path: path.read_bytes()
        for path in workspace.generated_strategies.rglob("*")
        if path.is_file()
    }
    for index in range(3):
        result = service.cleanup(f"unknown_{index}", revision_ids, apply=apply)
        assert isinstance(result, ErrorResponse)
        assert not result.success and result.code == "strategy_not_found"
    assert {
        path: path.read_bytes()
        for path in workspace.generated_strategies.rglob("*")
        if path.is_file()
    } == before
    assert revisions.load("fixture") == current
    if not populated:
        assert not (workspace.generated_strategies / ".locks").exists()


@pytest.mark.parametrize("change", ["missing", "new_current"])
def test_cleanup_rechecks_current_after_lock_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    workspace, _, revisions, service = _setup(tmp_path)
    first = _draft(revisions)
    _draft(revisions)
    pointer = workspace.generated_strategies / "fixture" / "current.json"
    original_lock = service._revisions.lifecycle_lock

    @contextmanager
    def change_current(strategy_id: str) -> Generator[None]:
        with original_lock(strategy_id):
            if change == "missing":
                pointer.unlink()
            else:
                pointer.write_text(json.dumps({"revision_id": first.revision_id}))
            yield

    monkeypatch.setattr(service._revisions, "lifecycle_lock", change_current)
    result = service.cleanup("fixture", [first.revision_id], apply=True)
    assert not result.success
    if change == "missing":
        assert isinstance(result, ErrorResponse)
        assert result.code == "strategy_not_found"
    else:
        assert isinstance(result, StrategyCleanupResult)
        assert result.revisions[0].outcome == "protected"
        assert "current_revision" in result.revisions[0].reasons
    assert revisions.load("fixture", first.revision_id) == first


@pytest.mark.parametrize("status", list(StrategyStatus))
def test_current_is_protected_in_every_lifecycle_state(
    tmp_path: Path, status: StrategyStatus
) -> None:
    _, _, revisions, service = _setup(tmp_path)
    current = _draft(revisions)
    current.status = status
    revisions.update(current)
    result = service.cleanup("fixture", [current.revision_id], apply=True)
    assert isinstance(result, StrategyCleanupResult)
    assert not result.success
    assert "current_revision" in result.revisions[0].reasons
    assert revisions.load("fixture") == current


@pytest.mark.parametrize(
    "status", [status for status in StrategyStatus if status != StrategyStatus.DRAFT]
)
def test_apply_rechecks_status_changed_since_preview(
    tmp_path: Path, status: StrategyStatus
) -> None:
    _, _, revisions, service = _setup(tmp_path)
    first = _draft(revisions)
    _draft(revisions)
    preview = service.cleanup("fixture", [first.revision_id])
    assert isinstance(preview, StrategyCleanupResult)
    assert preview.revisions[0].outcome == "eligible"
    first.status = status
    revisions.update(first)
    result = service.cleanup("fixture", [first.revision_id], apply=True)
    assert isinstance(result, StrategyCleanupResult)
    assert not result.success
    assert result.revisions[0].reasons == [f"status:{status.value}"]
    assert revisions.load("fixture", first.revision_id) == first


@pytest.mark.parametrize("reference", ["job", "run", "manifest"])
def test_historical_and_orphaned_references_are_protected(
    tmp_path: Path, reference: str
) -> None:
    workspace, store, revisions, service = _setup(tmp_path)
    first = _draft(revisions)
    _draft(revisions)
    if reference == "job":
        store.upsert_job(
            {
                "job_id": "failed_job",
                "job_type": "backtest",
                "status": "failed",
                "strategy_name": "fixture",
                "revision_id": first.revision_id,
                "created_at": first.created_at,
            }
        )
    elif reference == "run":
        store.create_run(
            run_id="old_run",
            job_id="missing_job",
            strategy_id="fixture",
            revision_id=first.revision_id,
            artifact_dir=workspace.runs / "old_run",
            metrics={},
        )
    else:
        ExecutionManifestStore(workspace).publish(
            "orphan", _manifest(first.revision_id)
        )
    result = service.cleanup("fixture", [first.revision_id], apply=True)
    assert isinstance(result, StrategyCleanupResult)
    assert not result.success
    assert result.revisions[0].outcome == "protected"
    assert result.revisions[0].reasons[0].startswith(f"{reference}:")
    assert revisions.load("fixture", first.revision_id) == first


@pytest.mark.parametrize("outcome", ["pending", "failed", "cancelled", "manifest_only"])
def test_orphaned_comparison_candidates_protect_drafts_regardless_of_outcome(
    tmp_path: Path, outcome: str
) -> None:
    workspace, store, revisions, service = _setup(tmp_path)
    first = _draft(revisions)
    _draft(revisions)
    comparisons = ComparisonStore(workspace)
    manifest = _comparison(first.revision_id)
    comparisons.initialize("orphan_comparison", manifest)
    if outcome in {"failed", "cancelled"}:
        comparisons.transition(
            "orphan_comparison",
            expected_hash=manifest.manifest_hash,
            candidate_id="candidate_1",
            status="running",
        )
        comparisons.finish_unfinished(
            "orphan_comparison",
            expected_hash=manifest.manifest_hash,
            active_status="failed" if outcome == "failed" else "cancelled",
            reason="Worker stopped",
        )
    elif outcome == "manifest_only":
        comparisons.path("orphan_comparison", "ledger.json").unlink()
    assert store.list_jobs() == []
    before = {
        path: path.read_bytes()
        for path in comparisons.path("orphan_comparison").parent.iterdir()
    }
    assert first.status == StrategyStatus.DRAFT
    result = service.cleanup("fixture", [first.revision_id], apply=True)
    assert isinstance(result, StrategyCleanupResult)
    assert not result.success and result.revisions[0].outcome == "protected"
    assert result.revisions[0].reasons == [
        "comparison:orphan_comparison:candidate_1",
        "comparison:orphan_comparison:candidate_2",
    ]
    assert revisions.load("fixture", first.revision_id) == first
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_comparison_job_with_unknown_strategy_blocks_cleanup(
    tmp_path: Path, status: str
) -> None:
    _, store, revisions, service = _setup(tmp_path)
    first = _draft(revisions)
    _draft(revisions)
    store.upsert_job(
        {
            "job_id": "comparison_unknown_strategy",
            "job_type": "comparison",
            "status": status,
            "strategy_name": None,
            "created_at": first.created_at,
        }
    )
    result = service.cleanup("fixture", [first.revision_id], apply=True)
    assert isinstance(result, ErrorResponse)
    assert result.code == "strategy_cleanup_blocked"
    assert "Unknown strategy reference" in result.error
    assert revisions.load("fixture", first.revision_id) == first


@pytest.mark.parametrize("strategy_id", ["other_generated", "kd_strategy"])
def test_unrelated_comparison_without_revision_allows_cleanup(
    tmp_path: Path, strategy_id: str
) -> None:
    workspace, _, revisions, service = _setup(tmp_path)
    first = _draft(revisions)
    _draft(revisions)
    comparisons = ComparisonStore(workspace)
    comparisons.initialize("unrelated", _comparison(None, strategy_id=strategy_id))
    before = comparisons.path("unrelated").read_bytes()
    result = service.cleanup("fixture", [first.revision_id], apply=True)
    assert isinstance(result, StrategyCleanupResult)
    assert result.success and result.revisions[0].outcome == "deleted"
    assert comparisons.path("unrelated").read_bytes() == before


@pytest.mark.parametrize(
    "corruption",
    [
        "invalid_json",
        "duplicate_root_key",
        "duplicate_child_revision",
        "bad_hash",
        "missing_strategy",
        "null_strategy",
        "unknown_strategy",
        "missing_revision",
        "invalid_revision",
    ],
)
@pytest.mark.parametrize("apply", [False, True])
def test_invalid_or_ambiguous_comparison_manifest_blocks_cleanup(
    tmp_path: Path, corruption: str, apply: bool
) -> None:
    workspace, _, revisions, service = _setup(tmp_path)
    first = _draft(revisions)
    _draft(revisions)
    comparisons = ComparisonStore(workspace)
    comparisons.initialize("orphan_comparison", _comparison(first.revision_id))
    path = comparisons.path("orphan_comparison")
    original = path.read_text()
    if corruption == "invalid_json":
        content = "{broken"
    elif corruption == "duplicate_root_key":
        content = '{"schema_version":1,' + original[1:]
    elif corruption == "duplicate_child_revision":
        content = original.replace(
            f'"revision_id":"{first.revision_id}"',
            f'"revision_id":"ambiguous","revision_id":"{first.revision_id}"',
            1,
        )
    else:
        payload = json.loads(original)
        for candidate in payload["candidates"]:
            child = candidate["manifest"]
            config = child["config"]
            if corruption == "missing_strategy":
                config.pop("strategy")
            elif corruption == "null_strategy":
                config["strategy"] = None
            elif corruption == "unknown_strategy":
                config["strategy"]["id"] = None
            elif corruption == "missing_revision":
                config["strategy"].pop("revision_id")
            elif corruption == "invalid_revision":
                config["strategy"]["revision_id"] = "unknown"
            child.pop("manifest_hash")
            child["manifest_hash"] = ExecutionManifest._digest(child)
        payload.pop("manifest_hash")
        payload["manifest_hash"] = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if corruption == "bad_hash":
            payload["manifest_hash"] = "0" * 64
        content = json.dumps(payload)
    path.write_text(content)
    result = service.cleanup("fixture", [first.revision_id], apply=apply)
    assert isinstance(result, ErrorResponse)
    assert result.code == "strategy_cleanup_blocked"
    assert revisions.load("fixture", first.revision_id) == first
    assert path.read_text() == content


@pytest.mark.parametrize(
    "artifact", ["ledger.json", "snapshot.parquet", "receipt.json", "unknown.json"]
)
def test_comparison_artifacts_without_manifest_block_cleanup(
    tmp_path: Path, artifact: str
) -> None:
    workspace, _, revisions, service = _setup(tmp_path)
    first = _draft(revisions)
    _draft(revisions)
    directory = workspace.root / "comparisons" / "incomplete"
    directory.mkdir(parents=True)
    path = directory / artifact
    path.write_bytes(b"preserved evidence")
    (directory / ".comparison.lock").touch()
    result = service.cleanup("fixture", [first.revision_id], apply=True)
    assert isinstance(result, ErrorResponse)
    assert result.code == "strategy_cleanup_blocked"
    assert "missing their manifest" in result.error
    assert revisions.load("fixture", first.revision_id) == first
    assert path.read_bytes() == b"preserved evidence"


@pytest.mark.parametrize(
    "invalid_path", ["root_file", "job_file", "manifest_directory"]
)
def test_invalid_comparison_storage_layout_blocks_cleanup(
    tmp_path: Path, invalid_path: str
) -> None:
    workspace, _, revisions, service = _setup(tmp_path)
    first = _draft(revisions)
    _draft(revisions)
    root = workspace.root / "comparisons"
    if invalid_path == "root_file":
        root.write_bytes(b"unexpected file")
    elif invalid_path == "job_file":
        root.mkdir()
        (root / "invalid_job").write_bytes(b"unexpected file")
    else:
        (root / "invalid_job" / "manifest.json").mkdir(parents=True)
    result = service.cleanup("fixture", [first.revision_id], apply=True)
    assert isinstance(result, ErrorResponse)
    assert result.code == "strategy_cleanup_blocked"
    assert revisions.load("fixture", first.revision_id) == first


@pytest.mark.parametrize("lock_only", [False, True])
def test_empty_or_lock_only_comparison_directory_does_not_reference_draft(
    tmp_path: Path, lock_only: bool
) -> None:
    workspace, _, revisions, service = _setup(tmp_path)
    first = _draft(revisions)
    _draft(revisions)
    directory = workspace.root / "comparisons" / "uninitialized"
    directory.mkdir(parents=True)
    if lock_only:
        (directory / ".comparison.lock").touch()
    result = service.cleanup("fixture", [first.revision_id], apply=True)
    assert isinstance(result, StrategyCleanupResult)
    assert result.success and result.revisions[0].outcome == "deleted"
    assert directory.is_dir()
    assert (directory / ".comparison.lock").exists() == lock_only


@pytest.mark.parametrize(
    "target",
    [
        "comparisons",
        "directory",
        "manifest.json",
        "ledger.json",
        "snapshot.parquet",
        "receipt.json",
        ".comparison.lock",
    ],
)
def test_symlinked_comparison_paths_block_cleanup_without_following_targets(
    tmp_path: Path, target: str
) -> None:
    workspace, _, revisions, service = _setup(tmp_path)
    first = _draft(revisions)
    _draft(revisions)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_bytes(b"retained")
    root = workspace.root / "comparisons"
    if target == "comparisons":
        root.symlink_to(outside, target_is_directory=True)
    elif target == "directory":
        root.mkdir()
        (root / "orphan_comparison").symlink_to(outside, target_is_directory=True)
    else:
        comparisons = ComparisonStore(workspace)
        comparisons.initialize("orphan_comparison", _comparison(first.revision_id))
        path = root / "orphan_comparison" / target
        path.unlink(missing_ok=True)
        path.symlink_to(sentinel)
    result = service.cleanup("fixture", [first.revision_id], apply=True)
    assert isinstance(result, ErrorResponse)
    assert result.code == "strategy_cleanup_blocked"
    assert revisions.load("fixture", first.revision_id) == first
    assert sentinel.read_bytes() == b"retained"


@pytest.mark.parametrize(
    "corruption",
    [
        "array",
        "null",
        "empty",
        "strategy_name",
        "job_id",
        "job_type",
        "config_path",
        "missing_revision",
        "null_revision",
        "invalid_revision",
    ],
)
@pytest.mark.parametrize("apply", [False, True])
def test_invalid_raw_job_history_blocks_cleanup_before_any_deletion(
    tmp_path: Path, corruption: str, apply: bool
) -> None:
    _, store, revisions, service = _setup(tmp_path)
    first = _draft(revisions)
    _draft(revisions)
    payload: Any = {
        "job_id": "historical_job",
        "job_type": "backtest",
        "status": "failed",
        "strategy_name": "fixture",
        "revision_id": first.revision_id,
        "config_path": first.config_path,
        "created_at": first.created_at,
    }
    store.upsert_job(payload)
    if corruption in {"array", "null", "empty"}:
        payload = {"array": [], "null": None, "empty": {}}[corruption]
    elif corruption == "missing_revision":
        payload.pop("revision_id")
    elif corruption == "null_revision":
        payload["revision_id"] = None
    elif corruption == "invalid_revision":
        payload["revision_id"] = "not_a_revision"
    else:
        payload[corruption] = "different_identity"
    with store.connect() as connection:
        connection.execute(
            "update jobs set payload = ? where job_id = ?",
            (json.dumps(payload), "historical_job"),
        )
    result = service.cleanup("fixture", [first.revision_id], apply=apply)
    assert isinstance(result, ErrorResponse)
    assert result.code == "strategy_cleanup_blocked"
    assert revisions.load("fixture", first.revision_id) == first


@pytest.mark.parametrize("reference", ["job", "run", "manifest"])
def test_missing_target_revision_without_legacy_evidence_blocks_cleanup(
    tmp_path: Path, reference: str
) -> None:
    workspace, store, revisions, service = _setup(tmp_path)
    first = _draft(revisions)
    _draft(revisions)
    if reference == "job":
        store.upsert_job(
            {
                "job_id": "ambiguous_job",
                "status": "done",
                "strategy_name": "fixture",
                "created_at": first.created_at,
            }
        )
    elif reference == "run":
        store.create_run(
            run_id="ambiguous_run",
            job_id="old_job",
            strategy_id="fixture",
            artifact_dir=workspace.runs / "ambiguous_run",
            metrics={},
        )
    else:
        original = _manifest(first.revision_id)
        config = original.config_copy()
        config["strategy"]["revision_id"] = None
        manifest = ExecutionManifest.create(
            strategy_execution=original.strategy_execution,
            kind="backtest",
            config=config,
        )
        ExecutionManifestStore(workspace).publish("ambiguous_manifest", manifest)
    result = service.cleanup("fixture", [first.revision_id], apply=True)
    assert isinstance(result, ErrorResponse)
    assert result.code == "strategy_cleanup_blocked"
    assert "Cannot establish the revision" in result.error
    assert revisions.load("fixture", first.revision_id) == first


@pytest.mark.parametrize("reference", ["job", "run"])
def test_unknown_indexed_strategy_identity_blocks_cleanup(
    tmp_path: Path, reference: str
) -> None:
    workspace, store, revisions, service = _setup(tmp_path)
    first = _draft(revisions)
    _draft(revisions)
    if reference == "job":
        store.upsert_job(
            {
                "job_id": "unknown_strategy",
                "status": "done",
                "strategy_name": None,
                "created_at": first.created_at,
            }
        )
    else:
        store.create_run(
            run_id="unknown_strategy",
            job_id="old_job",
            strategy_id="",
            artifact_dir=workspace.runs / "unknown_strategy",
            metrics={},
        )
    result = service.cleanup("fixture", [first.revision_id], apply=True)
    assert isinstance(result, ErrorResponse)
    assert result.code == "strategy_cleanup_blocked"
    assert revisions.load("fixture", first.revision_id) == first


@pytest.mark.parametrize(
    "identity", [None, "", 42, False, [], {}, "missing", "missing_strategy", "mapping"]
)
@pytest.mark.parametrize("apply", [False, True])
def test_unknown_manifest_strategy_identity_blocks_cleanup(
    tmp_path: Path, identity: Any, apply: bool
) -> None:
    workspace, _, revisions, service = _setup(tmp_path)
    first = _draft(revisions)
    _draft(revisions)
    original = _manifest(first.revision_id)
    config = original.config_copy()
    if identity == "missing_strategy":
        config.pop("strategy")
    elif identity == "mapping":
        config["strategy"] = None
    elif identity == "missing":
        config["strategy"].pop("id")
    else:
        config["strategy"]["id"] = identity
    # Historical JSON can carry a valid digest without satisfying today's
    # request schema. Cleanup must still establish its strategy identity.
    payload = original.model_dump(mode="json", exclude={"manifest_hash"})
    payload["config"] = config
    manifest = ExecutionManifest.model_validate(
        {**payload, "manifest_hash": ExecutionManifest._digest(payload)}
    )
    ExecutionManifestStore(workspace).publish("unidentified_manifest", manifest)
    result = service.cleanup("fixture", [first.revision_id], apply=apply)
    assert isinstance(result, ErrorResponse)
    assert result.code == "strategy_cleanup_blocked"
    assert revisions.load("fixture", first.revision_id) == first


@pytest.mark.parametrize("reference", ["job", "run"])
def test_explicit_flat_source_legacy_history_does_not_reference_new_drafts(
    tmp_path: Path, reference: str
) -> None:
    workspace, store, revisions, service = _setup(tmp_path)
    first = _draft(revisions)
    _draft(revisions)
    source = workspace.generated_strategies / "fixture.py"
    source.write_text("legacy source")
    config = (
        workspace.configs / "fixture.yaml"
        if reference == "job"
        else workspace.runs / "legacy_run" / "config.yaml"
    )
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        yaml.safe_dump({"strategy": {"id": "fixture", "source_path": str(source)}})
    )
    if reference == "job":
        store.upsert_job(
            {
                "job_id": "legacy_job",
                "status": "done",
                "strategy_name": "fixture",
                "config_path": str(config),
                "created_at": first.created_at,
            }
        )
    else:
        store.create_run(
            run_id="legacy_run",
            job_id="old_job",
            strategy_id="fixture",
            artifact_dir=config.parent,
            metrics={},
        )
    before = source.read_bytes(), config.read_bytes()
    result = service.cleanup("fixture", [first.revision_id], apply=True)
    assert isinstance(result, StrategyCleanupResult)
    assert result.success and result.revisions[0].outcome == "deleted"
    assert (source.read_bytes(), config.read_bytes()) == before


@pytest.mark.parametrize("strategy_id", ["other_generated", "kd_strategy"])
def test_valid_unrelated_and_bundled_jobs_without_revision_allow_cleanup(
    tmp_path: Path, strategy_id: str
) -> None:
    workspace, store, revisions, service = _setup(tmp_path)
    first = _draft(revisions)
    _draft(revisions)
    store.upsert_job(
        {
            "job_id": "unrelated",
            "status": "done",
            "strategy_name": strategy_id,
            "created_at": first.created_at,
        }
    )
    store.create_run(
        run_id="unrelated",
        job_id="unrelated",
        strategy_id=strategy_id,
        artifact_dir=workspace.runs / "unrelated",
        metrics={},
    )
    result = service.cleanup("fixture", [first.revision_id], apply=True)
    assert isinstance(result, StrategyCleanupResult)
    assert result.success and result.revisions[0].outcome == "deleted"


@pytest.mark.parametrize("content", ["strategy: [", "strategy: {}", "strategy: null"])
def test_unverifiable_legacy_config_cannot_clear_an_unknown_revision_reference(
    tmp_path: Path, content: str
) -> None:
    workspace, store, revisions, service = _setup(tmp_path)
    first = _draft(revisions)
    _draft(revisions)
    config = workspace.configs / "fixture.yaml"
    config.write_text(content)
    store.upsert_job(
        {
            "job_id": "legacy_job",
            "status": "done",
            "strategy_name": "fixture",
            "config_path": str(config),
            "created_at": first.created_at,
        }
    )
    result = service.cleanup("fixture", [first.revision_id], apply=True)
    assert isinstance(result, ErrorResponse)
    assert result.code == "strategy_cleanup_blocked"
    assert revisions.load("fixture", first.revision_id) == first
    assert config.read_text() == content


@pytest.mark.parametrize("target", ["source", "revision", "cache", "runs", "manifest"])
def test_symlinked_cleanup_paths_fail_closed_without_touching_targets(
    tmp_path: Path, target: str
) -> None:
    workspace, _, revisions, service = _setup(tmp_path)
    first = _draft(revisions)
    _draft(revisions)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep"
    sentinel.write_text("retained")
    if target == "source":
        path = Path(first.source_path)
        path.unlink()
        path.symlink_to(sentinel)
    elif target == "revision":
        path = Path(first.source_path).parent
        path.rename(outside / "original")
        path.symlink_to(outside / "original", target_is_directory=True)
    elif target == "cache":
        (Path(first.source_path).parent / "__pycache__").symlink_to(
            outside, target_is_directory=True
        )
    elif target == "runs":
        workspace.runs.rmdir()
        workspace.runs.symlink_to(outside, target_is_directory=True)
    else:
        directory = workspace.runs / "orphan"
        directory.mkdir()
        (directory / "manifest.json").symlink_to(sentinel)
    result = service.cleanup("fixture", [first.revision_id], apply=True)
    assert not result.success
    assert sentinel.read_text() == "retained"
    assert Path(first.source_path).parent.exists()


@pytest.mark.parametrize("tamper", ["metadata", "source", "unknown_file", "manifest"])
def test_unknown_or_corrupt_content_is_preserved_for_inspection(
    tmp_path: Path, tamper: str
) -> None:
    workspace, _, revisions, service = _setup(tmp_path)
    first = _draft(revisions)
    _draft(revisions)
    directory = Path(first.source_path).parent
    if tamper == "manifest":
        directory = workspace.runs / "orphan"
        directory.mkdir()
        path = directory / "manifest.json"
    else:
        path = (
            directory
            / {
                "metadata": "metadata.json",
                "source": "strategy.py",
                "unknown_file": "my_notes.txt",
            }[tamper]
        )
    path.write_text("unexpected content")
    result = service.cleanup("fixture", [first.revision_id], apply=True)
    assert not result.success
    assert path.read_text() == "unexpected content"
    assert Path(first.source_path).parent.exists()


def test_apply_rechecks_current_pointer_and_references_after_preview(
    tmp_path: Path,
) -> None:
    workspace, _, revisions, service = _setup(tmp_path)
    first = _draft(revisions)
    _draft(revisions)
    result = service.cleanup("fixture", [first.revision_id])
    assert isinstance(result, StrategyCleanupResult)
    assert result.revisions[0].outcome == "eligible"
    (workspace.generated_strategies / "fixture" / "current.json").write_text(
        json.dumps({"revision_id": first.revision_id})
    )
    ExecutionManifestStore(workspace).publish(
        "late_reference", _manifest(first.revision_id)
    )
    result = service.cleanup("fixture", [first.revision_id], apply=True)
    assert isinstance(result, StrategyCleanupResult)
    assert not result.success
    assert set(result.revisions[0].reasons) == {
        "current_revision",
        "manifest:late_reference",
    }


@pytest.mark.parametrize(
    "failure", [OSError("disk unavailable"), TimeoutError("timed out")]
)
def test_deletion_failure_is_reported_and_lock_is_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: OSError
) -> None:
    workspace, _, revisions, service = _setup(tmp_path)
    first = _draft(revisions)
    _draft(revisions)

    def fail_delete(*_args: Any, **_kwargs: Any) -> None:
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(
            "tradingdev.adapters.storage.strategy_revisions.shutil.rmtree", fail_delete
        )
        result = service.cleanup("fixture", [first.revision_id], apply=True)
    assert isinstance(result, StrategyCleanupResult)
    assert not result.success
    assert result.revisions[0].outcome == "failed"
    assert result.revisions[0].reasons == [str(failure)]
    assert revisions.load("fixture", first.revision_id) == first
    assert not list(workspace.root.rglob(".pending-*"))
    result = service.cleanup("fixture", [first.revision_id], apply=True)
    assert result.success
    assert not Path(first.source_path).parent.exists()


def test_cleanup_waits_for_validation_then_rechecks_eligibility(tmp_path: Path) -> None:
    workspace, _, revisions, service = _setup(tmp_path)
    first = _draft(revisions)
    _draft(revisions)
    entered = Event()

    def cleanup() -> StrategyCleanupResult | ErrorResponse:
        entered.set()
        return service.cleanup("fixture", [first.revision_id], apply=True)

    with ThreadPoolExecutor(max_workers=1) as executor:
        with revisions.lifecycle_lock("fixture"):
            pending = executor.submit(cleanup)
            assert entered.wait(timeout=2)
            assert not pending.done()
            first.status = StrategyStatus.VALIDATED
            # Nested acquisition through another store must be reentrant.
            StrategyRevisionStore(workspace).update(first)
        result = pending.result(timeout=5)
    assert isinstance(result, StrategyCleanupResult)
    assert not result.success
    assert result.revisions[0].reasons == ["status:validated"]
    assert revisions.load("fixture", first.revision_id) == first


def test_lifecycle_lock_timeout_preserves_files_and_releases_waiter(
    tmp_path: Path,
) -> None:
    workspace, _, revisions, service = _setup(tmp_path)
    first = _draft(revisions)
    _draft(revisions)

    def locked_cleanup() -> str:
        from tradingdev.adapters.storage.strategy_revisions import StrategyRevisionError

        try:
            with StrategyRevisionStore(workspace).lifecycle_lock("fixture", timeout=0):
                raise AssertionError("The held lifecycle lock must reject this attempt")
        except StrategyRevisionError as exc:
            return str(exc)

    with (
        ThreadPoolExecutor(max_workers=1) as executor,
        revisions.lifecycle_lock("fixture"),
    ):
        result = executor.submit(locked_cleanup).result(timeout=5)
    assert "busy; retry" in result
    assert revisions.load("fixture", first.revision_id) == first
    assert service.cleanup("fixture", [first.revision_id], apply=True).success
    assert not Path(first.source_path).parent.exists()


@pytest.mark.parametrize("dry_run", [False, True])
def test_service_validation_holds_revision_until_check_and_update_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dry_run: bool
) -> None:
    """Exercise actual lifecycle orchestration, replacing only contract execution."""
    workspace, store, revisions, cleanup = _setup(tmp_path)
    first = _draft(revisions)
    _draft(revisions)
    service = StrategyService(workspace, store=store)
    if dry_run:
        assert service.record_validation_status(
            "fixture",
            {
                "revision_id": first.revision_id,
                "checked_at": first.created_at,
                "success": True,
            },
        )["success"]
    entered = Event()
    release = Event()
    cleanup_entered = Event()

    def check(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        entered.set()
        assert release.wait(timeout=5)
        assert Path(first.source_path).is_file()
        return {"diagnostics": [], "signal_analysis": {}}

    def clean() -> StrategyCleanupResult | ErrorResponse:
        cleanup_entered.set()
        return cleanup.cleanup("fixture", [first.revision_id], apply=True)

    monkeypatch.setattr(service, "_quality_gate_diagnostics", lambda _path: [])
    monkeypatch.setattr(service._contract_checker, "check", check)
    with ThreadPoolExecutor(max_workers=2) as executor:
        validation = executor.submit(
            service.dry_run if dry_run else service.validate,
            "fixture",
            first.revision_id,
        )
        try:
            assert entered.wait(timeout=2)
            deletion = executor.submit(clean)
            assert cleanup_entered.wait(timeout=2)
            with pytest.raises(FutureTimeoutError):
                deletion.result(timeout=0.1)
        finally:
            release.set()
        assert validation.result(timeout=5)["success"]
        result = deletion.result(timeout=5)
    assert isinstance(result, StrategyCleanupResult)
    assert not result.success
    assert result.revisions[0].reasons == [
        "status:runnable" if dry_run else "status:validated"
    ]
    assert Path(first.source_path).is_file()


def test_bad_identifiers_and_legacy_storage_are_never_deleted(tmp_path: Path) -> None:
    workspace, _, revisions, service = _setup(tmp_path)
    first = _draft(revisions)
    legacy = workspace.generated_strategies / "legacy.json"
    legacy.write_text("{}")
    assert not service.cleanup("../outside").success
    result = service.cleanup("fixture", ["../outside"], apply=True)
    assert isinstance(result, StrategyCleanupResult)
    assert not result.success and result.revisions[0].outcome == "protected"
    assert not service.cleanup("legacy").success
    assert legacy.read_text() == "{}"
    assert revisions.load("fixture") == first
