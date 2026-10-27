"""Complete ordered outcomes, immutable evidence and failure-safe publication."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import pytest

from tradingdev.adapters.storage.comparisons import ComparisonStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.domain.comparison import ComparisonCandidate, ComparisonManifest
from tradingdev.domain.execution import ExecutionManifest, ManifestError
from tradingdev.domain.strategies.execution import StrategyExecution

if TYPE_CHECKING:
    from pathlib import Path

    from tradingdev.adapters.storage.comparisons import ComparisonArtifact
    from tradingdev.domain.comparison_results import (
        CandidateStatus,
        ComparisonLedger,
        ComparisonReceipt,
    )


def _manifest(*, count: int = 3, label: str = "Window") -> ComparisonManifest:
    return ComparisonManifest.create(
        [
            ComparisonCandidate(
                candidate_id=f"candidate_{index}",
                label=f"{label} {index}",
                rationale="Fixed before execution",
                manifest=ExecutionManifest.create(
                    kind="backtest",
                    config={
                        "strategy": {
                            "id": "fixture",
                            "parameters": {"period": index},
                        },
                        "backtest": {
                            "symbol": "BTC/USDT",
                            "timeframe": "1h",
                            "start_date": "2024-01-01",
                            "end_date": "2024-01-08",
                            "init_cash": 10000,
                            "fees": 0,
                        },
                    },
                    strategy_execution=StrategyExecution(
                        kind="generated", constructor_kwargs={"period": index}
                    ),
                ),
            )
            for index in range(1, count + 1)
        ]
    )


@pytest.fixture
def store(tmp_path: Path) -> ComparisonStore:
    return ComparisonStore(WorkspacePaths(tmp_path / "workspace"))


def _transition(
    store: ComparisonStore,
    manifest: ComparisonManifest,
    index: int,
    status: CandidateStatus,
    *,
    reason: str | None = None,
) -> ComparisonLedger:
    return store.transition(
        "job_1",
        expected_hash=manifest.manifest_hash,
        candidate_id=f"candidate_{index}",
        status=status,
        reason=reason,
    )


def _snapshot(
    store: ComparisonStore,
    manifest: ComparisonManifest,
    *,
    content: bytes = b"actual shared snapshot bytes",
    binding: str = '{"row_count": 100, "symbol": "BTC/USDT"}',
) -> ComparisonReceipt:
    return store.publish_snapshot(
        "job_1",
        expected_hash=manifest.manifest_hash,
        snapshot_bytes=content,
        data_binding_json=binding,
        source_dataset_id="dataset_1",
    )


def test_initialize_retains_all_reserved_rows_and_no_execution_data(
    store: ComparisonStore,
) -> None:
    manifest = _manifest(count=8)
    ledger = store.initialize("job_1", manifest)
    assert [row.candidate_id for row in ledger.candidates] == [
        f"candidate_{i}" for i in range(1, 9)
    ]
    assert all(
        row.status == "pending" and row.reason is None for row in ledger.candidates
    )
    assert all(
        row.run_id == f"job_1_candidate_{i}"
        for i, row in enumerate(ledger.candidates, 1)
    )
    assert [row.child_manifest_hash for row in ledger.candidates] == [
        candidate.manifest.manifest_hash for candidate in manifest.candidates
    ]
    assert not ledger.is_complete
    assert not store.path("job_1", "snapshot.parquet").exists()
    assert not store.path("job_1", "receipt.json").exists()
    assert store.initialize("job_1", manifest) == ledger
    with pytest.raises(ManifestError):
        store.initialize("job_1", _manifest(label="Different"))


def test_success_and_failure_retries_cannot_replay_or_replace_outcomes(
    store: ComparisonStore,
) -> None:
    manifest = _manifest(count=2)
    store.initialize("job_1", manifest)
    with pytest.raises(ManifestError, match="pending to succeeded"):
        _transition(store, manifest, 1, "succeeded")
    _transition(store, manifest, 1, "running")
    success = _transition(store, manifest, 1, "succeeded")
    before = store.path("job_1", "ledger.json").read_bytes()
    assert _transition(store, manifest, 1, "succeeded") == success
    assert store.path("job_1", "ledger.json").read_bytes() == before
    for status in ("running", "failed", "cancelled"):
        with pytest.raises(ValueError):
            _transition(store, manifest, 1, status, reason="overwrite")
    _transition(store, manifest, 2, "running")
    failed = _transition(store, manifest, 2, "failed", reason="Invalid strategy")
    assert failed.is_complete
    assert (
        _transition(store, manifest, 2, "failed", reason="Invalid strategy") == failed
    )
    with pytest.raises(ManifestError):
        _transition(store, manifest, 2, "failed", reason="Changed reason")
    assert store.initialize("job_1", manifest) == failed
    assert all("metrics" not in row.model_dump() for row in failed.candidates)


@pytest.mark.parametrize("active_status", ["failed", "cancelled"])
def test_interrupted_work_preserves_success_and_marks_every_unfinished_row(
    store: ComparisonStore, active_status: Any
) -> None:
    manifest = _manifest()
    store.initialize("job_1", manifest)
    _transition(store, manifest, 1, "running")
    _transition(store, manifest, 1, "succeeded")
    _transition(store, manifest, 2, "running")
    ledger = store.finish_unfinished(
        "job_1",
        expected_hash=manifest.manifest_hash,
        reason="Worker stopped",
        active_status=active_status,
    )
    assert [row.status for row in ledger.candidates] == [
        "succeeded",
        active_status,
        "not_run",
    ]
    assert ledger.is_complete
    assert [row.reason for row in ledger.candidates] == [
        None,
        "Worker stopped",
        "Worker stopped",
    ]
    assert (
        store.finish_unfinished(
            "job_1", expected_hash=manifest.manifest_hash, reason="Later observation"
        )
        == ledger
    )


def test_all_failed_and_data_load_failure_still_have_complete_ledgers(
    store: ComparisonStore,
) -> None:
    manifest = _manifest(count=2)
    store.initialize("job_1", manifest)
    for index in (1, 2):
        _transition(store, manifest, index, "running")
        ledger = _transition(
            store, manifest, index, "failed", reason="Execution failed"
        )
    assert ledger.is_complete
    assert len(ledger.candidates) == 2
    store.initialize("job_2", manifest)
    skipped = store.finish_unfinished(
        "job_2", expected_hash=manifest.manifest_hash, reason="Data unavailable"
    )
    assert skipped.is_complete
    assert [row.status for row in skipped.candidates] == ["not_run", "not_run"]
    assert not store.path("job_2", "receipt.json").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("job_id", "other"),
        ("manifest_hash", "0" * 64),
        ("schema_version", "1"),
        ("schema_version", True),
        ("schema_version", 1.0),
    ],
)
def test_invalid_ledger_identity_or_types_fail_closed(
    store: ComparisonStore, field: str, value: object
) -> None:
    manifest = _manifest()
    ledger = store.initialize("job_1", manifest).model_dump(mode="json")
    ledger[field] = value
    store.path("job_1", "ledger.json").write_text(json.dumps(ledger))
    with pytest.raises(ManifestError):
        store.load_ledger("job_1", expected_hash=manifest.manifest_hash)


@pytest.mark.parametrize(
    "change", ["missing", "reorder", "child_hash", "run_id", "reason", "metrics"]
)
def test_ledger_cannot_hide_candidates_or_invent_bindings(
    store: ComparisonStore, change: str
) -> None:
    manifest = _manifest()
    payload = store.initialize("job_1", manifest).model_dump(mode="json")
    rows = payload["candidates"]
    if change == "missing":
        rows.pop()
    elif change == "reorder":
        rows.reverse()
    elif change == "child_hash":
        rows[0]["child_manifest_hash"] = "0" * 64
    elif change == "run_id":
        rows[0]["run_id"] = "unrelated"
    elif change == "reason":
        rows[0]["status"] = "failed"
    else:
        rows[0]["metrics"] = {"return": 0}
    store.path("job_1", "ledger.json").write_text(json.dumps(payload))
    with pytest.raises(ManifestError):
        store.load_ledger("job_1", expected_hash=manifest.manifest_hash)


@pytest.mark.parametrize(
    "content", [None, "{broken", "{}", '{"value": NaN}', '{"value": 1,"value": 2}']
)
def test_missing_malformed_or_ambiguous_ledger_is_never_recreated_on_read(
    store: ComparisonStore, content: str | None
) -> None:
    manifest = _manifest()
    store.initialize("job_1", manifest)
    path = store.path("job_1", "ledger.json")
    if content is None:
        path.unlink()
    else:
        path.write_text(content)
    with pytest.raises(ManifestError):
        store.load_ledger("job_1", expected_hash=manifest.manifest_hash)


def test_snapshot_receipt_binds_actual_bytes_without_changing_approved_manifest(
    store: ComparisonStore,
) -> None:
    manifest = _manifest()
    store.initialize("job_1", manifest)
    approved = store.path("job_1").read_bytes()
    receipt = _snapshot(store, manifest)
    assert (
        receipt.snapshot_sha256
        == hashlib.sha256(b"actual shared snapshot bytes").hexdigest()
    )
    assert receipt.data_binding_json == '{"row_count":100,"symbol":"BTC/USDT"}'
    assert _snapshot(store, manifest) == receipt
    loaded, snapshot = store.load_snapshot(
        "job_1", expected_hash=manifest.manifest_hash
    )
    assert loaded == receipt and snapshot == b"actual shared snapshot bytes"
    assert store.path("job_1").read_bytes() == approved
    for kwargs in ({"content": b"replacement"}, {"binding": '{"row_count":99}'}):
        with pytest.raises(ManifestError, match="cannot be replaced"):
            _snapshot(store, manifest, **kwargs)


def test_metadata_query_does_not_read_snapshot_but_still_validates_receipt(
    store: ComparisonStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    store.initialize("job_1", manifest)
    receipt = _snapshot(store, manifest)
    original_read = store._read

    def read(job_id: str, artifact: ComparisonArtifact) -> bytes:
        assert artifact != "snapshot.parquet", "Status queries must not load the frame"
        return original_read(job_id, artifact)

    monkeypatch.setattr(store, "_read", read)
    assert store.load_receipt("job_1", expected_hash=manifest.manifest_hash) == receipt
    store.path("job_1", "receipt.json").write_text('{"job_id":"other"}')
    with pytest.raises(ManifestError, match="valid comparison receipt"):
        store.load_receipt("job_1", expected_hash=manifest.manifest_hash)


@pytest.mark.parametrize(
    "binding",
    [
        "[]",
        "null",
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":1e999}',
        '{"x":1,"x":2}',
        '{"nested":{"x":1,"x":2}}',
    ],
)
def test_snapshot_requires_finite_unambiguous_object_binding(
    store: ComparisonStore, binding: str
) -> None:
    manifest = _manifest()
    store.initialize("job_1", manifest)
    with pytest.raises(ValueError):
        _snapshot(store, manifest, binding=binding)
    assert not store.path("job_1", "snapshot.parquet").exists()


@pytest.mark.parametrize("artifact", ["receipt.json", "snapshot.parquet"])
def test_snapshot_loader_rejects_missing_and_changed_artifacts(
    store: ComparisonStore, artifact: ComparisonArtifact
) -> None:
    manifest = _manifest()
    store.initialize("job_1", manifest)
    _snapshot(store, manifest)
    path = store.path("job_1", artifact)
    original = path.read_bytes()
    path.unlink()
    with pytest.raises(ManifestError):
        store.load_snapshot("job_1", expected_hash=manifest.manifest_hash)
    if artifact == "receipt.json":
        payload = json.loads(original)
        payload["job_id"] = "other"
        path.write_text(json.dumps(payload))
    else:
        path.write_bytes(b"different bytes")
    with pytest.raises(ManifestError):
        store.load_snapshot("job_1", expected_hash=manifest.manifest_hash)


def test_snapshot_cannot_be_published_before_complete_initialization(
    store: ComparisonStore,
) -> None:
    with pytest.raises(ManifestError):
        _snapshot(store, _manifest())
    assert not store.path("job_1", "snapshot.parquet").exists()


def test_concurrent_initializers_never_mix_bundle_and_ledger(
    store: ComparisonStore,
) -> None:
    manifests = [_manifest(label="A"), _manifest(label="B")]

    def initialize(manifest: ComparisonManifest) -> str | None:
        try:
            store.initialize("job_1", manifest)
        except ManifestError:
            return None
        return manifest.manifest_hash

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(initialize, manifests))
    hashes = [result for result in results if result is not None]
    assert len(hashes) == 1
    assert (
        store.load_ledger("job_1", expected_hash=hashes[0]).manifest_hash == hashes[0]
    )
    assert not list(store.path("job_1").parent.glob(".*-*"))


def test_concurrent_candidate_updates_do_not_lose_outcomes(
    store: ComparisonStore,
) -> None:
    manifest = _manifest(count=8)
    store.initialize("job_1", manifest)

    def finish(index: int) -> None:
        _transition(store, manifest, index, "running")
        _transition(store, manifest, index, "failed", reason=f"Failure {index}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(finish, range(1, 9)))
    ledger = store.load_ledger("job_1", expected_hash=manifest.manifest_hash)
    assert ledger.is_complete
    assert [row.reason for row in ledger.candidates] == [
        f"Failure {i}" for i in range(1, 9)
    ]


def test_failed_ledger_replace_preserves_prior_state_and_cleans_temporary_file(
    store: ComparisonStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    initial = store.initialize("job_1", manifest)

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("replace interrupted")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        _transition(store, manifest, 1, "running")
    assert store.load_ledger("job_1", expected_hash=manifest.manifest_hash) == initial
    assert not list(store.path("job_1").parent.glob(".*-*"))


def test_initialization_recovers_interruption_without_exposing_partial_ledger(
    store: ComparisonStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    link = os.link

    def fail_manifest(source: Path, target: Path) -> None:
        if target.name == "manifest.json":
            raise OSError("manifest interrupted")
        link(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(os, "link", fail_manifest)
        with pytest.raises(OSError, match="interrupted"):
            store.initialize("job_1", manifest)
    assert store.path("job_1", "ledger.json").is_file()
    assert not store.path("job_1", "manifest.json").exists()
    with pytest.raises(ManifestError):
        store.load_ledger("job_1", expected_hash=manifest.manifest_hash)
    with pytest.raises(ManifestError):
        store.initialize("job_1", _manifest(label="Different"))
    assert not store.path("job_1", "manifest.json").exists()
    ledger = store.initialize("job_1", manifest)
    assert all(row.status == "pending" for row in ledger.candidates)
    assert not list(store.path("job_1").parent.glob(".*-*"))


def test_initialization_cannot_recreate_lost_ledger_after_success(
    store: ComparisonStore,
) -> None:
    manifest = _manifest(count=1)
    store.initialize("job_1", manifest)
    _transition(store, manifest, 1, "running")
    _transition(store, manifest, 1, "succeeded")
    store.path("job_1", "ledger.json").unlink()
    with pytest.raises(ManifestError):
        store.initialize("job_1", manifest)
    assert not store.path("job_1", "ledger.json").exists()


def test_failed_receipt_publication_rolls_back_only_new_snapshot(
    store: ComparisonStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    store.initialize("job_1", manifest)
    link = os.link

    def fail_receipt(source: Path, target: Path) -> None:
        if target.name == "receipt.json":
            raise OSError("receipt interrupted")
        link(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(os, "link", fail_receipt)
        with pytest.raises(OSError, match="interrupted"):
            _snapshot(store, manifest)
    assert not store.path("job_1", "snapshot.parquet").exists()
    assert not store.path("job_1", "receipt.json").exists()
    assert not list(store.path("job_1").parent.glob(".*-*"))
    _snapshot(store, manifest)


def test_crash_between_snapshot_and_receipt_allows_only_identical_recovery(
    store: ComparisonStore,
) -> None:
    manifest = _manifest()
    store.initialize("job_1", manifest)
    store.path("job_1", "snapshot.parquet").write_bytes(b"original bytes")
    with pytest.raises(ManifestError, match="cannot be replaced"):
        _snapshot(store, manifest, content=b"different")
    _snapshot(store, manifest, content=b"original bytes")
    assert (
        store.load_snapshot("job_1", expected_hash=manifest.manifest_hash)[1]
        == b"original bytes"
    )


@pytest.mark.parametrize("job_id", ["", "../escape", "/absolute", "a/b", "x" * 117])
def test_job_ids_cannot_escape_or_overflow_reserved_child_ids(
    store: ComparisonStore, job_id: str
) -> None:
    with pytest.raises(ValueError):
        store.initialize(job_id, _manifest())


@pytest.mark.parametrize(
    "artifact",
    [
        "manifest.json",
        "ledger.json",
        "receipt.json",
        "snapshot.parquet",
        ".comparison.lock",
    ],
)
def test_artifact_symlinks_are_rejected_without_touching_target(
    store: ComparisonStore, tmp_path: Path, artifact: ComparisonArtifact
) -> None:
    manifest = _manifest()
    store.initialize("job_1", manifest)
    path = store.path("job_1", artifact)
    path.unlink(missing_ok=True)
    outside = tmp_path / "outside"
    outside.write_bytes(b"untouched")
    path.symlink_to(outside)
    with pytest.raises(ManifestError, match="path leaves workspace"):
        store.path("job_1", artifact)
    assert outside.read_bytes() == b"untouched"


@pytest.mark.parametrize("component", ["workspace", "comparisons", "job_1"])
def test_replaced_parent_directory_is_rejected(
    store: ComparisonStore, tmp_path: Path, component: str
) -> None:
    target = tmp_path / "outside"
    target.mkdir()
    parent = tmp_path / "workspace"
    if component != "workspace":
        parent /= "comparisons"
    if component == "job_1":
        parent /= "job_1"
    parent.parent.mkdir(parents=True, exist_ok=True)
    parent.symlink_to(target, target_is_directory=True)
    with pytest.raises(ManifestError, match="path leaves workspace"):
        store.initialize("job_1", _manifest())
    assert list(target.iterdir()) == []
