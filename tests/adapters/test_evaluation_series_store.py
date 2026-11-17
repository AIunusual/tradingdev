"""Evaluation series artifacts are immutable, bounded and verifiable."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from tradingdev.adapters.storage.evaluation_series import EvaluationSeriesStore
from tradingdev.domain.ml.evaluation.series import (
    EvaluationSeriesBundle,
    build_bundle,
    build_signal_series,
    frame_bytes,
)

if TYPE_CHECKING:
    from pathlib import Path

_TIMESTAMPS = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
_SIGNAL = [1.0, 2.0, 3.0, 2.0, 3.0, 4.0]
_FORWARD = [0.1, 0.3, 0.2, 0.5, 0.4, 0.8]


def _bundle(**fields: object) -> EvaluationSeriesBundle:
    frame = build_signal_series(_SIGNAL, _FORWARD, _TIMESTAMPS, window=3)
    values: dict[str, object] = {
        "kind": "signal_evaluation",
        "evaluation_id": "eval_0123456789abcdef",
        "dataset_id": "dataset-1",
        "dataset_hash": "b" * 64,
        "settings": {"window": 3, "horizon": 1},
        "method_family": "rule",
        "producer_source": "backend_rule",
        "signal_fingerprint": "f" * 16,
        "training_status": "no_fitting",
        "created_at": "2024-01-01T00:00:00+00:00",
    }
    values.update(fields)
    return build_bundle(frame, **values)  # type: ignore[arg-type]


def test_roundtrip_and_tamper_rejection(tmp_path: Path) -> None:
    store = EvaluationSeriesStore(tmp_path)
    bundle = _bundle()

    artifact_id = store.save(bundle)
    loaded = store.load(artifact_id)
    pd.testing.assert_frame_equal(loaded.frame, bundle.frame)

    path = store.path_for(artifact_id) / "series.parquet"
    changed = bundle.frame.copy()
    changed.loc[2, "signal"] = 99.0
    changed.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="content hash"):
        store.load(artifact_id)


def test_reevaluation_keeps_old_artifacts(tmp_path: Path) -> None:
    store = EvaluationSeriesStore(tmp_path)
    first_id = store.save(_bundle())
    second_id = store.save(_bundle(evaluation_id="eval_fedcba9876543210"))

    assert first_id != second_id
    assert store.load(first_id).manifest.evaluation_id == "eval_0123456789abcdef"
    assert store.load(second_id).manifest.evaluation_id == "eval_fedcba9876543210"


def test_page_is_bounded_and_export_returns_full_rows(tmp_path: Path) -> None:
    store = EvaluationSeriesStore(tmp_path)
    for index in range(3):
        store.save(_bundle(entry_id=f"entry-{index}"))

    first = store.page(limit=2)
    second = store.page(limit=2, offset=int(first["next_offset"]))

    assert first["total"] == 3
    assert len(first["items"]) == 2
    assert first["next_offset"] == 2
    assert len(second["items"]) == 1
    assert second["next_offset"] is None
    artifact_id = first["items"][0]["artifact_id"]
    exported = store.export(artifact_id)
    assert len(exported["rows"]) == 6
    assert exported["rows"][0]["pearson_ic"] is None


def test_rows_pages_reassemble_to_the_full_export(tmp_path: Path) -> None:
    store = EvaluationSeriesStore(tmp_path)
    artifact_id = store.save(_bundle())
    pages = [
        store.rows_page(artifact_id, offset=offset, limit=2) for offset in (0, 2, 4)
    ]

    assembled = [row for page in pages for row in page["rows"]]

    assert [page["next_offset"] for page in pages] == [2, 4, None]
    assert assembled == store.export(artifact_id)["rows"]


def test_prune_keeps_only_protected_artifacts(tmp_path: Path) -> None:
    store = EvaluationSeriesStore(tmp_path)
    keep_id = store.save(_bundle(entry_id="keep"))
    drop_id = store.save(_bundle(entry_id="drop"))

    removed = store.prune(protected={keep_id})

    assert removed == [drop_id]
    assert store.path_for(keep_id).exists()
    assert not store.path_for(drop_id).exists()


def test_report_protection_marker_survives_unprotected_cleanup(tmp_path: Path) -> None:
    store = EvaluationSeriesStore(tmp_path)
    cited_id = store.save(_bundle(entry_id="cited"))
    transient_id = store.save(_bundle(entry_id="transient"))

    store.protect(cited_id)
    removed = store.prune(protected=set())

    assert removed == [transient_id]
    assert store.path_for(cited_id).exists()
    assert store.load(cited_id).manifest.artifact_id == cited_id


def test_failed_publication_leaves_no_partial_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = EvaluationSeriesStore(tmp_path)
    first_id = store.save(_bundle())
    second = _bundle(evaluation_id="eval_fedcba9876543210")
    second_id = second.manifest.artifact_id

    def fail_replace(*args: object, **kwargs: object) -> None:
        raise OSError("publication failed")

    monkeypatch.setattr(
        "tradingdev.adapters.storage.evaluation_series.os.replace", fail_replace
    )
    with pytest.raises(OSError, match="publication failed"):
        store.save(second)

    assert not store.path_for(second_id).exists()
    assert not list((store.root / ".staging").iterdir())
    assert store.load(first_id).manifest.artifact_id == first_id


def test_failed_cleanup_keeps_the_artifact_that_could_not_be_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = EvaluationSeriesStore(tmp_path)
    store.save(_bundle(entry_id="first"))
    store.save(_bundle(entry_id="second"))
    ordered_ids = sorted(
        path.name for path in store.path_for("0" * 64).parent.iterdir()
    )
    removed_first, kept_id = ordered_ids
    original = shutil.rmtree

    def fail_on_second(path: str | Path, *args: Any, **kwargs: Any) -> None:
        if str(path).endswith(kept_id):
            raise OSError("cleanup failed")
        original(path, *args, **kwargs)

    monkeypatch.setattr(
        "tradingdev.adapters.storage.evaluation_series.shutil.rmtree", fail_on_second
    )
    with pytest.raises(OSError, match="cleanup failed"):
        store.prune(protected=set())

    assert not store.path_for(removed_first).exists()
    assert store.path_for(kept_id).exists()


def test_malformed_ids_are_rejected(tmp_path: Path) -> None:
    store = EvaluationSeriesStore(tmp_path)
    with pytest.raises(ValueError, match="malformed"):
        store.path_for("not-a-hash")
    with pytest.raises(ValueError, match="malformed"):
        store.protect("not-a-hash")


def test_saving_the_same_bundle_twice_is_idempotent(tmp_path: Path) -> None:
    store = EvaluationSeriesStore(tmp_path)
    bundle = _bundle()

    first = store.save(bundle)
    second = store.save(bundle)

    assert first == second
    assert store.load(first).manifest == bundle.manifest


def test_missing_artifact_is_unavailable(tmp_path: Path) -> None:
    store = EvaluationSeriesStore(tmp_path)
    with pytest.raises(FileNotFoundError, match="unavailable"):
        store.load("0" * 64)


def test_incomplete_and_unreadable_artifacts_are_rejected(tmp_path: Path) -> None:
    store = EvaluationSeriesStore(tmp_path)
    bundle = _bundle()
    incomplete = store.path_for("1" * 64)
    incomplete.mkdir(parents=True)
    (incomplete / "manifest.json").write_text(
        json.dumps(bundle.manifest.model_dump(mode="json")), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="incomplete"):
        store.load("1" * 64)

    unreadable = store.path_for("2" * 64)
    unreadable.mkdir(parents=True)
    (unreadable / "manifest.json").write_text("{not-json", encoding="utf-8")
    (unreadable / "series.parquet").write_bytes(frame_bytes(bundle.frame))
    with pytest.raises(ValueError, match="unreadable evaluation series manifest"):
        store.load("2" * 64)


def test_directory_id_must_match_the_manifest(tmp_path: Path) -> None:
    store = EvaluationSeriesStore(tmp_path)
    bundle = _bundle()
    wrong_id = "3" * 64
    path = store.path_for(wrong_id)
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(
        json.dumps(bundle.manifest.model_dump(mode="json")), encoding="utf-8"
    )
    (path / "series.parquet").write_bytes(frame_bytes(bundle.frame))

    with pytest.raises(ValueError, match="does not match its directory"):
        store.load(wrong_id)
    with pytest.raises(ValueError, match="does not match its directory"):
        store.page()


def test_bounds_are_enforced_for_discovery_and_rows(tmp_path: Path) -> None:
    store = EvaluationSeriesStore(tmp_path)
    artifact_id = store.save(_bundle())
    for limit in (0, 201):
        with pytest.raises(ValueError):
            store.page(limit=limit)
    with pytest.raises(ValueError):
        store.page(offset=-1)
    for limit in (0, 201):
        with pytest.raises(ValueError):
            store.rows_page(artifact_id, limit=limit)
    for offset in (-1, 7):
        with pytest.raises(ValueError):
            store.rows_page(artifact_id, offset=offset)


def test_export_bound_is_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(EvaluationSeriesStore, "MAX_EXPORT_ROWS", 5)
    store = EvaluationSeriesStore(tmp_path)
    artifact_id = store.save(_bundle())

    with pytest.raises(ValueError, match="export bound"):
        store.export(artifact_id)


def test_prune_ignores_non_directory_entries(tmp_path: Path) -> None:
    store = EvaluationSeriesStore(tmp_path)
    artifact_id = store.save(_bundle())
    stray = store.path_for("0" * 64).parent / "stray.txt"
    stray.write_text("keep", encoding="utf-8")

    assert store.prune(protected={artifact_id}) == []
    assert stray.exists()
