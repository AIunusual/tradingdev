"""Evaluation series artifacts are immutable, bounded and verifiable."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from tradingdev.adapters.storage.evaluation_series import EvaluationSeriesStore
from tradingdev.domain.ml.evaluation.series import (
    EvaluationSeriesBundle,
    build_bundle,
    build_signal_series,
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


def test_prune_keeps_only_protected_artifacts(tmp_path: Path) -> None:
    store = EvaluationSeriesStore(tmp_path)
    keep_id = store.save(_bundle(entry_id="keep"))
    drop_id = store.save(_bundle(entry_id="drop"))

    removed = store.prune(protected={keep_id})

    assert removed == [drop_id]
    assert store.path_for(keep_id).exists()
    assert not store.path_for(drop_id).exists()
