"""Pointwise evaluation series keep original coordinates and honest gaps."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.ml.evaluation.series import (
    EvaluationSeriesBundle,
    build_bundle,
    build_signal_series,
    to_json_records,
    verify_series,
)

_TIMESTAMPS = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")


def _manifest_fields(**overrides: object) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "kind": "signal_evaluation",
        "evaluation_id": "eval_abcdef0123456789",
        "dataset_id": "dataset-1",
        "dataset_hash": "a" * 64,
        "settings": {"window": 3, "horizon": 1, "n_quantiles": 5},
        "method_family": "rule",
        "producer_source": "backend_rule",
        "signal_fingerprint": "f" * 16,
        "training_status": "no_fitting",
        "created_at": "2024-01-01T00:00:00+00:00",
    }
    fields.update(overrides)
    return fields


def test_rolling_ic_matches_independently_computed_correlations() -> None:
    signal = [1.0, 2.0, 3.0, 1.0, 1.0, 1.0]
    forward = [0.1, 0.3, 0.2, 0.5, 0.4, 0.8]

    frame = build_signal_series(signal, forward, _TIMESTAMPS, window=3)

    expected_pearson = float(np.corrcoef(signal[:3], forward[:3])[0, 1])
    rank_signal = pd.Series(signal[:3]).rank().to_numpy()
    rank_forward = pd.Series(forward[:3]).rank().to_numpy()
    expected_spearman = float(np.corrcoef(rank_signal, rank_forward)[0, 1])
    assert frame.loc[2, "pearson_ic"] == pytest.approx(expected_pearson)
    assert frame.loc[2, "spearman_ic"] == pytest.approx(expected_spearman)


def test_warmup_and_constant_windows_keep_their_original_coordinates() -> None:
    signal = [1.0, 2.0, 3.0, 1.0, 1.0, 1.0]
    forward = [0.1, 0.3, 0.2, 0.5, 0.4, 0.8]

    frame = build_signal_series(signal, forward, _TIMESTAMPS, window=3)

    assert frame["source_row"].tolist() == list(range(6))
    assert frame["timestamp"].tolist() == [
        timestamp.isoformat() for timestamp in _TIMESTAMPS
    ]
    assert frame.loc[0, "availability_reason"] == "ic_warmup"
    assert pd.isna(frame.loc[0, "pearson_ic"])
    assert frame.loc[5, "availability_reason"] == "constant_window"
    assert pd.isna(frame.loc[5, "spearman_ic"])
    assert frame.loc[2, "availability_reason"] == "ok"


def test_changed_values_are_rejected_against_the_saved_content_hash() -> None:
    frame = build_signal_series(
        [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
        [0.1, 0.3, 0.2, 0.5, 0.4, 0.8],
        _TIMESTAMPS,
        window=3,
    )
    bundle = build_bundle(frame, **_manifest_fields())

    frame.loc[2, "signal"] = 99.0
    tampered = EvaluationSeriesBundle(manifest=bundle.manifest, frame=frame)

    with pytest.raises(ValueError, match="content hash"):
        verify_series(tampered)


def test_changed_settings_are_rejected_against_the_manifest_identity() -> None:
    frame = build_signal_series(
        [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
        [0.1, 0.3, 0.2, 0.5, 0.4, 0.8],
        _TIMESTAMPS,
        window=3,
    )
    bundle = build_bundle(frame, **_manifest_fields())
    changed = EvaluationSeriesBundle(
        manifest=bundle.manifest.model_copy(
            update={"settings": {"window": 99, "horizon": 1}}
        ),
        frame=bundle.frame,
    )

    with pytest.raises(ValueError, match="manifest identity"):
        verify_series(changed)


def test_reordered_timestamps_are_rejected_even_with_a_matching_hash() -> None:
    frame = build_signal_series(
        [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
        [0.1, 0.3, 0.2, 0.5, 0.4, 0.8],
        _TIMESTAMPS,
        window=3,
    )
    swapped = frame.copy()
    stamps = swapped["timestamp"].tolist()
    stamps[1], stamps[2] = stamps[2], stamps[1]
    swapped["timestamp"] = stamps
    rebuilt = build_bundle(swapped, **_manifest_fields())

    with pytest.raises(ValueError, match="chronological"):
        verify_series(rebuilt)


def test_dataset_pairing_reports_relationship_without_claiming_holdout() -> None:
    frame = build_signal_series(
        [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
        [0.1, 0.3, 0.2, 0.5, 0.4, 0.8],
        _TIMESTAMPS,
        window=3,
    )
    different = build_bundle(
        frame,
        **_manifest_fields(
            training_status="fitted",
            training_dataset_id="train-1",
            dataset_id="eval-1",
        ),
    )
    unknown = build_bundle(
        frame,
        **_manifest_fields(training_status="unknown", training_dataset_id=""),
    )

    assert different.manifest.holdout_relationship == "different_dataset"
    assert unknown.manifest.holdout_relationship == "unknown"
    assert "out_of_sample" not in different.manifest.model_dump()


def test_json_export_keeps_missing_values_as_null_not_zero() -> None:
    frame = build_signal_series(
        [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
        [0.1, 0.3, 0.2, 0.5, 0.4, 0.8],
        _TIMESTAMPS,
        window=3,
    )

    records = to_json_records(frame)

    assert records[0]["pearson_ic"] is None
    assert records[0]["source_row"] == 0
    assert records[2]["pearson_ic"] is not None
