"""Pointwise evaluation series keep original coordinates and honest gaps."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.ml.evaluation import series as series_module
from tradingdev.domain.ml.evaluation.metrics import signal_metrics_with_series
from tradingdev.domain.ml.evaluation.series import (
    FOLD_COLUMNS,
    SIGNAL_COLUMNS,
    EvaluationSeriesBundle,
    EvaluationSeriesManifest,
    build_bundle,
    build_fold_series,
    build_signal_series,
    compute_artifact_id,
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


def test_pearson_underflow_preserves_the_scored_spearman_values() -> None:
    signal = [1e-200, 2e-200, 3e-200, 4e-200]
    forward = [0.1, 0.3, 0.2, 0.7]
    metrics, pearson, spearman = signal_metrics_with_series(signal, forward, window=3)

    frame = build_signal_series(
        signal,
        forward,
        _TIMESTAMPS[:4],
        window=3,
        pearson_ic=pearson,
        spearman_ic=spearman,
    )

    assert metrics["ic_mean"] is None
    assert metrics["rank_ic_mean"] == pytest.approx(0.5)
    assert frame["pearson_ic"].isna().all()
    assert frame["pearson_availability_reason"].tolist() == [
        "ic_warmup",
        "ic_warmup",
        "non_finite",
        "non_finite",
    ]
    assert frame["spearman_ic"].iloc[2:].tolist() == pytest.approx([0.5, 0.5])
    assert frame["spearman_availability_reason"].iloc[2:].tolist() == ["ok", "ok"]


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


def _series_frame() -> pd.DataFrame:
    return build_signal_series(
        [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
        [0.1, 0.3, 0.2, 0.5, 0.4, 0.8],
        _TIMESTAMPS,
        window=3,
    )


def _rebound(
    manifest: EvaluationSeriesManifest, **updates: object
) -> EvaluationSeriesManifest:
    changed = manifest.model_copy(update=updates)
    return changed.model_copy(
        update={"artifact_id": compute_artifact_id(changed.model_dump())}
    )


def test_build_signal_series_rejects_mismatched_and_short_inputs() -> None:
    with pytest.raises(ValueError, match="equal length"):
        build_signal_series([1.0, 2.0], [0.1], _TIMESTAMPS, window=2)
    with pytest.raises(ValueError, match="at least 2"):
        build_signal_series([1.0] * 6, [0.1] * 6, _TIMESTAMPS, window=1)
    with pytest.raises(ValueError, match="not enough rows"):
        build_signal_series([1.0, 2.0], [0.1, 0.2], _TIMESTAMPS[:2], window=3)
    with pytest.raises(ValueError):
        build_signal_series([1.0] * 6, [0.1] * 6, ["not-a-date"] * 6, window=3)


def test_non_finite_correlation_keeps_a_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        series_module, "_correlation", lambda *args, **kwargs: float("nan")
    )

    frame = build_signal_series(
        [1.0, 2.0, 3.0], [0.1, 0.3, 0.2], _TIMESTAMPS[:3], window=3
    )

    assert frame.loc[2, "availability_reason"] == "non_finite"
    assert pd.isna(frame.loc[2, "pearson_ic"])


def test_one_ic_failure_does_not_erase_the_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = series_module._correlation

    def fail_pearson(left: Any, right: Any, *, rank: bool) -> float:
        if not rank:
            raise ZeroDivisionError("pearson unavailable")
        return original(left, right, rank=rank)

    monkeypatch.setattr(series_module, "_correlation", fail_pearson)

    frame = build_signal_series(
        [1.0, 2.0, 3.0, 4.0], [0.1, 0.3, 0.2, 0.7], _TIMESTAMPS[:4], window=3
    )

    assert pd.isna(frame.loc[2, "pearson_ic"])
    assert frame.loc[2, "pearson_availability_reason"] == "non_finite"
    assert frame.loc[2, "spearman_ic"] is not None
    assert frame.loc[2, "spearman_availability_reason"] == "ok"


def test_build_bundle_rejects_wrong_columns_and_empty_frames() -> None:
    frame = _series_frame()
    with pytest.raises(ValueError, match="needs columns"):
        build_bundle(frame.drop(columns=["signal"]), **_manifest_fields())
    with pytest.raises(ValueError, match="at least one row"):
        build_bundle(frame.iloc[0:0], **_manifest_fields())


def test_build_fold_series_requires_integer_coordinates() -> None:
    with pytest.raises(ValueError, match="fold series needs fold"):
        build_fold_series([{}])


def test_manifest_roundtrip_keeps_tuple_columns() -> None:
    bundle = build_bundle(_series_frame(), **_manifest_fields())
    record = bundle.manifest.model_dump(mode="json")

    assert EvaluationSeriesManifest.from_record(record).columns == SIGNAL_COLUMNS
    assert FOLD_COLUMNS[-1] == "eval_end"


def test_verify_rejects_column_and_row_count_mismatches() -> None:
    bundle = build_bundle(_series_frame(), **_manifest_fields())

    with pytest.raises(ValueError, match="columns do not match"):
        verify_series(
            EvaluationSeriesBundle(
                _rebound(bundle.manifest, columns=("source_row",)), bundle.frame
            )
        )
    with pytest.raises(ValueError, match="row count does not match"):
        verify_series(
            EvaluationSeriesBundle(_rebound(bundle.manifest, row_count=5), bundle.frame)
        )


def test_verify_rejects_wrong_ic_availability_claims() -> None:
    frame = _series_frame()
    frame.loc[2, "pearson_ic"] = np.nan
    with pytest.raises(ValueError, match="available IC cell is missing"):
        verify_series(build_bundle(frame, **_manifest_fields()))

    frame = _series_frame()
    frame.loc[0, "pearson_ic"] = 0.1
    with pytest.raises(ValueError, match="unavailable IC cell carries a value"):
        verify_series(build_bundle(frame, **_manifest_fields()))


def test_fold_series_keeps_symbol_for_panel_predictions() -> None:
    record = {
        "fold": 0,
        "source_row": 4,
        "symbol": "BTC/USDT",
        "timestamp": "2024-01-01T00:00:00+00:00",
        "prediction": 1.0,
        "target": 1.0,
        "forward_return": 0.1,
        "train_start_row": 0,
        "train_end_row": 3,
        "eval_start_row": 4,
        "eval_end_row": 5,
        "train_start": "2024-01-01T00:00:00+00:00",
        "train_end": "2024-01-01T03:00:00+00:00",
        "eval_start": "2024-01-01T04:00:00+00:00",
        "eval_end": "2024-01-01T05:00:00+00:00",
    }

    frame = build_fold_series(
        [record, {**record, "source_row": 5, "symbol": "ETH/USDT"}]
    )

    assert frame["symbol"].tolist() == ["BTC/USDT", "ETH/USDT"]


def test_verify_rejects_duplicate_fold_coordinates() -> None:
    record = {
        "fold": 0,
        "source_row": 4,
        "timestamp": "2024-01-01T00:00:00+00:00",
        "prediction": 1.0,
        "target": 1.0,
        "forward_return": 0.1,
        "train_start_row": 0,
        "train_end_row": 3,
        "eval_start_row": 4,
        "eval_end_row": 5,
        "train_start": "2024-01-01T00:00:00+00:00",
        "train_end": "2024-01-01T03:00:00+00:00",
        "eval_start": "2024-01-01T04:00:00+00:00",
        "eval_end": "2024-01-01T05:00:00+00:00",
    }
    frame = build_fold_series([record, {**record, "source_row": 4}])
    bundle = build_bundle(
        frame,
        **_manifest_fields(
            kind="fold_validation",
            settings={},
            method_family="ml_model",
            producer_source="backend_model",
            training_status="fitted",
            training_dataset_id="train-1",
        ),
    )

    with pytest.raises(ValueError, match="repeat a source coordinate"):
        verify_series(bundle)
