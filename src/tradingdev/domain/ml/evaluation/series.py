"""Versioned, immutable pointwise evaluation series.

The scalar :class:`~tradingdev.domain.ml.evaluation.artifacts.SignalEvaluationBundle`
answers how a signal scored in aggregate. This module answers a different
question: which exact rows produced that score. A series is bound to the
producer's dataset identity, settings and content hash so a reader can verify
it long after the model file is gone. Missing IC values keep their original
row coordinates and carry a reason; they are never dropped, renumbered or
zero-filled.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass
from statistics import correlation
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SIGNAL_COLUMNS: tuple[str, ...] = (
    "source_row",
    "timestamp",
    "signal",
    "realized_return",
    "pearson_ic",
    "spearman_ic",
    "availability_reason",
)
FOLD_COLUMNS: tuple[str, ...] = (
    "fold",
    "source_row",
    "timestamp",
    "prediction",
    "target",
    "forward_return",
    "train_start_row",
    "train_end_row",
    "eval_start_row",
    "eval_end_row",
    "train_start",
    "train_end",
    "eval_start",
    "eval_end",
)
_FOLD_TEXT_COLUMNS = frozenset(
    {"timestamp", "train_start", "train_end", "eval_start", "eval_end"}
)

_FOLD_INT_COLUMNS = frozenset(
    {
        "fold",
        "source_row",
        "train_start_row",
        "train_end_row",
        "eval_start_row",
        "eval_end_row",
    }
)

SeriesKind = Literal["signal_evaluation", "fold_validation"]
TrainingStatus = Literal["fitted", "no_fitting", "unknown"]
HoldoutRelationship = Literal[
    "in_sample", "different_dataset", "not_applicable", "unknown"
]


class SeriesArtifactModel(BaseModel):
    """Reject coercion and non-finite numbers in a frozen artifact record."""

    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, allow_inf_nan=False
    )


class EvaluationSeriesManifest(SeriesArtifactModel):
    """Identity and verification data travelling with one saved series."""

    schema_version: Literal[1] = 1
    artifact_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: SeriesKind
    evaluation_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
    dataset_id: str
    dataset_hash: str
    settings: dict[str, Any]
    method_family: str
    producer_source: str
    signal_fingerprint: str = ""
    entry_id: str = ""
    training_dataset_id: str = ""
    training_status: TrainingStatus = "unknown"
    holdout_relationship: HoldoutRelationship = "unknown"
    index_convention: Literal["source_row_utc"] = "source_row_utc"
    columns: tuple[str, ...]
    row_count: int = Field(ge=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: str

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> EvaluationSeriesManifest:
        """Validate a stored manifest without filling in missing identity."""
        values = dict(record)
        if isinstance(values.get("columns"), list):
            values["columns"] = tuple(values["columns"])
        return cls.model_validate(values)


@dataclass(frozen=True)
class EvaluationSeriesBundle:
    """A frame plus the manifest that makes it a verifiable artifact."""

    manifest: EvaluationSeriesManifest
    frame: pd.DataFrame


def frame_bytes(frame: pd.DataFrame) -> bytes:
    """Canonical Parquet bytes; both writer and verifier use this function."""
    stream = io.BytesIO()
    frame.to_parquet(stream, index=False)
    return stream.getvalue()


def compute_artifact_id(fields: Mapping[str, Any]) -> str:
    """Hash the manifest identity without its self-referential artifact ID."""
    payload = {key: value for key, value in fields.items() if key != "artifact_id"}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _holdout_relationship(
    *, training_status: TrainingStatus, training_dataset_id: str, dataset_id: str
) -> HoldoutRelationship:
    """Report the recorded pairing; never assert a holdout from IDs alone."""
    if training_status == "no_fitting":
        return "not_applicable"
    if training_status == "fitted" and training_dataset_id and dataset_id:
        return "in_sample" if training_dataset_id == dataset_id else "different_dataset"
    return "unknown"


def build_bundle(
    frame: pd.DataFrame,
    *,
    kind: SeriesKind,
    evaluation_id: str,
    dataset_id: str,
    dataset_hash: str,
    settings: Mapping[str, Any],
    method_family: str,
    producer_source: str,
    signal_fingerprint: str = "",
    entry_id: str = "",
    training_dataset_id: str = "",
    training_status: TrainingStatus = "unknown",
    created_at: str,
) -> EvaluationSeriesBundle:
    """Bind a frame to its producer and content, deriving stable IDs."""
    expected = SIGNAL_COLUMNS if kind == "signal_evaluation" else FOLD_COLUMNS
    if tuple(frame.columns) != expected:
        raise ValueError(
            f"{kind} needs columns {list(expected)}, got {list(frame.columns)}"
        )
    if frame.empty:
        raise ValueError("an evaluation series must contain at least one row")
    fields: dict[str, Any] = {
        "schema_version": 1,
        "kind": kind,
        "evaluation_id": evaluation_id,
        "dataset_id": dataset_id,
        "dataset_hash": dataset_hash,
        "settings": dict(settings),
        "method_family": method_family,
        "producer_source": producer_source,
        "signal_fingerprint": signal_fingerprint,
        "entry_id": entry_id,
        "training_dataset_id": training_dataset_id,
        "training_status": training_status,
        "holdout_relationship": _holdout_relationship(
            training_status=training_status,
            training_dataset_id=training_dataset_id,
            dataset_id=dataset_id,
        ),
        "index_convention": "source_row_utc",
        "columns": tuple(frame.columns),
        "row_count": int(len(frame)),
        "content_hash": hashlib.sha256(frame_bytes(frame)).hexdigest(),
        "created_at": created_at,
    }
    fields["artifact_id"] = compute_artifact_id(fields)
    return EvaluationSeriesBundle(
        manifest=EvaluationSeriesManifest.model_validate(fields), frame=frame
    )


def _finite(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)  # type: ignore[arg-type]
    return number if math.isfinite(number) else None


def _correlation(left: Sequence[float], right: Sequence[float], *, rank: bool) -> float:
    """Independent correlation for one window, with no rolling-state reuse."""
    if rank:
        left_values: list[float] = pd.Series(left, dtype=float).rank().tolist()
        right_values: list[float] = pd.Series(right, dtype=float).rank().tolist()
    else:
        left_values = [float(value) for value in left]
        right_values = [float(value) for value in right]
    return float(correlation(left_values, right_values))


def build_signal_series(
    signal: Sequence[float],
    forward_returns: Sequence[float],
    timestamps: Sequence[Any] | pd.DatetimeIndex,
    *,
    window: int,
) -> pd.DataFrame:
    """Save every scored row with rolling IC and its availability reason."""
    if len(signal) != len(forward_returns) or len(signal) != len(timestamps):
        raise ValueError(
            "signal, forward_returns and timestamps must have equal length "
            f"(got {len(signal)}, {len(forward_returns)}, {len(timestamps)})"
        )
    if window < 2:
        raise ValueError("window must be at least 2")
    if len(signal) < window:
        raise ValueError(
            f"not enough rows for an IC window of {window}: got {len(signal)}"
        )
    utc = pd.to_datetime(pd.Series(list(timestamps)), utc=True)
    if utc.isna().any():
        raise ValueError("evaluation timestamps must all be parseable")

    pearson: list[float | None] = []
    spearman: list[float | None] = []
    reasons: list[str] = []
    for index in range(len(signal)):
        if index + 1 < window:
            pearson.append(None)
            spearman.append(None)
            reasons.append("ic_warmup")
            continue
        window_signal = [
            float(value) for value in signal[index - window + 1 : index + 1]
        ]
        window_forward = [
            float(value) for value in forward_returns[index - window + 1 : index + 1]
        ]
        if len(set(window_signal)) < 2 or len(set(window_forward)) < 2:
            pearson.append(None)
            spearman.append(None)
            reasons.append("constant_window")
            continue
        try:
            pair = (
                _correlation(window_signal, window_forward, rank=False),
                _correlation(window_signal, window_forward, rank=True),
            )
        except ValueError, ZeroDivisionError:
            pair = (float("nan"), float("nan"))
        values = tuple(_finite(value) for value in pair)
        if any(value is None for value in values):
            pearson.append(None)
            spearman.append(None)
            reasons.append("non_finite")
            continue
        pearson.append(values[0])
        spearman.append(values[1])
        reasons.append("ok")

    return pd.DataFrame(
        {
            "source_row": range(len(signal)),
            "timestamp": [timestamp.isoformat() for timestamp in utc],
            "signal": [float(value) for value in signal],
            "realized_return": [float(value) for value in forward_returns],
            "pearson_ic": pearson,
            "spearman_ic": spearman,
            "availability_reason": reasons,
        }
    )


def build_fold_series(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Normalise fold predictions into the canonical immutable frame."""
    data: dict[str, list[Any]] = {name: [] for name in FOLD_COLUMNS}
    for record in records:
        for name in FOLD_COLUMNS:
            value = record.get(name)
            if name in _FOLD_TEXT_COLUMNS:
                data[name].append(str(value))
            elif name in _FOLD_INT_COLUMNS:
                if value is None:
                    raise ValueError(f"fold series needs {name}")
                data[name].append(int(value))
            else:
                data[name].append(_finite(value))
    return pd.DataFrame(data, columns=list(FOLD_COLUMNS))


def to_json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Project NaNs to null so a reader never sees a fabricated zero."""
    records: list[dict[str, Any]] = []
    for values in frame.itertuples(index=False, name=None):
        records.append(
            {
                name: None
                if pd.isna(value)
                else value.item()
                if hasattr(value, "item")
                else value
                for name, value in zip(frame.columns, values, strict=True)
            }
        )
    return records


def verify_series(bundle: EvaluationSeriesBundle) -> None:
    """Refuse changed values, settings, identity or misaligned coordinates."""
    manifest = bundle.manifest
    if hashlib.sha256(frame_bytes(bundle.frame)).hexdigest() != manifest.content_hash:
        raise ValueError("evaluation series content hash does not match its manifest")
    if compute_artifact_id(manifest.model_dump()) != manifest.artifact_id:
        raise ValueError("evaluation series manifest identity does not match its ID")
    if tuple(bundle.frame.columns) != manifest.columns:
        raise ValueError("evaluation series columns do not match their manifest")
    if len(bundle.frame) != manifest.row_count:
        raise ValueError("evaluation series row count does not match its manifest")
    if manifest.kind == "signal_evaluation":
        rows = bundle.frame["source_row"].tolist()
        if rows != list(range(manifest.row_count)):
            raise ValueError(
                "evaluation series source rows are not in their original order"
            )
        for reason, pearson, spearman in zip(
            bundle.frame["availability_reason"],
            bundle.frame["pearson_ic"],
            bundle.frame["spearman_ic"],
            strict=True,
        ):
            if reason == "ok":
                if _finite(pearson) is None or _finite(spearman) is None:
                    raise ValueError("an available IC cell is missing its value")
            elif _finite(pearson) is not None or _finite(spearman) is not None:
                raise ValueError("an unavailable IC cell carries a value")
    else:
        if bundle.frame["source_row"].duplicated().any():
            raise ValueError("fold prediction rows repeat a source coordinate")
