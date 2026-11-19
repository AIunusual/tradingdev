"""Application orchestration for ML fold validation and final model training."""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from tradingdev.adapters.storage.evaluation_series import EvaluationSeriesStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths, now_iso
from tradingdev.adapters.storage.ml_datasets import MLDatasetStore
from tradingdev.domain.ml.evaluation import signal_metrics
from tradingdev.domain.ml.evaluation.series import (
    build_bundle,
    build_fold_series,
)
from tradingdev.domain.ml.leaderboard import signal_fingerprint
from tradingdev.domain.ml.models.registry import (
    RegisteredModelKind,
    create_direction_model,
    get_model_kind,
)
from tradingdev.domain.ml.registry import (
    ModelArtifactStore,
    build_model_artifact,
    content_sha256,
)
from tradingdev.domain.ml.validation import get_validation_scheme
from tradingdev.domain.ml.validation.temporal import temporal_folds

if TYPE_CHECKING:
    import numpy.typing as npt

    from tradingdev.adapters.storage.filesystem import WorkspacePaths


@dataclass(frozen=True)
class _ValidationRequest:
    """A validation request that has been resolved and checked."""

    scheme: str
    label_horizon: int
    """Initial purge horizon; saved label intervals may exclude more dates."""
    n_folds: int
    embargo: int
    min_train_bars: int


# Cap rolling IC windows at the fold length.
_FOLD_IC_WINDOW = 60


class MLTrainingService:
    """Validate immutable datasets, fit models and publish their artifacts."""

    def __init__(self, workspace: WorkspacePaths) -> None:
        self._workspace = workspace

    def train_model(
        self,
        *,
        dataset_id: str,
        model_kind: str,
        parameters: dict[str, Any] | None = None,
        validation: str | None = None,
        n_folds: int = 5,
        embargo: int = 0,
        min_train_bars: int | None = None,
    ) -> dict[str, Any]:
        """Validate folds, fit all rows and publish a content-bound model artifact."""
        frame, manifest = MLDatasetStore(self._workspace.ml_datasets).load(dataset_id)
        try:
            kind = get_model_kind(model_kind)
        except KeyError as exc:
            raise ValueError(f"unknown model kind: {model_kind}") from exc

        params = self._validated_parameters(kind, parameters or {})
        # Reject unusable validation geometry before fitting.
        request = self._validation_request(
            scheme=validation,
            n_folds=n_folds,
            embargo=embargo,
            min_train_bars=min_train_bars,
            n_bars=frame["timestamp"].nunique(),
            label_horizon=int(manifest["horizon"]),
        )

        feature_columns = list(manifest["feature_columns"])

        validation_summary = None
        if request is not None:
            validation_summary = self._validate_in_folds(
                request,
                dataset_id=dataset_id,
                dataset_hash=str(manifest["content_hash"]),
                model_kind=model_kind,
                params=params,
                frame=frame,
                feature_columns=feature_columns,
            )

        model = create_direction_model(model_kind, **params)
        # Explicit feature selection excludes labels and their observation times.
        model.train(frame[feature_columns + ["target"]])

        # Fingerprint declared defaults plus overrides, not estimator reporting.
        effective_parameters = kind.config_cls(**params).model_dump()
        blob = pickle.dumps(model)
        artifact = build_model_artifact(
            model_name=model_kind,
            framework=kind.framework,
            kind="direction",
            parameters=effective_parameters,
            feature_names=tuple(feature_columns),
            content_hash=content_sha256(blob),
            n_train_samples=int(len(frame)),
            training_dataset_id=dataset_id,
            feature_importance=model.feature_importance(),
        )
        store = ModelArtifactStore(self._workspace.models)
        store.save(model_bytes=blob, artifact=artifact)

        predictions = model.predict(frame).to_numpy()
        in_sample_accuracy = float((predictions == frame["target"].to_numpy()).mean())
        return {
            "success": True,
            "model": {
                "model_id": artifact.model_id,
                "model_name": model_kind,
                "framework": kind.framework,
                "kind": "direction",
                "dataset_id": dataset_id,
                "n_train_samples": int(len(frame)),
                "n_features": len(manifest["feature_columns"]),
                "parameters": artifact.parameters,
                "feature_importance": [
                    {"feature": name, "importance": value}
                    for name, value in artifact.feature_importance
                ],
                "in_sample_accuracy": in_sample_accuracy,
                "validation": validation_summary,
                "path": str(store.path_for(artifact.model_id)),
            },
        }

    @staticmethod
    def _validation_request(
        *,
        scheme: str | None,
        n_folds: int,
        embargo: int,
        min_train_bars: int | None,
        n_bars: int,
        label_horizon: int,
    ) -> _ValidationRequest | None:
        """Resolve the advertised scheme and validate geometry before any fitting."""
        if scheme is None:
            return None
        get_validation_scheme(scheme)
        if n_folds < 1:
            raise ValueError(f"n_folds must be at least 1, got {n_folds}")
        if embargo < 0:
            raise ValueError(f"embargo must not be negative, got {embargo}")
        if label_horizon < 1:
            raise ValueError(
                "this dataset has no label horizon, so the purge width cannot "
                "be determined; rebuild it with an explicit horizon"
            )
        floor = min_train_bars
        if floor is None:
            # Reserve room for purge/embargo inside the first training block.
            floor = max(1, n_bars // max(1, 2 * (n_folds + 1)))
        return _ValidationRequest(
            scheme=scheme,
            label_horizon=label_horizon,
            n_folds=n_folds,
            embargo=embargo,
            min_train_bars=floor,
        )

    def _validate_in_folds(
        self,
        request: _ValidationRequest,
        *,
        dataset_id: str,
        dataset_hash: str,
        model_kind: str,
        params: dict[str, Any],
        frame: pd.DataFrame,
        feature_columns: list[str],
    ) -> dict[str, Any]:
        """Fit and score separate fold models using the saved label horizon.

        Keep all symbols on a date together and check actual label intervals
        after geometric purging. These scores do not describe the final model,
        which is subsequently fitted on every row.
        """
        splits = temporal_folds(
            frame,
            scheme=request.scheme,
            horizon=request.label_horizon,
            n_folds=request.n_folds,
            embargo=request.embargo,
            min_train_dates=request.min_train_bars,
        )
        model_kind_info = get_model_kind(model_kind)
        effective_parameters = model_kind_info.config_cls(**params).model_dump()

        folds: list[dict[str, Any]] = []
        fold_rows: list[dict[str, Any]] = []
        all_predictions: list[float] = []
        for index, split in enumerate(splits):
            train_rows = [int(row) for row in split.train_rows]
            test_rows = [int(row) for row in split.test_rows]
            train_frame = frame.iloc[train_rows]
            test_frame = frame.iloc[test_rows]
            fold_model = create_direction_model(model_kind, **params)
            fold_model.train(train_frame[feature_columns + ["target"]])
            predictions = fold_model.predict(test_frame).to_numpy()
            all_predictions.extend(float(value) for value in predictions)
            folds.append(
                {
                    "fold": index,
                    "train_bars": int(len(train_frame)),
                    "test_bars": int(len(test_frame)),
                    "purged_bars": split.purged_rows,
                    "embargoed_bars": split.embargoed_rows,
                    **self._frame_scores(test_frame, predictions),
                }
            )
            for position, (source_row, prediction) in enumerate(
                zip(test_frame.index, predictions, strict=True)
            ):
                fold_rows.append(
                    {
                        "fold": index,
                        "source_row": int(source_row),
                        "symbol": (
                            str(test_frame["symbol"].iloc[position])
                            if "symbol" in test_frame.columns
                            else ""
                        ),
                        "timestamp": pd.to_datetime(
                            test_frame["timestamp"].iloc[position], utc=True
                        ).isoformat(),
                        "prediction": float(prediction),
                        "target": float(test_frame["target"].iloc[position]),
                        "forward_return": float(
                            test_frame["forward_return"].iloc[position]
                        ),
                        "train_start_row": train_rows[0],
                        "train_end_row": train_rows[-1],
                        "eval_start_row": test_rows[0],
                        "eval_end_row": test_rows[-1],
                        "train_start": pd.to_datetime(
                            frame["timestamp"].iloc[train_rows[0]], utc=True
                        ).isoformat(),
                        "train_end": pd.to_datetime(
                            frame["timestamp"].iloc[train_rows[-1]], utc=True
                        ).isoformat(),
                        "eval_start": pd.to_datetime(
                            frame["timestamp"].iloc[test_rows[0]], utc=True
                        ).isoformat(),
                        "eval_end": pd.to_datetime(
                            frame["timestamp"].iloc[test_rows[-1]], utc=True
                        ).isoformat(),
                    }
                )

        series_frame = build_fold_series(fold_rows)
        fingerprint = signal_fingerprint(all_predictions)
        bundle = build_bundle(
            series_frame,
            kind="fold_validation",
            evaluation_id=f"fold_{dataset_id}",
            dataset_id=dataset_id,
            dataset_hash=dataset_hash,
            settings={
                "scheme": request.scheme,
                "n_folds": request.n_folds,
                "embargo": request.embargo,
                "label_horizon": request.label_horizon,
                "min_train_bars": request.min_train_bars,
                "model_kind": model_kind,
                "parameters": dict(params),
                "candidate": {
                    "model_id": "",
                    "method_name": "",
                    "parameters": effective_parameters,
                },
            },
            method_family="ml_model",
            producer_source="backend_model",
            signal_fingerprint=fingerprint,
            training_dataset_id=dataset_id,
            training_status="fitted",
            created_at=now_iso(),
        )

        return {
            "scheme": request.scheme,
            "label_horizon": int(request.label_horizon),
            "embargo": int(request.embargo),
            "n_folds": len(folds),
            "purged_bars": sum(split.purged_rows for split in splits),
            "embargoed_bars": sum(split.embargoed_rows for split in splits),
            "leakage_audit_clean": True,
            "artifact_trained_on": "all_bars",
            "fold_predictions_available": True,
            "fold_series_artifact_id": EvaluationSeriesStore(
                self._workspace.ml_evaluation_series
            ).save(bundle, protect=True),
            "folds": folds,
            "aggregate": self._aggregated_scores(folds),
        }

    def _frame_scores(
        self, frame: pd.DataFrame, predictions: npt.NDArray[Any]
    ) -> dict[str, Any]:
        """Average per-symbol scores without rolling across asset boundaries."""
        scored = frame.assign(prediction=predictions)
        groups = (
            [group for _, group in scored.groupby("symbol")]
            if "symbol" in scored.columns
            else [scored]
        )
        return self._aggregated_scores(
            [
                self._fold_scores(
                    predictions=group["prediction"].to_numpy(),
                    forward_returns=group["forward_return"].to_numpy(),
                    targets=group["target"].to_numpy(),
                )
                for group in (item.sort_values("timestamp") for item in groups)
            ]
        )

    def _fold_scores(
        self,
        *,
        predictions: npt.NDArray[Any],
        forward_returns: npt.NDArray[Any],
        targets: npt.NDArray[Any],
    ) -> dict[str, Any]:
        """Score with a fold-bounded IC window; map non-finite scores to None."""
        accuracy = float((predictions == targets).mean())
        scores: dict[str, Any] = {"accuracy": self._finite(accuracy)}
        if len(predictions) < 2:
            return {**scores, "ic_mean": None, "rank_ic_mean": None, "hit_rate": None}
        window = max(2, min(_FOLD_IC_WINDOW, len(predictions)))
        raw = signal_metrics(
            pd.Series(predictions, dtype=float),
            pd.Series(forward_returns, dtype=float),
            window=window,
        )
        return {
            **scores,
            "ic_mean": self._finite(raw.get("ic_mean")),
            "rank_ic_mean": self._finite(raw.get("rank_ic_mean")),
            "hit_rate": self._finite(raw.get("hit_rate")),
        }

    @staticmethod
    def _aggregated_scores(folds: list[dict[str, Any]]) -> dict[str, Any]:
        """Average available fold scores; leave unmeasured metrics as None."""
        aggregate: dict[str, Any] = {}
        for key in ("accuracy", "ic_mean", "rank_ic_mean", "hit_rate"):
            present = [
                float(fold[key])
                for fold in folds
                if fold.get(key) is not None and math.isfinite(float(fold[key]))
            ]
            aggregate[key] = sum(present) / len(present) if present else None
        return aggregate

    @staticmethod
    def _validated_parameters(
        kind: RegisteredModelKind, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Reject undeclared parameters rather than silently training with defaults."""
        allowed = set(kind.config_cls.model_fields)
        unknown = sorted(set(parameters) - allowed)
        if unknown:
            raise ValueError(
                f"unknown parameters for {kind.name}: {', '.join(unknown)}"
            )
        return dict(parameters)

    @staticmethod
    def _finite(value: object) -> float | int | None:
        """Copy a finite number, mapping missing or non-finite inputs to None."""
        if value is None:
            return None
        number = float(value)  # type: ignore[arg-type]
        return number if math.isfinite(number) else None
