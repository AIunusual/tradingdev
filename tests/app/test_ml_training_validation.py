"""Independently verify purged fold geometry, scores and saved-model behavior."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
from tests.data_cache import seed_market_cache

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.ml_service import MLService
from tradingdev.domain.ml.models import create_direction_model
from tradingdev.domain.ml.validation import (
    audit_validation_splits,
    build_validation_splits,
    validation_scheme_names,
)

if TYPE_CHECKING:
    from pathlib import Path

_ROWS = 300
_METRIC_KEYS = ("accuracy", "ic_mean", "rank_ic_mean", "hit_rate")


def _write_cached_bars(workspace: WorkspacePaths) -> None:
    """Write oscillating OHLCV bars, exactly what ensure_data would cache."""
    index = np.arange(_ROWS, dtype=float)
    close = 100.0 + 10.0 * np.sin(index / 5.0) + index * 0.01
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=_ROWS, freq="h", tz="UTC"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(_ROWS, 10.0),
        }
    )
    seed_market_cache(workspace.processed_data, frame)


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspacePaths:
    paths = WorkspacePaths(tmp_path / "workspace")
    paths.ensure()
    _write_cached_bars(paths)
    return paths


@pytest.fixture
def service(workspace: WorkspacePaths) -> MLService:
    return MLService(workspace=workspace)


def _build(service: MLService, **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "feature_set": "returns",
        "horizon": 5,
    }
    kwargs.update(overrides)
    dataset = service.build_dataset(**kwargs)["dataset"]
    assert isinstance(dataset, dict)
    return dataset


def _train(service: MLService, dataset_id: str, **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "dataset_id": dataset_id,
        "model_kind": "xgboost_direction",
        "parameters": {"n_estimators": 5, "max_depth": 2},
    }
    kwargs.update(overrides)
    model = service.train_model(**kwargs)["model"]
    assert isinstance(model, dict)
    return model


def _validation(
    service: MLService, dataset_id: str, **overrides: Any
) -> dict[str, Any]:
    model = _train(service, dataset_id, **overrides)
    validation = model["validation"]
    assert isinstance(validation, dict)
    return validation


class TestContractMatchesTheRegistry:
    def test_the_advertised_schemes_are_the_buildable_ones(
        self, service: MLService
    ) -> None:
        advertised = [
            item["name"] for item in service.get_contract()["validation_schemes"]
        ]
        assert tuple(advertised) == validation_scheme_names()

    def test_every_advertised_scheme_is_described(self, service: MLService) -> None:
        for item in service.get_contract()["validation_schemes"]:
            assert item["description"].strip()

    @pytest.mark.parametrize("scheme", validation_scheme_names())
    def test_every_advertised_scheme_trains(
        self, service: MLService, scheme: str
    ) -> None:
        dataset_id = _build(service)["dataset_id"]
        validation = _validation(service, dataset_id, validation=scheme, n_folds=3)
        assert validation["scheme"] == scheme
        assert validation["n_folds"] == 3


class TestNoSchemeMeansNoValidation:
    def test_validation_is_absent_when_none_is_asked_for(
        self, service: MLService
    ) -> None:
        model = _train(service, _build(service)["dataset_id"])
        assert model["validation"] is None

    def test_in_sample_accuracy_is_still_reported(self, service: MLService) -> None:
        model = _train(service, _build(service)["dataset_id"])
        assert model["in_sample_accuracy"] is not None


class TestWalkForward:
    def test_returns_one_fold_per_request(self, service: MLService) -> None:
        dataset_id = _build(service)["dataset_id"]
        validation = _validation(
            service, dataset_id, validation="purged_walk_forward", n_folds=4
        )
        assert validation["n_folds"] == 4
        assert len(validation["folds"]) == 4
        assert [fold["fold"] for fold in validation["folds"]] == [0, 1, 2, 3]

    def test_folds_are_anchored_so_history_grows(self, service: MLService) -> None:
        validation = _validation(
            service,
            _build(service)["dataset_id"],
            validation="purged_walk_forward",
            n_folds=3,
        )
        train_bars = [fold["train_bars"] for fold in validation["folds"]]
        assert train_bars == sorted(train_bars)
        assert len(set(train_bars)) == len(train_bars)

    def test_folds_never_train_on_the_whole_dataset(self, service: MLService) -> None:
        dataset = _build(service)
        model = _train(
            service,
            dataset["dataset_id"],
            validation="purged_walk_forward",
            n_folds=3,
        )
        for fold in model["validation"]["folds"]:
            assert 0 < fold["train_bars"] < model["n_train_samples"]
            assert fold["test_bars"] > 0

    def test_the_purge_is_the_datasets_own_label_horizon(
        self, service: MLService
    ) -> None:
        for horizon in (1, 5):
            dataset_id = _build(service, horizon=horizon)["dataset_id"]
            validation = _validation(
                service, dataset_id, validation="purged_walk_forward", n_folds=3
            )
            assert validation["label_horizon"] == horizon
            assert [fold["purged_bars"] for fold in validation["folds"]] == [
                horizon
            ] * 3

    def test_a_longer_horizon_purges_more(self, service: MLService) -> None:
        short = _validation(
            service,
            _build(service, horizon=1)["dataset_id"],
            validation="purged_walk_forward",
            n_folds=3,
        )
        long = _validation(
            service,
            _build(service, horizon=12)["dataset_id"],
            validation="purged_walk_forward",
            n_folds=3,
        )
        assert short["purged_bars"] < long["purged_bars"]
        assert long["folds"][0]["train_bars"] < short["folds"][0]["train_bars"]

    def test_embargo_is_charged_on_top_of_the_purge(self, service: MLService) -> None:
        dataset_id = _build(service)["dataset_id"]
        bare = _validation(
            service,
            dataset_id,
            validation="purged_walk_forward",
            n_folds=3,
            embargo=0,
        )
        widened = _validation(
            service,
            dataset_id,
            validation="purged_walk_forward",
            n_folds=3,
            embargo=6,
        )
        assert bare["embargoed_bars"] == 0
        assert widened["embargoed_bars"] > 0
        assert widened["purged_bars"] == bare["purged_bars"]
        for before, after in zip(bare["folds"], widened["folds"], strict=True):
            assert before["train_bars"] - after["train_bars"] == 6
            # Widening the gap must not move the window being judged.
            assert before["test_bars"] == after["test_bars"]


class TestKFold:
    def test_test_windows_together_cover_every_bar(self, service: MLService) -> None:
        dataset = _build(service)
        validation = _validation(
            service, dataset["dataset_id"], validation="purged_kfold", n_folds=4
        )
        assert sum(fold["test_bars"] for fold in validation["folds"]) == dataset["rows"]

    def test_kfold_purges_the_right_edge_and_both_sides_of_middle_folds(
        self, service: MLService
    ) -> None:
        validation = _validation(
            service,
            _build(service)["dataset_id"],
            validation="purged_kfold",
            n_folds=3,
        )
        assert validation["folds"][0]["purged_bars"] == 5
        assert validation["folds"][1]["purged_bars"] == 10

    def test_the_two_schemes_do_not_cut_the_same_windows(
        self, service: MLService
    ) -> None:
        dataset_id = _build(service)["dataset_id"]
        walk = _validation(
            service, dataset_id, validation="purged_walk_forward", n_folds=3
        )
        kfold = _validation(service, dataset_id, validation="purged_kfold", n_folds=3)
        assert [fold["test_bars"] for fold in walk["folds"]] != [
            fold["test_bars"] for fold in kfold["folds"]
        ]


class TestTheAuditIsVerifiedNotAssumed:
    """``leakage_audit_clean`` is a claim; these tests re-derive it."""

    @pytest.mark.parametrize("scheme", validation_scheme_names())
    @pytest.mark.parametrize("embargo", [0, 4])
    def test_rebuilt_splits_audit_clean(
        self, service: MLService, scheme: str, embargo: int
    ) -> None:
        dataset = _build(service)
        n_folds = 3
        validation = _validation(
            service,
            dataset["dataset_id"],
            validation=scheme,
            n_folds=n_folds,
            embargo=embargo,
        )
        assert validation["leakage_audit_clean"] is True
        rebuilt = build_validation_splits(
            scheme,
            n_bars=dataset["rows"],
            horizon=validation["label_horizon"],
            n_folds=n_folds,
            embargo=embargo,
            min_train_bars=1,
        )
        assert (
            audit_validation_splits(rebuilt, horizon=validation["label_horizon"]) == ()
        )

    @pytest.mark.parametrize("scheme", validation_scheme_names())
    def test_the_reported_fold_geometry_is_the_real_geometry(
        self, service: MLService, scheme: str
    ) -> None:
        dataset = _build(service)
        n_folds = 3
        validation = _validation(
            service,
            dataset["dataset_id"],
            validation=scheme,
            n_folds=n_folds,
            embargo=2,
        )
        rebuilt = build_validation_splits(
            scheme,
            n_bars=dataset["rows"],
            horizon=validation["label_horizon"],
            n_folds=n_folds,
            embargo=2,
            min_train_bars=1,
        )
        assert [
            (fold["train_bars"], fold["test_bars"], fold["purged_bars"])
            for fold in validation["folds"]
        ] == [
            (split.train_bars, split.test_bars, split.purged_bars) for split in rebuilt
        ]


class TestFoldScores:
    def test_every_fold_reports_a_score(self, service: MLService) -> None:
        validation = _validation(
            service,
            _build(service)["dataset_id"],
            validation="purged_walk_forward",
            n_folds=3,
        )
        for fold in validation["folds"]:
            for key in _METRIC_KEYS:
                assert key in fold
            assert fold["accuracy"] is not None
            assert 0.0 <= fold["accuracy"] <= 1.0

    def test_the_aggregate_is_the_mean_of_the_folds(self, service: MLService) -> None:
        validation = _validation(
            service,
            _build(service)["dataset_id"],
            validation="purged_walk_forward",
            n_folds=3,
        )
        for key in _METRIC_KEYS:
            observed = [fold[key] for fold in validation["folds"] if fold[key]]
            expected = sum(observed) / len(observed)
            assert validation["aggregate"][key] == pytest.approx(expected)

    def test_scores_are_finite_or_explicitly_none(self, service: MLService) -> None:
        validation = _validation(
            service,
            _build(service)["dataset_id"],
            validation="purged_kfold",
            n_folds=4,
        )
        for fold in validation["folds"]:
            for key in _METRIC_KEYS:
                value = fold[key]
                assert value is None or math.isfinite(value)


class TestTheArtifactIsUnaffected:
    def test_the_stored_model_is_still_fit_on_every_bar(
        self, service: MLService
    ) -> None:
        """Folds assess the procedure; the saved model fits all dataset rows."""
        dataset = _build(service)
        model = _train(
            service,
            dataset["dataset_id"],
            validation="purged_walk_forward",
            n_folds=3,
        )
        assert model["validation"]["artifact_trained_on"] == "all_bars"
        assert model["n_train_samples"] == dataset["rows"]

    def test_asking_for_validation_does_not_change_the_model_identity(
        self, service: MLService
    ) -> None:
        dataset_id = _build(service)["dataset_id"]
        assert (
            _train(service, dataset_id)["model_id"]
            == _train(service, dataset_id, validation="purged_kfold", n_folds=3)[
                "model_id"
            ]
        )


class TestRejections:
    def test_an_unknown_scheme_is_rejected_and_names_the_alternatives(
        self, service: MLService
    ) -> None:
        with pytest.raises(ValueError, match="unknown validation scheme: rolling"):
            _train(
                service,
                _build(service)["dataset_id"],
                validation="rolling",
                n_folds=3,
            )
        with pytest.raises(ValueError) as caught:
            _train(
                service,
                _build(service)["dataset_id"],
                validation="rolling",
                n_folds=3,
            )
        for name in validation_scheme_names():
            assert name in str(caught.value)

    def test_a_fold_count_of_zero_is_rejected(self, service: MLService) -> None:
        with pytest.raises(ValueError, match="n_folds must be at least 1"):
            _train(
                service,
                _build(service)["dataset_id"],
                validation="purged_walk_forward",
                n_folds=0,
            )

    def test_a_negative_embargo_is_rejected(self, service: MLService) -> None:
        with pytest.raises(ValueError, match="embargo must not be negative"):
            _train(
                service,
                _build(service)["dataset_id"],
                validation="purged_kfold",
                embargo=-1,
            )

    def test_a_training_floor_the_data_cannot_meet_is_rejected(
        self, service: MLService
    ) -> None:
        with pytest.raises(ValueError, match="training bars"):
            _train(
                service,
                _build(service)["dataset_id"],
                validation="purged_walk_forward",
                n_folds=3,
                min_train_bars=10_000,
            )

    def test_a_rejected_validation_leaves_no_artifact(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(service)["dataset_id"]
        with pytest.raises(ValueError):
            _train(service, dataset_id, validation="not_a_scheme")
        assert list(workspace.models.iterdir()) == []

    def test_rejections_do_not_depend_on_the_model_kind(
        self, service: MLService
    ) -> None:
        dataset_id = _build(service)["dataset_id"]
        with pytest.raises(ValueError, match="unknown validation scheme"):
            _train(
                service,
                dataset_id,
                model_kind="lightgbm_direction",
                validation="rolling",
            )

    def test_an_unusable_request_is_refused_before_anything_is_fitted(
        self, service: MLService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tradingdev.app import ml_training as module

        fitted: list[str] = []

        def spy(kind: str, **kwargs: Any) -> Any:
            fitted.append(kind)
            return create_direction_model(kind, **kwargs)

        monkeypatch.setattr(module, "create_direction_model", spy)
        with pytest.raises(ValueError, match="unknown validation scheme"):
            _train(service, _build(service)["dataset_id"], validation="rolling")
        assert fitted == []
