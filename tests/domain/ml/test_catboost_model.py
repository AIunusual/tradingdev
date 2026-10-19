"""Tests for the CatBoost direction model."""

import pickle
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.ml.features.features import FeatureEngineer
from tradingdev.domain.ml.models.catboost_model import CatBoostDirectionModel
from tradingdev.domain.ml.registry.model_registry import (
    compute_model_id,
    content_sha256,
)
from tradingdev.domain.ml.schemas import CatBoostModelConfig


def test_repeat_training_has_identical_artifact_bytes_and_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from catboost import CatBoostClassifier

    frame = pd.DataFrame(
        {"feature": np.arange(60, dtype=float), "target": [-1, 0, 1] * 20}
    )
    original_fit = CatBoostClassifier.fit
    original_probabilities: list[np.ndarray[Any, Any]] = []

    def fit_with_distinct_metadata(
        estimator: CatBoostClassifier, *args: Any, **kwargs: Any
    ) -> CatBoostClassifier:
        original_fit(estimator, *args, **kwargs)
        original_probabilities.append(estimator.predict_proba(frame[["feature"]]))
        metadata = estimator.get_metadata()
        metadata["model_guid"] = f"random-{len(original_probabilities)}"
        metadata["train_finish_time"] = f"2024-01-0{len(original_probabilities)}"
        return estimator

    monkeypatch.setattr(CatBoostClassifier, "fit", fit_with_distinct_metadata)
    config = CatBoostModelConfig(n_estimators=5, max_depth=2, n_jobs=1)
    models = [CatBoostDirectionModel(config) for _ in range(2)]
    blobs: list[bytes] = []
    for index, model in enumerate(models):
        model.train(frame)
        blob = pickle.dumps(model)
        blobs.append(blob)
        restored = pickle.loads(blob)
        assert isinstance(restored, CatBoostDirectionModel)
        np.testing.assert_array_equal(
            model.predict_proba(frame).to_numpy(), original_probabilities[index]
        )
        pd.testing.assert_frame_equal(
            restored.predict_proba(frame), model.predict_proba(frame)
        )
    assert blobs[0] == blobs[1]
    identifiers = [
        compute_model_id(
            model_name="catboost_direction",
            parameters=config.model_dump(),
            feature_names=("feature",),
            content_hash=content_sha256(blob),
        )
        for blob in blobs
    ]
    assert identifiers[0] == identifiers[1]

    changed = CatBoostDirectionModel(config)
    changed.train(frame.assign(target=[1, 0, -1] * 20))
    assert pickle.dumps(changed) != blobs[0]


class TestCatBoostDirectionModel:
    def _make_feature_df(self, df: pd.DataFrame) -> pd.DataFrame:
        fe = FeatureEngineer(lookback=6)
        return fe.transform(df, include_target=True)

    def test_train_and_predict_shape(self, large_ohlcv_df: pd.DataFrame) -> None:
        feat = self._make_feature_df(large_ohlcv_df)
        config = CatBoostModelConfig(n_estimators=10, max_depth=3)
        model = CatBoostDirectionModel(config=config)
        model.train(feat)
        preds = model.predict(feat)
        assert len(preds) == len(feat)

    def test_predict_before_train_raises(self, large_ohlcv_df: pd.DataFrame) -> None:
        feat = self._make_feature_df(large_ohlcv_df)
        model = CatBoostDirectionModel(config=CatBoostModelConfig())
        with pytest.raises(RuntimeError, match="not been trained"):
            model.predict(feat)

    def test_predict_proba_columns(self, large_ohlcv_df: pd.DataFrame) -> None:
        feat = self._make_feature_df(large_ohlcv_df)
        config = CatBoostModelConfig(n_estimators=10, max_depth=3)
        model = CatBoostDirectionModel(config=config)
        model.train(feat)
        proba = model.predict_proba(feat)
        assert isinstance(proba, pd.DataFrame)
        assert len(proba) == len(feat)
        row_sums = proba.sum(axis=1)
        assert (abs(row_sums - 1.0) < 0.01).all()

    def test_early_stopping_with_eval(self, large_ohlcv_df: pd.DataFrame) -> None:
        feat = self._make_feature_df(large_ohlcv_df)
        split = int(len(feat) * 0.8)
        train_feat = feat.iloc[:split]
        val_feat = feat.iloc[split:]

        config = CatBoostModelConfig(
            n_estimators=50,
            max_depth=3,
            early_stopping_rounds=5,
        )
        model = CatBoostDirectionModel(config=config)
        model.train(train_feat, eval_df=val_feat)
        preds = model.predict(val_feat)
        assert len(preds) == len(val_feat)
