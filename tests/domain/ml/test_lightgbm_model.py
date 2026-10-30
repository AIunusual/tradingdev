"""Tests for the LightGBM direction model."""

import pandas as pd
import pytest

from tradingdev.domain.ml.features.features import FeatureEngineer
from tradingdev.domain.ml.models.lightgbm_model import LightGBMDirectionModel
from tradingdev.domain.ml.schemas import LightGBMModelConfig


class TestLightGBMDirectionModel:
    def _make_feature_df(self, df: pd.DataFrame) -> pd.DataFrame:
        fe = FeatureEngineer(lookback=6)
        return fe.transform(df, include_target=True)

    def test_predict_before_train_raises(self, large_ohlcv_df: pd.DataFrame) -> None:
        feat = self._make_feature_df(large_ohlcv_df)
        model = LightGBMDirectionModel(config=LightGBMModelConfig())
        with pytest.raises(RuntimeError, match="not been trained"):
            model.predict(feat)

    def test_prediction_contract(self, large_ohlcv_df: pd.DataFrame) -> None:
        feat = self._make_feature_df(large_ohlcv_df)
        feat.index = pd.Index(range(100, 100 + len(feat)), name="sample")
        config = LightGBMModelConfig(n_estimators=10, max_depth=3)
        model = LightGBMDirectionModel(config=config)
        model.train(feat)
        proba = model.predict_proba(feat)
        assert isinstance(proba, pd.DataFrame)
        assert len(proba) == len(feat)
        pd.testing.assert_index_equal(proba.index, feat.index)
        assert list(proba.columns) == sorted(feat["target"].unique())
        pd.testing.assert_series_equal(
            model.predict(feat), proba.idxmax(axis=1).astype(float)
        )
        row_sums = proba.sum(axis=1)
        assert (abs(row_sums - 1.0) < 0.01).all()

    def test_early_stopping_with_eval(self) -> None:
        train_feat = pd.DataFrame({"feature": [-1.0, 1.0] * 50, "target": [-1, 1] * 50})
        # Each tree can improve training, but worsens the inverted validation
        # labels. Validation must select the first round and stop after patience.
        val_feat = train_feat.assign(target=-train_feat["target"])

        config = LightGBMModelConfig(
            n_estimators=50,
            max_depth=3,
            early_stopping_rounds=5,
            n_jobs=1,
        )
        model = LightGBMDirectionModel(config=config)
        model.train(train_feat, eval_df=val_feat)
        classifier = model._model
        assert classifier is not None
        assert classifier.best_iteration_ == 1
        assert classifier.n_estimators_ == classifier.best_iteration_
        history = classifier.evals_result_["valid_0"]["binary_logloss"]
        assert len(history) == 1 + config.early_stopping_rounds
