"""Tests for the CatBoost direction model."""

import pandas as pd
import pytest

from tradingdev.domain.ml.features.features import FeatureEngineer
from tradingdev.domain.ml.models.catboost_model import CatBoostDirectionModel
from tradingdev.domain.ml.schemas import CatBoostModelConfig


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
        """Training with eval_df for early stopping should not error."""
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
