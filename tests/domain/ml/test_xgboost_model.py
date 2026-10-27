"""Tests for the XGBoost direction model."""

import pandas as pd
import pytest

from tradingdev.domain.ml.features.features import FeatureEngineer
from tradingdev.domain.ml.models.xgboost_model import XGBoostDirectionModel
from tradingdev.domain.ml.schemas import XGBoostModelConfig


class TestXGBoostDirectionModel:
    def _make_feature_df(self, df: pd.DataFrame) -> pd.DataFrame:
        fe = FeatureEngineer(lookback=6)
        return fe.transform(df, include_target=True)

    def test_train_and_predict_shape(self, large_ohlcv_df: pd.DataFrame) -> None:
        feat = self._make_feature_df(large_ohlcv_df)
        config = XGBoostModelConfig(n_estimators=10, max_depth=3)
        model = XGBoostDirectionModel(config=config)
        model.train(feat)
        preds = model.predict(feat)
        assert len(preds) == len(feat)

    def test_predict_before_train_raises(self, large_ohlcv_df: pd.DataFrame) -> None:
        feat = self._make_feature_df(large_ohlcv_df)
        model = XGBoostDirectionModel(config=XGBoostModelConfig())
        with pytest.raises(RuntimeError, match="not been trained"):
            model.predict(feat)

    def test_predict_proba_columns(self, large_ohlcv_df: pd.DataFrame) -> None:
        feat = self._make_feature_df(large_ohlcv_df)
        config = XGBoostModelConfig(n_estimators=10, max_depth=3)
        model = XGBoostDirectionModel(config=config)
        model.train(feat)
        proba = model.predict_proba(feat)
        assert isinstance(proba, pd.DataFrame)
        assert len(proba) == len(feat)
        # Probabilities should sum to ~1
        row_sums = proba.sum(axis=1)
        assert (abs(row_sums - 1.0) < 0.01).all()

    def test_early_stopping_with_eval(self, large_ohlcv_df: pd.DataFrame) -> None:
        """Training with eval_df for early stopping should not error."""
        feat = self._make_feature_df(large_ohlcv_df)
        split = int(len(feat) * 0.8)
        train_feat = feat.iloc[:split]
        val_feat = feat.iloc[split:]

        config = XGBoostModelConfig(
            n_estimators=50,
            max_depth=3,
            early_stopping_rounds=5,
        )
        model = XGBoostDirectionModel(config=config)
        model.train(train_feat, eval_df=val_feat)
        preds = model.predict(val_feat)
        assert len(preds) == len(val_feat)

    @pytest.mark.parametrize("empty_input", ["training", "validation"])
    @pytest.mark.parametrize("already_trained", [False, True])
    def test_empty_training_inputs_preserve_model_state(
        self,
        large_ohlcv_df: pd.DataFrame,
        empty_input: str,
        already_trained: bool,
    ) -> None:
        feat = self._make_feature_df(large_ohlcv_df.iloc[:200])
        model = XGBoostDirectionModel(XGBoostModelConfig(n_estimators=2, n_jobs=1))
        before_predictions: pd.DataFrame | None = None
        if already_trained:
            model.train(feat)
            before_predictions = model.predict_proba(feat)
        before_model = model._model
        before_features = model._feature_names.copy()
        changed = feat.rename(columns={"log_return": "replacement_feature"})
        train = changed.iloc[:0] if empty_input == "training" else changed
        validation = changed.iloc[:0] if empty_input == "validation" else None

        with pytest.raises(
            ValueError, match=f"{empty_input.title()} data.*labelled row"
        ):
            model.train(train, eval_df=validation)

        assert model._model is before_model
        assert model._feature_names == before_features
        if before_predictions is not None:
            pd.testing.assert_frame_equal(model.predict_proba(feat), before_predictions)
        else:
            with pytest.raises(RuntimeError, match="not been trained"):
                model.predict_proba(feat)

    def test_no_mature_labels_cannot_create_a_trained_model(
        self, large_ohlcv_df: pd.DataFrame
    ) -> None:
        frame = large_ohlcv_df.iloc[:300]
        features = FeatureEngineer(lookback=5, target_horizon=300).transform(frame)
        assert features.empty
        model = XGBoostDirectionModel(XGBoostModelConfig(n_estimators=2, n_jobs=1))

        with pytest.raises(ValueError, match="Training data.*labelled row"):
            model.train(features)

        inference = FeatureEngineer(lookback=5).transform(frame, include_target=False)
        with pytest.raises(RuntimeError, match="not been trained"):
            model.predict_proba(inference)
