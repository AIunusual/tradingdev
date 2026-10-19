"""Tests for the enumerable model zoo registry."""

import pandas as pd
import pytest

from tradingdev.domain.ml.features.features import FeatureEngineer
from tradingdev.domain.ml.models import (
    CatBoostDirectionModel,
    LightGBMDirectionModel,
    XGBoostDirectionModel,
    create_direction_model,
    get_model_kind,
    list_model_kinds,
)
from tradingdev.domain.ml.models.registry import ModelKind, register


def _features(df: pd.DataFrame) -> pd.DataFrame:
    return FeatureEngineer(lookback=6).transform(df, include_target=True)


def test_zoo_lists_the_three_boosters() -> None:
    kinds = list_model_kinds()
    assert kinds == ["catboost_direction", "lightgbm_direction", "xgboost_direction"]
    for name in kinds:
        kind = get_model_kind(name)
        assert isinstance(kind, ModelKind)
        assert kind.config_cls is not None
        assert kind.model_cls is not None


def test_create_direction_model_builds_the_right_class() -> None:
    lgbm = create_direction_model("lightgbm_direction", n_estimators=10, max_depth=2)
    assert isinstance(lgbm, LightGBMDirectionModel)

    cb = create_direction_model("catboost_direction", n_estimators=10, max_depth=2)
    assert isinstance(cb, CatBoostDirectionModel)

    xgb = create_direction_model("xgboost_direction", n_estimators=10, max_depth=2)
    assert isinstance(xgb, XGBoostDirectionModel)


def test_unknown_kind_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_model_kind("does_not_exist")
    with pytest.raises(KeyError):
        create_direction_model("does_not_exist")


def test_register_rejects_duplicates() -> None:
    duplicate = ModelKind(
        name="lightgbm_direction",
        description="collision",
        config_cls=LightGBMDirectionModel,
        model_cls=LightGBMDirectionModel,
        framework="lightgbm",
    )
    with pytest.raises(ValueError, match="duplicate model kind"):
        register(duplicate)


def test_zoo_model_trains_and_predicts(large_ohlcv_df: pd.DataFrame) -> None:
    feat = _features(large_ohlcv_df)
    model = create_direction_model("lightgbm_direction", n_estimators=10, max_depth=3)
    model.train(feat)
    preds = model.predict(feat)
    assert isinstance(preds, pd.Series)
    assert len(preds) == len(feat)
