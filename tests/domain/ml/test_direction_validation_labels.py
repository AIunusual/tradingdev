"""Validation answers must never be replaced with known training classes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.ml.models.registry import create_direction_model
from tradingdev.domain.ml.schemas import XGBoostModelConfig
from tradingdev.domain.strategies.bundled.xgboost_strategy.config import (
    XGBoostStrategyConfig,
)
from tradingdev.domain.strategies.bundled.xgboost_strategy.strategy import (
    XGBoostStrategy,
)


@pytest.mark.parametrize(
    "kind", ["xgboost_direction", "lightgbm_direction", "catboost_direction"]
)
@pytest.mark.parametrize("targets", [[0, 0], [-1, 0, 1]])
def test_unseen_validation_classes_are_rejected_without_replacing_fitted_state(
    kind: str, targets: list[int]
) -> None:
    train = pd.DataFrame(
        {"feature": np.tile([0.0, 1.0], 30), "target": np.tile([-1, 1], 30)}
    )
    known = train.iloc[:4].copy()
    model = create_direction_model(
        kind, n_estimators=3, max_depth=2, n_jobs=1, early_stopping_rounds=2
    )
    model.train(train, eval_df=known)
    prediction = model.predict(known)
    probabilities = model.predict_proba(known)
    changed = train.rename(columns={"feature": "different_feature"})
    if -1 in targets:
        changed["target"] = train["target"].replace({-1: 0})
    evaluation = pd.DataFrame(
        {"different_feature": np.arange(len(targets), dtype=float), "target": targets}
    )
    original = evaluation.copy(deep=True)
    with pytest.raises(ValueError, match="Validation targets contain unseen classes"):
        model.train(changed, eval_df=evaluation)
    pd.testing.assert_frame_equal(evaluation, original)
    pd.testing.assert_series_equal(model.predict(known), prediction)
    pd.testing.assert_frame_equal(model.predict_proba(known), probabilities)


@pytest.mark.parametrize("kind", ["lightgbm_direction", "catboost_direction"])
@pytest.mark.parametrize("empty_part", ["train", "validation"])
def test_empty_inputs_preserve_fitted_direction_model(
    kind: str, empty_part: str
) -> None:
    train = pd.DataFrame(
        {"feature": np.tile([0.0, 1.0], 30), "target": np.tile([-1, 1], 30)}
    )
    model = create_direction_model(kind, n_estimators=3, max_depth=2, n_jobs=1)
    model.train(train)
    before = model.predict_proba(train)
    with pytest.raises(ValueError, match="at least one labelled row"):
        model.train(
            train.iloc[:0] if empty_part == "train" else train,
            eval_df=train.iloc[:0] if empty_part == "validation" else train,
        )
    pd.testing.assert_frame_equal(model.predict_proba(train), before)


@pytest.mark.parametrize(
    ("kind", "estimator"),
    [
        ("xgboost_direction", "xgboost.XGBClassifier"),
        ("lightgbm_direction", "lightgbm.LGBMClassifier"),
        ("catboost_direction", "catboost.CatBoostClassifier"),
    ],
)
def test_failed_estimator_fit_preserves_complete_previous_state(
    kind: str, estimator: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = pd.DataFrame(
        {"feature": np.tile([0.0, 1.0], 30), "target": np.tile([-1, 1], 30)}
    )
    model = create_direction_model(kind, n_estimators=3, max_depth=2, n_jobs=1)
    model.train(train)
    before = model.predict_proba(train)
    prediction = model.predict(train)
    changed = train.rename(columns={"feature": "different_feature"})
    changed["target"] = train["target"].replace({-1: 0})

    def reject(*args: object, **kwargs: object) -> None:
        raise ValueError("estimator fit failed")

    monkeypatch.setattr(f"{estimator}.fit", reject)
    with pytest.raises(ValueError, match="estimator fit failed"):
        model.train(changed)
    pd.testing.assert_frame_equal(model.predict_proba(train), before)
    pd.testing.assert_series_equal(model.predict(train), prediction)


def test_real_lookback_search_rejects_unseen_validation_answers() -> None:
    close = np.concatenate(
        (
            np.tile([100.0, 101.0], 240),
            np.tile([100.0, 100.0, 101.0, 101.0, 99.0, 99.0], 20),
        )
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01", periods=len(close), freq="h", tz="UTC"
            ),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.arange(len(close)) + 100.0,
        }
    )
    strategy = XGBoostStrategy(
        XGBoostStrategyConfig(
            lookback_candidates=[6],
            validation_ratio=0.2,
            model=XGBoostModelConfig(n_estimators=3, max_depth=2, n_jobs=1),
        )
    )
    with pytest.raises(ValueError, match="Validation targets contain unseen classes"):
        strategy.fit(frame)
    assert strategy.get_parameters()["best_lookback"] is None
    with pytest.raises(RuntimeError, match="not fitted"):
        strategy.generate_signals(frame)
