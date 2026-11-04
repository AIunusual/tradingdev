"""Threshold selection uses finite net PnL in both execution modes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.ml.threshold_optimizer import ThresholdOptimizer
from tradingdev.domain.strategies.bundled.safety_volume_strategy.config import (
    SafetyVolumeStrategyConfig,
)
from tradingdev.domain.strategies.bundled.safety_volume_strategy.strategy import (
    SafetyVolumeStrategy,
)

if TYPE_CHECKING:
    from tradingdev.domain.backtest.base_engine import BaseBacktestEngine
    from tradingdev.domain.ml.features.features import FeatureEngineer
    from tradingdev.domain.ml.features.risk_features import RiskFeatureEngineer
    from tradingdev.domain.ml.models.xgboost_model import XGBoostDirectionModel


@pytest.mark.parametrize(
    ("scores", "expected"),
    [([None, float("inf"), -2.0, 0.0], 0.8), ([None, -1.0, None, -2.0], 0.9)],
)
def test_threshold_search_uses_net_pnl_and_handles_unavailable_returns(
    scores: list[float | None], expected: float
) -> None:
    values = iter(scores)
    frame = pd.DataFrame({"close": [100.0, 101.0]})
    engine = cast(
        "BaseBacktestEngine",
        SimpleNamespace(
            run=lambda frame: SimpleNamespace(
                metrics={
                    "total_return": None,
                    "total_pnl": next(values),
                    "total_trades": 1,
                }
            )
        ),
    )
    model = cast(
        "XGBoostDirectionModel",
        SimpleNamespace(predict_proba=lambda frame: pd.DataFrame({1: [0.95, 0.95]})),
    )
    features = cast(
        "FeatureEngineer", SimpleNamespace(transform=lambda frame, **kwargs: frame)
    )
    actual = ThresholdOptimizer(engine).search(
        frame,
        model,
        candidates=[0.5, 0.6, 0.7, 0.8],
        default_threshold=0.9,
        feature_engineer=features,
    )
    assert actual == expected


def test_safety_volume_threshold_selects_highest_pnl_without_capital(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = iter([None, -4.0, -1.0])
    frame = pd.DataFrame({"close": [100.0, 101.0]})
    engine = cast(
        "BaseBacktestEngine",
        SimpleNamespace(
            run=lambda frame: SimpleNamespace(
                metrics={
                    "total_return": None,
                    "total_pnl": next(values),
                    "total_volume": 100,
                }
            )
        ),
    )
    model = cast(
        "XGBoostDirectionModel",
        SimpleNamespace(predict_proba=lambda frame: pd.DataFrame({1: [0.95, 0.95]})),
    )
    features = cast(
        "RiskFeatureEngineer", SimpleNamespace(transform=lambda frame, **kwargs: frame)
    )
    strategy = SafetyVolumeStrategy(
        config=SafetyVolumeStrategyConfig(risk_threshold_candidates=[0.5, 0.6, 0.7]),
        backtest_engine=engine,
    )
    monkeypatch.setattr(
        strategy, "_compute_sma_directions", lambda frame: np.ones(len(frame))
    )
    assert strategy._search_threshold(frame, frame, model, features, None) == 0.7
