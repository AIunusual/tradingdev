"""Bundled fit routines share objective direction and missing-value semantics."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.strategies.bundled.glft_ml_strategy.config import (
    GLFTMLStrategyConfig,
)
from tradingdev.domain.strategies.bundled.glft_ml_strategy.strategy import (
    GLFTMLStrategy,
)
from tradingdev.domain.strategies.bundled.glft_strategy.config import GLFTStrategyConfig
from tradingdev.domain.strategies.bundled.glft_strategy.strategy import GLFTStrategy
from tradingdev.domain.strategies.bundled.quantile_strategy import strategy as quantile
from tradingdev.domain.strategies.bundled.quantile_strategy.config import (
    QuantileStrategyConfig,
)

if TYPE_CHECKING:
    from tradingdev.domain.backtest.base_engine import BaseBacktestEngine


def _engine(first: float | None) -> BaseBacktestEngine:
    scores = iter([first, 0.1])
    return cast(
        "BaseBacktestEngine",
        SimpleNamespace(
            run=lambda frame: SimpleNamespace(
                metrics={"max_drawdown_amount": next(scores), "daily_pnl_mean": 0.0}
            )
        ),
    )


@pytest.mark.parametrize("first", [0.5, None])
def test_glft_fit_selects_minimum_available_amount_drawdown(
    first: float | None,
    sample_ohlcv_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tradingdev.domain.strategies.bundled.glft_strategy import strategy as module

    monkeypatch.setattr(module, "estimate_n_jobs", lambda *args, **kwargs: 1)
    strategy = GLFTStrategy(
        config=GLFTStrategyConfig(
            gamma_candidates=[0.0, 200.0],
            kappa_candidates=[1000.0],
            ema_window_candidates=[10],
            max_holding_bars_candidates=[30],
            vol_window_candidates=[10],
            min_entry_edge_candidates=[0.0012],
            profit_target_ratio_candidates=[1.0],
            target_metric="max_drawdown_amount",
        ),
        backtest_engine=_engine(first),
    )
    strategy.fit(sample_ohlcv_df)
    assert strategy.get_parameters()["best_gamma"] == 200.0


@pytest.mark.parametrize("first", [0.5, None])
def test_glft_ml_search_selects_minimum_available_amount_drawdown(
    first: float | None,
    sample_ohlcv_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = GLFTMLStrategy(
        config=GLFTMLStrategyConfig(
            gamma_candidates=[0.0, 200.0],
            kappa_candidates=[1000.0],
            ema_window_candidates=[10],
            max_holding_bars_candidates=[30],
            min_entry_edge_candidates=[0.0012],
            profit_target_ratio_candidates=[1.0],
            confidence_threshold_candidates=[0.55],
            vol_type="realized",
            target_metric="max_drawdown_amount",
        ),
        backtest_engine=_engine(first),
    )
    monkeypatch.setattr(
        strategy._ml_model,
        "predict_proba",
        lambda frame: pd.DataFrame(
            {0: np.full(len(frame), 0.1), 1: np.full(len(frame), 0.9)},
            index=frame.index,
        ),
    )
    strategy._grid_search_glft(sample_ohlcv_df)
    assert strategy.get_parameters()["best_gamma"] == 200.0


@pytest.mark.parametrize("first", [0.5, None])
def test_quantile_search_selects_minimum_available_amount_drawdown(
    first: float | None,
    sample_ohlcv_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = SimpleNamespace(
        transform=lambda frame, **kwargs: frame.assign(target_regime=0),
        get_feature_names=lambda: ["close"],
    )
    monkeypatch.setattr(quantile, "QuantileFeatureEngineer", lambda **kwargs: features)
    monkeypatch.setattr(
        quantile, "_train_regime_classifier", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(quantile, "estimate_n_jobs", lambda *args, **kwargs: 1)

    def evaluate(*args: Any) -> tuple[tuple[Any, ...], dict[str, float | None]]:
        params = args[4]
        return params, {
            "max_drawdown_amount": first if params[1] == 0.001 else 0.1,
            "daily_pnl_mean": 0.0,
        }

    monkeypatch.setattr(quantile, "_evaluate_regime_combo", evaluate)
    strategy = quantile.QuantileStrategy(
        config=QuantileStrategyConfig(
            horizon_candidates=[30],
            min_entry_edge_candidates=[0.001, 0.002],
            dynamic_sizing=False,
            target_metric="max_drawdown_amount",
        ),
        backtest_engine=_engine(first),
    )
    strategy.fit(sample_ohlcv_df)
    assert strategy.get_parameters()["best_min_confidence"] == 0.002
