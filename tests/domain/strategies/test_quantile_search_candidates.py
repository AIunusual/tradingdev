"""Default and explicit sizing candidates must reach a real trained classifier."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tradingdev.domain.backtest.base_engine import BaseBacktestEngine
from tradingdev.domain.backtest.result import BacktestResult
from tradingdev.domain.strategies.bundled.quantile_strategy import strategy as quantile
from tradingdev.domain.strategies.bundled.quantile_strategy.config import (
    QuantileStrategyConfig,
)

if TYPE_CHECKING:
    import pandas as pd


class _MetricEngine(BaseBacktestEngine):
    """Isolate candidate selection from transaction economics."""

    def __init__(self, *, eligible: bool = True) -> None:
        super().__init__()
        self.calls = 0
        self.eligible = eligible

    def run(self, df: pd.DataFrame) -> BacktestResult:
        self.calls += 1
        assert len(df) and {"signal", "size_weight"}.issubset(df.columns)
        return BacktestResult(
            metrics={
                "total_volume": 1.0 if self.eligible else None,
                "daily_pnl_mean": 0.0,
            },
            equity_curve=np.zeros(len(df)),
        )


@pytest.mark.parametrize("dynamic", [True, False])
@pytest.mark.parametrize("candidates", [[], [0.008, 0.009]])
def test_sizing_search_uses_candidates_or_single_configured_value(
    large_ohlcv_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    dynamic: bool,
    candidates: list[float],
) -> None:
    monkeypatch.setattr(quantile, "estimate_n_jobs", lambda *args, **kwargs: 1)
    config = QuantileStrategyConfig(
        n_estimators=3,
        max_depth=2,
        horizon=2,
        dynamic_sizing=dynamic,
        edge_for_full_size_candidates=candidates,
    )
    engine = _MetricEngine()
    strategy = quantile.QuantileStrategy(config, engine)
    strategy.fit(large_ohlcv_df)
    expected = candidates if dynamic and candidates else [config.edge_for_full_size]
    assert engine.calls == len(expected)
    assert strategy.get_parameters()["best_edge_for_full_size"] == expected[0]
    assert "signal" in strategy.generate_signals(large_ohlcv_df)


def test_executed_but_ineligible_candidates_still_fail(
    large_ohlcv_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(quantile, "estimate_n_jobs", lambda *args, **kwargs: 1)
    engine = _MetricEngine(eligible=False)
    strategy = quantile.QuantileStrategy(
        QuantileStrategyConfig(n_estimators=3, max_depth=2, horizon=2), engine
    )
    with pytest.raises(ValueError, match="finite eligible objective"):
        strategy.fit(large_ohlcv_df)
    assert engine.calls == 1, "The failure must follow an evaluated candidate"
    with pytest.raises(RuntimeError, match="no trained regime model"):
        strategy.generate_signals(large_ohlcv_df)
