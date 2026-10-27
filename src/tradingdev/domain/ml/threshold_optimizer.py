"""Threshold optimisation for signal probability strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tradingdev.domain.ml.features.features import FeatureEngineer
from tradingdev.domain.optimization.grid_search import finite_metric_value
from tradingdev.shared.utils.logger import setup_logger

if TYPE_CHECKING:
    from tradingdev.domain.backtest.base_engine import (
        BaseBacktestEngine,
    )
    from tradingdev.domain.ml.models.xgboost_model import (
        XGBoostDirectionModel,
    )

logger = setup_logger(__name__)


class ThresholdOptimizer:
    """Find the lowest signal threshold with non-negative P&L."""

    def __init__(
        self,
        engine: BaseBacktestEngine,
        min_bars_between_trades: int = 1,
    ) -> None:
        self._engine = engine
        self._cooldown = min_bars_between_trades

    def search(
        self,
        val_df: pd.DataFrame,
        model: XGBoostDirectionModel,
        candidates: list[float],
        default_threshold: float,
        feature_engineer: FeatureEngineer | None = None,
    ) -> float:
        """Search for the lowest threshold where P&L >= 0.

        Args:
            val_df: Validation OHLCV DataFrame.
            model: Trained model with ``predict_proba``.
            candidates: Sorted threshold candidates.
            default_threshold: Fallback if no candidate works.

        Returns:
            Selected threshold value.
        """
        fe = feature_engineer if feature_engineer is not None else FeatureEngineer()
        source = val_df.reset_index(drop=True)
        feat_val = fe.transform(source, include_target=False)
        if len(feat_val) == 0:
            return default_threshold

        proba = model.predict_proba(feat_val).reindex(source.index)

        for threshold in sorted(candidates):
            signals = apply_threshold(proba, threshold, self._cooldown)
            signal_df = val_df.copy()
            signal_df["signal"] = signals.values

            result = self._engine.run(signal_df)
            total_pnl = finite_metric_value(result.metrics.get("total_pnl"))
            total_trades = result.metrics.get("total_trades", 0)

            logger.info(
                "Threshold %.2f: net_pnl=%s, trades=%s",
                threshold,
                total_pnl,
                total_trades,
            )

            if total_pnl is not None and total_pnl >= 0:
                logger.info(
                    "Selected threshold=%.2f (lowest with P&L>=0, trades=%d)",
                    threshold,
                    total_trades,
                )
                return threshold

        logger.warning(
            "No threshold achieved P&L>=0; using default=%.2f",
            default_threshold,
        )
        return default_threshold


def apply_threshold(
    proba: pd.DataFrame,
    threshold: float,
    cooldown: int = 1,
) -> pd.Series[int]:
    """Convert probability DataFrame to signals."""
    p_long = proba[1] if 1 in proba.columns else pd.Series(0.0, index=proba.index)
    p_short = proba[-1] if -1 in proba.columns else pd.Series(0.0, index=proba.index)

    signals = pd.Series(0, index=proba.index, dtype=int)
    ready = proba.notna().all(axis=1)
    long_mask = ready & (p_long >= threshold) & (p_long >= p_short)
    short_mask = ready & (p_short >= threshold) & (~long_mask)
    signals = signals.where(~long_mask, 1)
    signals = signals.where(~short_mask, -1)

    if cooldown > 1:
        last_bar = -cooldown
        for i in range(len(signals)):
            if signals.iloc[i] != 0:
                if i - last_bar < cooldown:
                    signals.iloc[i] = 0
                else:
                    last_bar = i

    return signals
