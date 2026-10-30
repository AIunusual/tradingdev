"""Configuration models for the bundled XGBoost strategy."""

from __future__ import annotations

from pydantic import BaseModel

from tradingdev.domain.ml.rolling_gen import RollingGenConfig
from tradingdev.domain.ml.schemas import XGBoostModelConfig


class XGBoostStrategyConfig(BaseModel):
    """XGBoost direction prediction strategy configuration."""

    model: XGBoostModelConfig = XGBoostModelConfig()
    lookback_candidates: list[int] = [12, 24, 48, 96, 168]
    retrain_interval: int = 24
    validation_ratio: float = 0.2
    signal_threshold: float = 0.55
    signal_threshold_candidates: list[float] | None = None
    target_horizon: int = 1
    min_bars_between_trades: int = 1
    monthly_volume_target: float | None = None
    rolling_gen: RollingGenConfig = RollingGenConfig()
    """Optional declarative rolling geometry; disabled unless enabled is set."""
