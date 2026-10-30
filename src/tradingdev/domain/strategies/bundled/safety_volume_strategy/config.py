"""Configuration models for the bundled safety-volume strategy."""

from pydantic import BaseModel

from tradingdev.domain.ml.rolling_gen import RollingGenConfig
from tradingdev.domain.ml.schemas import XGBoostModelConfig
from tradingdev.domain.strategies.bundled.indicator_parameters import (
    MovingAveragePeriod,
)


class SafetyVolumeStrategyConfig(BaseModel):
    """Safety-first volume strategy configuration."""

    risk_model: XGBoostModelConfig = XGBoostModelConfig()
    risk_threshold: float = 0.5
    risk_threshold_candidates: list[float] | None = None
    target_holding_bars: int = 5
    max_acceptable_loss_pct: float = 0.003
    fee_rate: float = 0.0011
    use_ml_direction: bool = False
    direction_model: XGBoostModelConfig = XGBoostModelConfig()
    sma_fast: MovingAveragePeriod = 5
    sma_slow: MovingAveragePeriod = 20
    min_holding_bars: int = 5
    max_holding_bars: int = 30
    lookback_candidates: list[int] = [360, 720, 1440]
    retrain_interval: int = 720
    validation_ratio: float = 0.2
    monthly_volume_target: float | None = None
    position_size: float = 3000.0
    rolling_gen: RollingGenConfig = RollingGenConfig()
    """Optional declarative rolling geometry; disabled unless enabled is set."""
