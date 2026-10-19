"""Configuration models for the bundled GLFT-ML strategy."""

from typing import Literal, Self

from pydantic import BaseModel, model_validator

from tradingdev.domain.ml.models.flaml_model import DEFAULT_ESTIMATOR_LIST
from tradingdev.domain.strategies.bundled.indicator_parameters import (
    MovingAveragePeriod,
)


class GLFTMLStrategyConfig(BaseModel):
    """GLFT plus ML direction prediction strategy configuration."""

    prediction_horizon: int = 5
    feature_lookback: int = 60
    ml_time_limit: int = 300
    ml_estimator_list: list[str] = list(DEFAULT_ESTIMATOR_LIST)
    ml_metric: str = "accuracy"
    random_seed: int = 42
    confidence_threshold: float = 0.55
    confidence_threshold_candidates: list[float] = [0.52, 0.55, 0.60]
    gamma: float = 0.0
    kappa: float = 1000.0
    ema_window: MovingAveragePeriod = 15
    vol_window: int = 30
    vol_type: Literal["realized", "parkinson", "implied"] = "implied"
    min_holding_bars: int = 5
    max_holding_bars: int = 13
    gamma_candidates: list[float] = [0.0, 200.0, 500.0]
    kappa_candidates: list[float] = [500.0, 1000.0]
    ema_window_candidates: list[MovingAveragePeriod] = [5, 15, 30, 75]
    max_holding_bars_candidates: list[int] = [8, 13, 30]
    min_entry_edge_candidates: list[float] = [0.0008, 0.0012, 0.002]
    profit_target_ratio_candidates: list[float] = [0.5, 0.75, 1.0]
    target_metric: str = "total_volume"
    min_entry_edge: float = 0.0012
    profit_target_ratio: float = 0.75
    strategy_sl: float = 0.003
    position_size: float = 3000.0
    monthly_volume_target: float | None = None
    fee_rate: float = 0.0002
    min_monthly_pnl: float | None = None

    @model_validator(mode="after")
    def glft_ml_max_gt_min_holding(self) -> Self:
        """Validate max_holding_bars > min_holding_bars."""
        if self.max_holding_bars <= self.min_holding_bars:
            msg = "max_holding_bars must be greater than min_holding_bars"
            raise ValueError(msg)
        return self
