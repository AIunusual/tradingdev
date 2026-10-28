"""KD Stochastic Oscillator crossover trading strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from joblib import Parallel, delayed

from tradingdev.domain.indicators.kd import KDIndicator
from tradingdev.domain.optimization.grid_search import (
    GridSearchResult,
    best_result,
    finite_metric_value,
    metric_direction,
    tuple_grid,
)
from tradingdev.domain.strategies.base import BaseStrategy
from tradingdev.domain.strategies.bundled.kd_strategy.config import KDStrategyConfig
from tradingdev.domain.strategies.signals import generate_checked_signals
from tradingdev.shared.utils.logger import setup_logger
from tradingdev.shared.utils.parallel import estimate_n_jobs

if TYPE_CHECKING:
    import pandas as pd

    from tradingdev.domain.backtest.base_engine import (
        BaseBacktestEngine,
    )
    from tradingdev.domain.backtest.schemas import ParallelConfig
    from tradingdev.domain.strategies.bundled.kd_strategy.config import KDFitConfig

logger = setup_logger(__name__)


def _evaluate_kd_combo(
    df: pd.DataFrame,
    trial_config: KDStrategyConfig,
    engine: BaseBacktestEngine,
    target: str,
) -> tuple[KDStrategyConfig, float | None]:
    """Evaluate a single KD parameter combination.

    This is a module-level function so it can be pickled by joblib.
    """
    trial = KDStrategy(config=trial_config)
    signals = generate_checked_signals(trial, df)
    result = engine.run(signals)
    return trial_config, finite_metric_value(result.metrics.get(target))


class KDStrategy(BaseStrategy):
    """KD crossover strategy using Stochastic Oscillator signals."""

    def __init__(
        self,
        config: KDStrategyConfig,
        fit_config: KDFitConfig | None = None,
        backtest_engine: BaseBacktestEngine | None = None,
        parallel_config: ParallelConfig | None = None,
    ) -> None:
        self._config = config
        self._fit_config = fit_config
        self._backtest_engine = backtest_engine
        self._parallel_config = parallel_config
        self._indicator = KDIndicator(
            k_period=config.k_period,
            d_period=config.d_period,
            smooth_k=config.smooth_k,
        )

    def fit(self, df: pd.DataFrame) -> None:
        """Grid search over KD parameter combinations."""
        if self._fit_config is None:
            return

        if self._backtest_engine is None:
            msg = "backtest_engine is required for fit()"
            raise RuntimeError(msg)

        engine = self._backtest_engine
        target = self._fit_config.target_metric
        direction = metric_direction(target)

        grid = tuple_grid(
            self._fit_config.k_period_range,
            self._fit_config.d_period_range,
            self._fit_config.smooth_k_range,
            self._fit_config.overbought_range,
            self._fit_config.oversold_range,
        )

        p_cfg = self._parallel_config
        n_jobs = estimate_n_jobs(
            df,
            safety_factor=p_cfg.safety_factor if p_cfg else 0.6,
            overhead_multiplier=p_cfg.overhead_multiplier if p_cfg else 3.0,
            reserve_cores=p_cfg.reserve_cores if p_cfg else 2,
        )
        logger.info(
            "KD grid search: %d combinations (n_jobs=%d)",
            len(grid),
            n_jobs,
        )

        configs = [
            KDStrategyConfig(
                k_period=k_p,
                d_period=d_p,
                smooth_k=sm_k,
                overbought=ob,
                oversold=os_,
            )
            for k_p, d_p, sm_k, ob, os_ in grid
        ]

        results: list[tuple[KDStrategyConfig, float | None]] = Parallel(
            n_jobs=n_jobs,
        )(delayed(_evaluate_kd_combo)(df, cfg, engine, target) for cfg in configs)

        best = best_result(
            (GridSearchResult(cfg.model_dump(), value, {}) for cfg, value in results),
            direction=direction,
        )
        best_config = KDStrategyConfig(**best.params)
        best_metric = best.metric_value

        self._config = best_config
        self._indicator = KDIndicator(
            k_period=best_config.k_period,
            d_period=best_config.d_period,
            smooth_k=best_config.smooth_k,
        )

        logger.info(
            "Best KD params: %s (%s=%.4f)",
            best_config.model_dump(),
            target,
            best_metric,
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate long/short signals based on KD crossovers."""
        result = self._indicator.calculate(df)

        k = result["stoch_k"]
        d = result["stoch_d"]
        k_prev = k.shift(1)
        d_prev = d.shift(1)

        golden_cross = (k > d) & (k_prev <= d_prev)
        death_cross = (k < d) & (k_prev >= d_prev)

        long_signal = golden_cross & (k < self._config.oversold)
        short_signal = death_cross & (k > self._config.overbought)

        result["signal"] = 0
        result.loc[long_signal, "signal"] = 1
        result.loc[short_signal, "signal"] = -1

        return result

    def get_parameters(self) -> dict[str, Any]:
        return self._config.model_dump()
