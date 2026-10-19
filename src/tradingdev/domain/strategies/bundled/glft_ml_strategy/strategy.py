"""GLFT + ML direction prediction strategy.

Combines the GLFT analytical spread model with a FLAML AutoML
direction predictor.  The ML model forecasts price direction over
a configurable horizon (e.g. 5 min); the GLFT model determines
optimal entry thresholds.  Entries are only taken when both agree.

Designed for limit-order execution (maker fees).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from tradingdev.domain import indicators
from tradingdev.domain.ml.features.direction_features import DirectionFeatureEngineer
from tradingdev.domain.ml.models.flaml_model import FLAMLAutoMLDirectionModel
from tradingdev.domain.optimization.grid_search import (
    finite_metric_value,
    is_better_metric,
    metric_direction,
    tuple_grid,
)
from tradingdev.domain.strategies.base import BaseStrategy
from tradingdev.shared.utils.logger import setup_logger

if TYPE_CHECKING:
    import numpy.typing as npt

    from tradingdev.domain.backtest.base_engine import BaseBacktestEngine
    from tradingdev.domain.backtest.schemas import ParallelConfig
    from tradingdev.domain.strategies.bundled.glft_ml_strategy.config import (
        GLFTMLStrategyConfig,
    )

logger = setup_logger(__name__)

# (gamma, kappa, ema_w, max_hold, entry_edge, pt_ratio)
_GLFTParamTuple = tuple[float, float, int, int, float, float]


class GLFTMLStrategy(BaseStrategy):
    """GLFT + FLAML AutoML direction prediction strategy.

    The ML model predicts price direction over the next N minutes.
    The GLFT model computes entry thresholds based on volatility.
    Entries require both ML direction agreement and GLFT deviation
    threshold to be met.

    Limit-order assumption: when the strategy enters, it places a
    limit order in advance based on the ML prediction.  The backtest
    uses maker fees to reflect this.
    """

    def __init__(
        self,
        config: GLFTMLStrategyConfig,
        backtest_engine: BaseBacktestEngine | None = None,
        parallel_config: ParallelConfig | None = None,
    ) -> None:
        self._config = config
        self._backtest_engine = backtest_engine
        self._parallel_config = parallel_config

        # ML components
        self._feature_eng = DirectionFeatureEngineer(
            lookback=config.feature_lookback,
            prediction_horizon=config.prediction_horizon,
        )
        num_cpus = self._compute_num_cpus()
        self._ml_model = FLAMLAutoMLDirectionModel(
            time_budget=config.ml_time_limit,
            estimator_list=config.ml_estimator_list,
            metric=config.ml_metric,
            verbosity=0,
            n_jobs=num_cpus,
            random_seed=config.random_seed,
        )
        self._ml_trained = False
        self._confidence_threshold: float = config.confidence_threshold

        # GLFT parameters (populated by fit or use defaults)
        self._best_gamma: float = config.gamma
        self._best_kappa: float = config.kappa
        self._best_ema_window: int = config.ema_window
        self._best_max_holding_bars: int = config.max_holding_bars
        self._best_min_entry_edge: float = config.min_entry_edge
        self._best_profit_target_ratio: float = config.profit_target_ratio
        self._best_confidence_threshold: float = config.confidence_threshold

    def _compute_num_cpus(self) -> int | None:
        """Derive ``n_jobs`` for the AutoML search from parallel config."""
        if self._parallel_config is None:
            return None
        from tradingdev.shared.utils.parallel import _get_performance_core_count

        perf = _get_performance_core_count()
        return max(1, perf - self._parallel_config.reserve_cores)

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> None:
        """Train ML model and grid-search GLFT parameters.

        Steps:
        1. Engineer features and run the AutoML direction search
        2. Generate ML direction predictions on training data
        3. Grid-search GLFT parameters using ML-filtered signals
        """
        if self._backtest_engine is None:
            logger.warning("No backtest engine; skipping fit()")
            return

        # --- Step 1: Train ML model ---
        logger.info(
            "GLFT-ML fit: engineering features (horizon=%d)...",
            self._config.prediction_horizon,
        )
        feat_df = self._feature_eng.transform(df, include_target=True)
        logger.info(
            "GLFT-ML fit: %d samples, %d features",
            len(feat_df),
            len(self._feature_eng.get_feature_names()),
        )

        self._ml_model.train(feat_df)
        self._ml_trained = True

        # --- Step 2: Get ML predictions on training data ---
        proba = self._ml_model.predict_proba(feat_df)
        # proba has columns [0, 1] for [down, up]
        # Convert to direction: 1 (up), -1 (down), 0 (uncertain)
        ml_directions = self._compute_ml_directions(
            proba,
            self._config.confidence_threshold,
        )

        # --- Step 3: Grid-search GLFT parameters ---
        self._grid_search_glft(feat_df, ml_directions)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate position-state signals via GLFT + ML."""
        if not self._ml_trained:
            logger.warning("ML model not trained; falling back to pure GLFT signals")
            return self._generate_pure_glft(df)

        feat_df = self._feature_eng.transform(df, include_target=False)

        proba = self._ml_model.predict_proba(feat_df)
        ml_directions = self._compute_ml_directions(
            proba,
            self._best_confidence_threshold,
        )

        close = np.asarray(feat_df["close"].astype(float).values)
        high = np.asarray(feat_df["high"].astype(float).values)
        low = np.asarray(feat_df["low"].astype(float).values)

        # Extract DVOL for implied volatility
        dvol: npt.NDArray[np.floating[Any]] | None = None
        if self._config.vol_type == "implied" and "dvol" in feat_df.columns:
            dvol = np.asarray(feat_df["dvol"].astype(float).values)

        ema = self._compute_ema(close, self._best_ema_window)
        sigma = self._compute_volatility(close, high, low, dvol=dvol)

        signals = self._run_ml_glft_state_machine(
            close=close,
            ema=ema,
            sigma=sigma,
            ml_dir=ml_directions,
            gamma=self._best_gamma,
            kappa=self._best_kappa,
            min_hold=self._config.min_holding_bars,
            max_hold=self._best_max_holding_bars,
            min_entry_edge=self._best_min_entry_edge,
            profit_target_ratio=self._best_profit_target_ratio,
            strategy_sl=self._config.strategy_sl,
        )

        result = feat_df.copy()
        result["signal"] = signals
        return result

    def get_parameters(self) -> dict[str, Any]:
        """Return strategy parameters."""
        params = self._config.model_dump()
        params["best_gamma"] = self._best_gamma
        params["best_kappa"] = self._best_kappa
        params["best_ema_window"] = self._best_ema_window
        params["best_max_holding_bars"] = self._best_max_holding_bars
        params["best_min_entry_edge"] = self._best_min_entry_edge
        params["best_profit_target_ratio"] = self._best_profit_target_ratio
        params["best_confidence_threshold"] = self._best_confidence_threshold
        params["ml_trained"] = self._ml_trained
        return params

    # ------------------------------------------------------------------
    # ML direction computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_ml_directions(
        proba: pd.DataFrame,
        confidence_threshold: float,
    ) -> npt.NDArray[np.float64]:
        """Convert class probabilities to direction signals.

        Args:
            proba: DataFrame with columns for class 0 (down) and 1 (up).
            confidence_threshold: Minimum probability to act.

        Returns:
            Array of 1 (up), -1 (down), 0 (uncertain).
        """
        # Class columns come from the fitted estimator; binary fits are [0, 1]
        cols = list(proba.columns)
        if len(cols) >= 2:
            p_down = np.asarray(proba.iloc[:, 0].values, dtype=np.float64)
            p_up = np.asarray(proba.iloc[:, 1].values, dtype=np.float64)
        else:
            # Fallback: single column assumed to be P(up)
            p_up = np.asarray(proba.iloc[:, 0].values, dtype=np.float64)
            p_down = 1.0 - p_up

        directions = np.zeros(len(proba), dtype=np.float64)
        directions[p_up >= confidence_threshold] = 1.0
        directions[p_down >= confidence_threshold] = -1.0
        return directions

    # ------------------------------------------------------------------
    # GLFT grid search
    # ------------------------------------------------------------------

    def _grid_search_glft(
        self,
        feat_df: pd.DataFrame,
        ml_directions: npt.NDArray[np.float64],
    ) -> None:
        """Grid-search GLFT parameters with ML direction filter."""
        assert self._backtest_engine is not None

        cfg = self._config
        target = cfg.target_metric
        direction = metric_direction(target)
        min_mp = cfg.min_monthly_pnl

        close = np.asarray(feat_df["close"].astype(float).values)
        high = np.asarray(feat_df["high"].astype(float).values)
        low = np.asarray(feat_df["low"].astype(float).values)

        dvol: npt.NDArray[np.floating[Any]] | None = None
        if cfg.vol_type == "implied" and "dvol" in feat_df.columns:
            dvol = np.asarray(feat_df["dvol"].astype(float).values)

        grid: list[_GLFTParamTuple] = [
            cast("_GLFTParamTuple", combo)
            for combo in tuple_grid(
                cfg.gamma_candidates,
                cfg.kappa_candidates,
                cfg.ema_window_candidates,
                cfg.max_holding_bars_candidates,
                cfg.min_entry_edge_candidates,
                cfg.profit_target_ratio_candidates,
            )
        ]

        # Also search confidence threshold
        conf_candidates = cfg.confidence_threshold_candidates

        best_value: float | None = None
        best_gamma = cfg.gamma
        best_kappa = cfg.kappa
        best_ema = cfg.ema_window
        best_max_hold = cfg.max_holding_bars
        best_entry_edge = cfg.min_entry_edge
        best_pt_ratio = cfg.profit_target_ratio
        best_conf = cfg.confidence_threshold

        total_combos = len(grid) * len(conf_candidates)
        logger.info(
            "GLFT-ML grid search: %d GLFT combos × %d conf thresholds = %d",
            len(grid),
            len(conf_candidates),
            total_combos,
        )

        n_filtered = 0
        for conf_thresh in conf_candidates:
            # Recompute ML directions for this threshold
            proba = self._ml_model.predict_proba(feat_df)
            ml_dir = self._compute_ml_directions(proba, conf_thresh)

            for params in grid:
                gamma, kappa, ema_w, max_hold, entry_edge, pt_ratio = params

                ema = self._compute_ema(close, ema_w)
                sigma = self._compute_volatility(
                    close,
                    high,
                    low,
                    dvol=dvol,
                )

                signals = self._run_ml_glft_state_machine(
                    close=close,
                    ema=ema,
                    sigma=sigma,
                    ml_dir=ml_dir,
                    gamma=gamma,
                    kappa=kappa,
                    min_hold=cfg.min_holding_bars,
                    max_hold=max_hold,
                    min_entry_edge=entry_edge,
                    profit_target_ratio=pt_ratio,
                    strategy_sl=cfg.strategy_sl,
                )

                sig_df = feat_df.copy()
                sig_df["signal"] = signals
                result = self._backtest_engine.run(sig_df)
                metrics = result.metrics

                if min_mp is not None:
                    dpm = finite_metric_value(metrics.get("daily_pnl_mean"))
                    monthly_pnl = dpm * 30 if dpm is not None else -math.inf
                    if monthly_pnl < min_mp:
                        n_filtered += 1
                        continue

                value = finite_metric_value(metrics.get(target))
                if is_better_metric(value, best_value, direction):
                    best_value = value
                    best_gamma = gamma
                    best_kappa = kappa
                    best_ema = ema_w
                    best_max_hold = max_hold
                    best_entry_edge = entry_edge
                    best_pt_ratio = pt_ratio
                    best_conf = conf_thresh

        if n_filtered > 0:
            logger.info(
                "GLFT-ML fit: %d/%d combos filtered (monthly_pnl < %.2f)",
                n_filtered,
                total_combos,
                min_mp if min_mp is not None else 0.0,
            )

        if best_value is None:
            raise ValueError(
                "No GLFT-ML parameter combination has a finite eligible objective"
            )

        self._best_gamma = best_gamma
        self._best_kappa = best_kappa
        self._best_ema_window = best_ema
        self._best_max_holding_bars = best_max_hold
        self._best_min_entry_edge = best_entry_edge
        self._best_profit_target_ratio = best_pt_ratio
        self._best_confidence_threshold = best_conf

        logger.info(
            "GLFT-ML fit complete: gamma=%.1f, kappa=%.1f, ema=%d, "
            "max_hold=%d, entry_edge=%.4f, pt_ratio=%.2f, "
            "conf_thresh=%.2f (%s=%.4f)",
            best_gamma,
            best_kappa,
            best_ema,
            best_max_hold,
            best_entry_edge,
            best_pt_ratio,
            best_conf,
            target,
            best_value,
        )

    # ------------------------------------------------------------------
    # EMA computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_ema(
        close: npt.NDArray[np.floating[Any]],
        ema_window: int,
    ) -> npt.NDArray[np.floating[Any]]:
        """Compute EMA on 1-min close; NaN until the window has warmed up."""
        return np.asarray(
            indicators.ema(pd.Series(close), ema_window),
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Volatility estimation
    # ------------------------------------------------------------------

    _MINUTES_PER_YEAR: float = 525_960.0

    def _compute_volatility(
        self,
        close: npt.NDArray[np.floating[Any]],
        high: npt.NDArray[np.floating[Any]],
        low: npt.NDArray[np.floating[Any]],
        dvol: npt.NDArray[np.floating[Any]] | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        """Compute per-bar volatility sigma."""
        if self._config.vol_type == "implied":
            if dvol is None:
                msg = "dvol array is required when vol_type is 'implied'"
                raise ValueError(msg)
            sigma = dvol / 100.0 / np.sqrt(self._MINUTES_PER_YEAR)
        else:
            window = self._config.vol_window
            log_ret = np.diff(np.log(np.maximum(close, 1e-10)))
            log_ret = np.concatenate([[0.0], log_ret])
            sigma = pd.Series(log_ret).rolling(window, min_periods=1).std().values

        sigma = np.where(
            np.isnan(sigma) | (sigma <= 0),
            1e-10,
            sigma,
        )
        return np.asarray(sigma, dtype=np.float64)

    # ------------------------------------------------------------------
    # ML-filtered GLFT state machine
    # ------------------------------------------------------------------

    @staticmethod
    def _run_ml_glft_state_machine(
        close: npt.NDArray[np.floating[Any]],
        ema: npt.NDArray[np.floating[Any]],
        sigma: npt.NDArray[np.floating[Any]],
        ml_dir: npt.NDArray[np.float64],
        gamma: float,
        kappa: float,
        min_hold: int,
        max_hold: int,
        min_entry_edge: float = 0.0012,
        profit_target_ratio: float = 1.0,
        strategy_sl: float = 0.005,
    ) -> npt.NDArray[np.floating[Any]]:
        """GLFT state machine with ML direction filter.

        Same as the pure GLFT state machine, but entries are only
        allowed when the ML direction prediction agrees:
        - ML=1 (up): only long entries allowed
        - ML=-1 (down): only short entries allowed
        - ML=0 (uncertain): no entries allowed
        """
        n = len(close)
        signals = np.zeros(n, dtype=np.float64)

        if gamma < 1e-12:
            spread_const = 1.0 / kappa
        else:
            spread_const = math.log(1.0 + gamma / kappa) / gamma

        state = 0
        bars_in_pos = 0
        entry_dev = 0.0

        for i in range(n):
            s = close[i]
            fair = ema[i]
            sig_sq = sigma[i] ** 2

            if fair <= 0 or s <= 0:
                signals[i] = float(state)
                if state != 0:
                    bars_in_pos += 1
                continue

            deviation = (s - fair) / fair

            if state == 0:
                # --- FLAT: evaluate entry ---
                tau = float(max_hold)
                glft_hs = gamma * sig_sq * tau / 2.0 + spread_const
                half_spread = max(glft_hs, min_entry_edge)

                want_long = deviation < -half_spread
                want_short = deviation > half_spread

                # ML direction filter
                md = ml_dir[i]
                if md >= 0.5:
                    want_short = False  # ML says up → only long
                elif md <= -0.5:
                    want_long = False  # ML says down → only short
                else:
                    want_long = False
                    want_short = False

                if want_long:
                    state = 1
                    bars_in_pos = 0
                    entry_dev = deviation
                elif want_short:
                    state = -1
                    bars_in_pos = 0
                    entry_dev = deviation
            else:
                # --- IN POSITION ---
                bars_in_pos += 1

                if bars_in_pos < min_hold:
                    signals[i] = float(state)
                    continue

                if bars_in_pos >= max_hold:
                    state = 0
                    signals[i] = 0.0
                    continue

                if strategy_sl > 0 and (
                    (state == 1 and deviation < entry_dev - strategy_sl)
                    or (state == -1 and deviation > entry_dev + strategy_sl)
                ):
                    state = 0
                    signals[i] = 0.0
                    continue

                if profit_target_ratio > 0:
                    target = entry_dev * (1.0 - profit_target_ratio)
                    if (state == 1 and deviation >= target) or (
                        state == -1 and deviation <= target
                    ):
                        state = 0

            signals[i] = float(state)

        return signals

    # ------------------------------------------------------------------
    # Fallback: pure GLFT (no ML)
    # ------------------------------------------------------------------

    def _generate_pure_glft(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals without ML (fallback)."""
        close = np.asarray(df["close"].astype(float).values)
        high = np.asarray(df["high"].astype(float).values)
        low = np.asarray(df["low"].astype(float).values)

        dvol: npt.NDArray[np.floating[Any]] | None = None
        if self._config.vol_type == "implied" and "dvol" in df.columns:
            dvol = np.asarray(df["dvol"].astype(float).values)

        ema = self._compute_ema(close, self._best_ema_window)
        sigma = self._compute_volatility(close, high, low, dvol=dvol)
        ml_dir = np.zeros(len(close), dtype=np.float64)

        signals = self._run_ml_glft_state_machine(
            close=close,
            ema=ema,
            sigma=sigma,
            ml_dir=ml_dir,
            gamma=self._best_gamma,
            kappa=self._best_kappa,
            min_hold=self._config.min_holding_bars,
            max_hold=self._best_max_holding_bars,
            min_entry_edge=self._best_min_entry_edge,
            profit_target_ratio=self._best_profit_target_ratio,
            strategy_sl=self._config.strategy_sl,
        )

        result = df.copy()
        result["signal"] = signals
        return result
