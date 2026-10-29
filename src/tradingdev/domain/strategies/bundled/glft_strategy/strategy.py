"""GLFT market-making strategy adapted for signal-based backtesting.

Based on Gueant, Lehalle & Fernandez-Tapia (2013),
"Dealing with the Inventory Risk: A Solution to the Market Making
Problem".  The model computes an optimal spread and reservation price
that adjust dynamically based on inventory risk and volatility.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from tradingdev.domain import indicators
from tradingdev.domain.optimization.grid_search import (
    finite_metric_value,
    is_better_metric,
    metric_direction,
    tuple_grid,
)
from tradingdev.domain.strategies.base import BaseStrategy
from tradingdev.domain.strategies.signals import generate_checked_signals
from tradingdev.shared.utils.logger import setup_logger
from tradingdev.shared.utils.parallel import estimate_n_jobs

if TYPE_CHECKING:
    import numpy.typing as npt

    from tradingdev.domain.backtest.base_engine import (
        BaseBacktestEngine,
    )
    from tradingdev.domain.backtest.schemas import ParallelConfig
    from tradingdev.domain.strategies.bundled.glft_strategy.config import (
        GLFTStrategyConfig,
    )

logger = setup_logger(__name__)

# Type alias for the full parameter tuple searched by fit()
_ParamTuple = tuple[float, float, int, int, int, float, int, float, int, float]


def _evaluate_glft_combo(
    df: pd.DataFrame,
    config: GLFTStrategyConfig,
    params: _ParamTuple,
    engine: BaseBacktestEngine,
    target: str,
) -> tuple[_ParamTuple, dict[str, Any]]:
    """Evaluate a single GLFT parameter combination.

    This is a module-level function so it can be pickled by joblib.

    Returns:
        Tuple of (params, metrics_dict) for the main process to
        apply constraint filtering.
    """
    (
        gamma,
        kappa,
        ema_w,
        max_hold,
        vol_win,
        entry_edge,
        trend_ema,
        pt_ratio,
        sig_agg,
        efs,
    ) = params
    trial = GLFTStrategy(config=config)
    trial._best_gamma = gamma
    trial._best_kappa = kappa
    trial._best_ema_window = ema_w
    trial._best_max_holding_bars = max_hold
    trial._best_vol_window = vol_win
    trial._best_min_entry_edge = entry_edge
    trial._best_trend_ema_window = trend_ema
    trial._best_profit_target_ratio = pt_ratio
    trial._best_signal_agg_minutes = sig_agg
    trial._best_edge_for_full_size = efs

    signals = generate_checked_signals(trial, df)
    result = engine.run(signals)
    return params, result.metrics


class GLFTStrategy(BaseStrategy):
    """GLFT market-making strategy for signal-based backtesting.

    Uses the Gueant-Lehalle-Fernandez-Tapia optimal market-making
    model to determine entry/exit thresholds based on:

    - **Reservation price**: mid-price adjusted for inventory risk
    - **Optimal spread**: adapts to volatility and time horizon

    The strategy is purely analytical -- no ML model required.
    """

    def __init__(
        self,
        config: GLFTStrategyConfig,
        backtest_engine: BaseBacktestEngine | None = None,
        parallel_config: ParallelConfig | None = None,
    ) -> None:
        self._config = config
        self._backtest_engine = backtest_engine
        self._parallel_config = parallel_config

        # Populated by fit() -- or use config defaults
        self._best_gamma: float = config.gamma
        self._best_kappa: float = config.kappa
        self._best_ema_window: int = config.ema_window
        self._best_max_holding_bars: int = config.max_holding_bars
        self._best_vol_window: int = config.vol_window
        self._best_min_entry_edge: float = config.min_entry_edge
        self._best_trend_ema_window: int = config.trend_ema_window
        self._best_profit_target_ratio: float = config.profit_target_ratio
        self._best_signal_agg_minutes: int = config.signal_agg_minutes
        self._best_edge_for_full_size: float = config.edge_for_full_size

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> None:
        """Grid-search over GLFT parameters.

        When ``min_monthly_pnl`` is set, only parameter combinations
        whose estimated monthly PnL (``daily_pnl_mean * 30``) meets the
        threshold are considered.  Among those, the combination with the
        best ``target_metric`` according to the metric catalog is selected.
        """
        if self._backtest_engine is None:
            logger.warning(
                "No backtest engine; skipping fit()",
            )
            return

        best_value: float | None = None
        best_gamma = self._config.gamma
        best_kappa = self._config.kappa
        best_ema = self._config.ema_window
        best_max_hold = self._config.max_holding_bars
        best_vol_win = self._config.vol_window
        best_entry_edge = self._config.min_entry_edge
        best_trend_ema = self._config.trend_ema_window
        best_pt_ratio = self._config.profit_target_ratio
        best_sig_agg = self._config.signal_agg_minutes
        best_efs = self._config.edge_for_full_size
        target = self._config.target_metric
        direction = metric_direction(target)
        min_mp = self._config.min_monthly_pnl

        # When using implied vol, vol_window is irrelevant
        vol_win_candidates = self._config.vol_window_candidates
        if self._config.vol_type == "implied":
            vol_win_candidates = [0]
            logger.info("vol_type='implied': vol_window_candidates ignored")

        # Only grid-search edge_for_full_size when dynamic sizing is on
        efs_candidates: list[float] = (
            self._config.edge_for_full_size_candidates
            if self._config.dynamic_sizing
            else [self._config.edge_for_full_size]
        )

        grid: list[_ParamTuple] = [
            cast("_ParamTuple", combo)
            for combo in tuple_grid(
                self._config.gamma_candidates,
                self._config.kappa_candidates,
                self._config.ema_window_candidates,
                self._config.max_holding_bars_candidates,
                vol_win_candidates,
                self._config.min_entry_edge_candidates,
                self._config.trend_ema_candidates,
                self._config.profit_target_ratio_candidates,
                self._config.signal_agg_minutes_candidates,
                efs_candidates,
            )
        ]

        p_cfg = self._parallel_config
        n_jobs = estimate_n_jobs(
            df,
            safety_factor=p_cfg.safety_factor if p_cfg else 0.6,
            overhead_multiplier=p_cfg.overhead_multiplier if p_cfg else 3.0,
            reserve_cores=p_cfg.reserve_cores if p_cfg else 2,
        )
        logger.info(
            "GLFT grid search: %d combinations (n_jobs=%d)",
            len(grid),
            n_jobs,
        )

        results: list[tuple[_ParamTuple, dict[str, Any]]] = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_glft_combo)(
                df,
                self._config,
                combo,
                self._backtest_engine,
                target,
            )
            for combo in grid
        )

        n_filtered = 0
        for params, metrics in results:
            if min_mp is not None:
                dpm = finite_metric_value(metrics.get("daily_pnl_mean"))
                monthly_pnl = dpm * 30 if dpm is not None else -math.inf
                if monthly_pnl < min_mp:
                    n_filtered += 1
                    continue

            value = finite_metric_value(metrics.get(target))

            if is_better_metric(value, best_value, direction):
                best_value = value
                best_gamma = params[0]
                best_kappa = params[1]
                best_ema = params[2]
                best_max_hold = params[3]
                best_vol_win = params[4]
                best_entry_edge = params[5]
                best_trend_ema = params[6]
                best_pt_ratio = params[7]
                best_sig_agg = params[8]
                best_efs = params[9]

        if n_filtered > 0:
            logger.info(
                "GLFT fit: %d/%d combos filtered (monthly_pnl < %.2f)",
                n_filtered,
                len(grid),
                min_mp if min_mp is not None else 0.0,
            )

        if best_value is None:
            raise ValueError(
                "No GLFT parameter combination has a finite eligible objective"
            )

        self._best_gamma = best_gamma
        self._best_kappa = best_kappa
        self._best_ema_window = best_ema
        self._best_max_holding_bars = best_max_hold
        self._best_vol_window = best_vol_win
        self._best_min_entry_edge = best_entry_edge
        self._best_trend_ema_window = best_trend_ema
        self._best_profit_target_ratio = best_pt_ratio
        self._best_signal_agg_minutes = best_sig_agg
        self._best_edge_for_full_size = best_efs

        logger.info(
            "GLFT fit complete: gamma=%.4f, kappa=%.4f, "
            "ema=%d, max_hold=%d, vol_win=%d, "
            "entry_edge=%.4f, trend_ema=%d, "
            "pt_ratio=%.2f, sig_agg=%d, efs=%.4f (%s=%.4f)",
            best_gamma,
            best_kappa,
            best_ema,
            best_max_hold,
            best_vol_win,
            best_entry_edge,
            best_trend_ema,
            best_pt_ratio,
            best_sig_agg,
            best_efs,
            target,
            best_value,
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate position-state signals via the GLFT model."""
        close = np.asarray(df["close"].astype(float).values)
        high = np.asarray(df["high"].astype(float).values)
        low = np.asarray(df["low"].astype(float).values)

        # Extract DVOL series when using implied volatility
        dvol: npt.NDArray[np.floating[Any]] | None = None
        if self._config.vol_type == "implied":
            if "dvol" not in df.columns:
                msg = "DataFrame must contain 'dvol' column when vol_type is 'implied'"
                raise ValueError(msg)
            dvol = np.asarray(df["dvol"].astype(float).values)

        ema = self._compute_ema(
            close,
            self._best_ema_window,
            self._best_signal_agg_minutes,
        )
        sigma = self._compute_volatility(close, high, low, dvol=dvol)

        # Trend filter: slow EMA direction (+1=up, -1=down, 0=flat,
        # NaN while the slow EMA is still warming up)
        trend_dir: npt.NDArray[np.floating[Any]] | None = None
        if self._best_trend_ema_window > 0:
            slow_ema = np.asarray(
                indicators.ema(pd.Series(close), self._best_trend_ema_window),
                dtype=np.float64,
            )
            # Trend = sign of slow EMA slope (diff)
            slope = np.diff(slow_ema, prepend=np.nan)
            trend_dir = np.sign(slope)

        dyn = self._config.dynamic_sizing
        min_weight = (
            self._config.min_position_size / self._config.position_size if dyn else 0.0
        )

        signals, size_weights = self._run_glft_state_machine(
            close=close,
            ema=ema,
            sigma=sigma,
            gamma=self._best_gamma,
            kappa=self._best_kappa,
            min_hold=self._config.min_holding_bars,
            max_hold=self._best_max_holding_bars,
            min_entry_edge=self._best_min_entry_edge,
            trend_dir=trend_dir,
            profit_target_ratio=self._best_profit_target_ratio,
            strategy_sl=self._config.strategy_sl,
            momentum_guard=self._config.momentum_guard,
            dynamic_sizing=dyn,
            min_weight=min_weight,
            edge_for_full_size=self._best_edge_for_full_size,
        )

        result = df.copy()
        result["signal"] = signals
        result["size_weight"] = size_weights
        return result

    def get_parameters(self) -> dict[str, Any]:
        """Return strategy parameters."""
        params = self._config.model_dump()
        params["best_gamma"] = self._best_gamma
        params["best_kappa"] = self._best_kappa
        params["best_ema_window"] = self._best_ema_window
        params["best_max_holding_bars"] = self._best_max_holding_bars
        params["best_vol_window"] = self._best_vol_window
        params["best_min_entry_edge"] = self._best_min_entry_edge
        params["best_trend_ema_window"] = self._best_trend_ema_window
        params["best_profit_target_ratio"] = self._best_profit_target_ratio
        params["best_signal_agg_minutes"] = self._best_signal_agg_minutes
        params["best_edge_for_full_size"] = self._best_edge_for_full_size
        return params

    # ------------------------------------------------------------------
    # Multi-timeframe EMA
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_ema(
        close: npt.NDArray[np.floating[Any]],
        ema_window: int,
        agg_minutes: int,
    ) -> npt.NDArray[np.floating[Any]]:
        """Compute EMA, optionally on aggregated N-min bars.

        When ``agg_minutes > 1``, the close series is resampled to
        N-minute bars (taking the last close in each window), the
        EMA is computed on these resampled values, and the result
        is mapped back to 1-minute resolution using only aggregated
        bars that have already closed. Bars before the first
        aggregated bar closes, and bars inside the EMA warm-up, are
        NaN so the strategy stays flat there.

        Returns an array of the same length as *close*.
        """
        if agg_minutes <= 1:
            return np.asarray(
                indicators.ema(pd.Series(close), ema_window),
                dtype=np.float64,
            )

        n = len(close)
        ag = agg_minutes

        # Resample: take close at end of each N-min window
        # Indices: ag-1, 2*ag-1, 3*ag-1, ...
        n_complete = (n // ag) * ag
        agg_close = close[ag - 1 : n_complete : ag]

        # EMA on resampled closes
        agg_ema = np.asarray(
            indicators.ema(pd.Series(agg_close), ema_window),
            dtype=np.float64,
        )

        # Map back to 1-min resolution.
        # At 1-min bar i, the most recently completed N-min bar
        # has index k = (i + 1) // ag - 1. k < 0 for bars before the
        # first window completes; those bars stay NaN so a later
        # bar's value is never used. The largest k is the last
        # aggregated index, so no upper bound is needed.
        k_idx = (np.arange(n) + 1) // ag - 1
        result = np.full(n, np.nan, dtype=np.float64)
        completed = k_idx >= 0
        result[completed] = agg_ema[k_idx[completed]]
        return result

    # ------------------------------------------------------------------
    # Volatility estimation
    # ------------------------------------------------------------------

    # Minutes in a year (365.25 * 24 * 60) — used for
    # annualized-to-per-bar volatility conversion.
    _MINUTES_PER_YEAR: float = 525_960.0

    def _compute_volatility(
        self,
        close: npt.NDArray[np.floating[Any]],
        high: npt.NDArray[np.floating[Any]],
        low: npt.NDArray[np.floating[Any]],
        dvol: npt.NDArray[np.floating[Any]] | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        """Compute volatility (sigma).

        When ``vol_type`` is ``"implied"``, *dvol* must be provided
        (annualized % from Deribit DVOL).  Otherwise a rolling
        historical estimator is used.
        """
        if self._config.vol_type == "implied":
            if dvol is None:
                msg = "dvol array is required when vol_type is 'implied'"
                raise ValueError(msg)
            # DVOL is annualized vol in % (e.g. 45 = 45%).
            # Convert to per-bar (1 min) sigma:
            #   sigma = DVOL / 100 / sqrt(minutes_per_year)
            sigma = dvol / 100.0 / np.sqrt(self._MINUTES_PER_YEAR)
        elif self._config.vol_type == "parkinson":
            window = self._best_vol_window
            # Parkinson estimator: uses high-low range
            log_hl = np.log(
                np.maximum(high, 1e-10) / np.maximum(low, 1e-10),
            )
            parkinson_var = log_hl**2 / (4.0 * math.log(2))
            sigma = np.sqrt(
                pd.Series(parkinson_var).rolling(window, min_periods=1).mean().values,
            )
        else:
            window = self._best_vol_window
            # Realized volatility: rolling std of log returns
            log_ret = np.diff(
                np.log(np.maximum(close, 1e-10)),
            )
            log_ret = np.concatenate([[0.0], log_ret])
            sigma = pd.Series(log_ret).rolling(window, min_periods=1).std().values

        # Keep unknown volatility missing; only floor nonpositive observations.
        sigma = np.where(sigma <= 0, 1e-10, sigma)
        return np.asarray(sigma, dtype=np.float64)

    # ------------------------------------------------------------------
    # GLFT state machine
    # ------------------------------------------------------------------

    @staticmethod
    def _run_glft_state_machine(
        close: npt.NDArray[np.floating[Any]],
        ema: npt.NDArray[np.floating[Any]],
        sigma: npt.NDArray[np.floating[Any]],
        gamma: float,
        kappa: float,
        min_hold: int,
        max_hold: int,
        min_entry_edge: float = 0.0012,
        trend_dir: npt.NDArray[np.floating[Any]] | None = None,
        profit_target_ratio: float = 1.0,
        strategy_sl: float = 0.005,
        momentum_guard: bool = True,
        dynamic_sizing: bool = False,
        min_weight: float = 0.0,
        edge_for_full_size: float = 0.005,
    ) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
        """GLFT state machine producing position-state signals.

        ``min_entry_edge`` is the minimum normalized price deviation
        from fair value required to open a position.  It should be
        at least ``fee_rate * 2`` (round-trip cost).

        When ``trend_dir`` is provided, entries are restricted to
        the trend direction (long only in uptrend, short only in
        downtrend).

        ``profit_target_ratio`` controls exit: the fraction of the
        entry deviation to capture before exiting.  1.0 means wait
        for full mean-reversion back to EMA (deviation ≈ 0).

        ``strategy_sl`` is the maximum additional adverse deviation
        (beyond entry deviation) before cutting losses.  0 disables.

        ``momentum_guard`` when True, only enters when deviation is
        narrowing (price moving back toward EMA), filtering out
        entries into momentum moves.

        When ``dynamic_sizing`` is True, the returned ``size_weights``
        array contains per-bar position-size weights in
        ``[min_weight, 1.0]``, proportional to ``|deviation| /
        edge_for_full_size``.  When False, weights are all 1.0.

        Returns a tuple of ``(signals, size_weights)`` where signals
        contains ``(1, -1, 0)``:

        - ``1``  = long position
        - ``-1`` = short position
        - ``0``  = flat (no position)
        """
        n = len(close)
        signals = np.zeros(n, dtype=np.float64)
        size_weights = np.ones(n, dtype=np.float64)

        # Pre-compute the constant part of the half-spread
        # (1/gamma) * ln(1 + gamma/kappa)  →  limit γ→0 is 1/kappa
        if gamma < 1e-12:
            spread_const = 1.0 / kappa
        else:
            spread_const = math.log(1.0 + gamma / kappa) / gamma

        state = 0  # 0=flat, 1=long, -1=short
        bars_in_pos = 0
        entry_dev = 0.0  # deviation at entry time
        current_weight = 1.0  # weight for current position

        for i in range(n):
            s = close[i]
            fair = ema[i]
            sig_sq = sigma[i] ** 2

            if fair <= 0 or s <= 0:
                signals[i] = float(state)
                if state != 0:
                    bars_in_pos += 1
                    size_weights[i] = current_weight
                continue

            # Normalized deviation from fair value
            deviation = (s - fair) / fair

            if state == 0:
                # --- FLAT: evaluate entry ---
                tau = float(max_hold)
                # GLFT half-spread in pure %-space.
                # γ and κ are dimensionless %-space params,
                # so the spread is price-independent and
                # only adapts to volatility (σ).
                glft_hs = gamma * sig_sq * tau / 2.0 + spread_const
                half_spread = max(glft_hs, min_entry_edge)

                want_long = deviation < -half_spread
                want_short = deviation > half_spread

                # Momentum guard: only enter when deviation
                # is narrowing (price reverting toward EMA)
                if momentum_guard and i > 0:
                    prev_fair = ema[i - 1]
                    if prev_fair > 0:
                        prev_dev = (close[i - 1] - prev_fair) / prev_fair
                        # Long: deviation increasing (less
                        # negative) = price recovering
                        if want_long and deviation <= prev_dev:
                            want_long = False
                        # Short: deviation decreasing (less
                        # positive) = price pulling back
                        if want_short and deviation >= prev_dev:
                            want_short = False

                # Apply trend filter. NaN means the slow EMA has not
                # warmed up, so no trend is known and nothing is entered.
                if trend_dir is not None:
                    td = trend_dir[i]
                    if np.isnan(td):
                        want_long = False
                        want_short = False
                    elif td > 0:
                        want_short = False
                    elif td < 0:
                        want_long = False

                if want_long or want_short:
                    state = 1 if want_long else -1
                    bars_in_pos = 0
                    entry_dev = deviation
                    if dynamic_sizing and edge_for_full_size > 0:
                        raw_w = abs(deviation) / edge_for_full_size
                        current_weight = min(max(raw_w, min_weight), 1.0)
                    else:
                        current_weight = 1.0
                    size_weights[i] = current_weight
            else:
                # --- IN POSITION ---
                bars_in_pos += 1
                size_weights[i] = current_weight

                if bars_in_pos < min_hold:
                    signals[i] = float(state)
                    continue

                # Force exit at max holding period
                if bars_in_pos >= max_hold:
                    state = 0
                    signals[i] = 0.0
                    continue

                # Strategy-level stop-loss: cut when adverse
                # deviation exceeds threshold beyond entry
                if strategy_sl > 0 and (
                    (state == 1 and deviation < entry_dev - strategy_sl)
                    or (state == -1 and deviation > entry_dev + strategy_sl)
                ):
                    state = 0
                    signals[i] = 0.0
                    continue

                # Profit target: exit when deviation reverts
                # toward EMA by the specified ratio.
                # target = entry_dev * (1 - ratio):
                #   ratio=1.0 → target=0 (full reversion)
                #   ratio=0.5 → target=half of entry_dev
                if profit_target_ratio > 0:
                    target = entry_dev * (1.0 - profit_target_ratio)
                    if (state == 1 and deviation >= target) or (
                        state == -1 and deviation <= target
                    ):
                        state = 0

            signals[i] = float(state)

        return signals, size_weights
