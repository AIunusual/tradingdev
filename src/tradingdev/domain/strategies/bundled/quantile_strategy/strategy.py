"""XGBoost regime-detection volume strategy.

Classifies the current market regime into four states:
- long_only: only upside opportunity (enter long)
- short_only: only downside opportunity (enter short)
- both: both directions possible (skip — direction is random)
- neither: no opportunity (skip)

Only enters when the model is confident that exactly one direction
has a profit opportunity, avoiding the dangerous "both" regime
where direction is a coin flip.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed

from tradingdev.domain.ml.features.quantile_features import QuantileFeatureEngineer
from tradingdev.domain.optimization.grid_search import (
    finite_metric_value,
    is_better_metric,
    metric_direction,
    tuple_grid,
)
from tradingdev.domain.strategies.base import BaseStrategy
from tradingdev.shared.utils.logger import setup_logger
from tradingdev.shared.utils.parallel import estimate_n_jobs

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tradingdev.domain.backtest.base_engine import BaseBacktestEngine
    from tradingdev.domain.backtest.schemas import ParallelConfig
    from tradingdev.domain.strategies.bundled.quantile_strategy.config import (
        QuantileStrategyConfig,
    )

logger = setup_logger(__name__)

# Grid search: (horizon, min_confidence, edge_for_full_size)
_ParamTuple = tuple[int, float, float]

# Regime class labels
_LONG_ONLY = 0
_SHORT_ONLY = 1
_BOTH = 2
_NEITHER = 3


class _RegimePredictor(Protocol):
    """Classifier interface needed by regime scoring."""

    def predict_proba(
        self, features: NDArray[np.float64], /
    ) -> NDArray[np.float32] | NDArray[np.float64]:
        """Return one four-class probability vector per feature row."""
        ...


def _predict_probabilities(
    model: _RegimePredictor, features: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Normalize model precision and reject malformed predictions."""
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    if probabilities.shape != (len(features), 4):
        raise ValueError(
            "Regime classifier probabilities must have shape "
            f"({len(features)}, 4); got {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("Regime classifier probabilities must be finite")
    return probabilities


def _timestamps(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Require ordered, unique time identities before combining scoring history."""
    if "timestamp" in frame.columns:
        stamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
    elif isinstance(frame.index, pd.DatetimeIndex):
        stamps = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
    else:
        raise ValueError("Quantile scoring requires timestamps on both frames")
    if stamps.hasnans or not stamps.is_monotonic_increasing or not stamps.is_unique:
        raise ValueError(
            "Quantile timestamps must be non-missing and strictly increasing"
        )
    return stamps


def _train_regime_classifier(
    feat_df: pd.DataFrame,
    feature_names: list[str],
    n_estimators: int = 50,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    subsample: float = 0.7,
    random_seed: int = 42,
) -> _RegimePredictor:
    """Train a 4-class regime classifier.

    Ensures all 4 classes are present by adding a tiny synthetic
    sample for any missing class (prevents XGBoost ValueError).
    """
    x: NDArray[np.float64] = feat_df[feature_names].to_numpy(dtype=np.float64)
    y: NDArray[np.int_] = feat_df["target_regime"].to_numpy(dtype=np.int_)

    # Ensure all 4 classes exist in training data
    present = set(np.unique(y))
    if present != {0, 1, 2, 3}:
        missing = {0, 1, 2, 3} - present
        x_pad = np.tile(x[:1], (len(missing), 1))
        y_pad = np.array(sorted(missing), dtype=int)
        x = np.concatenate([x, x_pad])
        y = np.concatenate([y, y_pad])

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        num_class=4,
        objective="multi:softprob",
        eval_metric="mlogloss",
        verbosity=0,
        random_state=random_seed,
    )
    model.fit(x, y, verbose=False)
    return model


def _run_regime_state_machine(
    close: NDArray[np.float64],
    proba: NDArray[np.float64],
    horizon: int,
    min_holding_bars: int,
    min_confidence: float,
    max_p_both: float,
    max_p_neither: float,
    strategy_sl: float,
    dynamic_sizing: bool,
    min_weight: float,
    edge_for_full_size: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """State machine using regime classification probabilities.

    Entry: P(long_only) or P(short_only) > min_confidence,
    AND P(both) < max_p_both (avoid coin-flip regimes),
    AND P(neither) < max_p_neither (avoid dead markets).
    Size: proportional to P(signal) - P(both).
    Missing probability rows flatten the position and reset all holding state.

    Returns:
        Tuple of (signals, size_weights).
    """
    n = len(close)
    signals = np.zeros(n, dtype=np.float64)
    size_weights = np.ones(n, dtype=np.float64)

    state = 0
    bars_in_pos = 0
    entry_price = 0.0
    current_weight = 1.0
    ready = np.isfinite(proba).all(axis=1)

    for i in range(n):
        if not ready[i]:
            state = 0
            bars_in_pos = 0
            entry_price = 0.0
            current_weight = 1.0
            continue

        p_lo = proba[i, _LONG_ONLY]
        p_so = proba[i, _SHORT_ONLY]
        p_both = proba[i, _BOTH]
        p_neither = proba[i, _NEITHER]

        if state == 0:
            # Filter: skip dangerous or dead regimes
            if p_both > max_p_both or p_neither > max_p_neither:
                signals[i] = 0.0
                continue

            long_edge = p_lo - p_both
            short_edge = p_so - p_both

            want_long = p_lo > min_confidence and long_edge > 0
            want_short = p_so > min_confidence and short_edge > 0

            if want_long and (not want_short or long_edge > short_edge):
                state = 1
                bars_in_pos = 0
                entry_price = close[i]
                if dynamic_sizing and edge_for_full_size > 0:
                    raw_w = long_edge / edge_for_full_size
                    current_weight = min(max(raw_w, min_weight), 1.0)
                else:
                    current_weight = 1.0
                size_weights[i] = current_weight
            elif want_short:
                state = -1
                bars_in_pos = 0
                entry_price = close[i]
                if dynamic_sizing and edge_for_full_size > 0:
                    raw_w = short_edge / edge_for_full_size
                    current_weight = min(max(raw_w, min_weight), 1.0)
                else:
                    current_weight = 1.0
                size_weights[i] = current_weight
        else:
            bars_in_pos += 1
            size_weights[i] = current_weight

            if bars_in_pos < min_holding_bars:
                signals[i] = float(state)
                continue

            # Strategy stop-loss
            if entry_price > 0 and strategy_sl > 0:
                pnl_pct = state * (close[i] - entry_price) / entry_price
                if pnl_pct < -strategy_sl:
                    state = 0
                    signals[i] = 0.0
                    continue

            # Horizon exit
            if bars_in_pos >= horizon:
                state = 0
                signals[i] = 0.0
                continue

        signals[i] = float(state)

    return signals, size_weights


def _evaluate_regime_combo(
    df: pd.DataFrame,
    feature_names: list[str],
    model: _RegimePredictor,
    config: QuantileStrategyConfig,
    params: _ParamTuple,
    engine: BaseBacktestEngine,
    source_frame: pd.DataFrame,
) -> tuple[_ParamTuple, dict[str, Any]]:
    """Evaluate a parameter combination on every source bar, including gaps."""
    horizon, min_confidence, edge_for_full_size = params

    x: NDArray[np.float64] = df[feature_names].to_numpy(dtype=np.float64)
    proba: NDArray[np.float64] = np.full((len(source_frame), 4), np.nan)
    positions = df.index.to_numpy(dtype=np.intp)
    if len(df):
        proba[positions] = _predict_probabilities(model, x)
    close_arr: NDArray[np.float64] = source_frame["close"].to_numpy(dtype=np.float64)

    dyn = config.dynamic_sizing
    min_weight = config.min_position_size / config.position_size if dyn else 0.0

    signals, size_weights = _run_regime_state_machine(
        close=close_arr,
        proba=proba,
        horizon=horizon,
        min_holding_bars=config.min_holding_bars,
        min_confidence=min_confidence,
        max_p_both=config.max_p_both,
        max_p_neither=config.max_p_neither,
        strategy_sl=config.strategy_sl,
        dynamic_sizing=dyn,
        min_weight=min_weight,
        edge_for_full_size=edge_for_full_size,
    )

    sig_df = source_frame.copy()
    sig_df["signal"] = signals
    sig_df["size_weight"] = size_weights

    result = engine.run(sig_df)
    return params, result.metrics


class QuantileStrategy(BaseStrategy):
    """Regime-detection strategy for volume trading.

    Classifies the market into four regimes (long_only, short_only,
    both, neither) and only enters when exactly one direction has
    a profitable opportunity.  Avoids the dangerous "both" regime
    where direction is essentially random.

    Supports chunked rolling retrain.
    """

    def __init__(
        self,
        config: QuantileStrategyConfig,
        backtest_engine: BaseBacktestEngine | None = None,
        parallel_config: ParallelConfig | None = None,
    ) -> None:
        self._config = config
        self._backtest_engine = backtest_engine
        self._parallel_config = parallel_config

        self._feature_eng: QuantileFeatureEngineer | None = None
        self._regime_model: _RegimePredictor | None = None
        self._feature_names: list[str] = []

        # Best params from fit()
        self._best_horizon: int = config.horizon
        self._best_min_confidence: float = config.min_entry_edge
        self._best_edge_for_full_size: float = config.edge_for_full_size

        # Train data for rolling retrain
        self._train_df: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame) -> None:
        """Train regime classifier and grid-search parameters."""
        if self._backtest_engine is None:
            logger.warning("No backtest engine; skipping fit()")
            return

        cfg = self._config
        df = df.assign(timestamp=_timestamps(df)).reset_index(drop=True)
        self._train_df = df.copy()

        horizons = cfg.horizon_candidates or [cfg.horizon]
        confidence_candidates = cfg.min_entry_edge_candidates or [cfg.min_entry_edge]
        efs_candidates: list[float] = (
            (cfg.edge_for_full_size_candidates or [cfg.edge_for_full_size])
            if cfg.dynamic_sizing
            else [cfg.edge_for_full_size]
        )

        best_value: float | None = None
        best_horizon = cfg.horizon
        best_confidence = cfg.min_entry_edge
        best_efs = cfg.edge_for_full_size
        target = cfg.target_metric
        direction = metric_direction(target)
        min_mp = cfg.min_monthly_pnl

        best_model: _RegimePredictor | None = None
        best_fe: QuantileFeatureEngineer | None = None
        best_fnames: list[str] = []

        for h in horizons:
            logger.info("Training regime classifier for horizon=%d", h)
            fe = QuantileFeatureEngineer(
                horizon=h,
                profit_target=cfg.profit_target,
            )
            feat_df = fe.transform(
                df,
                include_target=True,
                target_type="regime",
            )
            fnames = [
                f
                for f in fe.get_feature_names()
                if f not in ("target_regime", "target_long", "target_short", "target")
            ]

            regime_model = _train_regime_classifier(
                feat_df,
                fnames,
                n_estimators=cfg.n_estimators,
                max_depth=cfg.max_depth,
                learning_rate=cfg.learning_rate,
                subsample=cfg.subsample,
                random_seed=cfg.random_seed,
            )
            # Evaluation does not need labels: immature tail observations can
            # still be scored, while feature gaps retain their source positions.
            scoring_feat = fe.transform(df, include_target=False)

            # Log class distribution
            y: NDArray[np.int_] = feat_df["target_regime"].to_numpy(dtype=np.int_)
            for cls, name in [
                (0, "long_only"),
                (1, "short_only"),
                (2, "both"),
                (3, "neither"),
            ]:
                logger.info(
                    "  %s: %.1f%%",
                    name,
                    float(np.mean(y == cls)) * 100,
                )

            grid: list[_ParamTuple] = [
                cast("_ParamTuple", combo)
                for combo in tuple_grid(
                    [h],
                    confidence_candidates,
                    efs_candidates,
                )
            ]

            p_cfg = self._parallel_config
            n_jobs = estimate_n_jobs(
                df,
                safety_factor=p_cfg.safety_factor if p_cfg else 0.6,
                overhead_multiplier=(p_cfg.overhead_multiplier if p_cfg else 3.0),
                reserve_cores=p_cfg.reserve_cores if p_cfg else 2,
            )
            logger.info(
                "Grid search (horizon=%d): %d combos (n_jobs=%d)",
                h,
                len(grid),
                n_jobs,
            )

            results: list[tuple[_ParamTuple, dict[str, Any]]] = Parallel(
                n_jobs=n_jobs,
            )(
                delayed(_evaluate_regime_combo)(
                    scoring_feat,
                    fnames,
                    regime_model,
                    cfg,
                    combo,
                    self._backtest_engine,
                    df,
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
                    best_horizon = params[0]
                    best_confidence = params[1]
                    best_efs = params[2]
                    best_model = regime_model
                    best_fe = fe
                    best_fnames = fnames

            if n_filtered > 0:
                logger.info(
                    "horizon=%d: %d/%d filtered",
                    h,
                    n_filtered,
                    len(grid),
                )

        if best_value is None:
            raise ValueError(
                "No quantile parameter combination has a finite eligible objective"
            )

        self._best_horizon = best_horizon
        self._best_min_confidence = best_confidence
        self._best_edge_for_full_size = best_efs
        self._regime_model = best_model
        self._feature_eng = best_fe
        self._feature_names = best_fnames

        logger.info(
            "Fit complete: horizon=%d, confidence=%.2f, efs=%.4f (%s=%.4f)",
            best_horizon,
            best_confidence,
            best_efs,
            target,
            best_value,
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals with chunked rolling retrain.

        Raises:
            RuntimeError: If :meth:`fit` has not produced a regime model.
                Training on the frame under evaluation would let earlier
                signals use later data, so the call is refused instead of
                silently reusing the frame as its own training set.
        """
        cfg = self._config

        if (
            self._regime_model is None
            or self._feature_eng is None
            or self._train_df is None
        ):
            msg = (
                "QuantileStrategy has no trained regime model: call fit(train_df) "
                "with a backtest engine before generate_signals(). Training on the "
                "frame being evaluated would let earlier signals use later data."
            )
            raise RuntimeError(msg)

        # --- Chunked rolling retrain ---
        horizon = self._best_horizon
        retrain_interval = cfg.retrain_interval
        train_window = cfg.train_window

        # Only the part of the training frame that precedes the scored frame is
        # history. ``walk_forward`` scores the training frame itself, so keeping
        # overlapping rows would concatenate it with its own copy: the prediction
        # context would span both copies, the duplicate timestamps would survive
        # the chunk filter, and the state machine would read late bars while
        # writing from bar 0.
        history = self._train_df
        scored_stamps = _timestamps(df)
        history_stamps = _timestamps(history)
        if not len(df):
            return df.assign(signal=0.0, size_weight=1.0)
        history = history.assign(timestamp=history_stamps)
        history = history[history_stamps < scored_stamps[0]]

        combined = pd.concat(
            [history, df.assign(timestamp=scored_stamps)],
            ignore_index=True,
        )
        test_offset = len(history)
        test_len = len(df)

        chunk_starts = list(range(0, test_len, retrain_interval))
        all_proba: NDArray[np.float64] = np.full((test_len, 4), np.nan)

        dyn = cfg.dynamic_sizing
        min_weight = cfg.min_position_size / cfg.position_size if dyn else 0.0

        for ci, chunk_start in enumerate(chunk_starts):
            chunk_end = min(chunk_start + retrain_interval, test_len)
            abs_start = test_offset + chunk_start
            abs_end = test_offset + chunk_end

            train_begin = max(0, abs_start - train_window)
            train_slice = combined.iloc[train_begin:abs_start].copy()

            regime_model = self._regime_model
            fnames = self._feature_names
            fe = self._feature_eng
            if len(train_slice) >= 200:
                logger.info(
                    "Chunk %d/%d: retrain on %d bars",
                    ci + 1,
                    len(chunk_starts),
                    len(train_slice),
                )
                retrain_fe = QuantileFeatureEngineer(
                    horizon=horizon,
                    profit_target=cfg.profit_target,
                )
                train_feat = retrain_fe.transform(
                    train_slice,
                    include_target=True,
                    target_type="regime",
                )
                # Warm-up and unknown horizon labels can remove every raw row.
                # Until observations mature, retain the model produced by fit().
                if not train_feat.empty:
                    regime_model = _train_regime_classifier(
                        train_feat,
                        fnames,
                        n_estimators=cfg.n_estimators,
                        max_depth=cfg.max_depth,
                        learning_rate=cfg.learning_rate,
                        subsample=cfg.subsample,
                        random_seed=cfg.random_seed,
                    )
                    fe = retrain_fe

            # Predict on chunk with context
            context_start = max(0, abs_start - 60)
            predict_ctx = combined.iloc[context_start:abs_end].reset_index(drop=True)
            predict_feat = fe.transform(
                predict_ctx,
                include_target=False,
            )
            offset = abs_start - context_start
            positions = predict_feat.index.to_numpy(dtype=np.intp) - offset
            keep = positions >= 0
            predict_feat = predict_feat.loc[keep]
            positions = positions[keep]
            if not len(predict_feat):
                continue

            x: NDArray[np.float64] = predict_feat[fnames].to_numpy(dtype=np.float64)
            all_proba[chunk_start + positions] = _predict_probabilities(regime_model, x)

        signals, size_weights = _run_regime_state_machine(
            close=df["close"].to_numpy(dtype=np.float64),
            proba=all_proba,
            horizon=horizon,
            min_holding_bars=cfg.min_holding_bars,
            min_confidence=self._best_min_confidence,
            max_p_both=cfg.max_p_both,
            max_p_neither=cfg.max_p_neither,
            strategy_sl=cfg.strategy_sl,
            dynamic_sizing=dyn,
            min_weight=min_weight,
            edge_for_full_size=self._best_edge_for_full_size,
        )
        return df.assign(signal=signals, size_weight=size_weights)

    def get_parameters(self) -> dict[str, Any]:
        """Return strategy parameters."""
        params = self._config.model_dump()
        params["best_horizon"] = self._best_horizon
        params["best_min_confidence"] = self._best_min_confidence
        params["best_edge_for_full_size"] = self._best_edge_for_full_size
        return params
