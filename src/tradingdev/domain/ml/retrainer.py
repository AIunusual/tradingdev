"""Rolling retraining loop for ML-based signal generation."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

from tradingdev.domain.ml.models.xgboost_model import XGBoostDirectionModel
from tradingdev.domain.ml.scoring_history import align_scoring_history
from tradingdev.shared.utils.logger import setup_logger

if TYPE_CHECKING:
    from tradingdev.domain.ml.features.features import FeatureEngineer
    from tradingdev.domain.ml.schemas import XGBoostModelConfig

logger = setup_logger(__name__)


class RollingRetrainer:
    """Generate signals via rolling prediction and retraining."""

    def __init__(
        self,
        model_config: XGBoostModelConfig,
        retrain_interval: int,
        threshold: float,
        cooldown: int,
        lookback: int,
    ) -> None:
        self._model_config = model_config
        self._retrain_interval = retrain_interval
        self._threshold = threshold
        self._cooldown = cooldown
        self._lookback = lookback

    def run(
        self,
        test_df: pd.DataFrame,
        train_data: pd.DataFrame,
        model: XGBoostDirectionModel,
        fe: FeatureEngineer,
    ) -> npt.NDArray[np.intp]:
        """Run the rolling retraining loop.

        Args:
            test_df: Test-period OHLCV DataFrame.
            train_data: Historical data for retraining context.
            model: Initially trained model.
            fe: Feature engineer instance.

        Returns:
            Array of signal values (1, -1, 0).

        Raises:
            ValueError: If the frames have no usable time identity, because the
                overlap with the training history cannot be detected without it.
            ValueError: If a non-empty scored frame leaves every bar without a
                usable feature row, because all-zero signals would otherwise be
                reported as a strategy that simply never trades.
        """
        history, scored = align_scoring_history(train_data, test_df)
        window_size = max(self._lookback * 10, len(history))
        combined = pd.concat([history, scored], ignore_index=True)
        test_offset = len(history)

        total_bars = len(scored)
        signals = np.zeros(total_bars, dtype=int)
        current_model = model
        last_trade_bar = -self._cooldown
        scored_bars = 0

        total_retrains = (total_bars - 1) // self._retrain_interval + 1
        retrain_count = 0
        t_start = time.monotonic()
        log_interval = max(total_bars // 20, 1)

        logger.info(
            "[signals] Starting: %d bars, retrain every %d "
            "bars (~%d retrains), threshold=%.2f",
            total_bars,
            self._retrain_interval,
            total_retrains,
            self._threshold,
        )

        for i in range(total_bars):
            combined_idx = test_offset + i

            if i % self._retrain_interval == 0:
                retrain_count += 1
                current_model = self._retrain(
                    combined,
                    combined_idx,
                    window_size,
                    fe,
                    retrain_count,
                    total_retrains,
                    i,
                    total_bars,
                    current_model,
                )

            if i > 0 and i % log_interval == 0:
                self._log_progress(i, total_bars, t_start)

            if i - last_trade_bar < self._cooldown:
                continue

            features = self._bar_features(combined, combined_idx, fe)
            if features is None:
                continue
            scored_bars += 1

            signal = self._signal(features, current_model)
            if signal != 0:
                signals[i] = signal
                last_trade_bar = i

        if total_bars and not scored_bars:
            needed = fe.context_rows
            available = len(combined)
            msg = (
                f"rolling prediction scored none of {total_bars} bars: the "
                f"feature engineer needs {needed} rows for a complete feature "
                f"row, and the history plus the scored frame hold {available}"
            )
            if available < needed:
                msg += "; provide more history before the scored window"
            else:
                msg += (
                    "; that is enough history, so the features are incomplete "
                    "for another reason (a flat price series leaves the "
                    "Bollinger width undefined, for example)"
                )
            raise ValueError(msg)

        elapsed = time.monotonic() - t_start
        n_signals = int(np.count_nonzero(signals))
        logger.info(
            "[signals] Complete: %d bars in %.1fs (%.0f bars/s), %d scored, "
            "%d signals emitted",
            total_bars,
            elapsed,
            total_bars / elapsed if elapsed > 0 else 0,
            scored_bars,
            n_signals,
        )
        return signals

    def _retrain(
        self,
        combined: pd.DataFrame,
        combined_idx: int,
        window_size: int,
        fe: FeatureEngineer,
        retrain_count: int,
        total_retrains: int,
        bar: int,
        total_bars: int,
        fallback_model: XGBoostDirectionModel,
    ) -> XGBoostDirectionModel:
        logger.info(
            "[signals] Retrain %d/%d at bar %d/%d",
            retrain_count,
            total_retrains,
            bar,
            total_bars,
        )
        start = max(0, combined_idx - window_size)
        window = combined.iloc[start:combined_idx].copy()

        if len(window) > self._lookback + 10:
            feat = fe.transform(window, include_target=True)
            if len(feat) >= 10:
                model = XGBoostDirectionModel(
                    config=self._model_config,
                )
                model.train(feat)
                return model
        return fallback_model

    def _bar_features(
        self,
        combined: pd.DataFrame,
        combined_idx: int,
        fe: FeatureEngineer,
    ) -> pd.DataFrame | None:
        """Return the feature row for ``combined_idx``, or None if it is unusable.

        The context must cover the features' own warm-up: the fixed indicators
        need a fixed number of bars whatever the lookback is, so sizing it as a
        multiple of the lookback alone left a short lookback with an empty frame
        and no prediction at all.  ``2 x lookback + 1`` stays as a floor so that
        a lookback large enough to cover the warm-up keeps the context, and
        therefore the seeded indicator values, it already had.
        """
        ctx_rows = max(2 * self._lookback + 1, fe.context_rows)
        ctx_start = max(0, combined_idx - ctx_rows + 1)
        ctx = combined.iloc[ctx_start : combined_idx + 1].copy()
        feat_pred = fe.transform(ctx, include_target=False)

        if combined_idx not in feat_pred.index:
            return None
        return feat_pred.loc[[combined_idx]]

    def _signal(
        self,
        features: pd.DataFrame,
        model: XGBoostDirectionModel,
    ) -> int:
        proba = model.predict_proba(features)
        p_long = float(proba[1].iloc[0]) if 1 in proba.columns else 0.0
        p_short = float(proba[-1].iloc[0]) if -1 in proba.columns else 0.0

        if p_long >= self._threshold and p_long >= p_short:
            return 1
        if p_short >= self._threshold:
            return -1
        return 0

    @staticmethod
    def _log_progress(i: int, total: int, t_start: float) -> None:
        elapsed = time.monotonic() - t_start
        bars_per_sec = i / elapsed
        eta = (total - i) / bars_per_sec
        logger.info(
            "[signals] Progress: %d/%d (%.1f%%) | %.0f bars/s | ETA %.0fs",
            i,
            total,
            i / total * 100,
            bars_per_sec,
            eta,
        )
