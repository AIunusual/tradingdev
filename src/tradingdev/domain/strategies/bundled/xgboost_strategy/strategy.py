"""XGBoost-based ML trading strategy with rolling retraining."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd  # noqa: TC002
from sklearn.metrics import accuracy_score

from tradingdev.domain.ml.features.features import FeatureEngineer
from tradingdev.domain.ml.models.xgboost_model import XGBoostDirectionModel
from tradingdev.domain.ml.retrainer import (
    RollingRetrainer,
)
from tradingdev.domain.ml.threshold_optimizer import (
    ThresholdOptimizer,
)
from tradingdev.domain.strategies.base import BaseStrategy
from tradingdev.shared.utils.logger import setup_logger

if TYPE_CHECKING:
    from tradingdev.domain.backtest.base_engine import (
        BaseBacktestEngine,
    )
    from tradingdev.domain.strategies.bundled.xgboost_strategy.config import (
        XGBoostStrategyConfig,
    )

logger = setup_logger(__name__)


class XGBoostStrategy(BaseStrategy):
    """XGBoost direction prediction strategy."""

    def __init__(
        self,
        config: XGBoostStrategyConfig,
        backtest_engine: BaseBacktestEngine | None = None,
    ) -> None:
        self._config = config
        self._backtest_engine = backtest_engine
        self._model: XGBoostDirectionModel | None = None
        self._feature_engineer: FeatureEngineer | None = None
        self._best_lookback: int | None = None
        self._best_threshold: float = config.signal_threshold
        self._train_data: pd.DataFrame | None = None
        self._rolling_trace: dict[str, Any] | None = None

    def fit(self, df: pd.DataFrame) -> None:
        """Optimise lookback and train the XGBoost model."""
        val_ratio = self._config.validation_ratio
        split_idx = int(len(df) * (1 - val_ratio))
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        best_lookback, best_model, best_fe = self._search_lookback(train_df, val_df)

        self._best_lookback = best_lookback
        self._feature_engineer = FeatureEngineer(
            lookback=best_lookback,
            target_horizon=self._config.target_horizon,
        )

        # Threshold search (volume-maximisation mode)
        if (
            self._config.signal_threshold_candidates
            and self._backtest_engine is not None
        ):
            optimizer = ThresholdOptimizer(
                engine=self._backtest_engine,
                min_bars_between_trades=(self._config.min_bars_between_trades),
            )
            self._best_threshold = optimizer.search(
                val_df,
                best_model,
                self._config.signal_threshold_candidates,
                self._config.signal_threshold,
                feature_engineer=best_fe,
            )

        # Retrain final model on full data
        feat_full = self._feature_engineer.transform(df, include_target=True)
        self._model = XGBoostDirectionModel(config=self._config.model)
        self._model.train(feat_full)
        self._train_data = df.copy()

        logger.info(
            "XGBoost fit complete: best_lookback=%d, threshold=%.2f",
            best_lookback,
            self._best_threshold,
        )

    def _search_lookback(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> tuple[
        int,
        XGBoostDirectionModel,
        FeatureEngineer,
    ]:
        """Find the best lookback by validation accuracy."""
        best_score = -1.0
        best_lb: int | None = None
        best_model: XGBoostDirectionModel | None = None
        best_fe: FeatureEngineer | None = None

        for lb in self._config.lookback_candidates:
            fe = FeatureEngineer(
                lookback=lb,
                target_horizon=self._config.target_horizon,
            )
            feat_train = fe.transform(train_df, include_target=True)
            feat_val = fe.transform(val_df, include_target=True)

            if len(feat_train) < 10 or len(feat_val) < 5:
                continue

            model = XGBoostDirectionModel(config=self._config.model)
            model.train(feat_train, eval_df=feat_val)

            preds = model.predict(feat_val)
            score = float(accuracy_score(feat_val["target"], preds))
            logger.info(
                "Lookback %d: val accuracy=%.4f",
                lb,
                score,
            )

            if score > best_score:
                best_score = score
                best_lb = lb
                best_model = model
                best_fe = fe

        if best_lb is None or best_model is None or best_fe is None:
            raise ValueError(
                "No lookback candidate has sufficient mature training and validation "
                "samples: at least 10 training rows and 5 validation rows are required "
                f"after feature and label filtering (target_horizon="
                f"{self._config.target_horizon})"
            )
        return best_lb, best_model, best_fe

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals via rolling prediction."""
        if (
            self._model is None
            or self._feature_engineer is None
            or self._best_lookback is None
            or self._train_data is None
        ):
            msg = "Strategy not fitted. Call fit() first."
            raise RuntimeError(msg)

        if self._config.rolling_gen.enabled:
            # Imported here so that a defect in the optional engine cannot stop
            # the classic path below from loading and running.
            from tradingdev.domain.ml.rolling_gen import (
                rolling_outcome,
                signals_from_probabilities,
                trace_payload,
            )

            outcome = rolling_outcome(
                self._train_data,
                df,
                self._config.rolling_gen.resolve(),
                model=self._model,
                model_config=self._config.model,
                fe=self._feature_engineer,
                lookback=self._best_lookback,
                retrain_interval=self._config.retrain_interval,
                label_horizon=self._config.target_horizon,
            )
            self._rolling_trace = trace_payload(outcome)
            signals = signals_from_probabilities(
                outcome.probabilities,
                threshold=self._best_threshold,
                cooldown=self._config.min_bars_between_trades,
            )
            result = df.copy()
            result["signal"] = signals
            return result

        retrainer = RollingRetrainer(
            model_config=self._config.model,
            retrain_interval=self._config.retrain_interval,
            threshold=self._best_threshold,
            cooldown=self._config.min_bars_between_trades,
            lookback=self._best_lookback,
        )
        signals = retrainer.run(
            df,
            self._train_data,
            self._model,
            self._feature_engineer,
        )

        result = df.copy()
        result["signal"] = signals
        return result

    def get_parameters(self) -> dict[str, Any]:
        """Return strategy parameters."""
        params = self._config.model_dump()
        params["best_lookback"] = self._best_lookback
        params["best_threshold"] = self._best_threshold
        return params

    def rolling_trace(self) -> dict[str, Any] | None:
        """Return the rolls that scored the last run, if the switch was on."""
        return self._rolling_trace
