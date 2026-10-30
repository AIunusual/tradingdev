"""Walk-forward validation for trading strategies."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tradingdev.domain.strategies.signals import generate_checked_signals
from tradingdev.domain.validation.config import PurgedWalkForwardConfig
from tradingdev.domain.validation.leakage_gate import (
    WalkForwardLeakageError,
    audit_fold_windows,
)
from tradingdev.domain.validation.purged_splitter import PurgedDataSplitter
from tradingdev.domain.validation.splitter import DataSplitter
from tradingdev.shared.utils.logger import setup_logger

if TYPE_CHECKING:
    from datetime import datetime

    import pandas as pd

    from tradingdev.domain.backtest.base_engine import (
        BaseBacktestEngine,
    )
    from tradingdev.domain.backtest.result import BacktestResult
    from tradingdev.domain.strategies.base import BaseStrategy
    from tradingdev.domain.validation.config import WalkForwardConfig

logger = setup_logger(__name__)


def _attach_rolling_trace(
    strategy: BaseStrategy,
    result: BacktestResult,
) -> None:
    """Store a strategy's rolling provenance beside the run's other metadata.

    The trace describes the signals just generated, so it is read before the next
    frame replaces it. A strategy without a rolling schedule returns ``None`` and
    leaves the result untouched.
    """
    trace = strategy.rolling_trace()
    if trace is not None:
        result.metric_metadata["rolling_trace"] = deepcopy(trace)


@dataclass
class WalkForwardResult:
    """Result from a single walk-forward fold."""

    fold_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_metrics: dict[str, Any] = field(default_factory=dict)
    test_metrics: dict[str, Any] = field(default_factory=dict)
    strategy_params: dict[str, Any] = field(default_factory=dict)
    train_backtest: BacktestResult | None = None
    test_backtest: BacktestResult | None = None


class WalkForwardValidator:
    """Orchestrate walk-forward validation cycles."""

    def __init__(
        self,
        config: WalkForwardConfig,
        engine: BaseBacktestEngine,
    ) -> None:
        self._config = config
        self._engine = engine
        self._splitter: DataSplitter | PurgedDataSplitter = (
            PurgedDataSplitter(config)
            if isinstance(config, PurgedWalkForwardConfig)
            else DataSplitter(config)
        )

    def validate(
        self,
        strategy: BaseStrategy,
        df: pd.DataFrame,
    ) -> list[WalkForwardResult]:
        """Run walk-forward validation.

        Args:
            strategy: Strategy instance (mutated by fit()).
            df: Full OHLCV DataFrame covering all periods.

        Returns:
            List of WalkForwardResult, one per fold.

        Raises:
            WalkForwardLeakageError: If ``config.label_horizon`` is set and the
                folds leak label information into training.
        """
        splits = self._splitter.split(df)
        self._check_leakage(df)

        results: list[WalkForwardResult] = []

        for i, (train_df, test_df) in enumerate(splits):
            result = self.run_fold(i, strategy, train_df, test_df)
            results.append(result)

        return results

    def _check_leakage(self, df: pd.DataFrame) -> None:
        """Block the run when training bars can see their own test window.

        Zero ``label_horizon`` disables this audit; it does not
        establish that the strategy has no forward-looking labels.
        """
        horizon = self._config.label_horizon
        if horizon < 1:
            return
        windows = self._splitter.split_windows(df)
        violations = audit_fold_windows(windows, horizon=horizon)
        if violations:
            logger.error(
                "Walk-forward leakage gate blocked %d folds: %s",
                len(windows),
                "; ".join(violations),
            )
            raise WalkForwardLeakageError(violations)
        logger.info(
            "Walk-forward leakage gate passed: %d folds, horizon=%d",
            len(windows),
            horizon,
        )

    def run_fold(
        self,
        fold_idx: int,
        strategy: BaseStrategy,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> WalkForwardResult:
        """Execute a single validation fold."""
        train_start = train_df["timestamp"].iloc[0]
        train_end = train_df["timestamp"].iloc[-1]
        test_start = test_df["timestamp"].iloc[0]
        test_end = test_df["timestamp"].iloc[-1]

        logger.info(
            "Fold %d: train [%s ~ %s] -> test [%s ~ %s]",
            fold_idx,
            train_start,
            train_end,
            test_start,
            test_end,
        )

        strategy.fit(train_df)

        train_signals = generate_checked_signals(strategy, train_df)
        train_result = self._engine.run(train_signals)
        train_metrics = train_result.metrics
        # Captured per frame, before the next call replaces it: the trace belongs
        # to the signals that were just generated, not to the fold as a whole.
        _attach_rolling_trace(strategy, train_result)

        test_signals = generate_checked_signals(strategy, test_df)
        test_result = self._engine.run(test_signals)
        test_metrics = test_result.metrics
        _attach_rolling_trace(strategy, test_result)

        logger.info(
            "Fold %d: train %s=%s, test %s=%s",
            fold_idx,
            self._config.target_metric,
            train_metrics.get(self._config.target_metric),
            self._config.target_metric,
            test_metrics.get(self._config.target_metric),
        )

        return WalkForwardResult(
            fold_index=fold_idx,
            train_start=train_start.to_pydatetime(),
            train_end=train_end.to_pydatetime(),
            test_start=test_start.to_pydatetime(),
            test_end=test_end.to_pydatetime(),
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            strategy_params=deepcopy(strategy.get_parameters()),
            train_backtest=train_result,
            test_backtest=test_result,
        )
