"""Tests for the opt-in purged walk-forward splitter.

The point of this splitter is that it satisfies the leakage gate *by
construction*. These tests prove that with the gate itself as the oracle, and
always run the plain splitter side by side so a green result cannot come from
a gate that simply stopped catching anything.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tradingdev.domain.backtest.signal_engine import SignalBacktestEngine
from tradingdev.domain.strategies.bundled.kd_strategy.config import KDStrategyConfig
from tradingdev.domain.strategies.bundled.kd_strategy.strategy import KDStrategy
from tradingdev.domain.validation.config import (
    AutomaticWalkForwardConfig,
    PurgedWalkForwardConfig,
)
from tradingdev.domain.validation.leakage_gate import (
    WalkForwardLeakageError,
    audit_fold_windows,
)
from tradingdev.domain.validation.purged_splitter import PurgedDataSplitter
from tradingdev.domain.validation.splitter import DataSplitter
from tradingdev.domain.validation.walk_forward import WalkForwardValidator

if TYPE_CHECKING:
    import pandas as pd

    from tradingdev.domain.validation.leakage_gate import FoldWindow


def _purged_windows(
    df: pd.DataFrame, *, horizon: int, embargo: int = 0, n_splits: int = 3
) -> list[FoldWindow]:
    config = PurgedWalkForwardConfig(
        mode="purged", n_splits=n_splits, label_horizon=horizon, embargo_bars=embargo
    )
    return PurgedDataSplitter(config).split_windows(df)


def _plain_windows(df: pd.DataFrame, *, n_splits: int = 3) -> list[FoldWindow]:
    config = AutomaticWalkForwardConfig(
        mode="automatic", n_splits=n_splits, train_ratio=0.7
    )
    return DataSplitter(config).split_windows(df)


class TestPurgedFoldsPassTheGate:
    @pytest.mark.parametrize("horizon", [1, 3, 5, 10])
    @pytest.mark.parametrize("embargo", [0, 2, 5])
    def test_purged_is_clean_where_plain_leaks(
        self, walk_forward_ohlcv_df: pd.DataFrame, horizon: int, embargo: int
    ) -> None:
        """The oracle test: same horizon, plain leaks and purged does not."""
        plain = _plain_windows(walk_forward_ohlcv_df)
        assert audit_fold_windows(plain, horizon=horizon), (
            "plain splitter must leak, otherwise this proof is vacuous"
        )
        purged = _purged_windows(
            walk_forward_ohlcv_df, horizon=horizon, embargo=embargo
        )
        assert audit_fold_windows(purged, horizon=horizon) == []

    @pytest.mark.parametrize("horizon", [1, 5, 10])
    def test_gap_is_at_least_horizon_plus_embargo(
        self, walk_forward_ohlcv_df: pd.DataFrame, horizon: int
    ) -> None:
        embargo = 4
        for train, test in _purged_windows(
            walk_forward_ohlcv_df, horizon=horizon, embargo=embargo
        ):
            assert test[0] - train[1] >= horizon + embargo

    def test_embargo_widens_the_gap(self, walk_forward_ohlcv_df: pd.DataFrame) -> None:
        narrow = _purged_windows(walk_forward_ohlcv_df, horizon=5, embargo=0)
        wide = _purged_windows(walk_forward_ohlcv_df, horizon=5, embargo=6)
        for (narrow_train, narrow_test), (wide_train, wide_test) in zip(
            narrow, wide, strict=True
        ):
            assert narrow_test[0] == wide_test[0]
            assert (wide_test[0] - wide_train[1]) - (
                narrow_test[0] - narrow_train[1]
            ) == 6

    def test_training_windows_still_precede_their_test(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        for train, test in _purged_windows(walk_forward_ohlcv_df, horizon=5):
            assert train[1] <= test[0]
            assert train[0] < train[1]
            assert test[0] < test[1]

    def test_fold_count_matches_n_splits(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        windows = _purged_windows(walk_forward_ohlcv_df, horizon=5, n_splits=4)
        assert len(windows) == 4


class TestPurgedSplitterContract:
    def test_requires_a_label_horizon(self) -> None:
        with pytest.raises(ValueError, match="label_horizon"):
            PurgedDataSplitter(
                PurgedWalkForwardConfig(mode="purged", n_splits=2, label_horizon=0)
            )

    def test_windows_match_the_frames(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        """Anti-drift lock, mirroring the one for DataSplitter."""
        config = PurgedWalkForwardConfig(
            mode="purged", n_splits=3, label_horizon=5, embargo_bars=2
        )
        splitter = PurgedDataSplitter(config)
        folds = splitter.split(walk_forward_ohlcv_df)
        windows = splitter.split_windows(walk_forward_ohlcv_df)
        assert len(folds) == len(windows)
        for (train_df, test_df), (train_win, test_win) in zip(
            folds, windows, strict=True
        ):
            assert walk_forward_ohlcv_df.iloc[train_win[0] : train_win[1]].index.equals(
                train_df.index
            )
            assert walk_forward_ohlcv_df.iloc[test_win[0] : test_win[1]].index.equals(
                test_df.index
            )


class TestValidatorWithPurgedSplit:
    def setup_method(self) -> None:
        self.engine = SignalBacktestEngine(init_cash=10_000.0, fees=0.0, slippage=0.0)

    def test_purged_run_completes_where_plain_is_blocked(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        """End-to-end: same label horizon, plain blocked and purged accepted."""
        blocked = WalkForwardValidator(
            config=AutomaticWalkForwardConfig(
                mode="automatic", n_splits=2, train_ratio=0.7, label_horizon=5
            ),
            engine=self.engine,
        )
        with pytest.raises(WalkForwardLeakageError):
            blocked.validate(
                KDStrategy(config=KDStrategyConfig()), walk_forward_ohlcv_df
            )

        purged = WalkForwardValidator(
            config=PurgedWalkForwardConfig(
                mode="purged", n_splits=2, label_horizon=5, embargo_bars=2
            ),
            engine=self.engine,
        )
        results = purged.validate(
            KDStrategy(config=KDStrategyConfig()), walk_forward_ohlcv_df
        )
        assert len(results) == 2

    def test_purged_split_without_horizon_is_a_config_error(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="label_horizon"):
            WalkForwardValidator(
                config=PurgedWalkForwardConfig.model_validate(
                    {"mode": "purged", "n_splits": 2}
                ),
                engine=self.engine,
            )
