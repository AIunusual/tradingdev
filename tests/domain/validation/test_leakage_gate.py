"""Tests for the walk-forward leakage gate.

The gate is a pure function of ``(windows, horizon)``, so the bulk of these
tests use small integer arrays and never touch a DataFrame, a strategy, or a
backtest engine. The last group locks ``DataSplitter.split`` and
``split_windows`` together so the audited geometry cannot drift from the
frames that actually get executed.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import pytest

from tradingdev.domain.backtest.schemas import WalkForwardConfig
from tradingdev.domain.backtest.signal_engine import SignalBacktestEngine
from tradingdev.domain.strategies.bundled.kd_strategy.config import KDStrategyConfig
from tradingdev.domain.strategies.bundled.kd_strategy.strategy import KDStrategy
from tradingdev.domain.validation.leakage_gate import (
    WalkForwardLeakageError,
    audit_fold_windows,
    enforce_no_leakage,
    fold_splits,
)
from tradingdev.domain.validation.splitter import DataSplitter
from tradingdev.domain.validation.walk_forward import WalkForwardValidator

if TYPE_CHECKING:
    import pandas as pd


class TestAuditFoldWindows:
    def test_naive_contiguous_fold_is_flagged(self) -> None:
        violations = audit_fold_windows([((0, 100), (100, 120))], horizon=5)
        assert len(violations) == 1
        assert "[95, 100)" in violations[0]
        assert "test start 100" in violations[0]

    def test_horizon_zero_disables_the_audit(self) -> None:
        assert audit_fold_windows([((0, 100), (100, 120))], horizon=0) == []

    def test_negative_horizon_disables_the_audit(self) -> None:
        assert audit_fold_windows([((0, 100), (100, 120))], horizon=-1) == []

    def test_a_gap_of_exactly_the_horizon_is_clean(self) -> None:
        """Purge width is derivable: a gap of `horizon` bars is sufficient."""
        assert audit_fold_windows([((0, 95), (100, 120))], horizon=5) == []

    def test_one_bar_short_of_the_horizon_is_flagged(self) -> None:
        violations = audit_fold_windows([((0, 96), (100, 120))], horizon=5)
        assert len(violations) == 1
        assert "[95, 96)" in violations[0]

    def test_a_wider_gap_is_clean(self) -> None:
        assert audit_fold_windows([((0, 80), (100, 120))], horizon=5) == []

    def test_every_fold_is_audited(self) -> None:
        violations = audit_fold_windows(
            [((0, 50), (50, 60)), ((0, 60), (60, 70))], horizon=3
        )
        assert len(violations) == 2
        assert "fold/0" in violations[0]
        assert "fold/1" in violations[1]

    def test_degenerate_windows_are_skipped(self) -> None:
        assert audit_fold_windows([((5, 5), (10, 20))], horizon=3) == []
        assert audit_fold_windows([((0, 10), (12, 12))], horizon=3) == []

    def test_empty_fold_list_is_clean(self) -> None:
        assert audit_fold_windows([], horizon=5) == []


class TestEnforceNoLeakage:
    def test_clean_folds_pass(self) -> None:
        enforce_no_leakage([((0, 90), (100, 120))], horizon=5)

    def test_leaking_folds_raise(self) -> None:
        with pytest.raises(WalkForwardLeakageError, match="leak label information"):
            enforce_no_leakage([((0, 100), (100, 120))], horizon=5)

    def test_error_carries_the_violations(self) -> None:
        with pytest.raises(WalkForwardLeakageError) as excinfo:
            enforce_no_leakage([((0, 100), (100, 120))], horizon=5)
        assert len(excinfo.value.violations) == 1
        assert "[95, 100)" in excinfo.value.violations[0]

    def test_disabled_horizon_never_raises(self) -> None:
        enforce_no_leakage([((0, 100), (100, 120))], horizon=0)


class TestFoldSplits:
    def test_converts_windows_into_named_splits(self) -> None:
        splits = fold_splits([((0, 10), (10, 20))])
        assert len(splits) == 1
        assert splits[0].name == "fold/0"
        assert splits[0].train == ((0, 10),)
        assert splits[0].test == (10, 20)

    def test_prefix_is_configurable(self) -> None:
        assert fold_splits([((0, 10), (10, 20))], name_prefix="wf")[0].name == "wf/0"


class TestSplitterWindowConsistency:
    """Lock `split` and `split_windows` together so the audit cannot drift."""

    @staticmethod
    def _assert_consistent(config: WalkForwardConfig, df: pd.DataFrame) -> None:
        splitter = DataSplitter(config)
        folds = splitter.split(df)
        windows = splitter.split_windows(df)
        assert len(folds) == len(windows)
        for (train_df, test_df), (train_win, test_win) in zip(
            folds, windows, strict=True
        ):
            assert df.iloc[train_win[0] : train_win[1]].index.equals(train_df.index)
            assert df.iloc[test_win[0] : test_win[1]].index.equals(test_df.index)

    def test_rolling_windows_match_the_frames(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        self._assert_consistent(
            WalkForwardConfig(n_splits=2, train_ratio=0.7, expanding=False),
            walk_forward_ohlcv_df,
        )

    def test_expanding_windows_match_the_frames(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        self._assert_consistent(
            WalkForwardConfig(n_splits=3, train_ratio=0.6, expanding=True),
            walk_forward_ohlcv_df,
        )

    def test_explicit_windows_match_the_frames(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        ts = walk_forward_ohlcv_df["timestamp"]
        boundary = ts.iloc[len(ts) // 2]
        self._assert_consistent(
            WalkForwardConfig(
                train_start=ts.iloc[0].to_pydatetime(),
                train_end=boundary.to_pydatetime(),
                test_start=(boundary + timedelta(days=1)).to_pydatetime(),
                test_end=ts.iloc[-1].to_pydatetime(),
            ),
            walk_forward_ohlcv_df,
        )


class TestValidatorGate:
    def setup_method(self) -> None:
        self.engine = SignalBacktestEngine(init_cash=10_000.0, fees=0.0, slippage=0.0)

    def test_default_horizon_does_not_block_a_run(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        """Regression guard: without a label horizon nothing is rejected."""
        config = WalkForwardConfig(n_splits=2, train_ratio=0.7)
        validator = WalkForwardValidator(config=config, engine=self.engine)
        results = validator.validate(
            KDStrategy(config=KDStrategyConfig()), walk_forward_ohlcv_df
        )
        assert len(results) == 2

    def test_label_horizon_blocks_leaking_folds(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        config = WalkForwardConfig(n_splits=2, train_ratio=0.7, label_horizon=5)
        validator = WalkForwardValidator(config=config, engine=self.engine)
        with pytest.raises(WalkForwardLeakageError) as excinfo:
            validator.validate(
                KDStrategy(config=KDStrategyConfig()), walk_forward_ohlcv_df
            )
        assert excinfo.value.violations
        assert any("fold/0" in item for item in excinfo.value.violations)

    def test_gate_runs_before_any_fold_executes(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        """A blocked run must not produce partial results."""
        config = WalkForwardConfig(n_splits=2, train_ratio=0.7, label_horizon=5)
        validator = WalkForwardValidator(config=config, engine=self.engine)
        with pytest.raises(WalkForwardLeakageError):
            validator.validate(
                KDStrategy(config=KDStrategyConfig()), walk_forward_ohlcv_df
            )
