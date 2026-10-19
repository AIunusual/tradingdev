"""Opt-in anchored strategy folds using the ML purge/embargo arithmetic.

The first block is training-only; later blocks test against previous data
minus label horizon and embargo. Unlike DataSplitter, explicit dates,
train_ratio and expanding do not control this split.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tradingdev.domain.ml.validation import purged_walk_forward_splits

if TYPE_CHECKING:
    import pandas as pd

    from tradingdev.domain.backtest.schemas import WalkForwardConfig
    from tradingdev.domain.validation.leakage_gate import FoldWindow


class PurgedDataSplitter:
    """Purged strategy folds with the DataSplitter split/split_windows interface."""

    def __init__(
        self,
        config: WalkForwardConfig,
        *,
        min_train_bars: int = 1,
    ) -> None:
        if config.label_horizon < 1:
            msg = (
                "PurgedDataSplitter needs a label horizon: set "
                "WalkForwardConfig.label_horizon to the number of bars a "
                "target looks ahead (got "
                f"{config.label_horizon})."
            )
            raise ValueError(msg)
        self._config = config
        self._min_train_bars = min_train_bars

    def split(self, df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate purged train/test splits from the DataFrame."""
        windows = self.split_windows(df)
        return [
            (df.iloc[train[0] : train[1]].copy(), df.iloc[test[0] : test[1]].copy())
            for train, test in windows
        ]

    def split_windows(self, df: pd.DataFrame) -> list[FoldWindow]:
        """Return half-open windows with label_horizon + embargo_bars before tests."""
        splits = purged_walk_forward_splits(
            len(df),
            horizon=self._config.label_horizon,
            n_splits=self._config.n_splits,
            embargo=self._config.embargo_bars,
            min_train_bars=self._min_train_bars,
        )
        windows: list[FoldWindow] = []
        for split in splits:
            if len(split.train) != 1:
                msg = (
                    f"{split.name}: anchored walk-forward keeps a single "
                    f"contiguous training block, got {len(split.train)}"
                )
                raise ValueError(msg)
            windows.append((split.train[0], split.test))
        return windows
