"""Data splitting strategies for walk-forward validation."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from tradingdev.domain.validation.config import (
    AutomaticWalkForwardConfig,
    ExplicitWalkForwardConfig,
)
from tradingdev.domain.validation.date_windows import (
    WalkForwardSplitError,
    inclusive_day_window,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd


class DataSplitter:
    """Generate (train, test) DataFrame pairs for validation.

    Supports:
        - Explicit date splits (single fold).
        - Rolling windows with fixed train size.
        - Expanding windows with growing train size.
    """

    def __init__(
        self, config: AutomaticWalkForwardConfig | ExplicitWalkForwardConfig
    ) -> None:
        if not isinstance(
            config, (AutomaticWalkForwardConfig, ExplicitWalkForwardConfig)
        ):
            raise TypeError("DataSplitter requires automatic or explicit configuration")
        self._config = config

    def split(self, df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate train/test splits from the DataFrame."""
        return [
            (df.iloc[train[0] : train[1]].copy(), df.iloc[test[0] : test[1]].copy())
            for train, test in self.split_windows(df)
        ]

    def split_windows(
        self, df: pd.DataFrame
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Positional ``[start, end)`` train/test windows, one per fold.

        Reports the integer geometry used by :meth:`split`,
        so callers can audit fold boundaries (purge/embargo) without
        re-deriving positions from the returned DataFrames. Both derive from
        :meth:`_explicit_masks` or the same auto geometry, so the audited
        windows cannot drift from the executed frames.
        """
        cfg = self._config

        if isinstance(cfg, ExplicitWalkForwardConfig):
            return self._explicit_windows(df)

        return self._auto_windows(df)

    def _explicit_masks(
        self, df: pd.DataFrame
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        """Boolean train/test masks for the explicit date window.

        Re-checks the series that actually reaches ``fit()`` rather than
        trusting the config alone: the frame must be ordered, both windows
        must select rows, and the last train timestamp must precede the first
        test timestamp.
        """
        cfg = self._config
        assert isinstance(cfg, ExplicitWalkForwardConfig)

        ts = df["timestamp"]
        if not ts.is_monotonic_increasing:
            msg = (
                "Walk-forward requires timestamps sorted ascending; "
                "positional fold windows cannot describe this frame"
            )
            raise WalkForwardSplitError(msg)
        train_lo, train_hi = inclusive_day_window(cfg.train_start, cfg.train_end)
        test_lo, test_hi = inclusive_day_window(cfg.test_start, cfg.test_end)
        train_mask = ((ts >= train_lo) & (ts < train_hi)).to_numpy()
        test_mask = ((ts >= test_lo) & (ts < test_hi)).to_numpy()

        train_pos = np.flatnonzero(train_mask)
        test_pos = np.flatnonzero(test_mask)
        if train_pos.size == 0 or test_pos.size == 0:
            empty = "train" if train_pos.size == 0 else "test"
            span = "empty frame" if ts.empty else f"data {ts.iloc[0]}..{ts.iloc[-1]}"
            msg = (
                f"Explicit walk-forward dates select no {empty} rows: "
                f"train {cfg.train_start}..{cfg.train_end}, "
                f"test {cfg.test_start}..{cfg.test_end}, {span}"
            )
            raise WalkForwardSplitError(msg)
        if ts.iloc[train_pos[-1]] >= ts.iloc[test_pos[0]]:
            msg = (
                "Train and test windows overlap: "
                f"train ends {ts.iloc[train_pos[-1]]}, "
                f"test starts {ts.iloc[test_pos[0]]}"
            )
            raise WalkForwardSplitError(msg)
        return train_mask, test_mask

    def _explicit_windows(
        self, df: pd.DataFrame
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Single window pair derived from the explicit date masks."""
        train_mask, test_mask = self._explicit_masks(df)
        train_pos = np.flatnonzero(train_mask)
        test_pos = np.flatnonzero(test_mask)
        return [
            (
                (int(train_pos[0]), int(train_pos[-1]) + 1),
                (int(test_pos[0]), int(test_pos[-1]) + 1),
            )
        ]

    def _auto_windows(
        self, df: pd.DataFrame
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Rolling or expanding window geometry shared by splitting and auditing."""
        cfg = self._config
        assert isinstance(cfg, AutomaticWalkForwardConfig)
        n = len(df)
        # Solve T / (T + S) = ratio with T + n_splits * S = n.
        # Round the initial training size down, then distribute every remaining
        # row over contiguous test windows whose sizes differ by at most one.
        # Interpret the decimal value persisted in JSON, without binary-rounding
        # errors crossing an integer boundary before floor.
        ratio = Fraction(str(cfg.train_ratio))
        train_size = int(n * ratio / (ratio + cfg.n_splits * (1 - ratio)))
        remaining = n - train_size
        if train_size < 1 or remaining < cfg.n_splits:
            raise WalkForwardSplitError(
                "Automatic walk-forward needs nonempty train and test windows; "
                f"received {n} rows for {cfg.n_splits} folds "
                f"and ratio {cfg.train_ratio}"
            )

        windows: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for i in range(cfg.n_splits):
            test_start_idx = train_size + remaining * i // cfg.n_splits
            test_end_idx = train_size + remaining * (i + 1) // cfg.n_splits
            train_end_idx = test_start_idx
            train_start_idx = 0 if cfg.expanding else train_end_idx - train_size

            windows.append(
                ((train_start_idx, train_end_idx), (test_start_idx, test_end_idx))
            )

        return windows
