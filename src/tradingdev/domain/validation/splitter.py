"""Data splitting strategies for walk-forward validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tradingdev.domain.validation.date_windows import (
    WalkForwardSplitError,
    inclusive_day_window,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd

    from tradingdev.domain.backtest.schemas import WalkForwardConfig


class DataSplitter:
    """Generate (train, test) DataFrame pairs for validation.

    Supports:
        - Explicit date splits (single fold).
        - Rolling windows with fixed train size.
        - Expanding windows with growing train size.
    """

    def __init__(self, config: WalkForwardConfig) -> None:
        self._config = config

    def split(self, df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate train/test splits from the DataFrame."""
        cfg = self._config

        if (
            cfg.train_start is not None
            and cfg.train_end is not None
            and cfg.test_start is not None
            and cfg.test_end is not None
        ):
            return self._explicit_split(df)

        return self._auto_split(df)

    def split_windows(
        self, df: pd.DataFrame
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Positional ``[start, end)`` train/test windows, one per fold.

        Mirrors :meth:`split` but reports integer geometry instead of frames,
        so callers can audit fold boundaries (purge/embargo) without
        re-deriving positions from the returned DataFrames. Both derive from
        :meth:`_explicit_masks` or the same auto geometry, so the audited
        windows cannot drift from the executed frames.
        """
        cfg = self._config

        if (
            cfg.train_start is not None
            and cfg.train_end is not None
            and cfg.test_start is not None
            and cfg.test_end is not None
        ):
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
        if (
            cfg.train_start is None
            or cfg.train_end is None
            or cfg.test_start is None
            or cfg.test_end is None
        ):
            msg = "Explicit split requires all four dates"
            raise WalkForwardSplitError(msg)

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
        """Rolling or expanding window geometry, mirroring ``_auto_split``."""
        cfg = self._config
        n = len(df)
        total_per_split = n // cfg.n_splits
        train_size = int(total_per_split * cfg.train_ratio)
        test_size = total_per_split - train_size

        windows: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for i in range(cfg.n_splits):
            if cfg.expanding:
                train_start_idx = 0
                train_end_idx = train_size + i * test_size
            else:
                train_start_idx = i * test_size
                train_end_idx = train_start_idx + train_size
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + test_size

            if test_end_idx > n:
                break

            windows.append(
                ((train_start_idx, train_end_idx), (test_start_idx, test_end_idx))
            )

        return windows

    def _explicit_split(
        self, df: pd.DataFrame
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Single split using explicit date boundaries."""
        train_mask, test_mask = self._explicit_masks(df)
        return [(df[train_mask].copy(), df[test_mask].copy())]

    def _auto_split(self, df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Rolling or expanding window splits."""
        cfg = self._config
        n = len(df)
        total_per_split = n // cfg.n_splits
        train_size = int(total_per_split * cfg.train_ratio)
        test_size = total_per_split - train_size

        splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        for i in range(cfg.n_splits):
            if cfg.expanding:
                train_end_idx = train_size + i * test_size
                test_start_idx = train_end_idx
                test_end_idx = test_start_idx + test_size
                train_df = df.iloc[:train_end_idx].copy()
            else:
                train_start_idx = i * test_size
                train_end_idx = train_start_idx + train_size
                test_start_idx = train_end_idx
                test_end_idx = test_start_idx + test_size
                train_df = df.iloc[train_start_idx:train_end_idx].copy()

            if test_end_idx > n:
                break

            test_df = df.iloc[test_start_idx:test_end_idx].copy()
            splits.append((train_df, test_df))

        return splits
