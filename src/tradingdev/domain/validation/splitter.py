"""Data splitting strategies for walk-forward validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from datetime import datetime

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
        re-deriving positions from the returned DataFrames. The two are kept
        consistent by a contract test rather than by construction.
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

    def _explicit_windows(
        self, df: pd.DataFrame
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Single window pair derived from the explicit date masks."""
        cfg = self._config
        ts = df["timestamp"]
        train_mask = (
            (ts >= _to_utc_ts(cfg.train_start))  # type: ignore[arg-type]
            & (ts < _to_utc_ts(cfg.train_end) + pd.Timedelta(days=1))  # type: ignore[arg-type]
        ).to_numpy()
        test_mask = (
            (ts >= _to_utc_ts(cfg.test_start))  # type: ignore[arg-type]
            & (ts < _to_utc_ts(cfg.test_end) + pd.Timedelta(days=1))  # type: ignore[arg-type]
        ).to_numpy()
        train_pos = np.flatnonzero(train_mask)
        test_pos = np.flatnonzero(test_mask)
        if train_pos.size == 0 or test_pos.size == 0:
            return []
        return [
            (
                (int(train_pos.min()), int(train_pos.max()) + 1),
                (int(test_pos.min()), int(test_pos.max()) + 1),
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
        cfg = self._config
        ts = df["timestamp"]

        train_mask = (ts >= _to_utc_ts(cfg.train_start)) & (  # type: ignore[arg-type]
            ts < _to_utc_ts(cfg.train_end) + pd.Timedelta(days=1)  # type: ignore[arg-type]
        )
        test_mask = (ts >= _to_utc_ts(cfg.test_start)) & (  # type: ignore[arg-type]
            ts < _to_utc_ts(cfg.test_end) + pd.Timedelta(days=1)  # type: ignore[arg-type]
        )
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


def _to_utc_ts(dt: datetime) -> pd.Timestamp:
    """Convert datetime to UTC-aware pandas Timestamp."""
    t = pd.Timestamp(dt)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t
