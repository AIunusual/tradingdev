"""An opt-in walk-forward splitter that purges and embargoes by construction.

:class:`DataSplitter` cuts a price series into train/test windows that are
disjoint in time, which is correct when a strategy only looks backwards. It is
not correct once training targets are interval-based labels: the label of the
bar at ``t`` is decided by prices up to ``t + horizon``, so the last bars of a
training window carry answers computed from test-window prices.

This splitter removes those bars instead of merely detecting them, reusing the
audited index arithmetic in
:mod:`tradingdev.domain.ml.validation.splits`. Folds produced here satisfy the
:mod:`leakage gate <tradingdev.domain.validation.leakage_gate>` by
construction, so a run is not blocked and its out-of-sample score is not
inflated.

It is deliberately a **separate, opt-in class** rather than a modification of
:class:`DataSplitter`:

* The purge width is a function of the label horizon, which
  :class:`DataSplitter` has no concept of.
* Purging is anchored --- block 0 is always training-only and every later
  block is tested once --- so there is no ``train_ratio`` and no ``expanding``
  window to configure. Replacing the existing splitter would silently change
  fold semantics for every current caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tradingdev.domain.ml.validation import purged_walk_forward_splits

if TYPE_CHECKING:
    import pandas as pd

    from tradingdev.domain.backtest.schemas import WalkForwardConfig
    from tradingdev.domain.validation.leakage_gate import FoldWindow


class PurgedDataSplitter:
    """Generate leak-free (train, test) DataFrame pairs for validation.

    Same surface as :class:`~tradingdev.domain.validation.splitter.DataSplitter`
    (``split`` and ``split_windows``) so the validator can swap between them.
    """

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
        """Positional ``[start, end)`` train/test windows, one per fold.

        Every window leaves a gap of at least ``label_horizon + embargo_bars``
        bars before its test window, which is exactly what the leakage gate
        checks.
        """
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
