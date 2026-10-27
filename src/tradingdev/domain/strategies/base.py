"""Abstract base class for all trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


class BaseStrategy(ABC):
    """Base interface for trading strategies.

    Strategies receive a DataFrame (optionally with indicator columns) and
    produce a ``signal`` column containing only ``1``, ``-1``, or ``0``.
    ``1`` selects long and ``-1`` selects short; ``-1`` is never a sell-only
    instruction to close a long position.

    In ``mode="signal"``, each row specifies a position state: ``1`` long,
    ``-1`` short, ``0`` flat. Repeat the state while it should persist;
    a one-row entry pulse followed by zero exits on the next signal change.
    ``signal_as_position`` has no effect in this mode.

    In ``mode="volume"``, nonzero signals select the position direction.
    Zero closes an existing position when ``signal_as_position=True``;
    with the default ``False``, zero prevents new entries but retains an
    existing position. An opposite nonzero signal closes and reverses it.

    Both modes shift signals by one bar. Signal-driven entries and exits use
    the following bar's open, falling back to close when open is absent.
    Stop-loss and take-profit rules are separate from these signal changes.
    """

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals.

        Args:
            df: OHLCV DataFrame (may already contain indicator columns).

        Returns:
            DataFrame with a ``signal`` column added.
        """
        ...

    def fit(self, df: pd.DataFrame) -> None:  # noqa: B027
        """Fit/optimize the strategy on training data.

        Default implementation is a no-op. Override in subclasses
        that support parameter optimization or model training.

        Args:
            df: Training OHLCV DataFrame.
        """

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """Return strategy parameter dictionary.

        Returns:
            Dictionary of parameter names to values.
        """
        ...
