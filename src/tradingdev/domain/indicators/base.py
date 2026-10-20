"""Abstract base class for all technical indicators."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


class BaseIndicator(ABC):
    """Base interface for technical indicators.

    Indicators receive an OHLCV DataFrame and return a new DataFrame
    with additional indicator columns appended.
    """

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicator values.

        Args:
            df: OHLCV DataFrame with at least ``[open, high, low, close, volume]``.

        Returns:
            DataFrame with new indicator columns appended.
        """
        ...

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """Return the indicator's parameter dictionary.

        Returns:
            Dictionary of parameter names to values.
        """
        ...
