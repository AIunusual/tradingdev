"""KD Stochastic Oscillator indicator."""

from typing import TYPE_CHECKING, Any

from tradingdev.domain import indicators
from tradingdev.domain.indicators.base import BaseIndicator

if TYPE_CHECKING:
    import pandas as pd


class KDIndicator(BaseIndicator):
    """Stochastic Oscillator (%K and %D).

    Appends ``stoch_k`` and ``stoch_d`` columns to the input DataFrame.
    """

    def __init__(
        self,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3,
    ) -> None:
        self._k_period = k_period
        self._d_period = d_period
        self._smooth_k = smooth_k

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate %K and %D values.

        Args:
            df: OHLCV DataFrame with ``high``, ``low``, ``close`` columns.

        Returns:
            DataFrame with ``stoch_k`` and ``stoch_d`` columns appended.
        """
        stoch = indicators.stochastic(
            df["high"],
            df["low"],
            df["close"],
            k=self._k_period,
            d=self._d_period,
            smooth_k=self._smooth_k,
        )

        result = df.copy()
        result["stoch_k"] = stoch.k
        result["stoch_d"] = stoch.d
        return result

    def get_parameters(self) -> dict[str, Any]:
        """Return indicator parameters."""
        return {
            "k_period": self._k_period,
            "d_period": self._d_period,
            "smooth_k": self._smooth_k,
        }
