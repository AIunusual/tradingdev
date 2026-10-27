"""Regime holding state remains consistent across unscorable bars."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tradingdev.domain.strategies.bundled.quantile_strategy.strategy import (
    _predict_probabilities,
    _run_regime_state_machine,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_prediction_boundary_accepts_and_normalizes_model_precision(
    precision: str,
) -> None:
    class _TypedModel:
        def predict_proba(
            self, features: NDArray[np.float64]
        ) -> NDArray[np.float32] | NDArray[np.float64]:
            if precision == "float32":
                return np.full((len(features), 4), 0.25, dtype=np.float32)
            return np.full((len(features), 4), 0.25, dtype=np.float64)

    probabilities = _predict_probabilities(
        _TypedModel(), np.ones((2, 3), dtype=np.float64)
    )

    assert probabilities.dtype == np.float64
    np.testing.assert_array_equal(probabilities, np.full((2, 4), 0.25))


def test_gap_resets_entry_price_minimum_hold_horizon_and_weight() -> None:
    """The post-gap position starts fresh even after a large price change."""
    probabilities = np.array(
        [
            [0.8, 0.0, 0.0, 0.2],
            [0.8, 0.0, 0.0, 0.2],
            [np.nan, np.nan, np.nan, np.nan],
            [0.4, 0.3, 0.1, 0.2],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    signals, weights = _run_regime_state_machine(
        close=np.array([100, 100, 100, 50, 40, 49, 50], dtype=np.float64),
        proba=probabilities,
        horizon=3,
        min_holding_bars=2,
        min_confidence=0.2,
        max_p_both=0.15,
        max_p_neither=0.2,
        strategy_sl=0.05,
        dynamic_sizing=True,
        min_weight=0.2,
        edge_for_full_size=1.0,
    )

    # The new entry is at 50, its first holding bar is protected by min hold,
    # 49 does not stop it out, and its own third holding bar closes it.
    np.testing.assert_array_equal(signals, [1, 1, 0, 1, 1, 1, 0])
    np.testing.assert_allclose(weights, [0.8, 0.8, 1.0, 0.3, 0.3, 0.3, 0.3])


@pytest.mark.parametrize("new_direction", [-1, 1])
def test_post_gap_stop_uses_the_new_entry_price(new_direction: int) -> None:
    probabilities = np.zeros((6, 4), dtype=np.float64)
    probabilities[:2, 0] = 1.0
    probabilities[2] = np.nan
    probabilities[3:, 0 if new_direction == 1 else 1] = 1.0

    signals, _ = _run_regime_state_machine(
        close=np.array(
            [100, 100, 100, 50, 50, 50 - 3 * new_direction], dtype=np.float64
        ),
        proba=probabilities,
        horizon=100,
        min_holding_bars=1,
        min_confidence=0.2,
        max_p_both=0.15,
        max_p_neither=0.2,
        strategy_sl=0.05,
        dynamic_sizing=False,
        min_weight=0.2,
        edge_for_full_size=1.0,
    )

    np.testing.assert_array_equal(signals, [1, 1, 0, new_direction, new_direction, 0])
