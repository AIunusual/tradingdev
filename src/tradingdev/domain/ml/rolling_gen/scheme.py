"""Declarative rolling-window geometry for ML signal generation.

A rolling scheme answers two questions for every bar that is being scored:
which rows may train the model, and which rows may feed the features that
score it.  Both answers are half-open slices of the concatenated frame, and
both are causal by construction: a training window always ends strictly
before the bar being scored, so no scheme can declare its way into the future.

This module holds only geometry.  It imports nothing from the engine, so the
window rules can be checked without a model, a feature engineer or market data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RollType(StrEnum):
    """Training-window shape, mirroring qlib's ``RollingGen`` roll types."""

    ROLL_SD = "ROLL_SD"
    """Fixed-length window sliding forward from the first bar."""

    ROLL_EX = "ROLL_EX"
    """Fixed origin, window end expanding with every bar."""


@dataclass(frozen=True)
class RollingScheme:
    """One named way to place the training window behind a scored bar.

    Every numeric field is optional.  An unset field follows the strategy's own
    classic parameter (see :func:`resolve_geometry`), which keeps the declared
    defaults usable for any bar length: a scheme written for 1-minute bars with
    a lookback of 24 cannot sensibly impose the same numbers on a strategy that
    looks back 720 bars, because a training window shorter than the feature
    warm-up would leave no labelled rows and silently stop retraining.
    """

    name: str
    description: str
    rtype: RollType
    step: int | None = None
    """Bars between retrains; unset follows the strategy's retrain interval."""

    window_size: int | None = None
    """Fixed training length for ``ROLL_SD``; unset follows ``10 x lookback``.

    Not applicable to ``ROLL_EX``, whose window starts at the first bar.
    """

    min_train_bars: int | None = None
    """Labelled rows a window needs before it may train.

    Unset follows ``lookback + 10``, the classic minimum expressed on the rows
    that actually survive feature warm-up and label maturity.
    """

    predict_lookback: int | None = None
    """Maximum preceding bars fed to the feature engineer when scoring one bar.

    The scored bar is added, giving up to ``predict_lookback + 1`` context rows;
    the slice is clipped to available history. Unset follows ``2 x lookback``.
    A value below ``lookback + 1`` cannot
    produce features for the scored bar at all, so it is rejected.
    """


@dataclass(frozen=True)
class RollingGeometry:
    """A scheme with every number filled in, ready to slice a frame."""

    rtype: RollType
    step: int
    min_train_bars: int
    predict_lookback: int
    window_size: int | None

    def train_window(self, combined_idx: int) -> tuple[int, int]:
        """Return the half-open training slice for the bar at ``combined_idx``.

        The stop is ``combined_idx`` itself: the scored bar and everything
        after it are excluded from training.  Every roll type shares this one
        expression, so a new scheme cannot widen the window past the bar it is
        scoring without changing the geometry for all of them.
        """
        if self.rtype is RollType.ROLL_EX:
            return 0, combined_idx
        if self.window_size is None:
            msg = "ROLL_SD requires a window_size"
            raise ValueError(msg)
        return max(0, combined_idx - self.window_size), combined_idx

    def predict_context(self, combined_idx: int) -> tuple[int, int]:
        """Return the half-open feature context ending at the scored bar."""
        return max(0, combined_idx - self.predict_lookback), combined_idx + 1


def resolve_geometry(
    scheme: RollingScheme,
    *,
    lookback: int,
    retrain_interval: int,
) -> RollingGeometry:
    """Fill the scheme's unset fields from the strategy's classic parameters.

    Args:
        scheme: The declared scheme, with any subset of overrides.
        lookback: The strategy's chosen feature lookback, in bars.
        retrain_interval: The strategy's classic retrain cadence, in bars.

    Returns:
        Concrete slice bounds for every bar.

    Raises:
        ValueError: If the strategy parameters or the declared overrides cannot
            form a usable window.
    """
    if lookback < 1:
        msg = f"lookback must be at least 1 bar, got {lookback}"
        raise ValueError(msg)

    step = retrain_interval if scheme.step is None else scheme.step
    if step < 1:
        msg = f"rolling step must be at least 1 bar, got {step}"
        raise ValueError(msg)

    min_train_bars = (
        lookback + 10 if scheme.min_train_bars is None else scheme.min_train_bars
    )
    if min_train_bars < 1:
        msg = f"min_train_bars must be at least 1, got {min_train_bars}"
        raise ValueError(msg)

    predict_lookback = (
        lookback * 2 if scheme.predict_lookback is None else scheme.predict_lookback
    )
    if predict_lookback < lookback + 1:
        msg = (
            "predict_lookback must be at least lookback + 1 "
            f"({lookback + 1}) so the scored bar keeps its features, "
            f"got {predict_lookback}"
        )
        raise ValueError(msg)

    window_size: int | None = None
    if scheme.rtype is RollType.ROLL_SD:
        window_size = (
            lookback * 10 if scheme.window_size is None else scheme.window_size
        )
        if window_size < 1:
            msg = f"window_size must be at least 1 bar, got {window_size}"
            raise ValueError(msg)

    return RollingGeometry(
        rtype=scheme.rtype,
        step=step,
        min_train_bars=min_train_bars,
        predict_lookback=predict_lookback,
        window_size=window_size,
    )
