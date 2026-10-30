"""Named rolling schemes, enumerable the same way validation schemes are.

The registry is the only place a scheme name becomes a geometry.  Unknown names
are rejected with the alternatives listed, so a typo fails before any model is
trained instead of quietly falling back to some default window.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from tradingdev.domain.ml.rolling_gen.scheme import RollingScheme, RollType

if TYPE_CHECKING:
    from tradingdev.domain.ml.rolling_gen.config import RollingSchemeConfig

_BUILTIN_SCHEMES: tuple[RollingScheme, ...] = (
    RollingScheme(
        name="roll_sd",
        description=(
            "Sliding window: the training set is a fixed number of bars that "
            "moves forward with each retrain (qlib ROLL_SD). The window length "
            "follows the strategy's lookback unless window_size is set."
        ),
        rtype=RollType.ROLL_SD,
    ),
    RollingScheme(
        name="roll_ex",
        description=(
            "Expanding window: the training set keeps its first bar and grows "
            "with every retrain (qlib ROLL_EX). The cadence and minimum size "
            "follow the strategy unless step or min_train_bars are set."
        ),
        rtype=RollType.ROLL_EX,
    ),
)

ROLLING_SCHEMES: dict[str, RollingScheme] = {
    scheme.name: scheme for scheme in _BUILTIN_SCHEMES
}


def rolling_scheme_names() -> tuple[str, ...]:
    """Names of the schemes that can be asked for, in a stable order."""
    return tuple(scheme.name for scheme in _BUILTIN_SCHEMES)


def get_rolling_scheme(name: str) -> RollingScheme:
    """Resolve a scheme; reject unknown names with the available alternatives."""
    try:
        return ROLLING_SCHEMES[name]
    except KeyError:
        raise ValueError(
            f"unknown rolling scheme: {name}; "
            f"choose one of {', '.join(rolling_scheme_names())}"
        ) from None


def resolve_rolling_scheme(cfg: RollingSchemeConfig) -> RollingScheme:
    """Overlay the declared overrides on the built-in scheme of the same name.

    Args:
        cfg: A strategy's ``rolling_gen`` block, holding a scheme name plus any
            fields the user chose to override.

    Returns:
        The named scheme with every non-``None`` override applied.

    Raises:
        ValueError: If the name is unknown, or if ``window_size`` is declared
            for ``ROLL_EX``, where a fixed length contradicts the expanding
            window and would otherwise be ignored without a word.
    """
    base = get_rolling_scheme(cfg.scheme)
    window_size = base.window_size if cfg.window_size is None else cfg.window_size
    if base.rtype is RollType.ROLL_EX and window_size is not None:
        msg = (
            f"rolling scheme {cfg.scheme!r} is an expanding window whose size "
            "grows from the first bar; window_size only applies to ROLL_SD"
        )
        raise ValueError(msg)
    return replace(
        base,
        step=base.step if cfg.step is None else cfg.step,
        window_size=window_size,
        min_train_bars=(
            base.min_train_bars if cfg.min_train_bars is None else cfg.min_train_bars
        ),
        predict_lookback=(
            base.predict_lookback
            if cfg.predict_lookback is None
            else cfg.predict_lookback
        ),
    )
