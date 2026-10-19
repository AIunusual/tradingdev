"""The named validation schemes the ML layer offers.

The contract advertises these names and ``train_model`` resolves the one it is
given, so both read this registry: a scheme that is advertised but cannot be
built - or one that can be built but is never advertised - is a contradiction
that this module makes impossible.

Every scheme here is purged and embargoed by construction. That is the whole
point of naming them: the caller asks for a split that does not leak, and gets
one, rather than being handed a plain time-series split and asked to trust it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tradingdev.domain.ml.validation.splits import (
    leakage_audit,
    purged_kfold_splits,
    purged_walk_forward_splits,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from tradingdev.domain.ml.validation.splits import Split


@dataclass(frozen=True)
class ValidationScheme:
    """One named way to cut bars into training and test windows."""

    name: str
    description: str
    fold_parameter: str
    """Name of the fold-count keyword the builder expects.

    The two builders disagree on what to call it (``n_splits`` versus
    ``n_folds``); recording it here means callers say "how many folds"
    once, whichever scheme they picked.
    """
    build: Callable[..., list[Split]]


_SCHEMES = (
    ValidationScheme(
        name="purged_walk_forward",
        description=(
            "Anchored walk-forward with purged/embargoed training bars; "
            "block 0 always trains, each later block is tested once."
        ),
        fold_parameter="n_splits",
        build=purged_walk_forward_splits,
    ),
    ValidationScheme(
        name="purged_kfold",
        description=(
            "Every bar falls in exactly one test window; training spans both "
            "sides with the label horizon purged and an embargo gap. "
            "Retrospective cross-validation, not a past-only deployment simulation."
        ),
        fold_parameter="n_folds",
        build=purged_kfold_splits,
    ),
)

VALIDATION_SCHEMES: dict[str, ValidationScheme] = {
    scheme.name: scheme for scheme in _SCHEMES
}


def validation_scheme_names() -> tuple[str, ...]:
    """Names of the schemes that can be asked for, in a stable order."""
    return tuple(scheme.name for scheme in _SCHEMES)


def get_validation_scheme(name: str) -> ValidationScheme:
    """Look a scheme up by name.

    Raises:
        ValueError: If the name is not one of :func:`validation_scheme_names`.
            The alternatives are named in the message: a caller that guesses a
            scheme name should not have to guess again.
    """
    try:
        return VALIDATION_SCHEMES[name]
    except KeyError:
        raise ValueError(
            f"unknown validation scheme: {name}; "
            f"choose one of {', '.join(validation_scheme_names())}"
        ) from None


def build_validation_splits(
    name: str,
    *,
    n_bars: int,
    horizon: int,
    n_folds: int = 5,
    embargo: int = 0,
    min_train_bars: int = 1,
) -> list[Split]:
    """Cut ``n_bars`` bars into folds under the named scheme.

    Args:
        name: Scheme name, see :func:`validation_scheme_names`.
        n_bars: Bars available to split.
        horizon: Label horizon in bars. It sets the purge width, because the
            label of bar ``t`` is decided by prices up to ``t + horizon``.
        n_folds: How many test windows to produce.
        embargo: Extra bars removed around each test window.
        min_train_bars: Smallest training window accepted per fold.

    Returns:
        One split per fold.

    Raises:
        ValueError: On an unknown scheme, or on geometry the scheme rejects
            (too few bars, too many folds, a horizon that does not fit).
    """
    scheme = get_validation_scheme(name)
    keywords: dict[str, Any] = {scheme.fold_parameter: n_folds}
    return scheme.build(
        n_bars,
        horizon=horizon,
        embargo=embargo,
        min_train_bars=min_train_bars,
        **keywords,
    )


def audit_validation_splits(
    splits: Sequence[Split],
    *,
    horizon: int,
) -> tuple[str, ...]:
    """Check finished splits for label leakage before anyone trains on them.

    The builders purge by construction, so this is a second opinion rather
    than the thing that makes them safe - but a split that fails it must not
    reach a model, and finding out here is far cheaper than finding out in a
    suspicious backtest.
    """
    return tuple(leakage_audit(splits, horizon=horizon))
