"""Leak-free validation splits for interval-based labels.

Purging and embargoing are index arithmetic, not heuristics applied inside a
model, so they can be unit tested without market data and audited after the
fact by :func:`leakage_audit`.

:mod:`schemes` names the cuts and is the single source both the advertised
contract and ``train_model`` read, so what is offered and what can be built
cannot drift apart.
"""

from __future__ import annotations

from tradingdev.domain.ml.validation.schemes import (
    VALIDATION_SCHEMES,
    ValidationScheme,
    audit_validation_splits,
    build_validation_splits,
    get_validation_scheme,
    validation_scheme_names,
)
from tradingdev.domain.ml.validation.splits import (
    Split,
    leakage_audit,
    purged_kfold_splits,
    purged_walk_forward_splits,
)

__all__ = [
    "VALIDATION_SCHEMES",
    "Split",
    "ValidationScheme",
    "audit_validation_splits",
    "build_validation_splits",
    "get_validation_scheme",
    "leakage_audit",
    "purged_kfold_splits",
    "purged_walk_forward_splits",
    "validation_scheme_names",
]
