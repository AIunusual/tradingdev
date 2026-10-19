"""Purged/embargoed index splits and label-overlap audits.

The schemes registry supplies both advertised names and training resolution.
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
