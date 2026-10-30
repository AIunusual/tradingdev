"""Declarative rolling schemes, off by default (qlib ``RollingGen`` counterpart).

Enable one through a strategy's ``rolling_gen`` block.  While it is disabled the
strategies keep their classic retraining loop and nothing here runs.
"""

from __future__ import annotations

from tradingdev.domain.ml.rolling_gen.config import (
    RollingGenConfig,
    RollingSchemeConfig,
)
from tradingdev.domain.ml.rolling_gen.engine import (
    rolling_probabilities,
    run_rolling,
    signals_from_probabilities,
)
from tradingdev.domain.ml.rolling_gen.registry import (
    ROLLING_SCHEMES,
    get_rolling_scheme,
    resolve_rolling_scheme,
    rolling_scheme_names,
)
from tradingdev.domain.ml.rolling_gen.scheme import (
    RollingGeometry,
    RollingScheme,
    RollType,
    resolve_geometry,
)

__all__ = [
    "ROLLING_SCHEMES",
    "RollType",
    "RollingGenConfig",
    "RollingGeometry",
    "RollingScheme",
    "RollingSchemeConfig",
    "get_rolling_scheme",
    "resolve_geometry",
    "resolve_rolling_scheme",
    "rolling_probabilities",
    "rolling_scheme_names",
    "run_rolling",
    "signals_from_probabilities",
]
