"""Feature cache identities bind bar values, symbol, timeframe, set and code.

Content hashing detects changes even when file size/mtime is unchanged.
Feature matrices can be reused across label horizons; the store owns I/O.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import talib

from tradingdev.domain.ml.features.registry import list_feature_sets

if TYPE_CHECKING:
    from collections.abc import Sequence

_DIGEST_LENGTH = 16


@dataclass(frozen=True)
class FeatureCacheIdentity:
    """Everything that decides whether a cached matrix may be reused."""

    symbol: str
    timeframe: str
    data_fingerprint: str
    feature_set: str
    code_fingerprint: str

    @property
    def key(self) -> str:
        """The cache key: a 16-char hex digest of the whole identity."""
        payload = "|".join(
            (
                self.symbol,
                self.timeframe,
                self.data_fingerprint,
                self.feature_set,
                self.code_fingerprint,
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_DIGEST_LENGTH]


def data_fingerprint(frame: pd.DataFrame) -> str:
    """Hash bar values; file size and mtime cannot establish identical inputs."""
    digest = hashlib.sha256()
    digest.update(",".join(str(column) for column in frame.columns).encode("utf-8"))
    digest.update(f"rows={len(frame)}".encode())
    hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype="uint64")
    digest.update(hashes.tobytes())
    return digest.hexdigest()[:_DIGEST_LENGTH]


def feature_cache_key(
    *,
    symbol: str,
    timeframe: str,
    data_fingerprint: str,
    feature_set: str,
    code_fingerprint: str,
) -> str:
    """Build the cache key for one (series, bars, feature set, code)."""
    return FeatureCacheIdentity(
        symbol=symbol,
        timeframe=timeframe,
        data_fingerprint=data_fingerprint,
        feature_set=feature_set,
        code_fingerprint=code_fingerprint,
    ).key


def _package_sources() -> list[tuple[str, bytes]]:
    """Read feature builders, helpers and their shared project indicator sources."""
    from tradingdev.domain import indicators
    from tradingdev.domain.ml import features as features_package  # local: circular

    sources: list[tuple[str, bytes]] = []
    for name, package in (("feature", features_package), ("indicator", indicators)):
        roots = list(getattr(package, "__path__", ()))
        if not roots:
            raise ValueError(
                f"cannot locate the {name} package source; pass a code_fingerprint "
                "explicitly instead"
            )
        package_sources: list[tuple[str, bytes]] = []
        for root in roots:
            directory = Path(root)
            for path in directory.rglob("*.py"):
                try:
                    relative = path.relative_to(directory).as_posix()
                    package_sources.append((f"{name}/{relative}", path.read_bytes()))
                except OSError as error:  # pragma: no cover - unreadable install
                    raise ValueError(
                        f"cannot read {name} source {path}: {error}"
                    ) from error
        if not package_sources:
            raise ValueError(
                f"the {name} package has no readable source; pass a "
                "code_fingerprint explicitly instead"
            )
        sources.extend(package_sources)
    return sorted(sources)


def _dependency_versions() -> dict[str, str]:
    """Versions of the numerical libraries used by the registered builders."""
    ta_version = getattr(talib, "__ta_version__", None)
    if not isinstance(ta_version, bytes) or not ta_version:
        raise ValueError("cannot read the TA-Lib C library version")
    return {
        "numpy": str(np.__version__),
        "pandas": str(pd.__version__),
        "TA-Lib": str(talib.__version__),
        "TA-Lib-C": ta_version.decode("ascii"),
    }


def feature_code_fingerprint(
    feature_sets: Sequence[object] | None = None,
) -> str:
    """Hash feature/indicator sources, numerical versions and declared columns.

    Use all registered sets unless supplied. Relative source paths make the
    identity independent of install location and Git. Unreadable source raises
    ``ValueError`` rather than allowing stale cache reuse.
    """
    sets: Sequence[object] = (
        list_feature_sets() if feature_sets is None else feature_sets
    )
    digest = hashlib.sha256()
    digest.update(b"feature-computation-v2")
    digest.update(json.dumps(_dependency_versions(), sort_keys=True).encode("utf-8"))
    for relative, source in _package_sources():
        digest.update(relative.encode("utf-8"))
        digest.update(source)
    for entry in sorted(sets, key=lambda item: str(getattr(item, "name", ""))):
        name = str(getattr(entry, "name", ""))
        columns = getattr(entry, "columns", ())
        digest.update(b"|set|")
        digest.update(name.encode("utf-8"))
        digest.update("|".join(str(column) for column in columns).encode("utf-8"))
    return digest.hexdigest()[:_DIGEST_LENGTH]
