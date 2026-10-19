"""Reusing a computed feature matrix, and proving the reuse is safe.

Turning bars into features is deterministic: the same bars through the same
builder produce the same columns. So the work only has to be done once per
(symbol, timeframe, bars, feature set, feature code) - which matters because
the same features are recomputed for every label horizon and re-stored in full
inside every dataset built from them.

Reuse is only safe if the key cannot collide with a different computation, so
every ingredient is here on purpose:

* **bar content**, hashed from the values rather than from file size and
  mtime: an edit that happens to keep the size, or a copy that keeps the
  mtime, would otherwise look unchanged and hand back stale features.
* **symbol and timeframe**, because two instruments can carry byte-identical
  bars in a test or in a synthetic feed, and those are different series.
* **feature set**, because a different set emits different columns.
* **feature code**, because a fixed indicator must invalidate features that
  were computed by the broken version. This is derived from the builders'
  source, and can be injected by a caller that knows better.

Nothing here touches the workspace or the clock; the store owns the files.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

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
    """Hash the *values* of a bar frame.

    Size and mtime are not used: a bar can be edited without changing its
    length, and a copied file can keep an old mtime. Hashing the values is
    the only way to know the inputs to the builder are the same inputs.
    """
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
    """Every Python file of the feature package, as ``(relative path, bytes)``.

    The whole package counts, not just the file a builder happens to live in:
    builders delegate to helpers (a change inside ``technical_features`` moves
    the RSI column just as much as a change in the builder itself), and the
    package is small enough that hashing it costs a few reads.
    """
    from tradingdev.domain.ml import features as features_package  # local: circular

    roots = list(getattr(features_package, "__path__", ()))
    if not roots:
        raise ValueError(
            "cannot locate the feature package source; pass a code_fingerprint "
            "explicitly instead"
        )
    sources: list[tuple[str, bytes]] = []
    for root in roots:
        directory = Path(root)
        for path in directory.rglob("*.py"):
            try:
                relative = path.relative_to(directory).as_posix()
                sources.append((relative, path.read_bytes()))
            except OSError as error:  # pragma: no cover - unreadable install
                raise ValueError(
                    f"cannot read feature source {path}: {error}"
                ) from error
    if not sources:
        raise ValueError(
            "the feature package has no readable source; pass a "
            "code_fingerprint explicitly instead"
        )
    return sorted(sources)


def feature_code_fingerprint(
    feature_sets: Sequence[object] | None = None,
) -> str:
    """Fingerprint the code that turns bars into features.

    Derived from the feature package's source and from each set's declared
    columns, rather than from a version constant or a git call, so it changes
    exactly when the code that produces the columns changes - and it can be
    computed anywhere, with no subprocess and no dependency on the working
    directory. A cache key built from it is therefore reproducible on another
    machine, which a ``git rev-parse`` of the local checkout is not.

    Args:
        feature_sets: Sets whose declarations to include; defaults to every
            registered set.

    Returns:
        A 16-char hex digest.

    Raises:
        ValueError: If the feature source cannot be read. A missing
            fingerprint must not silently become "unchanged": that would let
            a fixed indicator keep serving features computed by the bug.
    """
    sets: Sequence[object] = (
        list_feature_sets() if feature_sets is None else feature_sets
    )
    digest = hashlib.sha256()
    digest.update(b"feature-package-v1")
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
