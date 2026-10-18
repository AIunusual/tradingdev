"""Filesystem storage for computed feature matrices.

A cache is only allowed to save work, never to change an answer, so this store
is built around the opposite of trust: everything it hands back is checked
against what the caller asked for *and* against what was written.

* the **request** must agree with the sidecar on symbol, timeframe, bar
  fingerprint, feature set and code fingerprint - the same five things that
  formed the key, so a key collision cannot hand back another computation;
* the **declared columns** must be the columns the caller is about to use, so a
  matrix built for a different feature set is not silently reused;
* the **content hash** of the parquet must still match, so an edited,
  truncated or half-written file is discarded rather than read;
* the **shape** must still match the sidecar, which catches a frame that was
  replaced by one of a different length.

Every failed check means the same thing: recompute. That is the safe direction
to fail in - a miss costs time, a wrong hit costs the correctness of every
dataset, model and leaderboard row built on top of it - so a rejection is
logged and reported as a miss, never raised.

A cache that only ever grows is the same class of bug as a cache that serves
stale data: it fills the disk with matrices nobody will ask for again, because
every new bar version, feature set or code change is a *new* key while the old
one stays behind forever. So the store is bounded: it keeps the entries that
were used most recently, sweeps the leftovers that can never be loaded again,
and does both after every write. Evicting is safe for the same reason a miss is
safe - the only thing lost is work that can be redone, and the recomputed
matrix is verified by the same checks as any other.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from tradingdev.adapters.storage.filesystem import WorkspacePaths, sha256_file

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from tradingdev.domain.ml.feature_cache import FeatureCacheIdentity

logger = logging.getLogger(__name__)

_SUFFIX = ".parquet"
_META_SUFFIX = ".json"
_TMP_SUFFIX = ".tmp"

# A cache of one matrix per (symbol, timeframe, bars, feature set, code) fills
# up slowly - it takes a code change or a data edit, not a run, to add an entry
# - so the default bound is generous enough that eviction never fires in normal
# use. It exists for the case nobody plans for: months of edits left running.
_DEFAULT_MAX_ENTRIES = 256
_DEFAULT_MAX_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class CacheStats:
    """How full the cache is, and what it is allowed to hold."""

    entries: int
    total_bytes: int
    max_entries: int | None
    max_bytes: int | None

    @property
    def over_limit(self) -> bool:
        """Whether the cache currently exceeds either bound."""
        return (self.max_entries is not None and self.entries > self.max_entries) or (
            self.max_bytes is not None and self.total_bytes > self.max_bytes
        )


@dataclass(frozen=True)
class EvictionReport:
    """What a sweep removed, so the caller can see the cache is bounded."""

    removed: tuple[str, ...]
    """Cache keys dropped because the cache was over its bound."""
    swept: tuple[str, ...]
    """Files that could never have been loaded again, removed as garbage."""
    freed_bytes: int
    remaining: int

    @property
    def empty(self) -> bool:
        """Whether nothing was removed."""
        return not self.removed and not self.swept


@dataclass(frozen=True)
class _Entry:
    """One cache entry as it stands on disk right now."""

    key: str
    matrix: Path
    meta: Path
    size: int
    """Bytes held by the matrix and its sidecar together."""
    used_at: float
    """When it was last written or read; the oldest is evicted first."""
    usable: bool
    """Whether its sidecar can still verify a load."""


class FeatureCacheStore:
    """One parquet plus one sidecar per feature matrix, keyed by identity."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        max_entries: int | None = _DEFAULT_MAX_ENTRIES,
        max_bytes: int | None = _DEFAULT_MAX_BYTES,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Open the cache.

        Args:
            root: Directory holding the matrices; defaults to the workspace.
            max_entries: Most matrices to keep. ``None`` keeps all of them.
            max_bytes: Most bytes to keep, counting matrices and sidecars.
                ``None`` is unbounded.
            clock: Source of the timestamps used to decide what is stale;
                defaults to wall time. Injected in tests so recency is a fact
                rather than something that depends on how fast the test ran.

        Raises:
            ValueError: If a bound is set to less than one. A bound of zero is
                a cache that keeps nothing, which is a different thing
                expressed by accident.
        """
        if root is None:
            root = WorkspacePaths().ml_feature_cache
        for name, value in (("max_entries", max_entries), ("max_bytes", max_bytes)):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._clock = clock or time.time

    def path_for(self, key: str) -> Path:
        """Where the matrix for ``key`` is stored."""
        return self.root / f"{key}{_SUFFIX}"

    def meta_path_for(self, key: str) -> Path:
        """Where the verification sidecar for ``key`` is stored."""
        return self.root / f"{key}{_META_SUFFIX}"

    def keys(self) -> tuple[str, ...]:
        """Cache keys that have both a matrix and its sidecar."""
        return tuple(
            sorted(
                path.name[: -len(_SUFFIX)]
                for path in self.root.glob(f"*{_SUFFIX}")
                if self.meta_path_for(path.name[: -len(_SUFFIX)]).exists()
            )
        )

    def load(
        self,
        identity: FeatureCacheIdentity,
        *,
        feature_columns: Sequence[str],
    ) -> pd.DataFrame | None:
        """Return the cached matrix for ``identity``, or ``None`` for a miss.

        Args:
            identity: What the matrix must have been computed from.
            feature_columns: The columns the caller will read from it. They
                are stored alongside the matrix and compared, so a matrix
                built for a different feature set cannot be reused even if
                the keys agreed.

        Returns:
            The stored frame, or ``None`` when there is nothing stored, when
            the stored entry describes a different computation, or when the
            files no longer match their recorded content.
        """
        key = identity.key
        record = self._read_meta(self.meta_path_for(key))
        if record is None:
            return self._miss(key, "no usable sidecar")
        if not self._describes(record, identity, feature_columns):
            return self._miss(key, "sidecar describes a different computation")
        parquet = self.path_for(key)
        if not parquet.exists():
            return self._miss(key, "matrix file is missing")
        if sha256_file(parquet) != record["content_hash"]:
            return self._miss(key, "matrix content no longer matches its record")
        try:
            frame = pd.read_parquet(parquet)
        except (OSError, ValueError) as error:
            return self._miss(key, f"matrix is unreadable ({error})")
        if len(frame) != int(record["rows"]):
            return self._miss(key, "matrix row count changed")
        if list(frame.columns) != list(record["columns"]):
            return self._miss(key, "matrix columns changed")
        # A hit makes an entry the newest one. Without this the cache would
        # evict by age rather than by use, and a matrix read on every run
        # would be thrown away before one written later and never read again.
        self._touch(key, record)
        return frame

    def save(
        self,
        identity: FeatureCacheIdentity,
        frame: pd.DataFrame,
        *,
        feature_columns: Sequence[str],
    ) -> Path:
        """Store a matrix and the sidecar that lets a later read verify it.

        The matrix is written first and moved into place before the sidecar
        exists, so a sidecar never points at a file that is not there yet.

        Writing is also when the cache is brought back within its bounds, and
        the entry just written is protected from that sweep: a bounded cache
        that could discard what it was asked to keep would not be a cache.
        """
        key = identity.key
        parquet = self.path_for(key)
        staging = parquet.with_name(f"{parquet.name}.tmp")
        frame.to_parquet(staging, index=False)
        os.replace(staging, parquet)
        record = {
            "cache_key": key,
            "symbol": identity.symbol,
            "timeframe": identity.timeframe,
            "data_fingerprint": identity.data_fingerprint,
            "feature_set": identity.feature_set,
            "code_fingerprint": identity.code_fingerprint,
            "feature_columns": [str(column) for column in feature_columns],
            "columns": [str(column) for column in frame.columns],
            "rows": int(len(frame)),
            "content_hash": sha256_file(parquet),
            "used_at": self._clock(),
        }
        self._write_meta(self.meta_path_for(key), record)
        report = self.evict(keep=key)
        if not report.empty:
            logger.info(
                "feature cache after writing %s: removed %d, swept %d, "
                "freed %d bytes, %d entries left",
                key,
                len(report.removed),
                len(report.swept),
                report.freed_bytes,
                report.remaining,
            )
        if self.stats().over_limit:
            # The entry just written cannot be dropped to make room, so a
            # matrix bigger than the whole budget stays over it. Say so
            # rather than leaving the bound quietly unmet.
            logger.warning(
                "feature cache is over its bound even after eviction; "
                "raise max_bytes or max_entries"
            )
        return parquet

    def stats(self) -> CacheStats:
        """How much the cache currently holds against what it may hold.

        Bytes counts every file the cache owns, including the ones a sweep
        would delete: what matters about a full disk is what is on it, not
        what is still loadable. Half-written ``.tmp`` files are left out -
        they are another write in progress, not cache content.
        """
        total = 0
        for path in self.root.iterdir():
            if path.is_file() and not path.name.endswith(_TMP_SUFFIX):
                try:
                    total += path.stat().st_size
                except OSError:  # pragma: no cover - vanishing file
                    continue
        return CacheStats(
            entries=len(self._entries()),
            total_bytes=total,
            max_entries=self.max_entries,
            max_bytes=self.max_bytes,
        )

    def evict(self, *, keep: str | None = None) -> EvictionReport:
        """Drop what can never be loaded, then the least recently used rest.

        Two different things are removed, for two different reasons:

        * **swept** - an orphan matrix, an orphan sidecar, or a pair whose
          sidecar can no longer verify anything. ``load`` already rejects all
          three, so they are pure garbage and are removed however much room
          there is;
        * **removed** - real entries, oldest use first, only while the cache
          is over one of its bounds.

        Args:
            keep: A key that must survive even if it is the oldest one, so a
                caller can sweep right after writing without throwing away
                what it just computed.

        Returns:
            What was removed and what is left.
        """
        swept: list[str] = []
        freed = 0
        for key in self._garbage():
            freed += self._delete(key)
            swept.append(key)

        removed: list[str] = []
        victims = sorted(
            (entry for entry in self._entries() if entry.key != keep),
            key=lambda entry: (entry.used_at, entry.key),
        )
        stats = self.stats()
        index = 0
        # Each iteration re-reads the disk: a deletion that fails - a file
        # locked by something else - must not be counted as freed space, and
        # the index always advances, so the loop cannot spin on one victim.
        while stats.over_limit and index < len(victims):
            victim = victims[index]
            index += 1
            freed += self._delete(victim.key)
            removed.append(victim.key)
            stats = self.stats()
        return EvictionReport(
            removed=tuple(removed),
            swept=tuple(swept),
            freed_bytes=freed,
            remaining=len(self._entries()),
        )

    def _garbage(self) -> list[str]:
        """Keys whose files can never satisfy a load, however much room there is.

        A matrix without its sidecar cannot be verified, a sidecar without its
        matrix has nothing to verify, and a sidecar that fails to parse
        verifies nothing. All three are what is left behind by an interrupted
        write or a partial delete.
        """
        metas = self._metas()
        matrices = self._matrices()
        garbage: list[str] = []
        for key, meta in metas.items():
            if key not in matrices or self._read_meta(meta) is None:
                garbage.append(key)
        garbage.extend(key for key in matrices if key not in metas)
        return sorted(set(garbage))

    def _matrices(self) -> dict[str, Path]:
        return {
            path.name[: -len(_SUFFIX)]: path for path in self.root.glob(f"*{_SUFFIX}")
        }

    def _metas(self) -> dict[str, Path]:
        return {
            path.name[: -len(_META_SUFFIX)]: path
            for path in self.root.glob(f"*{_META_SUFFIX}")
        }

    def _entries(self) -> list[_Entry]:
        """Every complete pair on disk, loadable or not."""
        metas = self._metas()
        entries = []
        for key, matrix in self._matrices().items():
            meta = metas.get(key)
            if meta is None:
                continue
            record = self._read_meta(meta)
            entries.append(
                _Entry(
                    key=key,
                    matrix=matrix,
                    meta=meta,
                    size=self._size(matrix) + self._size(meta),
                    used_at=self._used_at(record, matrix, meta),
                    usable=record is not None,
                )
            )
        return entries

    @staticmethod
    def _size(path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:  # pragma: no cover - vanishing file
            return 0

    def _used_at(
        self, record: dict[str, Any] | None, matrix: Path, meta: Path
    ) -> float:
        """When an entry was last written or read, oldest-first if unknown.

        Entries written before recency was recorded have no ``used_at``, and
        guessing that they are new would evict everything else first; their
        file mtimes are the honest fallback.
        """
        raw = record.get("used_at") if record is not None else None
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return float(raw)
        try:
            return max(matrix.stat().st_mtime, meta.stat().st_mtime)
        except OSError:  # pragma: no cover - vanishing file
            return 0.0

    def _touch(self, key: str, record: dict[str, Any]) -> None:
        """Record that an entry was used just now.

        Best effort by design: a cache that cannot update its bookkeeping is a
        cache that recomputes a little more, not one that fails to answer.
        """
        record["used_at"] = self._clock()
        try:
            self._write_meta(self.meta_path_for(key), record)
        except OSError as error:  # pragma: no cover - read-only cache
            logger.warning("could not record use of %s: %s", key, error)

    def _delete(self, key: str) -> int:
        """Remove one entry's files, returning the bytes freed.

        Tolerates a file that cannot be removed and reports it as freeing
        nothing: the outcome is a cache that stays a little too full, which is
        better than a write that fails because housekeeping could not run.
        """
        freed = 0
        for path in (self.path_for(key), self.meta_path_for(key)):
            if not path.exists():
                continue
            size = self._size(path)
            try:
                path.unlink()
            except OSError as error:  # pragma: no cover - locked file
                logger.warning("could not remove %s: %s", path.name, error)
                continue
            freed += size
        return freed

    @staticmethod
    def _describes(
        record: dict[str, Any],
        identity: FeatureCacheIdentity,
        feature_columns: Sequence[str],
    ) -> bool:
        """Whether a sidecar describes exactly the requested computation."""
        return (
            record.get("symbol") == identity.symbol
            and record.get("timeframe") == identity.timeframe
            and record.get("data_fingerprint") == identity.data_fingerprint
            and record.get("feature_set") == identity.feature_set
            and record.get("code_fingerprint") == identity.code_fingerprint
            and list(record.get("feature_columns", ()))
            == [str(column) for column in feature_columns]
        )

    @staticmethod
    def _read_meta(path: Path) -> dict[str, Any] | None:
        """Read a sidecar, treating an unreadable or incomplete one as a miss."""
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(record, dict):
            return None
        # A sidecar without these cannot verify anything, so it is a miss.
        required = ("content_hash", "rows", "columns", "feature_columns")
        return record if all(key in record for key in required) else None

    @staticmethod
    def _write_meta(path: Path, record: dict[str, Any]) -> None:
        payload = json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True)
        staging = path.with_name(f"{path.name}.tmp")
        staging.write_text(payload, encoding="utf-8")
        os.replace(staging, path)

    @staticmethod
    def _miss(key: str, reason: str) -> None:
        """Log a rejected entry: a miss is safe, but a silent one is a hazard."""
        logger.info("feature cache miss for %s: %s", key, reason)
