"""Filesystem persistence for signal-evaluation leaderboards.

The store is one JSON-lines file, one record per evaluated candidate, kept in
``entry_id`` order so the file is byte-stable no matter what order the
evaluations happened in.

Records are keyed by :func:`build_entry_id`, so re-evaluating the same model on
the same dataset under the same settings rewrites one row instead of adding a
second. That matters beyond tidiness: the number of candidates is what decides
how significant the winner has to be, and a duplicated candidate would make the
leaderboard quietly claim more evidence than it has.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.domain.ml.leaderboard import LeaderboardEntry

_FILE_NAME = "leaderboard.jsonl"


class LeaderboardStore:
    """Append-and-update storage of evaluated candidates."""

    def __init__(self, root: Path | None = None) -> None:
        if root is None:
            root = WorkspacePaths().ml_leaderboard
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        """Where the records live."""
        return self.root / _FILE_NAME

    def entries(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        horizon: int | None = None,
        feature_set: str | None = None,
    ) -> tuple[LeaderboardEntry, ...]:
        """Stored candidates, optionally narrowed by measurement conditions.

        Raises:
            ValueError: If a stored line cannot be read back. A record that
                silently vanishes would shrink the comparison and make the
                winner look stronger, so corruption is never skipped.
        """
        return tuple(
            entry
            for entry in self._read()
            if entry.context.matches(
                symbol=symbol,
                timeframe=timeframe,
                horizon=horizon,
                feature_set=feature_set,
            )
        )

    def record(self, entry: LeaderboardEntry) -> bool:
        """Store a candidate, replacing any earlier row for the same id.

        Returns:
            ``True`` if an existing record was replaced, ``False`` if this is
            the first record for the id.
        """
        existing = {item.entry_id: item for item in self._read()}
        replaced = entry.entry_id in existing
        existing[entry.entry_id] = entry
        ordered = sorted(existing.values(), key=lambda item: item.entry_id)
        payload = "".join(
            json.dumps(item.to_record(), ensure_ascii=False, sort_keys=True) + "\n"
            for item in ordered
        )
        # Write-then-rename: a reader never sees a half-written file, and the
        # previous records stay readable if this write fails.
        staging = self.path.with_suffix(self.path.suffix + ".tmp")
        staging.write_text(payload, encoding="utf-8")
        os.replace(staging, self.path)
        return replaced

    def _read(self) -> list[LeaderboardEntry]:
        if not self.path.exists():
            return []
        entries: list[LeaderboardEntry] = []
        for number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                entries.append(LeaderboardEntry.from_record(json.loads(line)))
            except (TypeError, ValueError, KeyError) as error:
                raise ValueError(
                    f"unreadable leaderboard record on line {number} of "
                    f"{self.path}: {error}"
                ) from error
        return entries
