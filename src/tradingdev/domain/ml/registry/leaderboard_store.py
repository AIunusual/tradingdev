"""Persist one JSON-lines record per candidate, ordered by entry_id.

Re-evaluation replaces that candidate instead of inflating comparison counts.
Writers lock the complete read/update/publish operation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from filelock import FileLock

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.domain.ml.leaderboard import LeaderboardEntry

_FILE_NAME = "leaderboard.jsonl"


class LeaderboardStore:
    """Append-and-update storage of evaluated candidates."""

    def __init__(self, root: Path | None = None, *, lock_timeout: float = 30) -> None:
        if root is None:
            root = WorkspacePaths().ml_leaderboard
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(self.root / ".leaderboard.lock", timeout=lock_timeout)

    @property
    def path(self) -> Path:
        """Where the records live."""
        return self.root / _FILE_NAME

    def entries(
        self,
        *,
        dataset_id: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        horizon: int | None = None,
        feature_set: str | None = None,
    ) -> tuple[LeaderboardEntry, ...]:
        """Filter stored candidates by dataset and measurement conditions.

        Corrupt records raise ValueError; skipping them would distort candidate
        counts and the reported significance threshold.
        """
        return tuple(
            entry
            for entry in self._read()
            if (dataset_id is None or entry.dataset_id == dataset_id)
            and entry.context.matches(
                symbol=symbol,
                timeframe=timeframe,
                horizon=horizon,
                feature_set=feature_set,
            )
        )

    def record(self, entry: LeaderboardEntry) -> bool:
        """Store a candidate; return whether it replaced an existing entry_id."""
        # The lock covers the complete read-modify-write, not only publication.
        # Its inode is retained so a waiting writer cannot acquire a new lock
        # file while another process still owns the old inode.
        with self._lock:
            existing = {item.entry_id: item for item in self._read()}
            replaced = entry.entry_id in existing
            existing[entry.entry_id] = entry
            ordered = sorted(existing.values(), key=lambda item: item.entry_id)
            payload = "".join(
                json.dumps(item.to_record(), ensure_ascii=False, sort_keys=True) + "\n"
                for item in ordered
            )
            # Same-directory replacement gives readers either the old complete
            # file or the new one. A failed write cannot leave partial records.
            staging: Path | None = None
            try:
                with NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    newline="",
                    prefix=".leaderboard-",
                    suffix=".tmp",
                    dir=self.root,
                    delete=False,
                ) as stream:
                    staging = Path(stream.name)
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(staging, self.path)
            finally:
                if staging is not None:
                    staging.unlink(missing_ok=True)
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
