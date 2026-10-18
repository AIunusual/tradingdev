"""Filesystem workspace adapter."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from tradingdev.shared.paths import resolve_workspace_root

if TYPE_CHECKING:
    from pathlib import Path


class WorkspacePaths:
    """Resolve and create TradingDev runtime workspace paths."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = resolve_workspace_root(root)

    @property
    def generated_strategies(self) -> Path:
        return self.root / "generated_strategies"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def ml_datasets(self) -> Path:
        return self.root / "ml_datasets"

    @property
    def ml_leaderboard(self) -> Path:
        return self.root / "ml_leaderboard"

    @property
    def ml_feature_cache(self) -> Path:
        return self.root / "ml_feature_cache"

    @property
    def feature_requests(self) -> Path:
        return self.root / "feature_requests"

    @property
    def raw_data(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed_data(self) -> Path:
        return self.root / "data" / "processed"

    def ensure(self) -> None:
        """Create the standard workspace directory tree."""
        for path in (
            self.root,
            self.generated_strategies,
            self.configs,
            self.runs,
            self.models,
            self.ml_datasets,
            self.ml_leaderboard,
            self.ml_feature_cache,
            self.feature_requests,
            self.raw_data,
            self.processed_data,
        ):
            path.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    """Return an ISO timestamp in UTC."""
    return datetime.now(UTC).isoformat()


def sha256_text(content: str) -> str:
    """Return the SHA-256 hash of a text payload."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hash of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON object from disk."""
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON object to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
