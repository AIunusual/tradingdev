"""Publish and verify immutable, content-addressed ML datasets."""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
from filelock import FileLock

from tradingdev.adapters.storage.filesystem import write_json

_ID = re.compile(r"ds_[0-9a-f]{16}")
_SCHEMA = 2


def _identity(metadata: dict[str, Any]) -> str:
    payload = json.dumps(metadata, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return "ds_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


class MLDatasetStore:
    """Keep dataset bytes and their metadata together under a verified identity."""

    def __init__(self, root: Path, *, lock_timeout: float = 30) -> None:
        self.root = root
        self._lock_timeout = lock_timeout

    def save(
        self, frame: pd.DataFrame, metadata: dict[str, Any]
    ) -> tuple[Path, dict[str, Any]]:
        """Publish a complete directory atomically; never replace an existing ID."""
        blob = frame.to_parquet(index=False)
        manifest = {
            **metadata,
            "schema_version": _SCHEMA,
            "rows": len(frame),
            "content_hash": hashlib.sha256(blob).hexdigest(),
        }
        dataset_id = _identity(manifest)
        manifest["dataset_id"] = dataset_id
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / dataset_id
        with FileLock(str(self.root / ".publish.lock"), timeout=self._lock_timeout):
            if destination.exists():
                _, existing = self.load(dataset_id)
                if existing != manifest:
                    raise ValueError(
                        "dataset identity collision; existing data preserved"
                    )
            else:
                with TemporaryDirectory(prefix=".publish-", dir=self.root) as temporary:
                    staging = Path(temporary) / dataset_id
                    staging.mkdir()
                    (staging / "dataset.parquet").write_bytes(blob)
                    write_json(staging / "manifest.json", manifest)
                    staging.rename(destination)
        return destination / "dataset.parquet", manifest

    def load(self, dataset_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Verify metadata and the exact bytes being read before exposing data."""
        if _ID.fullmatch(dataset_id) is None:
            raise ValueError(f"malformed dataset_id: {dataset_id}")
        directory = self.root / dataset_id
        try:
            raw = json.loads((directory / "manifest.json").read_text("utf-8"))
            blob = (directory / "dataset.parquet").read_bytes()
        except FileNotFoundError as exc:
            raise ValueError(
                f"unknown dataset_id or incomplete dataset: {dataset_id}"
            ) from exc
        if not isinstance(raw, dict) or raw.get("schema_version") != _SCHEMA:
            raise ValueError("unsupported dataset schema; rebuild the dataset")
        manifest: dict[str, Any] = raw
        identity = {
            key: value for key, value in manifest.items() if key != "dataset_id"
        }
        if (
            manifest.get("dataset_id") != dataset_id
            or _identity(identity) != dataset_id
        ):
            raise ValueError("dataset metadata identity mismatch")
        if hashlib.sha256(blob).hexdigest() != manifest.get("content_hash"):
            raise ValueError("dataset content hash mismatch")
        frame = pd.read_parquet(io.BytesIO(blob))
        if len(frame) != manifest.get("rows"):
            raise ValueError("dataset row count mismatch")
        return frame, manifest
