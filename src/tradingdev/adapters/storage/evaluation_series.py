"""Atomic, content-bound publication of pointwise evaluation series."""

from __future__ import annotations

import io
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from filelock import FileLock

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.domain.ml.evaluation.series import (
    EvaluationSeriesBundle,
    EvaluationSeriesManifest,
    frame_bytes,
    to_json_records,
    verify_series,
)

_ARTIFACT_ID = re.compile(r"^[0-9a-f]{64}$")
_READ_ERRORS = (ValueError, TypeError, KeyError, OSError)


class EvaluationSeriesStore:
    """Publish immutable series artifacts without rewriting older evidence."""

    MAX_PAGE_SIZE = 200
    MAX_EXPORT_ROWS = 100_000
    MAX_EXPORT_BYTES = 8 * 1024 * 1024

    def __init__(self, root: Path | None = None, *, lock_timeout: float = 30) -> None:
        self.root = Path(root or WorkspacePaths().ml_evaluation_series)
        self.root.mkdir(parents=True, exist_ok=True)
        self._artifacts = self.root / "artifacts"
        self._artifacts.mkdir(parents=True, exist_ok=True)
        self._protected = self.root / "protected"
        self._protected.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(
            self.root / ".evaluation-series.lock", timeout=lock_timeout
        )
        self._verified: dict[
            str,
            tuple[
                tuple[tuple[int, int, int], tuple[int, int, int]],
                EvaluationSeriesBundle,
            ],
        ] = {}

    @property
    def lock(self) -> FileLock:
        """The lock shared with publication, protection and cleanup callers."""
        return self._lock

    def path_for(self, artifact_id: str) -> Path:
        """Where one immutable artifact lives."""
        if not _ARTIFACT_ID.fullmatch(artifact_id):
            raise ValueError(f"malformed evaluation series artifact ID: {artifact_id}")
        return self._artifacts / artifact_id

    def save(self, bundle: EvaluationSeriesBundle, *, protect: bool = False) -> str:
        """Publish a verified bundle, returning its stable artifact ID."""
        with self._lock:
            artifact_id = self._save_locked(bundle)
            if protect:
                self._protect_locked(artifact_id)
        return artifact_id

    def _save_locked(self, bundle: EvaluationSeriesBundle) -> str:
        verify_series(bundle)
        artifact_id = bundle.manifest.artifact_id
        destination = self.path_for(artifact_id)
        if destination.exists():
            existing = self._load_from(destination)
            if existing.manifest != bundle.manifest:
                raise ValueError(
                    "a different series was published under this artifact ID"
                )
            return artifact_id
        staging = self.root / ".staging" / uuid4().hex
        try:
            staging.mkdir(parents=True)
            (staging / "series.parquet").write_bytes(frame_bytes(bundle.frame))
            (staging / "manifest.json").write_text(
                json.dumps(
                    bundle.manifest.model_dump(mode="json"),
                    sort_keys=True,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            os.replace(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        self._verified.pop(artifact_id, None)
        return artifact_id

    def load(self, artifact_id: str) -> EvaluationSeriesBundle:
        """Read a bundle, refusing a missing or altered artifact."""
        path = self.path_for(artifact_id)
        if not path.exists():
            raise FileNotFoundError(f"evaluation series {artifact_id} is unavailable")
        return self._load_verified(artifact_id, path)

    @staticmethod
    def _signature(path: Path) -> tuple[int, int, int]:
        stat = path.stat()
        return (stat.st_mtime_ns, stat.st_size, stat.st_ino)

    def _load_verified(self, artifact_id: str, path: Path) -> EvaluationSeriesBundle:
        manifest_path = path / "manifest.json"
        series_path = path / "series.parquet"
        if not manifest_path.exists() or not series_path.exists():
            raise ValueError(f"incomplete evaluation series artifact: {path.name}")
        signature = (self._signature(manifest_path), self._signature(series_path))
        cached = self._verified.get(artifact_id)
        if cached is not None and cached[0] == signature:
            return cached[1]
        raw = series_path.read_bytes()
        bundle = self._load_from(path, raw)
        self._verified[artifact_id] = (signature, bundle)
        return bundle

    def _load_from(
        self, path: Path, raw: bytes | None = None
    ) -> EvaluationSeriesBundle:
        manifest_path = path / "manifest.json"
        series_path = path / "series.parquet"
        if not manifest_path.exists() or not series_path.exists():
            raise ValueError(f"incomplete evaluation series artifact: {path.name}")
        try:
            manifest = EvaluationSeriesManifest.from_record(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
            raw = series_path.read_bytes() if raw is None else raw
            frame = pd.read_parquet(io.BytesIO(raw))
        except (ValueError, TypeError, KeyError, OSError) as error:
            raise ValueError(
                f"unreadable evaluation series manifest: {path.name}: {error}"
            ) from error
        if manifest.artifact_id != path.name:
            raise ValueError(
                "evaluation series artifact ID does not match its directory"
            )
        bundle = EvaluationSeriesBundle(manifest=manifest, frame=frame)
        verify_series(bundle, content_bytes=raw)
        return bundle

    def page(
        self,
        *,
        kind: str | None = None,
        dataset_id: str | None = None,
        entry_id: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Discover saved series with a hard page-size bound."""
        if limit < 1 or limit > self.MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {self.MAX_PAGE_SIZE}")
        if offset < 0:
            raise ValueError("offset must not be negative")
        selected = [
            manifest
            for manifest in self._manifests()
            if (kind is None or manifest.kind == kind)
            and (dataset_id is None or manifest.dataset_id == dataset_id)
            and (entry_id is None or manifest.entry_id == entry_id)
        ]
        selected.sort(key=lambda item: (item.created_at, item.artifact_id))
        end = offset + limit
        items = [item.model_dump(mode="json") for item in selected[offset:end]]
        return {
            "items": items,
            "total": len(selected),
            "offset": offset,
            "next_offset": end if end < len(selected) else None,
        }

    def rows_page(
        self, artifact_id: str, *, offset: int = 0, limit: int = 50
    ) -> dict[str, Any]:
        """Read one verified artifact's rows in bounded, reassemblable pages."""
        if limit < 1 or limit > self.MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {self.MAX_PAGE_SIZE}")
        if offset < 0:
            raise ValueError("offset must not be negative")
        bundle = self.load(artifact_id)
        total = int(len(bundle.frame))
        if offset > total:
            raise ValueError(f"offset exceeds series row count ({total})")
        end = min(offset + limit, total)
        return {
            "manifest": bundle.manifest.model_dump(mode="json"),
            "rows": to_json_records(bundle.frame.iloc[offset:end]),
            "total": total,
            "offset": offset,
            "next_offset": end if end < total else None,
        }

    def export(self, artifact_id: str) -> dict[str, Any]:
        """Return the full saved series; oversized artifacts fail explicitly."""
        bundle = self.load(artifact_id)
        if bundle.manifest.row_count > self.MAX_EXPORT_ROWS:
            raise ValueError(
                f"evaluation series has {bundle.manifest.row_count} rows, above "
                f"the export bound of {self.MAX_EXPORT_ROWS}"
            )
        rows = to_json_records(bundle.frame)
        payload = {
            "manifest": bundle.manifest.model_dump(mode="json"),
            "rows": rows,
        }
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > self.MAX_EXPORT_BYTES:
            raise ValueError(
                "evaluation series export exceeds the byte bound of "
                f"{self.MAX_EXPORT_BYTES}; use row paging"
            )
        return payload

    def protect(self, artifact_id: str) -> None:
        """Mark one existing artifact as evidence a report still cites."""
        with self._lock:
            self._protect_locked(artifact_id)

    def _protect_locked(self, artifact_id: str) -> None:
        path = self.path_for(artifact_id)
        if not path.exists():
            raise FileNotFoundError(f"evaluation series {artifact_id} is unavailable")
        self._load_verified(artifact_id, path)
        (self._protected / artifact_id).touch(exist_ok=True)

    def prune(self, *, protected: set[str] | frozenset[str]) -> list[str]:
        """Remove only unreferenced artifacts; never touch protected evidence."""
        removed: list[str] = []
        with self._lock:
            for path in sorted(self._artifacts.iterdir()):
                if not path.is_dir() or path.name in protected:
                    continue
                if (self._protected / path.name).exists():
                    continue
                shutil.rmtree(path)
                self._verified.pop(path.name, None)
                removed.append(path.name)
        return removed

    def _manifests(self) -> list[EvaluationSeriesManifest]:
        manifests: list[EvaluationSeriesManifest] = []
        for path in sorted(self._artifacts.iterdir()):
            if not path.is_dir():
                continue
            manifest_path = path / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = EvaluationSeriesManifest.from_record(
                    json.loads(manifest_path.read_text(encoding="utf-8"))
                )
            except _READ_ERRORS:
                # A corrupt index entry is reported through `load`, never
                # silently omitted from discovery.
                raise ValueError(
                    f"unreadable evaluation series manifest: {path.name}"
                ) from None
            if manifest.artifact_id != path.name:
                raise ValueError(
                    "evaluation series artifact ID does not match its directory"
                )
            manifests.append(manifest)
        return manifests
