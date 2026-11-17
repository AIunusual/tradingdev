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
from pydantic import TypeAdapter

from tradingdev.adapters.storage.filesystem import write_json
from tradingdev.domain.ml.market_evidence import (
    RawMarketEvidence,
    evaluated_rows_fingerprint,
)

_ID = re.compile(r"ds_[0-9a-f]{16}")
_LEGACY_SCHEMA = 2
_SCHEMA = 3
_RAW_MARKET = TypeAdapter(RawMarketEvidence)


def _identity(metadata: dict[str, Any]) -> str:
    payload = json.dumps(metadata, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return "ds_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _evaluation_rows(frame: pd.DataFrame, manifest: dict[str, Any]) -> dict[str, str]:
    raw = _RAW_MARKET.validate_python(manifest["raw_market"], strict=True)
    is_panel = manifest["kind"] == "panel"
    symbols = manifest["symbols"] if is_panel else [manifest["symbol"]]
    horizon = manifest["horizon"]
    if (
        type(horizon) is not int
        or horizon < 1
        or manifest["kind"] not in ("single", "panel")
        or not symbols
        or [series["symbol"] for series in raw["series"]] != sorted(set(symbols))
        or any(
            series["source"] != manifest["source"]
            or series["market_type"] != manifest["market_type"]
            or series["timeframe"] != manifest["timeframe"]
            or re.fullmatch(r"[0-9a-f]{64}", series["fingerprint"]) is None
            or not series["first_utc"].endswith("+00:00")
            or not series["last_utc"].endswith("+00:00")
            for series in raw["series"]
        )
        or not {"timestamp", "label_start", "label_end", "raw_row"} <= set(frame)
    ):
        raise ValueError("invalid dataset raw market evidence")
    if is_panel:
        if "symbol" not in frame or set(frame["symbol"]) != set(symbols):
            raise ValueError(
                "dataset panel membership does not match raw market evidence"
            )
        groups = dict(iter(frame.groupby("symbol")))
    else:
        groups = {symbols[0]: frame}
    return {
        series["symbol"]: evaluated_rows_fingerprint(
            groups[series["symbol"]], series=series, horizon=horizon
        )
        for series in raw["series"]
    }


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
            "schema_version": _SCHEMA if "raw_market" in metadata else _LEGACY_SCHEMA,
            "rows": len(frame),
            "content_hash": hashlib.sha256(blob).hexdigest(),
        }
        if manifest["schema_version"] == _SCHEMA:
            manifest["evaluation_rows"] = _evaluation_rows(frame, manifest)
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
        except OSError as exc:
            raise ValueError(f"cannot read saved dataset: {dataset_id}") from exc
        if not isinstance(raw, dict) or raw.get("schema_version") not in (
            _LEGACY_SCHEMA,
            _SCHEMA,
        ):
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
        if manifest["schema_version"] == _SCHEMA and (
            "raw_market" not in manifest
            or _evaluation_rows(frame, manifest) != manifest.get("evaluation_rows")
        ):
            raise ValueError("dataset evaluation row evidence mismatch")
        return frame, manifest
