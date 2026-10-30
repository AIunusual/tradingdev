"""Publish immutable data generations with an atomic successful-query receipt."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from uuid import uuid4

from filelock import FileLock

from tradingdev.domain.data.cache import CacheReceipt, DataIdentity

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    import pandas as pd


class DataCache:
    """Serialize refreshes and keep already-published generation paths readable."""

    def __init__(
        self, raw_dir: Path, processed_dir: Path, identity: DataIdentity
    ) -> None:
        self.identity = identity
        self.raw_dir = raw_dir / "cache-v2" / identity.key
        self.processed_dir = processed_dir / "cache-v2" / identity.key

    def lock(self, key: str) -> FileLock:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        return FileLock(self.processed_dir / f"{key}.lock", timeout=60)

    def receipt(self, key: str) -> CacheReceipt | None:
        path = self.processed_dir / f"{key}.json"
        if not path.exists():
            return None
        receipt = CacheReceipt.model_validate_json(path.read_bytes())
        if receipt.identity != self.identity:
            raise ValueError("Cache identity does not match the requested data")
        if not self.processed_path(receipt).is_file():
            raise ValueError("Cache receipt references a missing data generation")
        return receipt

    def processed_path(self, receipt: CacheReceipt) -> Path:
        return self.processed_dir / f"{receipt.generation}.parquet"

    def raw_path(self, receipt: CacheReceipt) -> Path:
        return self.raw_dir / f"{receipt.generation}.csv"

    def publish(
        self,
        key: str,
        *,
        raw: pd.DataFrame,
        frame: pd.DataFrame,
        query_start: datetime,
        query_end: datetime,
        fetched_at: datetime,
        save_raw: Callable[[pd.DataFrame, Path], None],
    ) -> CacheReceipt:
        """Call under lock; a failed refresh never changes the previous pointer.

        Readers can retain old immutable paths after dropping the lock. Published
        generations are retained; only this attempt's unpublished files are removed.
        """
        receipt = CacheReceipt(
            identity=self.identity,
            generation=uuid4().hex,
            query_start=query_start,
            query_end=query_end,
            fetched_at=fetched_at,
        )
        previous = self.receipt(key)
        if previous is not None and (
            receipt.fetched_at < previous.fetched_at
            or not receipt.covers(previous.query_start, previous.query_end)
        ):
            raise ValueError(
                "Refresh cannot replace newer or wider cached query evidence"
            )
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        raw_path, processed_path = self.raw_path(receipt), self.processed_path(receipt)
        published = False
        try:
            with (
                TemporaryDirectory(prefix=".pending-", dir=self.raw_dir) as raw_temp,
                TemporaryDirectory(
                    prefix=".pending-", dir=self.processed_dir
                ) as processed_temp,
            ):
                raw_stage = Path(raw_temp) / "raw.csv"
                data_stage = Path(processed_temp) / "data.parquet"
                receipt_stage = Path(processed_temp) / "receipt.json"
                save_raw(raw, raw_stage)
                frame.to_parquet(data_stage, index=False)
                receipt_stage.write_text(receipt.model_dump_json(), encoding="utf-8")
                for path in (raw_stage, data_stage, receipt_stage):
                    with path.open("rb") as stream:
                        os.fsync(stream.fileno())
                raw_stage.replace(raw_path)
                data_stage.replace(processed_path)
                receipt_stage.replace(self.processed_dir / f"{key}.json")
                published = True
            return receipt
        finally:
            if not published:
                raw_path.unlink(missing_ok=True)
                processed_path.unlink(missing_ok=True)
