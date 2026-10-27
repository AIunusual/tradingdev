"""Immutable, content-addressed inputs shared by independent executions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING, Self

import pandas as pd

from tradingdev.domain.execution import ManifestError

if TYPE_CHECKING:
    from pathlib import Path

    from tradingdev.app.data_service import LoadedDataset
    from tradingdev.domain.execution import ExecutionManifest


def execution_data_binding(manifest: ExecutionManifest) -> str:
    """Bind saved source settings and period without applying current defaults."""
    config = manifest.config_copy()
    backtest = config.get("backtest")
    keys = ("symbol", "timeframe", "start_date", "end_date")
    if (
        not isinstance(config.get("data"), dict)
        or not isinstance(backtest, dict)
        or any(not isinstance(backtest.get(key), str) for key in keys)
    ):
        raise ManifestError("Execution dataset requires complete saved data settings")
    return json.dumps(
        {
            "data": config["data"],
            "market": {key: backtest[key] for key in keys},
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


@dataclass(frozen=True)
class ExecutionDataset:
    """The complete merged frame, detached from caches and mutable Python objects.

    ``payload`` is the actual Parquet artifact to persist, including feature
    columns and pandas index metadata. ``source_dataset_id`` describes the
    original data request; ``dataset_id`` identifies these captured bytes.
    """

    payload: bytes = field(repr=False)
    data_binding_json: str
    processed_path: Path
    source_dataset_id: str
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        payload = bytes(self.payload)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "content_sha256", hashlib.sha256(payload).hexdigest())

    @classmethod
    def capture(cls, manifest: ExecutionManifest, dataset: LoadedDataset) -> Self:
        """Serialize a verified execution's fully loaded frame exactly once."""
        binding = execution_data_binding(manifest)
        payload = dataset.frame.to_parquet(engine="pyarrow", index=None)
        restored = pd.read_parquet(BytesIO(payload), engine="pyarrow")
        try:
            pd.testing.assert_frame_equal(
                dataset.frame,
                restored,
                check_exact=True,
                check_index_type=True,
                check_column_type=True,
                check_freq=False,
            )
        except AssertionError as exc:
            raise ValueError(
                "Execution dataset cannot preserve its values, index and dtypes "
                "in Parquet"
            ) from exc
        return cls(
            payload=payload,
            data_binding_json=binding,
            processed_path=dataset.processed_path,
            source_dataset_id=dataset.dataset_id,
        )

    @property
    def dataset_id(self) -> str:
        """Identify the captured contents independently of their cache path."""
        return f"sha256:{self.content_sha256}"

    def frame_for(self, manifest: ExecutionManifest) -> pd.DataFrame:
        """Decode fresh data only for the identical sources and research period."""
        if execution_data_binding(manifest) != self.data_binding_json:
            raise ManifestError(
                "Execution dataset does not match the manifest's data settings "
                "and research period"
            )
        return pd.read_parquet(BytesIO(self.payload), engine="pyarrow")
