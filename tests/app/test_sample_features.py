"""Feature-cache sampling preserves joins without unbounded reads or expansion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow.dataset as arrow_dataset
import pytest
from tests.data_cache import cached_market_path, seed_market_cache

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.data_service import DataService
from tradingdev.domain.backtest.schemas import BacktestConfig

if TYPE_CHECKING:
    from pathlib import Path


def _service(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    timestamps: pd.DatetimeIndex,
    *,
    end: str | None = None,
) -> tuple[DataService, BacktestConfig]:
    service = DataService(WorkspacePaths(root / "workspace"))
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            **dict.fromkeys(("open", "high", "low", "close", "volume"), 100.0),
        }
    )
    seed_market_cache(root / "workspace/data/processed", frame)

    class Crawler:
        def fetch(self, *_args: Any, **_kwargs: Any) -> pd.DataFrame:
            pytest.fail("A cached sample must not fetch data")

    monkeypatch.setattr(
        "tradingdev.app.data_service.create_crawler", lambda *_: Crawler()
    )
    monkeypatch.setattr("tradingdev.app.data_service.DeribitDVOLCrawler", Crawler)
    config = BacktestConfig(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date=end or timestamps[-1].ceil("s").tz_localize(None).isoformat(),
        init_cash=10000,
    )
    return service, config


def _raw(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "data": {
            "requirements": {
                "market": {"symbol": "BTC/USDT", "timeframe": "1h"},
                "features": features,
            }
        }
    }


@pytest.mark.parametrize("feature_type", ["custom", "funding_rate", "dvol"])
@pytest.mark.parametrize(
    "storage", ["naive", "UTC", "America/New_York", "strings", "integers"]
)
def test_feature_cache_reads_projected_batches_and_only_sample_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    feature_type: str,
    storage: str,
) -> None:
    timestamps = pd.date_range("2024-01-01", periods=64, freq="h", tz="UTC")
    service, config = _service(tmp_path, monkeypatch, timestamps)
    stored = pd.date_range("2023-01-01", periods=10000, freq="h", tz="UTC")
    frame = pd.DataFrame({"timestamp": stored, "value": 42.0, "unused": "payload"})
    if storage == "naive":
        frame["timestamp"] = stored.tz_localize(None)
    elif storage == "strings":
        frame["timestamp"] = stored.astype(str)
    elif storage == "integers":
        frame["timestamp"] = stored.as_unit("ns").astype("int64")
    else:
        frame["timestamp"] = stored.tz_convert(storage)
    path = tmp_path / "feature.parquet"
    frame.sample(frac=1, random_state=42).to_parquet(
        path, index=False, row_group_size=127
    )
    before = path.read_bytes()
    original_dataset = arrow_dataset.dataset
    observations: list[dict[str, Any]] = []
    batches: list[int] = []

    class Dataset:
        def __init__(self, source: Path) -> None:
            self.source = source
            self.dataset = original_dataset(source, format="parquet")
            self.schema = self.dataset.schema

        def scanner(self, **kwargs: Any) -> Any:
            scanner = self.dataset.scanner(**kwargs)
            if self.source != path:
                return scanner
            observations.append(kwargs)

            class Scanner:
                def to_batches(self) -> Any:
                    for batch in scanner.to_batches():
                        batches.append(len(batch))
                        yield batch

            return Scanner()

    monkeypatch.setattr(arrow_dataset, "dataset", lambda source, **_: Dataset(source))

    def unexpected_read(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Sample features must not use a full Parquet read")

    monkeypatch.setattr(pd, "read_parquet", unexpected_read)
    result = service.load_sample(
        _raw(
            [
                {
                    "type": feature_type,
                    "source": "local",
                    "column": "value",
                    "path": str(path),
                }
            ]
        ),
        config,
        max_rows=64,
        output_dir=tmp_path / "sample",
    )
    assert len(result.frame) == 64
    assert result.frame["value"].eq(42).all()
    assert "unused" not in result.frame
    assert len(observations) == 1
    assert observations[0]["columns"] == ["timestamp", "value"]
    assert observations[0]["batch_size"] == 64
    assert observations[0]["batch_readahead"] == 0
    assert observations[0]["fragment_readahead"] == 0
    assert (observations[0]["filter"] is None) == (storage in {"strings", "integers"})
    assert batches and max(batches) <= 64
    assert path.read_bytes() == before


@pytest.mark.parametrize("preferred_column", [False, True])
def test_dvol_uses_original_range_default_cache_and_column_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preferred_column: bool
) -> None:
    timestamps = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    service, config = _service(tmp_path, monkeypatch, timestamps, end="2025-12-31")
    path = tmp_path / "workspace/data/processed/btc_dvol_1h_2024_2025.parquet"
    frame = pd.DataFrame({"timestamp": timestamps, "dvol_close": 51.0})
    if preferred_column:
        frame["volatility"] = 42.0
    frame.to_parquet(path, index=False)
    before = path.read_bytes()
    result = service.load_sample(
        _raw([{"type": "dvol", "source": "deribit", "column": "volatility"}]),
        config,
        max_rows=4,
        output_dir=tmp_path / "sample",
    )
    assert result.frame["volatility"].eq(42.0 if preferred_column else 51.0).all()
    assert not (tmp_path / "sample/feature-0.parquet").exists()
    assert path.read_bytes() == before


@pytest.mark.parametrize("sample", [False, True])
@pytest.mark.parametrize(
    ("feature_type", "stored_column"),
    [
        ("custom", "value"),
        ("funding_rate", "value"),
        ("dvol", "value"),
        ("dvol", "dvol_close"),
    ],
)
def test_feature_loading_only_fills_from_earlier_matching_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample: bool,
    feature_type: str,
    stored_column: str,
) -> None:
    timestamps = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    service, config = _service(tmp_path, monkeypatch, timestamps)
    path = tmp_path / "feature.parquet"
    pd.DataFrame(
        {
            "timestamp": [
                timestamps[0] - pd.Timedelta(hours=1),
                timestamps[1],
                timestamps[3],
                timestamps[4],
                *([timestamps[0] + pd.Timedelta(minutes=30)] * 1000),
                timestamps[-1] + pd.Timedelta(hours=1),
            ],
            stored_column: [999.0, 42.0, None, 43.0, *([999.0] * 1000), 999.0],
        }
    ).sample(frac=1, random_state=42).to_parquet(path, index=False, row_group_size=7)
    market_path = cached_market_path(tmp_path / "workspace/data/processed")
    before = {source: source.read_bytes() for source in (path, market_path)}
    raw = _raw(
        [
            {
                "type": feature_type,
                "source": "local",
                "column": "value",
                "path": str(path),
            }
        ]
    )
    result = (
        service.load_sample(raw, config, max_rows=6, output_dir=tmp_path / "sample")
        if sample
        else service.load(raw, config)
    )
    pd.testing.assert_series_equal(
        result.frame["value"],
        pd.Series([float("nan"), 42.0, 42.0, 42.0, 43.0, 43.0], name="value"),
    )
    assert result.frame["timestamp"].tolist() == timestamps.tolist()
    assert {source: source.read_bytes() for source in before} == before
    if sample:
        pd.testing.assert_frame_equal(
            pd.read_parquet(result.processed_path), result.frame
        )


@pytest.mark.parametrize("empty_file", [False, True])
def test_feature_sampling_keeps_missing_values_when_no_timestamps_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_file: bool
) -> None:
    timestamps = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    service, config = _service(tmp_path, monkeypatch, timestamps)
    frame = pd.DataFrame(
        {"timestamp": timestamps + pd.Timedelta(minutes=30), "value": 42.0}
    )
    path = tmp_path / "feature.parquet"
    (frame.head(0) if empty_file else frame).to_parquet(path, index=False)
    result = service.load_sample(
        _raw(
            [
                {
                    "type": "custom",
                    "source": "local",
                    "column": "value",
                    "path": str(path),
                }
            ]
        ),
        config,
        max_rows=3,
        output_dir=tmp_path / "sample",
    )
    assert result.frame["value"].isna().all()


def test_duplicate_features_preserve_order_when_join_fits_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamps = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    service, config = _service(tmp_path, monkeypatch, timestamps)
    path = tmp_path / "feature.parquet"
    pd.DataFrame(
        {"timestamp": [timestamps[0], timestamps[0]], "value": [42.0, 43.0]}
    ).to_parquet(path, index=False)
    result = service.load_sample(
        _raw(
            [
                {
                    "type": "custom",
                    "source": "local",
                    "column": "value",
                    "path": str(path),
                }
            ]
        ),
        config,
        max_rows=3,
        output_dir=tmp_path / "sample",
    )
    assert result.frame["timestamp"].tolist() == [timestamps[0], *timestamps]
    assert result.frame["value"].tolist() == [42.0, 43.0, 43.0]


@pytest.mark.parametrize("duplicates", [3, 1000])
def test_duplicate_feature_overflow_is_rejected_before_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, duplicates: int
) -> None:
    timestamps = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    service, config = _service(tmp_path, monkeypatch, timestamps)
    path = tmp_path / "feature.parquet"
    pd.DataFrame({"timestamp": [timestamps[0]] * duplicates, "value": 42.0}).to_parquet(
        path, index=False
    )

    def unexpected_merge(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail(
            "An oversized feature join must be rejected before materializing it"
        )

    monkeypatch.setattr(pd.DataFrame, "merge", unexpected_merge)
    with pytest.raises(
        ValueError, match="Feature joins exceeded the sample row budget"
    ):
        service.load_sample(
            _raw(
                [
                    {
                        "type": "custom",
                        "source": "local",
                        "column": "value",
                        "path": str(path),
                    }
                ]
            ),
            config,
            max_rows=3,
            output_dir=tmp_path / "sample",
        )
    assert not (tmp_path / "sample/sample.parquet").exists()


def test_multiple_features_check_left_key_multiplicity_before_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamps = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    service, config = _service(tmp_path, monkeypatch, timestamps)
    features = []
    for column in ("first", "second"):
        path = tmp_path / f"{column}.parquet"
        pd.DataFrame({"timestamp": [timestamps[0]] * 2, column: 42.0}).to_parquet(
            path, index=False
        )
        features.append(
            {"type": "custom", "source": "local", "column": column, "path": str(path)}
        )
    merge = pd.DataFrame.merge
    joined_rows = []

    def observe_merge(left: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        result = merge(left, *args, **kwargs)
        joined_rows.append(len(result))
        return result

    monkeypatch.setattr(pd.DataFrame, "merge", observe_merge)
    with pytest.raises(
        ValueError, match="Feature joins exceeded the sample row budget"
    ):
        service.load_sample(
            _raw(features), config, max_rows=4, output_dir=tmp_path / "sample"
        )
    assert joined_rows == [3]


def test_feature_nanosecond_window_preserves_inclusive_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamps = pd.date_range(
        "2024-01-01 00:00:00.000000001", periods=3, freq="ns", tz="UTC"
    )
    service, config = _service(tmp_path, monkeypatch, timestamps)
    path = tmp_path / "feature.parquet"
    pd.DataFrame({"timestamp": timestamps, "value": [1.0, 2.0, 3.0]}).to_parquet(
        path, index=False
    )
    result = service.load_sample(
        _raw(
            [
                {
                    "type": "custom",
                    "source": "local",
                    "column": "value",
                    "path": str(path),
                }
            ]
        ),
        config,
        max_rows=3,
        output_dir=tmp_path / "sample",
    )
    assert result.frame["value"].tolist() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("invalid", ["timestamp", "value", "timestamp_value"])
def test_invalid_feature_cache_fails_without_creating_a_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    timestamps = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    service, config = _service(tmp_path, monkeypatch, timestamps)
    frame = pd.DataFrame({"timestamp": timestamps, "value": 42.0})
    if invalid == "timestamp_value":
        frame["timestamp"] = ["2024-01-01", "not-a-timestamp", "2024-01-02"]
    else:
        frame = frame.drop(columns=invalid)
    path = tmp_path / "feature.parquet"
    frame.to_parquet(path, index=False)
    before = path.read_bytes()
    with pytest.raises((ValueError, KeyError)):
        service.load_sample(
            _raw(
                [
                    {
                        "type": "custom",
                        "source": "local",
                        "column": "value",
                        "path": str(path),
                    }
                ]
            ),
            config,
            max_rows=3,
            output_dir=tmp_path / "sample",
        )
    assert not (tmp_path / "sample/sample.parquet").exists()
    assert path.read_bytes() == before
