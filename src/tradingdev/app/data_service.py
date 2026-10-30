"""Application service for market data access."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from tradingdev.adapters.storage.data_cache import DataCache
from tradingdev.adapters.storage.feature_samples import read_feature_sample
from tradingdev.adapters.storage.filesystem import (
    WorkspacePaths,
    sha256_file,
    sha256_text,
)
from tradingdev.app.contracts.common import DEFAULT_PAGE_LIMIT, Page
from tradingdev.app.contracts.data import CachedDataset
from tradingdev.app.data_manager import DataManager
from tradingdev.app.market_sample import load_market_sample
from tradingdev.app.pagination import paginate
from tradingdev.domain.backtest.schemas import BacktestConfig
from tradingdev.domain.data.cache import CacheReceipt, DataIdentity
from tradingdev.domain.data.crawlers.deribit_dvol import DeribitDVOLCrawler
from tradingdev.domain.data.crawlers.registry import create_crawler, describe_sources
from tradingdev.domain.data.loader import DataLoader
from tradingdev.domain.data.processor import DataProcessor
from tradingdev.domain.data.requirements import (
    DataRequirement,
    FeatureSpec,
    MarketDataSpec,
)
from tradingdev.domain.data.schemas import DataConfig, MarketDataRequest
from tradingdev.shared.utils.config import load_config


@dataclass(frozen=True)
class LoadedDataset:
    """Loaded market data and cache metadata."""

    frame: pd.DataFrame
    processed_path: Path
    dataset_id: str


class DataService:
    """Load market data from explicit data requirements."""

    def __init__(self, workspace: WorkspacePaths | None = None) -> None:
        self._workspace = workspace or WorkspacePaths()
        self._workspace.ensure()
        self._loader = DataLoader()

    def data_config(self, raw_config: dict[str, Any]) -> DataConfig:
        """Build DataConfig with workspace defaults."""
        raw = raw_config.get("data", {})
        data = dict(raw) if isinstance(raw, dict) else {}
        raw_dir, processed_dir = self._data_dirs()
        data.setdefault("raw_dir", str(raw_dir))
        data.setdefault("processed_dir", str(processed_dir))
        requirements = data.get("requirements", {})
        if isinstance(requirements, dict):
            market = requirements.get("market", {})
            if isinstance(market, dict) and "source" in market:
                data["source"] = market["source"]
        return DataConfig(**data)

    def requirements(
        self,
        raw_config: dict[str, Any],
        backtest_config: BacktestConfig,
    ) -> DataRequirement:
        """Parse explicit data requirements, defaulting to market data only.

        ``requirements.market.source`` is the single source-selection field;
        when omitted it inherits ``data.source`` so older configs keep
        working.
        """
        raw = raw_config.get("data", {})
        data = raw if isinstance(raw, dict) else {}
        default_source = str(data.get("source", "binance_vision"))
        req = data.get("requirements") if isinstance(data, dict) else None
        if isinstance(req, dict):
            market = req.get("market")
            if isinstance(market, dict) and "source" not in market:
                req = {**req, "market": {**market, "source": default_source}}
            return DataRequirement(**req)
        return DataRequirement(
            market=MarketDataSpec(
                source=default_source,
                symbol=backtest_config.symbol,
                timeframe=backtest_config.timeframe,
            ),
            features=[],
        )

    def execution_config(
        self,
        raw_config: dict[str, Any],
        backtest_config: BacktestConfig,
    ) -> dict[str, Any]:
        """Resolve data locations now, without fetching or versioning their contents."""
        data = self.data_config(raw_config).model_dump(mode="json")
        for key in ("raw_dir", "processed_dir"):
            data[key] = str(self._resolve_data_path(data[key]))
        requirements = self.requirements(raw_config, backtest_config)
        if (
            requirements.market.symbol != backtest_config.symbol
            or requirements.market.timeframe != backtest_config.timeframe
        ):
            raise ValueError(
                "data.requirements.market symbol/timeframe must match backtest; "
                "the backtest fields determine the actual data request"
            )
        self._check_market_settings(requirements.market, str(data["market_type"]))
        request = self._request_from_backtest(backtest_config)
        features = []
        for feature in requirements.features:
            resolved = feature.model_dump(mode="json")
            path = (
                self._resolve_data_path(feature.path)
                if feature.path
                else self._default_dvol_path(request)
                if feature.type == "dvol"
                else None
            )
            resolved["path"] = str(path) if path is not None else None
            raw_path = (
                self._resolve_data_path(feature.raw_path)
                if feature.raw_path
                else path.with_suffix(".csv")
                if feature.type == "dvol" and path is not None
                else None
            )
            resolved["raw_path"] = str(raw_path) if raw_path is not None else None
            features.append(resolved)
        data["requirements"] = {
            "market": requirements.market.model_dump(mode="json"),
            "features": features,
        }
        return data

    @staticmethod
    def _check_market_settings(market: MarketDataSpec, market_type: str) -> None:
        """Reject known configuration errors without constructing any provider."""
        sources = {item["source"]: item for item in describe_sources()}
        details = sources.get(market.source)
        if details is None:
            raise ValueError(
                f"Unknown data source {market.source!r}. "
                f"Known sources: {', '.join(sorted(sources))}"
            )
        timeframes = details["timeframes"]
        if timeframes is not None and market.timeframe not in timeframes:
            raise ValueError(
                f"Data source {market.source!r} does not support "
                f"timeframe {market.timeframe!r}. Supported: {', '.join(timeframes)}"
            )
        market_types = details["market_types"]
        if market_types and market_type not in market_types:
            raise ValueError(
                f"Data source {market.source!r} does not support "
                f"market_type {market_type!r}. Supported: {', '.join(market_types)}"
            )

    def load(
        self,
        raw_config: dict[str, Any],
        backtest_config: BacktestConfig,
    ) -> LoadedDataset:
        """Load OHLCV data and merge declared feature sources."""
        data_config = self.data_config(raw_config)
        requirements = self.requirements(raw_config, backtest_config)
        request = self._request_from_backtest(backtest_config)
        frame, processed_path = self._load_market(
            request,
            data_config,
            requirements.market.source,
        )

        for feature in requirements.features:
            frame = self._merge_feature(frame, feature, request)

        dataset_id = self._dataset_id(request, requirements, data_config)
        return LoadedDataset(
            frame=frame, processed_path=processed_path, dataset_id=dataset_id
        )

    def ensure(
        self,
        *,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        source: str = "binance_vision",
        market_type: str = "futures/um",
    ) -> LoadedDataset:
        """Ensure OHLCV data for the requested range is cached locally."""
        request = MarketDataRequest(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        data_config = self.data_config(
            {"data": {"source": source, "market_type": market_type}}
        )
        frame, processed_path = self._load_market(request, data_config, source)
        requirements = DataRequirement(
            market=MarketDataSpec(source=source, symbol=symbol, timeframe=timeframe),
            features=[],
        )
        return LoadedDataset(
            frame=frame,
            processed_path=processed_path,
            dataset_id=self._dataset_id(request, requirements, data_config),
        )

    def fetch_market(
        self,
        raw_config: dict[str, Any],
        backtest_config: BacktestConfig,
    ) -> pd.DataFrame:
        """Fetch supplementary OHLCV without reading or publishing cache files.

        Only the declared market is fetched; strategy feature sources are not
        merged. The response stays in memory, so cancellation leaves no temporary
        raw or processed dataset behind.
        """
        config = self.data_config(raw_config)
        market = self.requirements(raw_config, backtest_config).market
        self._check_market_settings(market, config.market_type)
        request = self._request_from_backtest(backtest_config)
        crawler = create_crawler(market.source, config)
        raw = crawler.fetch(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start=request.start_date,
            end=request.end_date,
        ).copy()
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
        frame = DataProcessor().process(raw)
        return frame.loc[
            frame["timestamp"].between(request.start_date, request.end_date)
        ].reset_index(drop=True)

    def load_cached(
        self,
        *,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        source: str = "binance_vision",
        market_type: str = "futures/um",
        raw_config: dict[str, Any] | None = None,
    ) -> LoadedDataset:
        """Read existing yearly files without fetching, replacing or deleting any."""
        request = MarketDataRequest(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        data_config = self.data_config(
            raw_config
            if raw_config is not None
            else {"data": {"source": source, "market_type": market_type}}
        )
        cache = self._market_cache(request, data_config)
        start, end = pd.Timestamp(request.start_date), pd.Timestamp(request.end_date)
        paths: list[Path] = []
        frames: list[pd.DataFrame] = []
        for year in range(start.year, end.year + 1):
            receipt = cache.receipt(str(year))
            lower, upper = self._year_bounds(request, year)
            if receipt is None or not receipt.covers(lower, upper):
                raise ValueError(
                    "no cached market query covers the requested interval; "
                    "run ensure_data first"
                )
            if not receipt.reusable(lower, upper):
                raise ValueError(
                    "cached-only loading requires a query acquired at least seven "
                    "days after the requested interval ended; for recent intervals, "
                    "wait until that boundary before fetching again"
                )
            path = cache.processed_path(receipt)
            paths.append(path)
            frames.append(self._loader.load_parquet(path))
        frame = pd.concat(frames, ignore_index=True).sort_values("timestamp")
        frame = frame.loc[
            (frame["timestamp"] >= start) & (frame["timestamp"] <= end)
        ].reset_index(drop=True)
        if frame.empty:
            raise ValueError(
                "no cached market data in the requested interval; run ensure_data first"
            )
        if frame["timestamp"].duplicated().any():
            raise ValueError("cached market data has duplicate timestamps")
        requirements = DataRequirement(
            market=MarketDataSpec(
                source=data_config.source, symbol=symbol, timeframe=timeframe
            ),
            features=[],
        )
        return LoadedDataset(
            frame, paths[0], self._dataset_id(request, requirements, data_config)
        )

    def load_sample(
        self,
        raw_config: dict[str, Any],
        backtest_config: BacktestConfig,
        *,
        max_rows: int,
        output_dir: Path,
    ) -> LoadedDataset:
        """Find cached or fetched historical bars through bounded interval searches.

        Sample files and fetched features live under output_dir. Existing market
        and feature caches are only read; the yearly cache manager is not used.
        """
        if not 1 <= max_rows <= 4096:
            raise ValueError("Sample size must be between 1 and 4096 rows")
        data_config = self.data_config(raw_config)
        requirements = self.requirements(raw_config, backtest_config)
        request = self._request_from_backtest(backtest_config)
        sample_config = data_config.model_copy(
            update={
                "raw_dir": str(output_dir / "raw"),
                "processed_dir": str(output_dir / "processed"),
            }
        )
        # Normal data loading constructs the provider even on a cache hit.
        # Check that boundary here too, without calling its network fetch.
        crawler = create_crawler(requirements.market.source, sample_config)
        frame = load_market_sample(request, data_config, crawler, max_rows=max_rows)
        features = [
            feature.model_copy(update={"path": str(self._default_dvol_path(request))})
            if feature.type == "dvol" and not feature.path
            else feature
            for feature in requirements.features
        ]
        request = request.model_copy(
            update={
                "start_date": frame["timestamp"].iloc[0],
                "end_date": frame["timestamp"].iloc[-1],
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, feature in enumerate(features):
            selected = feature
            if feature.type == "dvol" and (
                not feature.path or not self._resolve_data_path(feature.path).exists()
            ):
                selected = feature.model_copy(
                    update={
                        "path": str(output_dir / f"feature-{index}.parquet"),
                        "raw_path": str(output_dir / f"feature-{index}.csv"),
                    }
                )
            frame = self._merge_feature(
                frame, selected, request, sample_max_rows=max_rows
            )
        processed_path = output_dir / "sample.parquet"
        frame.to_parquet(processed_path, index=False)
        # This identity describes the exact sample, not the full requested data.
        return LoadedDataset(frame, processed_path, sha256_file(processed_path))

    def list_available_data(
        self, raw_config: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Inventory identified cache receipts, reading no candle data."""
        directory = Path(self.data_config(raw_config or {}).processed_dir)
        grouped: dict[str, dict[str, Any]] = {}
        for path in sorted(directory.glob("cache-v2/*/[0-9][0-9][0-9][0-9].json")):
            receipt = CacheReceipt.model_validate_json(path.read_bytes())
            identity = receipt.identity
            if identity.kind != "ohlcv" or path.parent.name != identity.key:
                raise ValueError(f"Invalid market cache identity: {path}")
            if not (path.parent / f"{receipt.generation}.parquet").is_file():
                raise ValueError(f"Missing market cache generation: {path}")
            entry = grouped.setdefault(
                identity.key,
                {**identity.model_dump(exclude={"kind"}), "years_available": []},
            )
            entry["years_available"].append(int(path.stem))
        return list(grouped.values())

    def list_available_data_page(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        source: str | None = None,
        market_type: str | None = None,
        offset: int = 0,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> Page[CachedDataset]:
        """Filter exact identities; inspect configs to inventory custom roots."""
        rows = [
            row
            for row in self.list_available_data()
            if (symbol is None or row["symbol"] == symbol)
            and (timeframe is None or row["timeframe"] == timeframe)
            and (source is None or row["source"] == source)
            and (market_type is None or row["market_type"] == market_type)
        ]
        rows.sort(
            key=lambda row: (
                str(row["source"]),
                str(row["market_type"]),
                str(row["symbol"]),
                str(row["timeframe"]),
            )
        )
        return paginate(
            rows, project=CachedDataset.model_validate, offset=offset, limit=limit
        )

    def data_available(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        *,
        raw_config: dict[str, Any] | None = None,
    ) -> bool:
        """Whether successful query evidence permits reuse for this exact request."""
        try:
            request = MarketDataRequest(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )
            cache = self._market_cache(request, self.data_config(raw_config or {}))
            for year in range(request.start_date.year, request.end_date.year + 1):
                receipt = cache.receipt(str(year))
                lower, upper = self._year_bounds(request, year)
                if receipt is None or not receipt.reusable(lower, upper):
                    return False
            return True
        except OSError, TypeError, ValueError:
            return False

    def inspect_dataset(self, config_path: Path | None = None) -> dict[str, Any]:
        """Return cached data plus feature-source status for an optional config."""
        raw_config = load_config(config_path) if config_path is not None else {}
        data_config = self.data_config(raw_config)
        raw_dir, processed_dir = (
            Path(data_config.raw_dir),
            Path(data_config.processed_dir),
        )
        response: dict[str, Any] = {
            "data_root": str(raw_dir.parent),
            "raw_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
            "datasets": self.list_available_data(raw_config),
            "requirements": None,
            "market_available": None,
            "market": None,
            "dataset_fingerprint": None,
            "features": [],
        }
        if config_path is None:
            return response

        backtest_config = BacktestConfig(**raw_config["backtest"])
        request = self._request_from_backtest(backtest_config)
        requirements = self.requirements(raw_config, backtest_config)
        response["requirements"] = requirements.model_dump(mode="json")
        response["market_available"] = self.data_available(
            request.symbol,
            request.timeframe,
            request.start_date.isoformat(),
            request.end_date.isoformat(),
            raw_config=raw_config,
        )
        response["market"] = self._inspect_market(request, data_config)
        response["dataset_fingerprint"] = self._dataset_fingerprint_report(
            request,
            requirements,
            response["market"],
            data_config,
        )
        response["features"] = [
            self._inspect_feature(feature, request) for feature in requirements.features
        ]
        return response

    def _request_from_backtest(
        self,
        backtest_config: BacktestConfig,
    ) -> MarketDataRequest:
        return MarketDataRequest(
            symbol=backtest_config.symbol,
            timeframe=backtest_config.timeframe,
            start_date=backtest_config.start_date,
            end_date=backtest_config.end_date,
        )

    def _load_market(
        self,
        request: MarketDataRequest,
        data_config: DataConfig,
        source: str,
    ) -> tuple[pd.DataFrame, Path]:
        crawler = create_crawler(source, data_config)
        manager = DataManager(
            data_config.model_copy(update={"source": source}), request, crawler=crawler
        )
        return manager.load()

    def _merge_feature(
        self,
        frame: pd.DataFrame,
        feature: FeatureSpec,
        request: MarketDataRequest,
        *,
        sample_max_rows: int | None = None,
    ) -> pd.DataFrame:
        path = self._resolve_data_path(feature.path) if feature.path else None
        if sample_max_rows is not None and path is not None and path.exists():
            merged = read_feature_sample(
                path,
                timestamps=frame["timestamp"],
                column=feature.column,
                max_rows=sample_max_rows,
                fallback_column="dvol_close" if feature.type == "dvol" else None,
            )
        elif feature.type == "dvol":
            feature_df = self._load_or_fetch_dvol(
                feature,
                request,
                sample_max_rows=sample_max_rows,
                sample_timestamps=pd.DatetimeIndex(frame["timestamp"].unique())
                if sample_max_rows is not None
                else None,
            )
            value_column = feature.column
            if (
                value_column not in feature_df.columns
                and "dvol_close" in feature_df.columns
            ):
                value_column = "dvol_close"
            merged = feature_df[["timestamp", value_column]].rename(
                columns={value_column: feature.column}
            )
        else:
            if not feature.path:
                msg = (
                    f"{feature.type} feature requires data.requirements.features[].path"
                )
                raise ValueError(msg)
            feature_df = self._loader.load_parquet(
                self._resolve_data_path(feature.path)
            )
            merged = feature_df[["timestamp", feature.column]]

        if sample_max_rows is not None:
            merged = merged.loc[merged["timestamp"].isin(frame["timestamp"])]
            counts = merged["timestamp"].value_counts()
            expected_rows = sum(
                max(int(counts.get(timestamp, 0)), 1)
                for timestamp in frame["timestamp"]
            )
            if expected_rows > sample_max_rows:
                raise ValueError("Feature joins exceeded the sample row budget")
        result = frame.merge(merged, on="timestamp", how="left")
        # Leading gaps are unknown; later observations cannot fill earlier bars.
        result[feature.column] = result[feature.column].ffill()
        return result

    def _load_or_fetch_dvol(
        self,
        feature: FeatureSpec,
        request: MarketDataRequest,
        *,
        sample_max_rows: int | None = None,
        sample_timestamps: pd.DatetimeIndex | None = None,
    ) -> pd.DataFrame:
        processed_path = (
            self._resolve_data_path(feature.path)
            if feature.path
            else self._default_dvol_path(request)
        )
        if processed_path.exists():
            if sample_max_rows is not None and sample_timestamps is not None:
                return read_feature_sample(
                    processed_path,
                    timestamps=pd.Series(sample_timestamps),
                    column=feature.column,
                    fallback_column="dvol_close",
                    max_rows=sample_max_rows,
                )
            return self._loader.load_parquet(processed_path)

        crawler = DeribitDVOLCrawler()
        currency = request.symbol.split("/")[0]
        if sample_max_rows is None:
            raw = crawler.fetch(
                symbol=currency,
                timeframe=request.timeframe,
                start=request.start_date,
                end=request.end_date,
            )
        else:
            raw = crawler.fetch_sample(
                symbol=currency,
                timeframe=request.timeframe,
                start=request.start_date,
                end=request.end_date,
                max_rows=sample_max_rows,
                timestamps=sample_timestamps,
            )
            if len(raw) > sample_max_rows:
                raise ValueError("Feature provider exceeded the sample row budget")
        raw_path = (
            self._resolve_data_path(feature.raw_path)
            if feature.raw_path
            else processed_path.with_suffix(".csv")
        )
        crawler.save_raw(raw, raw_path)
        if raw["timestamp"].dt.tz is None:
            raw["timestamp"] = raw["timestamp"].dt.tz_localize("UTC")
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(processed_path, index=False)
        return raw

    def _default_dvol_path(self, request: MarketDataRequest) -> Path:
        symbol = request.symbol.split("/")[0].lower()
        start_year = request.start_date.year
        end_year = request.end_date.year
        return self._processed_data_dir() / (
            f"{symbol}_dvol_{request.timeframe}_{start_year}_{end_year}.parquet"
        )

    def _data_dirs(self) -> tuple[Path, Path]:
        data_root = os.environ.get("TRADINGDEV_DATA_ROOT")
        if data_root:
            root = Path(data_root).expanduser().resolve()
            return root / "raw", root / "processed"
        return self._workspace.raw_data, self._workspace.processed_data

    def _processed_data_dir(self) -> Path:
        return self._data_dirs()[1]

    @staticmethod
    def _market_cache(request: MarketDataRequest, config: DataConfig) -> DataCache:
        return DataCache(
            Path(config.raw_dir),
            Path(config.processed_dir),
            DataIdentity.market(config, request),
        )

    @staticmethod
    def _year_bounds(
        request: MarketDataRequest, year: int
    ) -> tuple[datetime, datetime]:
        return (
            max(request.start_date, datetime(year, 1, 1, tzinfo=UTC)),
            min(
                request.end_date, datetime(year, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)
            ),
        )

    def _resolve_data_path(self, value: str | None) -> Path:
        if not value:
            msg = "data path is required"
            raise ValueError(msg)
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return (Path.cwd() / path).resolve()

    def _inspect_market(
        self, request: MarketDataRequest, config: DataConfig
    ) -> dict[str, Any]:
        cache = self._market_cache(request, config)
        paths: list[Path] = []
        errors: list[str] = []
        queries: list[dict[str, Any]] = []
        for year in range(request.start_date.year, request.end_date.year + 1):
            try:
                receipt = cache.receipt(str(year))
                if receipt is not None:
                    paths.append(cache.processed_path(receipt))
                    queries.append(
                        receipt.model_dump(
                            mode="json", exclude={"identity", "generation", "version"}
                        )
                    )
            except (OSError, ValueError) as error:
                errors.append(str(error))
        report: dict[str, Any] = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "source": cache.identity.source,
            "market_type": cache.identity.market_type,
            "queries": queries,
            "paths": [str(path) for path in paths],
            "exists": len(paths) == request.end_date.year - request.start_date.year + 1,
            "rows": None,
            "columns": [],
            "start_timestamp": None,
            "end_timestamp": None,
            "timezone": None,
            "missing_values": {},
            "errors": errors,
        }
        frames: list[pd.DataFrame] = []
        for path in paths:
            if not path.exists():
                continue
            try:
                frames.append(self._loader.load_parquet(path))
            except Exception as exc:  # noqa: BLE001
                report["errors"].append(f"{path}: {exc}")
        if not frames:
            return report

        frame = pd.concat(frames, ignore_index=True)
        report["rows"] = int(len(frame))
        report["columns"] = [str(column) for column in frame.columns]
        report["missing_values"] = {
            str(column): int(count)
            for column, count in frame.isna().sum().to_dict().items()
        }
        if "timestamp" in frame.columns and not frame.empty:
            timestamps = frame["timestamp"]
            report["start_timestamp"] = str(timestamps.min())
            report["end_timestamp"] = str(timestamps.max())
            tz = getattr(timestamps.dt, "tz", None)
            report["timezone"] = str(tz) if tz is not None else None
        return report

    def _inspect_feature(
        self,
        feature: FeatureSpec,
        request: MarketDataRequest,
    ) -> dict[str, Any]:
        path = (
            self._default_dvol_path(request)
            if feature.type == "dvol" and not feature.path
            else self._resolve_data_path(feature.path)
            if feature.path
            else None
        )
        exists = bool(path and path.exists())
        missing_values: int | None = None
        rows: int | None = None
        if exists and path is not None:
            try:
                frame = self._loader.load_parquet(path)
                rows = len(frame)
                if feature.column in frame.columns:
                    missing_values = int(frame[feature.column].isna().sum())
                elif feature.type == "dvol" and "dvol_close" in frame.columns:
                    missing_values = int(frame["dvol_close"].isna().sum())
            except Exception:  # noqa: BLE001
                missing_values = None
        return {
            "type": feature.type,
            "source": feature.source,
            "column": feature.column,
            "path": str(path) if path is not None else None,
            "exists": exists,
            "rows": rows,
            "missing_values": missing_values,
        }

    def _dataset_fingerprint_report(
        self,
        request: MarketDataRequest,
        requirements: DataRequirement,
        market_report: object,
        data_config: DataConfig,
    ) -> dict[str, str]:
        file_hashes = []
        if isinstance(market_report, dict):
            for value in market_report.get("paths", []):
                path = Path(str(value))
                if path.exists():
                    file_hashes.append({"path": str(path), "sha256": sha256_file(path)})
        dataset_id = self._dataset_id(request, requirements, data_config)
        payload = {
            "dataset_id": dataset_id,
            "files": file_hashes,
            "requirements": requirements.model_dump(mode="json"),
        }
        return {
            "dataset_id": dataset_id,
            "fingerprint": sha256_text(json.dumps(payload, sort_keys=True)),
        }

    def _dataset_id(
        self,
        request: MarketDataRequest,
        requirements: DataRequirement,
        data_config: DataConfig,
    ) -> str:
        payload = {
            "market": DataIdentity.market(data_config, request).model_dump(mode="json"),
            "request": request.model_dump(mode="json"),
            "features": [
                feature.model_dump(mode="json") for feature in requirements.features
            ],
        }
        return "query-sha256:" + sha256_text(
            json.dumps(payload, sort_keys=True, ensure_ascii=False)
        )
