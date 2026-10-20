"""Application service for market data access."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tradingdev.adapters.storage.feature_samples import read_feature_sample
from tradingdev.adapters.storage.filesystem import (
    WorkspacePaths,
    sha256_file,
    sha256_text,
)
from tradingdev.app.market_sample import load_market_sample
from tradingdev.domain.backtest.schemas import BacktestConfig
from tradingdev.domain.data.crawlers.deribit_dvol import DeribitDVOLCrawler
from tradingdev.domain.data.crawlers.registry import create_crawler
from tradingdev.domain.data.data_manager import DataManager, market_data_filename
from tradingdev.domain.data.loader import DataLoader
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

    _DATA_FILE = re.compile(r"^([a-z0-9]+)_([a-z0-9]+)_(\d{4})(_partial)?\.parquet$")

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

        dataset_id = self._dataset_id(request, requirements)
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
    ) -> LoadedDataset:
        """Ensure OHLCV data for the requested range is cached locally."""
        request = MarketDataRequest(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        data_config = self.data_config({"data": {"source": source}})
        frame, processed_path = self._load_market(request, data_config, source)
        requirements = DataRequirement(
            market=MarketDataSpec(source=source, symbol=symbol, timeframe=timeframe),
            features=[],
        )
        return LoadedDataset(
            frame=frame,
            processed_path=processed_path,
            dataset_id=self._dataset_id(request, requirements),
        )

    def load_cached(
        self,
        *,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        source: str = "binance_vision",
    ) -> LoadedDataset:
        """Read existing yearly files without fetching, replacing or deleting any."""
        request = MarketDataRequest(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        # Validate the provider name, but never invoke its fetch methods.
        create_crawler(source, self.data_config({"data": {"source": source}}))
        start = pd.Timestamp(request.start_date)
        end = pd.Timestamp(request.end_date)
        start = (
            start.tz_localize("UTC")
            if start.tzinfo is None
            else start.tz_convert("UTC")
        )
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        if start > end:
            raise ValueError("start_date must not follow end_date")
        paths: list[Path] = []
        frames: list[pd.DataFrame] = []
        for year in range(start.year, end.year + 1):
            directory = self._processed_data_dir()
            path = directory / market_data_filename(symbol, timeframe, year)
            if not path.is_file():
                path = directory / market_data_filename(
                    symbol, timeframe, year, partial=True
                )
            if not path.is_file():
                raise ValueError(
                    f"no cached market data for {symbol} {timeframe} in {year}; "
                    "run ensure_data first"
                )
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
            market=MarketDataSpec(source=source, symbol=symbol, timeframe=timeframe),
            features=[],
        )
        return LoadedDataset(frame, paths[0], self._dataset_id(request, requirements))

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

    def list_available_data(self) -> list[dict[str, Any]]:
        """List cached OHLCV parquet files from the workspace data root."""
        processed_dir = self._processed_data_dir()
        if not processed_dir.exists():
            return []

        grouped: dict[tuple[str, str], set[int]] = {}
        for path in sorted(processed_dir.glob("*.parquet")):
            match = self._DATA_FILE.match(path.name)
            if not match:
                continue
            symbol, timeframe = match.group(1), match.group(2)
            grouped.setdefault((symbol, timeframe), set()).add(int(match.group(3)))
        return [
            {
                "symbol": symbol.upper(),
                "timeframe": timeframe,
                "years_available": sorted(years),
            }
            for (symbol, timeframe), years in sorted(grouped.items())
        ]

    def data_available(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> bool:
        """Return whether yearly processed parquet files are already cached."""
        processed_dir = self._processed_data_dir()
        if not processed_dir.exists():
            return False
        try:
            start_year = int(start_date[:4])
            end_year = int(end_date[:4])
        except ValueError, IndexError:
            return False
        for year in range(start_year, end_year + 1):
            complete = processed_dir / market_data_filename(symbol, timeframe, year)
            partial = processed_dir / market_data_filename(
                symbol, timeframe, year, partial=True
            )
            if not complete.exists() and not partial.exists():
                return False
        return True

    def inspect_dataset(self, config_path: Path | None = None) -> dict[str, Any]:
        """Return cached data plus feature-source status for an optional config."""
        raw_dir, processed_dir = self._data_dirs()
        response: dict[str, Any] = {
            "data_root": str(raw_dir.parent),
            "raw_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
            "datasets": self.list_available_data(),
            "requirements": None,
            "market_available": None,
            "market": None,
            "dataset_fingerprint": None,
            "features": [],
        }
        if config_path is None:
            return response

        raw_config = load_config(config_path)
        backtest_config = BacktestConfig(**raw_config["backtest"])
        request = self._request_from_backtest(backtest_config)
        requirements = self.requirements(raw_config, backtest_config)
        response["requirements"] = requirements.model_dump(mode="json")
        response["market_available"] = self.data_available(
            request.symbol,
            request.timeframe,
            request.start_date.date().isoformat(),
            request.end_date.date().isoformat(),
        )
        response["market"] = self._inspect_market(request)
        response["dataset_fingerprint"] = self._dataset_fingerprint_report(
            request,
            requirements,
            response["market"],
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
        manager = DataManager(data_config, request, crawler=crawler)
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
        result[feature.column] = result[feature.column].ffill().bfill()
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

    def _market_paths(self, request: MarketDataRequest) -> list[Path]:
        return [
            self._processed_data_dir()
            / market_data_filename(request.symbol, request.timeframe, year)
            for year in range(
                request.start_date.year,
                request.end_date.year + 1,
            )
        ]

    def _resolve_data_path(self, value: str | None) -> Path:
        if not value:
            msg = "data path is required"
            raise ValueError(msg)
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return (Path.cwd() / path).resolve()

    def _inspect_market(self, request: MarketDataRequest) -> dict[str, Any]:
        paths = self._market_paths(request)
        report: dict[str, Any] = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "paths": [str(path) for path in paths],
            "exists": all(path.exists() for path in paths),
            "rows": 0,
            "columns": [],
            "start_timestamp": None,
            "end_timestamp": None,
            "timezone": None,
            "missing_values": {},
            "errors": [],
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
    ) -> dict[str, str]:
        file_hashes = []
        if isinstance(market_report, dict):
            for value in market_report.get("paths", []):
                path = Path(str(value))
                if path.exists():
                    file_hashes.append({"path": str(path), "sha256": sha256_file(path)})
        dataset_id = self._dataset_id(request, requirements)
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
    ) -> str:
        feature_key = ",".join(
            f"{feature.type}:{feature.source}:{feature.column}:{feature.path or ''}"
            for feature in requirements.features
        )
        return (
            f"{request.symbol}:{request.timeframe}:"
            f"{request.start_date.date()}:{request.end_date.date()}:"
            f"{feature_key}"
        )
