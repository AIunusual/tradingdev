"""Acquire the fixed baseline for a run's market without choosing it per user."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from tradingdev.app.data_service import DataService
from tradingdev.domain.backtest.metrics import timestamp_index
from tradingdev.domain.backtest.schemas import BacktestConfig
from tradingdev.domain.benchmark import (
    BENCHMARKS,
    BenchmarkDefinition,
    BenchmarkEvidence,
    asset_class_for,
    benchmark_source,
)
from tradingdev.domain.data.crawlers.registry import describe_sources
from tradingdev.domain.data.schemas import DataConfig, MarketDataRequest
from tradingdev.shared.utils.logger import setup_logger

if TYPE_CHECKING:
    import pandas as pd

logger = setup_logger(__name__)


class BenchmarkService:
    """Resolve the asset class's baseline and capture its closes once per run.

    The baseline is supplementary evidence: a missing provider, an empty range
    or an unsupported cost setting is recorded as an explicit reason and never
    fails the strategy execution it accompanies.
    """

    def __init__(self, data_service: DataService | None = None) -> None:
        self._data_service = data_service or DataService()

    def evidence(
        self,
        frame: pd.DataFrame,
        raw_config: dict[str, Any],
        backtest_config: BacktestConfig,
    ) -> BenchmarkEvidence:
        """Return baseline evidence for the market this run actually loaded."""
        market = self._data_service.requirements(raw_config, backtest_config).market
        asset_class = asset_class_for(market.source, market.symbol)
        definition = None if asset_class is None else BENCHMARKS.get(asset_class)
        if definition is None:
            return BenchmarkEvidence(
                status="undefined",
                asset_class=asset_class,
                reason="undefined_asset_class",
                detail=(
                    f"資產類別 {asset_class or '未分類'} 沒有定義 benchmark；"
                    "不以其他市場的指數代替。"
                ),
                source=market.source,
            )
        source = benchmark_source(definition, market.source)
        market_type = _market_type(raw_config, source)
        reused = definition.symbol == market.symbol and source == market.source
        try:
            benchmark_frame = (
                frame
                if reused
                else self._load(definition.symbol, source, raw_config, backtest_config)
            )
            timestamps = timestamp_index(benchmark_frame)
            if timestamps is None:
                return _unavailable(
                    definition,
                    asset_class,
                    source,
                    market_type,
                    reused,
                    "missing_timestamps",
                    "基準行情沒有可用的時間戳，無法與回測 K 線對齊。",
                )
            closes = np.asarray(benchmark_frame["close"], dtype=np.float64)
            if len(closes) != len(timestamps):
                raise ValueError("Benchmark closes do not align with its timestamps")
            if not np.all(np.isfinite(closes)) or np.any(closes <= 0):
                return _unavailable(
                    definition,
                    asset_class,
                    source,
                    market_type,
                    reused,
                    "invalid_benchmark_prices",
                    "基準收盤價必須是有限正數。",
                )
        except Exception as exc:  # noqa: BLE001 - the run must survive this
            logger.warning("Benchmark %s unavailable: %s", definition.id, exc)
            return _unavailable(
                definition,
                asset_class,
                source,
                market_type,
                reused,
                "benchmark_data_unavailable",
                str(exc),
            )
        return BenchmarkEvidence(
            status="available",
            asset_class=asset_class,
            definition=definition,
            source=source,
            market_type=market_type,
            dataset_id=_range_id(
                definition.symbol, source, market_type, backtest_config
            ),
            reused_run_frame=reused,
            timestamps=timestamps,
            closes=closes,
        )

    def _load(
        self,
        symbol: str,
        source: str,
        raw_config: dict[str, Any],
        backtest_config: BacktestConfig,
    ) -> pd.DataFrame:
        """Load the baseline over the run's own period, timeframe and venue."""
        config = deepcopy(raw_config)
        config["backtest"] = {**backtest_config.model_dump(), "symbol": symbol}
        data = config.get("data")
        data = {} if not isinstance(data, dict) else data
        data["source"] = source
        requirements = data.get("requirements")
        if isinstance(requirements, dict):
            resolved = deepcopy(requirements)
            market = resolved.get("market")
            if isinstance(market, dict):
                resolved["market"] = {**market, "symbol": symbol, "source": source}
            # Strategy features describe the traded asset, not the baseline.
            resolved["features"] = []
            data["requirements"] = resolved
        config["data"] = data
        # Ordinary yearly caches do not prove provider/market identity and may
        # delete old partial files before a failed refresh. Supplementary reads
        # must neither reuse nor mutate that user data (issues #73 and #74).
        return self._data_service.fetch_market(
            config, BacktestConfig.model_validate(config["backtest"])
        )


def _market_type(raw_config: dict[str, Any], source: str) -> str | None:
    data = raw_config.get("data")
    details = next(item for item in describe_sources() if item["source"] == source)
    if not details["market_types"]:
        return None
    config = DataConfig.model_validate(data if isinstance(data, dict) else {})
    return config.market_type


def _range_id(
    symbol: str, source: str, market_type: str | None, backtest_config: BacktestConfig
) -> str:
    """Describe the baseline request; the run's own binding stays authoritative."""
    request = MarketDataRequest(
        symbol=symbol,
        timeframe=backtest_config.timeframe,
        start_date=backtest_config.start_date,
        end_date=backtest_config.end_date,
    )
    return (
        f"{source}:{market_type or '-'}:{symbol}:{request.timeframe}:"
        f"{request.start_date.isoformat()}:{request.end_date.isoformat()}"
    )


def _unavailable(
    definition: BenchmarkDefinition,
    asset_class: str | None,
    source: str,
    market_type: str | None,
    reused: bool,
    reason: str,
    detail: str,
) -> BenchmarkEvidence:
    return BenchmarkEvidence(
        status="unavailable",
        asset_class=asset_class,
        definition=definition,
        reason=reason,
        detail=detail,
        source=source,
        market_type=market_type,
        reused_run_frame=reused,
    )
