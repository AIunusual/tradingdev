"""Data MCP tools."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import ValidationError

from tradingdev.app.contracts.common import (
    DEFAULT_PAGE_LIMIT,
    Page,
    PageLimit,
    PageOffset,
)
from tradingdev.app.contracts.data import (
    CachedDataset,
    DatasetInspection,
    DataSourceDescription,
    EnsureDataResponse,
)
from tradingdev.domain.data.crawlers.registry import available_sources, describe_sources
from tradingdev.domain.data.schemas import MarketDataRequest
from tradingdev.mcp.tools.input_errors import invalid_arguments

if TYPE_CHECKING:
    from mcp.server import MCPServer

    from tradingdev.app.data_service import DataService


def register(mcp: MCPServer, service: DataService) -> None:
    """Register data tools."""

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def list_available_data(
        symbol: str | None = None,
        timeframe: str | None = None,
        source: str | None = None,
        market_type: str | None = None,
        offset: PageOffset = 0,
        limit: PageLimit = DEFAULT_PAGE_LIMIT,
    ) -> Page[CachedDataset]:
        """List cached OHLCV identities: source, market, symbol, timeframe.

        Exact filters; sorted in that order. Reads receipts, excluding unidentified
        legacy files. Follow next_offset; restart at 0 after changes (no snapshot).
        inspect_dataset(config_path) reads custom roots, query bounds and observations.
        """
        return service.list_available_data_page(
            symbol=symbol,
            timeframe=timeframe,
            source=source,
            market_type=market_type,
            offset=offset,
            limit=limit,
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def list_data_sources() -> list[DataSourceDescription]:
        """Discover sources, assets, symbols, intervals and market settings.

        Set data.requirements.market.source in YAML. null timeframes means upstream-
        dependent. Discovery and preparation check local capabilities offline;
        execution checks historical data after approval.
        """
        return [
            DataSourceDescription.model_validate(item) for item in describe_sources()
        ]

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def inspect_dataset(config_path: str | None = None) -> DatasetInspection:
        """Inspect local caches, optionally against an existing run config.

        No downloads. Missing caches do not block preparation; approved backtests
        fetch independently. ensure_data requires an explicit download request.
        Query coverage does not prove complete candles.
        """
        path = Path(config_path) if config_path else None
        if path is not None and not path.is_file():
            raise ToolError(f"config_path must point to an existing file: {path}")
        return DatasetInspection.model_validate(service.inspect_dataset(path))

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=False,
            open_world_hint=True,
        )
    )
    def ensure_data(
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        source: str = "binance_vision",
        market_type: str = "futures/um",
    ) -> EnsureDataResponse:
        """Cache queried OHLCV for the requested interval.

        Requires an explicit download request. Preparation uses code/settings and
        synthetic fixtures. Approved backtests fetch independently; no prerequisite.
        Choose source/market via list_data_sources. Recent ranges refresh each load;
        reuse needs a query acquired >=7 days after the interval ended. Queries may be
        empty and do not prove complete candles.
        """
        try:
            MarketDataRequest.model_validate(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "start_date": start_date,
                    "end_date": end_date,
                }
            )
        except ValidationError as exc:
            raise invalid_arguments(exc) from exc
        sources = available_sources()
        if source not in sources:
            raise ToolError(
                f"Unknown data source {source!r}. Known sources: {', '.join(sources)}"
            )
        dataset = service.ensure(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            source=source,
            market_type=market_type,
        )
        return EnsureDataResponse(
            success=True,
            rows=len(dataset.frame),
            processed_path=str(dataset.processed_path),
            dataset_id=dataset.dataset_id,
        )
