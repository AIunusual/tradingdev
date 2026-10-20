"""Data MCP tools."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import ValidationError

from tradingdev.app.contracts.data import (
    CachedDataset,
    DatasetInspection,
    EnsureDataResponse,
)
from tradingdev.domain.data.crawlers.registry import available_sources
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
    def list_available_data() -> list[CachedDataset]:
        """List cached OHLCV datasets; use inspect_dataset to inspect file contents."""
        return [
            CachedDataset.model_validate(row) for row in service.list_available_data()
        ]

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def list_data_sources() -> list[str]:
        """List registered source names before selecting a source for ensure_data."""
        return available_sources()

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

        This does not fetch data. Use ensure_data to acquire missing market data.
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
    ) -> EnsureDataResponse:
        """Fetch and cache OHLCV data for the requested range when needed.

        Select a source from list_data_sources. Completed-year downloads may
        replace partial caches. Inspect the cache before starting a backtest.
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
        )
        return EnsureDataResponse(
            success=True,
            rows=len(dataset.frame),
            processed_path=str(dataset.processed_path),
            dataset_id=dataset.dataset_id,
        )
