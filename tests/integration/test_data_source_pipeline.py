"""Integration tests: each data source feeds the existing backtest pipeline.

Both tests stub only the external transport (ccxt client / HTTP client) and
run the real crawler, DataManager yearly cache, DataService market-data loading,
strategy loader, and backtest engine end to end.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from tests.data_cache import cached_market_path
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.data_service import DataService
from tradingdev.app.strategy_service import StrategyService
from tradingdev.domain.strategies.loader import StrategyLoader

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch

_DAY_MS = 86_400_000
_DAY_S = 86_400
_YEAR_2024_START_S = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp())


def _raw_config(source: str, symbol: str) -> dict[str, Any]:
    return {
        "strategy": {
            "id": "kd_crossover",
            "parameters": {
                "k_period": 14,
                "d_period": 3,
                "smooth_k": 3,
                "overbought": 80.0,
                "oversold": 20.0,
            },
        },
        "backtest": {
            "symbol": symbol,
            "timeframe": "1d",
            "start_date": "2024-01-01",
            "end_date": "2024-02-29",
            "init_cash": 10_000.0,
            "fees": 0.0,
            "slippage": 0.0,
        },
        "data": {
            "requirements": {
                "market": {
                    "source": source,
                    "symbol": symbol,
                    "timeframe": "1d",
                },
                "features": [],
            }
        },
    }


def _service(tmp_path: Path) -> tuple[BacktestService, WorkspacePaths]:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = BacktestService(
        data_service=DataService(workspace),
        strategy_loader=StrategyLoader(workspace_root=workspace.root),
        strategy_gate=StrategyService(workspace),
    )
    return service, workspace


def _daily_close(i: int) -> float:
    return 100.0 + 10.0 * ((i % 20) / 20.0) - 5.0 * ((i % 7) / 7.0)


class _FakeBinance:
    """ccxt.binance stand-in returning deterministic daily candles for 2024."""

    instances: list[_FakeBinance] = []

    def __init__(self, _params: dict[str, Any] | None = None) -> None:
        self.rateLimit = 0
        self.calls = 0
        _FakeBinance.instances.append(self)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int | None = None,
        limit: int = 1000,
    ) -> list[list[float]]:
        self.calls += 1
        base_ms = _YEAR_2024_START_S * 1000
        candles = []
        for i in range(366):
            ts = base_ms + i * _DAY_MS
            if since is not None and ts < since:
                continue
            close = _daily_close(i)
            candles.append([ts, close - 0.5, close + 1.0, close - 1.0, close, 1000.0])
        return candles[:limit]


class _FakeYahooResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.status_code = 200

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _FakeYahooClient:
    """httpx.Client stand-in serving deterministic daily candles."""

    requested_symbols: list[str] = []
    fail_benchmark = False

    def __init__(self, **_kwargs: Any) -> None:
        pass

    def get(self, _url: str, params: dict[str, Any]) -> _FakeYahooResponse:
        symbol = _url.rsplit("/", 1)[-1]
        type(self).requested_symbols.append(symbol)
        if symbol == "^GSPC" and self.fail_benchmark:
            return _FakeYahooResponse(
                {"chart": {"result": None, "error": {"code": "Not Found"}}}
            )
        period1 = int(params["period1"])
        period2 = int(params["period2"])
        # Yahoo's transport interval is half-open, unlike the crawler contract.
        timestamps = list(range(period1, period2, _DAY_S))
        closes = [
            _daily_close((ts - _YEAR_2024_START_S) // _DAY_S) for ts in timestamps
        ]
        return _FakeYahooResponse(
            {
                "chart": {
                    "result": [
                        {
                            "timestamp": timestamps,
                            "indicators": {
                                "quote": [
                                    {
                                        "open": [c - 0.5 for c in closes],
                                        "high": [c + 1.0 for c in closes],
                                        "low": [c - 1.0 for c in closes],
                                        "close": closes,
                                        "volume": [1000.0] * len(closes),
                                    }
                                ]
                            },
                        }
                    ],
                    "error": None,
                }
            }
        )


def test_ccxt_crypto_source_flows_into_backtest_pipeline(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRADINGDEV_DATA_ROOT", raising=False)
    monkeypatch.setattr(
        "tradingdev.domain.data.crawlers.binance_api.ccxt.binance",
        _FakeBinance,
    )
    _FakeBinance.instances = []
    service, workspace = _service(tmp_path)

    run = service.run_raw_config(_raw_config("binance_api", "BTC/USDT"))

    assert run.mode == "simple"
    assert "total_return" in run.metrics
    assert run.dataset_id.startswith("query-sha256:")
    cached = cached_market_path(
        workspace.processed_data, timeframe="1d", source="binance_api"
    )
    assert cached.exists()
    assert sum(inst.calls for inst in _FakeBinance.instances) >= 1

    # Second run must reuse the yearly cache without hitting the exchange.
    calls_before = sum(inst.calls for inst in _FakeBinance.instances)
    service.run_raw_config(_raw_config("binance_api", "BTC/USDT"))
    assert sum(inst.calls for inst in _FakeBinance.instances) == calls_before


@pytest.mark.parametrize("benchmark_available", [True, False])
def test_yahoo_finance_equity_source_flows_into_backtest_pipeline(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    benchmark_available: bool,
) -> None:
    monkeypatch.delenv("TRADINGDEV_DATA_ROOT", raising=False)
    monkeypatch.setattr(
        "tradingdev.domain.data.crawlers.yahoo_finance.httpx.Client",
        _FakeYahooClient,
    )
    _FakeYahooClient.requested_symbols = []
    monkeypatch.setattr(_FakeYahooClient, "fail_benchmark", not benchmark_available)
    service, workspace = _service(tmp_path)

    run = service.run_raw_config(_raw_config("yahoo_finance", "AAPL"))

    assert run.mode == "simple"
    assert "total_return" in run.metrics
    assert run.dataset_id.startswith("query-sha256:")
    cached = cached_market_path(
        workspace.processed_data, symbol="AAPL", timeframe="1d", source="yahoo_finance"
    )
    assert cached.exists()
    assert Counter(_FakeYahooClient.requested_symbols) == {"AAPL": 1, "^GSPC": 1}
    original_cache = cached.read_bytes()

    # Trade data reuse and supplementary benchmark acquisition are independent.
    rerun = service.run_raw_config(_raw_config("yahoo_finance", "AAPL"))
    assert Counter(_FakeYahooClient.requested_symbols) == {"AAPL": 1, "^GSPC": 2}
    assert cached.read_bytes() == original_cache
    assert list(workspace.processed_data.rglob("*.parquet")) == [cached]
    assert rerun.metrics == run.metrics
    for result in (run, rerun):
        benchmark = result.pipeline.benchmark
        assert benchmark is not None
        assert benchmark.reused_run_frame is False
        if benchmark_available:
            assert benchmark.status == "available"
            assert benchmark.timestamps is not None
            assert len(benchmark.timestamps) == 60
            assert benchmark.timestamps[0] == datetime(2024, 1, 1, tzinfo=UTC)
            assert benchmark.timestamps[-1] == datetime(2024, 2, 29, tzinfo=UTC)
        else:
            assert benchmark.status == "unavailable"
            assert benchmark.reason == "benchmark_data_unavailable"
