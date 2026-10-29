"""The baseline follows the saved market requirement, never a caller choice."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import pytest

from tradingdev.app.benchmark_service import BenchmarkService
from tradingdev.domain.backtest.schemas import BacktestConfig
from tradingdev.domain.data.requirements import DataRequirement, MarketDataSpec

if TYPE_CHECKING:
    from tradingdev.app.data_service import DataService


class _StubDataService:
    """Serve prepared frames and record every baseline request made."""

    def __init__(self, frames: dict[str, Any]) -> None:
        self.frames = frames
        self.loaded: list[tuple[str, str, dict[str, Any]]] = []

    def requirements(
        self, raw_config: dict[str, Any], backtest_config: BacktestConfig
    ) -> DataRequirement:
        del backtest_config
        market = raw_config["data"]["requirements"]["market"]
        features = raw_config["data"]["requirements"].get("features", [])
        return DataRequirement.model_validate(
            {"market": MarketDataSpec(**market), "features": features}
        )

    def load(
        self, raw_config: dict[str, Any], backtest_config: BacktestConfig
    ) -> SimpleNamespace:
        self.loaded.append(
            (backtest_config.symbol, raw_config["data"]["source"], raw_config)
        )
        frame = self.frames[backtest_config.symbol]
        if isinstance(frame, Exception):
            raise frame
        return SimpleNamespace(frame=frame)


def _frame(closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    stamps = pd.date_range(start, periods=len(closes), freq="h", tz="UTC")
    return pd.DataFrame({"timestamp": stamps, "close": closes})


def _config(
    symbol: str, source: str, *, market_type: str | None = "futures/um"
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "source": source,
        "requirements": {
            "market": {"source": source, "symbol": symbol, "timeframe": "1h"},
            "features": [{"type": "dvol", "source": "deribit", "column": "dvol_close"}],
        },
    }
    if market_type is not None:
        data["market_type"] = market_type
    return {
        "data": data,
        "backtest": {
            "symbol": symbol,
            "timeframe": "1h",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-01-02T00:00:00",
            "init_cash": 1000.0,
            "mode": "signal",
        },
    }


def _evidence(
    frames: dict[str, Any], config: dict[str, Any], run_frame: pd.DataFrame
) -> tuple[Any, _StubDataService]:
    stub = _StubDataService(frames)
    service = BenchmarkService(data_service=cast("DataService", stub))
    backtest = BacktestConfig.model_validate(config["backtest"])
    return service.evidence(run_frame, config, backtest), stub


def test_other_crypto_pair_loads_the_fixed_btc_baseline() -> None:
    config = _config("ETH/USDT", "binance_vision")
    evidence, stub = _evidence(
        {"BTC/USDT": _frame([100.0, 101.0])}, config, _frame([1.0, 1.0])
    )
    assert evidence.status == "available"
    assert evidence.definition is not None
    assert evidence.definition.symbol == "BTC/USDT"
    assert evidence.closes is not None
    assert evidence.closes.tolist() == [100.0, 101.0]
    assert evidence.reused_run_frame is False
    assert stub.loaded[0][0] == "BTC/USDT"
    # The baseline request never inherits the strategy's feature sources.
    assert stub.loaded[0][2]["data"]["requirements"]["features"] == []
    assert stub.loaded[0][2]["data"]["requirements"]["market"]["symbol"] == "BTC/USDT"


def test_trading_the_baseline_itself_reuses_the_loaded_run_frame() -> None:
    config = _config("BTC/USDT", "binance_vision")
    frame = _frame([100.0, 101.0])
    evidence, stub = _evidence({}, config, frame)
    assert evidence.status == "available"
    assert evidence.reused_run_frame is True
    assert evidence.closes is not None and evidence.closes.tolist() == [100.0, 101.0]
    assert stub.loaded == []


def test_us_listing_loads_the_sp500_index() -> None:
    config = _config("AAPL", "yahoo_finance", market_type=None)
    evidence, stub = _evidence(
        {"^GSPC": _frame([4000.0, 4010.0])}, config, _frame([1.0, 1.0])
    )
    assert evidence.status == "available"
    assert evidence.definition is not None
    assert evidence.definition.symbol == "^GSPC"
    assert stub.loaded[0][:2] == ("^GSPC", "yahoo_finance")


def test_undefined_asset_class_never_loads_another_market() -> None:
    config = _config("2330.TW", "yahoo_finance", market_type=None)
    evidence, stub = _evidence({}, config, _frame([1.0, 1.0]))
    assert evidence.status == "undefined"
    assert evidence.reason == "undefined_asset_class"
    assert evidence.definition is None
    assert stub.loaded == []


def test_provider_failure_is_recorded_instead_of_failing_the_run() -> None:
    config = _config("ETH/USDT", "binance_vision")
    evidence, _ = _evidence(
        {"BTC/USDT": RuntimeError("network down")}, config, _frame([1.0, 1.0])
    )
    assert evidence.status == "unavailable"
    assert evidence.reason == "benchmark_data_unavailable"
    assert evidence.detail == "network down"


def test_baseline_without_timestamps_cannot_be_aligned() -> None:
    config = _config("ETH/USDT", "binance_vision")
    evidence, _ = _evidence(
        {"BTC/USDT": pd.DataFrame({"close": [100.0, 101.0]})},
        config,
        _frame([1.0, 1.0]),
    )
    assert evidence.status == "unavailable"
    assert evidence.reason == "missing_timestamps"


@pytest.mark.parametrize("closes", [[100.0, 0.0], [100.0, float("nan")], [100.0, -1.0]])
def test_nonpositive_or_missing_baseline_prices_are_refused(
    closes: list[float],
) -> None:
    config = _config("ETH/USDT", "binance_vision")
    evidence, _ = _evidence({"BTC/USDT": _frame(closes)}, config, _frame([1.0, 1.0]))
    assert evidence.status == "unavailable"
    assert evidence.reason == "invalid_benchmark_prices"


def test_benchmark_frames_are_validated_for_ordering() -> None:
    config = _config("ETH/USDT", "binance_vision")
    stamps = pd.to_datetime(["2024-01-02", "2024-01-01"], utc=True)
    frame = pd.DataFrame({"timestamp": stamps, "close": [100.0, 101.0]})
    evidence, _ = _evidence({"BTC/USDT": frame}, config, _frame([1.0, 1.0]))
    assert evidence.status == "unavailable"
    assert evidence.reason == "benchmark_data_unavailable"


def test_baseline_range_identity_records_the_requested_period() -> None:
    config = _config("ETH/USDT", "binance_vision")
    evidence, _ = _evidence(
        {"BTC/USDT": _frame([100.0, 101.0])}, config, _frame([1.0, 1.0])
    )
    assert evidence.dataset_id == "BTC/USDT:1h:2024-01-01:2024-01-02"
