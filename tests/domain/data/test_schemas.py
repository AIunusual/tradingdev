"""Tests for Pydantic data schemas."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tradingdev.domain.backtest.schemas import BacktestConfig
from tradingdev.domain.data.schemas import MarketDataRequest, OHLCVBar
from tradingdev.domain.strategies.bundled.kd_strategy.config import KDStrategyConfig


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2024-01-01", "2024-01-01T01:00:00Z"),
        ("2024-01-01T08:00:00+08:00", "2023-12-31T17:00:00-08:00"),
    ],
)
def test_market_request_normalizes_instants_before_selecting_years(
    start: str, end: str
) -> None:
    request = MarketDataRequest.model_validate(
        {"symbol": "BTC/USDT", "timeframe": "1h", "start_date": start, "end_date": end}
    )
    assert request.start_date == datetime(2024, 1, 1, tzinfo=UTC)
    assert request.end_date == datetime(2024, 1, 1, 1, tzinfo=UTC)
    assert request.start_date.tzinfo is UTC
    assert request.end_date.tzinfo is UTC


@pytest.mark.parametrize("reverse", [False, True])
def test_market_request_checks_order_after_timezone_conversion(reverse: bool) -> None:
    values = {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01T08:00:00+08:00",
        "end_date": "2023-12-31T15:00:00-08:00" if reverse else "2024-01-01",
    }
    if reverse:
        with pytest.raises(
            ValidationError, match="end_date must not precede start_date"
        ):
            MarketDataRequest.model_validate(values)
    else:
        request = MarketDataRequest.model_validate(values)
        assert request.start_date == request.end_date


class TestOHLCVBar:
    def test_valid_bar(self) -> None:
        bar = OHLCVBar(
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            open=42000.0,
            high=42500.0,
            low=41800.0,
            close=42200.0,
            volume=1234.5,
        )
        assert bar.close == 42200.0

    def test_negative_volume_rejected(self) -> None:
        with pytest.raises(ValidationError, match="volume must be non-negative"):
            OHLCVBar(
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                open=42000.0,
                high=42500.0,
                low=41800.0,
                close=42200.0,
                volume=-1.0,
            )

    def test_zero_volume_accepted(self) -> None:
        bar = OHLCVBar(
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            open=42000.0,
            high=42500.0,
            low=41800.0,
            close=42200.0,
            volume=0.0,
        )
        assert bar.volume == 0.0


class TestBacktestConfig:
    def test_valid_signal_config(self) -> None:
        config = BacktestConfig(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            init_cash=10_000.0,
            mode="signal",
        )
        assert config.init_cash == 10_000.0
        assert config.fees == 0.0006

    def test_valid_volume_config(self) -> None:
        config = BacktestConfig(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            mode="volume",
        )
        assert config.init_cash is None
        assert config.monthly_max_loss == 1500.0

    def test_signal_mode_requires_init_cash(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="init_cash is required"):
            BacktestConfig(
                symbol="BTC/USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 12, 31),
                mode="signal",
            )

    def test_default_values(self) -> None:
        config = BacktestConfig(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            mode="volume",
        )
        assert config.slippage == 0.0005


class TestKDStrategyConfig:
    def test_defaults(self) -> None:
        config = KDStrategyConfig()
        assert config.k_period == 14
        assert config.d_period == 3
        assert config.smooth_k == 3
        assert config.overbought == 80.0
        assert config.oversold == 20.0

    def test_custom_values(self) -> None:
        config = KDStrategyConfig(
            k_period=9, d_period=5, smooth_k=5, overbought=70.0, oversold=30.0
        )
        assert config.k_period == 9
        assert config.overbought == 70.0

    def test_overbought_out_of_range(self) -> None:
        with pytest.raises(ValidationError, match="overbought must be between"):
            KDStrategyConfig(overbought=150.0)

    def test_oversold_out_of_range(self) -> None:
        with pytest.raises(ValidationError, match="oversold must be between"):
            KDStrategyConfig(oversold=-10.0)
