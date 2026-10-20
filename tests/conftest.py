"""Shared test fixtures."""

import os
from datetime import UTC
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

pytest_plugins = ["tests.llm_plugin"]


def pytest_configure(config: pytest.Config) -> None:
    """Isolate plotting caches before QuantStats is imported during collection."""
    directory = TemporaryDirectory(prefix="tradingdev-plot-cache-")
    config.add_cleanup(directory.cleanup)
    environment = pytest.MonkeyPatch()
    config.add_cleanup(environment.undo)
    environment.setenv("MPLCONFIGDIR", directory.name)


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """Remove each test's files during teardown, including failed tests.

    Pytest's built-in fixture retains recent runs by default. TemporaryDirectory
    reports cleanup errors instead of silently retaining strategy/data artifacts.
    Tests that launch processes must stop them before this fixture is torn down.
    """
    original_cwd = Path.cwd()
    with TemporaryDirectory(prefix="tradingdev-test-") as directory:
        path = Path(directory).resolve()
        try:
            yield path
        finally:
            # A test may request monkeypatch before tmp_path, making its chdir
            # undo run after our teardown. Windows cannot remove the current cwd.
            try:
                current_cwd = Path.cwd()
            except FileNotFoundError:
                os.chdir(original_cwd)
            else:
                if current_cwd.is_relative_to(path):
                    os.chdir(original_cwd)


def _make_ohlcv(
    n: int,
    seed: int,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Helper to build a synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(seed=seed)

    timestamps = pd.date_range(start=start, periods=n, freq="h", tz=UTC)

    returns = rng.normal(0.0001, 0.005, size=n)
    close = 40_000.0 * np.cumprod(1 + returns)

    noise = rng.uniform(0.001, 0.005, size=n)
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = close * (1 + rng.normal(0, 0.002, size=n))
    volume = rng.uniform(100, 1000, size=n)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with 100 rows of 1h BTC/USDT data.

    Generates a random walk price series around 40,000 with realistic
    OHLCV structure.
    """
    return _make_ohlcv(100, seed=42)


@pytest.fixture
def sample_ohlcv_with_kd() -> pd.DataFrame:
    """OHLCV data extended to 200 rows for KD indicator calculation.

    KD requires enough bars for the lookback period (default 14+3+3).
    """
    return _make_ohlcv(200, seed=123)


@pytest.fixture
def large_ohlcv_df() -> pd.DataFrame:
    """1000-row OHLCV DataFrame for ML strategy tests."""
    return _make_ohlcv(1000, seed=456)


@pytest.fixture
def walk_forward_ohlcv_df() -> pd.DataFrame:
    """500-row OHLCV DataFrame spanning ~20 days, for walk-forward tests."""
    return _make_ohlcv(500, seed=789)


@pytest.fixture
def sample_ohlcv_with_dvol_df() -> pd.DataFrame:
    """100-row OHLCV DataFrame with a synthetic ``dvol`` column.

    DVOL values are uniform random between 30 and 80 (annualized %).
    """
    df = _make_ohlcv(100, seed=42)
    rng = np.random.default_rng(seed=42)
    df["dvol"] = rng.uniform(30.0, 80.0, size=100)
    return df
