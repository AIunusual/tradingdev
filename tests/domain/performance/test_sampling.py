"""Frequency eligibility comes from declared bars, never elapsed timestamp gaps."""

import pytest

from tradingdev.domain.performance.sampling import (
    bar_duration_ns,
    daily_observation_unavailable_reason,
)

_DAY_NS = 86_400_000_000_000


@pytest.mark.parametrize(
    "frequency",
    ["1m", "1min", "60m", "1440m", "1T", "1h", "24h", "1d", "D", "86400s", "1ms"],
)
def test_daily_or_finer_frequency_supports_daily_observations(frequency: str) -> None:
    assert daily_observation_unavailable_reason(frequency) is None


@pytest.mark.parametrize(
    "frequency",
    ["1w", "1wk", "W", "1M", "1mo", "MS", "ME", "3d", "7D", "25h", "1441m"],
)
def test_coarse_bars_cannot_supply_daily_observations(frequency: str) -> None:
    assert (
        daily_observation_unavailable_reason(frequency) == "unsupported_daily_sampling"
    )


@pytest.mark.parametrize(
    "frequency",
    [
        "",
        "unknown",
        "0d",
        "0M",
        "-1d",
        "1.5h",
        "1m2s",
        "inf",
        "1",
        "1x",
        "9" * 5000 + "m",
    ],
)
def test_unknown_or_invalid_frequency_does_not_claim_daily_observations(
    frequency: str,
) -> None:
    assert daily_observation_unavailable_reason(frequency) == "unknown_bar_frequency"


@pytest.mark.parametrize(
    "frequency,expected",
    [
        ("1h", 3_600_000_000_000),
        ("4h", 4 * 3_600_000_000_000),
        ("30m", 30 * 60_000_000_000),
        ("1d", _DAY_NS),
        ("D", _DAY_NS),
        ("1w", 7 * _DAY_NS),
        ("1wk", 7 * _DAY_NS),
        ("h", 3_600_000_000_000),
    ],
)
def test_a_declared_bar_has_a_fixed_duration(frequency: str, expected: int) -> None:
    assert bar_duration_ns(frequency) == expected


@pytest.mark.parametrize(
    "frequency", ["1M", "1mo", "MS", "ME", "", "bogus", "0h", "1.5h", "1m2s", "1x"]
)
def test_a_bar_without_a_fixed_duration_is_refused(frequency: str) -> None:
    """Months and years vary; a calendar cannot be assumed from the label."""
    assert bar_duration_ns(frequency) is None
