"""Explicit research inputs for scripted fixtures, never novice user replies."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import yaml

from tradingdev.domain.research import NewResearchSpec


def research_for_config(
    config: str | dict[str, Any], *, strategy_outline: str = "Fixture strategy"
) -> NewResearchSpec:
    """Preserve the existing fixture's choices instead of adding product defaults.

    Missing annualization stays unavailable. New default-policy tests must construct
    their own research input; model conversations must supply actual user messages.
    """
    raw = yaml.safe_load(config) if isinstance(config, str) else config
    assert isinstance(raw, dict)
    backtest = raw["backtest"]
    assert isinstance(backtest, dict)
    annualization = backtest.get("periods_per_year")
    payload: dict[str, Any] = {
        "kind": "new",
        "strategy_outline": strategy_outline,
        **{
            name: backtest[name]
            for name in ("symbol", "timeframe", "start_date", "end_date")
        },
        "annualization": "unavailable" if annualization is None else annualization,
    }
    for name in ("init_cash", "fees", "slippage"):
        if name in backtest and backtest[name] is not None:
            payload[name] = backtest[name]
    return NewResearchSpec.model_validate(payload)


def with_fixture_market_source(config: str) -> str:
    """Make the old fixture's Binance market assumption explicit in its YAML."""
    raw = yaml.safe_load(config)
    assert isinstance(raw, dict)
    result = deepcopy(raw)
    data = result.setdefault("data", {})
    assert isinstance(data, dict)
    requirements = data.get("requirements", {})
    assert isinstance(requirements, dict)
    market = requirements.get("market", {})
    assert isinstance(market, dict)
    if "source" not in data and "source" not in market:
        data["source"] = "binance_vision"
    if (
        data.get("source") == "binance_vision"
        or market.get("source") == "binance_vision"
    ):
        data.setdefault("market_type", "futures/um")
    return yaml.safe_dump(result, sort_keys=False)
