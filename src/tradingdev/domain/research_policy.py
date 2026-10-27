"""Single maintenance source for guided research simulation assumptions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import JsonValue

RESEARCH_POLICY_VERSION = "research-v1"
DEFAULT_INIT_CASH = 10_000.0
DEFAULT_FEES = 0.0006
DEFAULT_SLIPPAGE = 0.0005
CRYPTO_DAILY_PERIODS_PER_YEAR = 365.0


def research_policy_payload() -> dict[str, JsonValue]:
    """Project policy for discovery; effective configurations freeze these values."""
    return {
        "policy_version": RESEARCH_POLICY_VERSION,
        "required_inputs": [
            "strategy_outline",
            "symbol",
            "start_date",
            "end_date",
            "timeframe",
        ],
        "simulation_defaults": {
            "init_cash": DEFAULT_INIT_CASH,
            "init_cash_unit": "quote_currency",
            "fees": DEFAULT_FEES,
            "slippage": DEFAULT_SLIPPAGE,
            "description": (
                "Simulation assumptions, not current exchange fees. Explicit valid "
                "values take priority; zero costs remain zero. Initial capital applies "
                "to signal mode and does not alter volume strategy semantics."
            ),
        },
        "annualization": {
            "basis": "observed UTC daily equity returns",
            "crypto_daily_periods_per_year": CRYPTO_DAILY_PERIODS_PER_YEAR,
            "market_policy": (
                "365 is inferred only for a provider whose registered asset classes "
                "are exclusively crypto. Mixed providers, including Yahoo, require "
                "an explicit positive factor or 'unavailable'."
            ),
            "sampling": (
                "Daily statistics require a declared bar frequency no coarser than "
                "one day. A factor does not make weekly/monthly bars into "
                "daily observations."
            ),
        },
        "provenance": (
            "request/project_default/selected_revision describe configuration origins, "
            "not proof of human input. Inheritance requires a selected immutable "
            "revision."
        ),
        "data_selection": (
            "Specify data.source or data.requirements.market.source. Providers with "
            "market type choices also require an explicit data.market_type."
        ),
    }
