"""Check reuse against recorded revision evidence without resolving new defaults."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tests.e2e.llm_client import ToolCall

_FIELDS = {
    "strategy_outline": ("research", "strategy_outline"),
    "symbol": ("backtest", "symbol"),
    "timeframe": ("backtest", "timeframe"),
    "start_date": ("backtest", "start_date"),
    "end_date": ("backtest", "end_date"),
    "init_cash": ("backtest", "init_cash"),
    "fees": ("backtest", "fees"),
    "slippage": ("backtest", "slippage"),
    "annualization": ("backtest", "periods_per_year"),
}


def _value(field: str, value: object) -> object:
    if field in {"start_date", "end_date"}:
        assert isinstance(value, str), f"Research {field} must record an ISO date"
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise AssertionError(f"Research {field} is not an ISO date") from exc
        return (
            parsed.replace(tzinfo=UTC)
            if parsed.tzinfo is None
            else parsed.astimezone(UTC)
        )
    if field in {"strategy_outline", "symbol", "timeframe"}:
        assert isinstance(value, str) and value.strip(), (
            f"Research {field} must record nonempty text"
        )
    elif field == "annualization" and value is None:
        return None
    else:
        assert (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(value)
        ), f"Research {field} must record a finite number"
    return value


def research_values(config: Any, *, context: str) -> dict[str, object]:
    """Project complete observed settings; never add defaults or scenario values."""
    assert isinstance(config, dict), (
        f"{context} must retain effective_config research evidence"
    )
    result: dict[str, object] = {}
    for field, (section, key) in _FIELDS.items():
        values = config.get(section)
        assert isinstance(values, dict) and key in values, (
            f"{context} effective_config is missing {section}.{key}"
        )
        result[field] = _value(field, values[key])
    return result


def assert_research_reused(
    research: Any, *, validated: ToolCall, dry_run: ToolCall
) -> None:
    """Require inherited or explicitly equal values from one verified revision.

    Callers bind both evidence records to the selected revision and check their
    successful lifecycle states. This checks observed values only: it neither
    fills missing evidence from a scenario nor rewrites input provenance.
    """
    expected = research_values(
        validated.result.get("effective_config"), context=validated.name
    )
    assert (
        research_values(dry_run.result.get("effective_config"), context=dry_run.name)
        == expected
    ), "Validated and dry-run effective_config research settings differ"
    _assert_inherited_values(research, expected)


def assert_research_matches_config(research: Any, selected_config: Any) -> None:
    """Compare an inherited request with independently bound immutable source YAML."""
    _assert_inherited_values(
        research, research_values(selected_config, context="Selected revision")
    )


def _assert_inherited_values(research: Any, expected: dict[str, object]) -> None:
    assert isinstance(research, dict) and research.get("kind") == (
        "inherit_selected_revision"
    ), "Prepared research must inherit the selected revision"
    for field, value in research.items():
        if field == "kind":
            continue
        assert field in _FIELDS, f"Unexpected research field {field!r}"
        if field == "annualization":
            assert value != "market", (
                "Fixed research reuse cannot reselect the market annualization policy"
            )
            if value == "unavailable":
                value = None
        assert _value(field, value) == expected[field], (
            f"Prepared research changes {field}: "
            f"selected revision has {expected[field]!r}; got {value!r}"
        )
