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


def _recorded_research(call: ToolCall) -> dict[str, object]:
    config = call.result.get("effective_config")
    assert isinstance(config, dict), (
        f"{call.name} must retain effective_config research evidence"
    )
    result: dict[str, object] = {}
    for field, (section, key) in _FIELDS.items():
        values = config.get(section)
        assert isinstance(values, dict) and key in values, (
            f"{call.name} effective_config is missing {section}.{key}"
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
    assert isinstance(research, dict) and research.get("kind") == (
        "inherit_selected_revision"
    ), "prepare_backtest research must inherit the selected revision"
    expected = _recorded_research(validated)
    assert _recorded_research(dry_run) == expected, (
        "Validated and dry-run effective_config research settings differ"
    )
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
            f"prepare_backtest research changes {field}: "
            f"selected revision has {expected[field]!r}; got {value!r}"
        )
