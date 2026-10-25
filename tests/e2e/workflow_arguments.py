"""Compare successful workflow inputs after the SDK's declared conversions.

The recorded ToolCall remains raw diagnostic evidence. Only fields inspected by
these workflow verifiers are normalized; source/YAML, identifiers and nested
strategy JSON values keep their original meaning. No omitted inputs are supplied.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from pydantic import JsonValue, TypeAdapter, ValidationError

from tradingdev.app.contracts.reports import ReportCommentary
from tradingdev.domain.presentation.confirmation import ConfirmationPresentation

if TYPE_CHECKING:
    from tests.e2e.llm_client import ToolCall

_PREPARE: dict[str, TypeAdapter[Any]] = {
    "presentation": TypeAdapter(ConfirmationPresentation),
    "parameters": TypeAdapter(dict[str, JsonValue] | None),
    "backtest_overrides": TypeAdapter(dict[str, JsonValue] | None),
    "minimum_history_bars": TypeAdapter(int),
    "sample_bars": TypeAdapter(int),
}
_PAGE: dict[str, TypeAdapter[Any]] = {
    "limit": TypeAdapter(int),
    "offset": TypeAdapter(int),
}
_FIELDS: dict[str, dict[str, TypeAdapter[Any]]] = {
    "build_dataset": {"horizon": TypeAdapter(int)},
    "train_model": {
        "parameters": TypeAdapter(dict[str, JsonValue] | None),
        "n_folds": TypeAdapter(int),
        "embargo": TypeAdapter(int),
        "min_train_bars": TypeAdapter(int | None),
    },
    "predict": _PAGE,
    "leaderboard": _PAGE,
    "evaluate_signal": {
        "window": TypeAdapter(int),
        "n_quantiles": TypeAdapter(int),
        "record": TypeAdapter(bool),
    },
    "prepare_backtest": _PREPARE,
    "prepare_walk_forward": _PREPARE,
    "prepare_optimization": {
        **_PREPARE,
        "param_ranges": TypeAdapter(dict[str, list[JsonValue]]),
    },
    "find_runs": {"parameters": TypeAdapter(dict[str, Any] | None)},
    "generate_report": {
        "run_ids": TypeAdapter(list[str]),
        "sections": TypeAdapter(list[str] | None),
        "commentary": TypeAdapter(list[ReportCommentary] | None),
    },
    "get_run_trades": _PAGE,
    "get_run_executions": _PAGE,
    "get_run_account_history": _PAGE,
    "cleanup_strategy_drafts": {
        "revision_ids": TypeAdapter(list[str] | None),
        "apply": TypeAdapter(bool),
    },
}


def effective_workflow_calls(calls: list[ToolCall]) -> list[ToolCall]:
    """Return comparison copies only for calls with actual successful responses."""
    normalized: list[ToolCall] = []
    for call in calls:
        if not isinstance(call.result, dict) or call.result.get("success") is not True:
            normalized.append(call)
            continue
        arguments = dict(call.arguments)
        for name, adapter in _FIELDS.get(call.name, {}).items():
            if name not in arguments:
                continue
            value = arguments[name]
            # Match FuncMetadata.pre_parse_json: only top-level objects, arrays
            # and null are replaced. Numeric/bool strings reach Pydantic as-is.
            if isinstance(value, str):
                try:
                    decoded = json.loads(value)
                except json.JSONDecodeError:
                    pass
                else:
                    if not isinstance(decoded, str | int | float):
                        value = decoded
            try:
                validated = adapter.validate_python(value)
            except ValidationError as exc:
                raise AssertionError(
                    f"Invalid successful {call.name} argument {name}: {exc}"
                ) from exc
            arguments[name] = adapter.dump_python(
                validated, mode="json", exclude_unset=True
            )
        normalized.append(replace(call, arguments=arguments))
    return normalized
