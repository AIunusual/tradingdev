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

from tradingdev.app.contracts.execution_requests import ParameterCandidateInput
from tradingdev.app.contracts.reports import ReportAnalysis, ReportCommentary
from tradingdev.domain.research import ResearchSpec, research_payload
from tradingdev.domain.research_evidence import EvidenceId, ResearchEvidenceDraft
from tradingdev.domain.validation.config import WalkForwardConfig

if TYPE_CHECKING:
    from tests.e2e.llm_client import ToolCall

_PREPARE: dict[str, TypeAdapter[Any]] = {
    "research": TypeAdapter(ResearchSpec),
    "evidence_id": TypeAdapter(EvidenceId),
    "parameters": TypeAdapter(dict[str, JsonValue] | None),
    "backtest_overrides": TypeAdapter(dict[str, JsonValue] | None),
}
_PAGE: dict[str, TypeAdapter[Any]] = {
    "limit": TypeAdapter(int),
    "offset": TypeAdapter(int),
}
_FIELDS: dict[str, dict[str, TypeAdapter[Any]]] = {
    "save_strategy": {"research": TypeAdapter(ResearchSpec)},
    "save_research_evidence": {"draft": TypeAdapter(ResearchEvidenceDraft)},
    "build_dataset": {"horizon": TypeAdapter(int)},
    "train_model": {
        "parameters": TypeAdapter(dict[str, JsonValue] | None),
        "n_folds": TypeAdapter(int),
        "embargo": TypeAdapter(int),
        "min_train_bars": TypeAdapter(int | None),
    },
    "predict": _PAGE,
    "leaderboard": {
        **_PAGE,
        "horizon": TypeAdapter(int | None),
        "allow_mixed": TypeAdapter(bool),
        "allow_parameter_axes": TypeAdapter(bool),
    },
    "evaluate_signal": {
        "window": TypeAdapter(int),
        "n_quantiles": TypeAdapter(int),
        "record": TypeAdapter(bool),
    },
    "prepare_backtest": _PREPARE,
    "prepare_comparison": {
        **{name: adapter for name, adapter in _PREPARE.items() if name != "parameters"},
        "candidates": TypeAdapter(list[ParameterCandidateInput]),
    },
    "prepare_walk_forward": {
        **_PREPARE,
        "validation_config": TypeAdapter(WalkForwardConfig | None),
    },
    "prepare_optimization": {
        **_PREPARE,
        "param_ranges": TypeAdapter(dict[str, list[JsonValue]]),
    },
    "find_runs": {"parameters": TypeAdapter(dict[str, Any] | None)},
    "generate_report": {
        "run_ids": TypeAdapter(list[str] | None),
        "ml_entry_ids": TypeAdapter(list[str] | None),
        "ml_allow_parameter_axes": TypeAdapter(bool),
        "sections": TypeAdapter(list[str] | None),
        "commentary": TypeAdapter(list[ReportCommentary] | None),
        "analysis": TypeAdapter(ReportAnalysis | None),
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
            arguments[name] = (
                research_payload(validated)
                if name == "research"
                else adapter.dump_python(validated, mode="json", exclude_unset=True)
            )
        normalized.append(replace(call, arguments=arguments))
    return normalized
