"""Failure-only comparison evidence without replaying large MCP payloads in logs."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tests.e2e.llm_client import ToolCall

_IDENTITY_FIELDS = {
    "plan_id",
    "job_id",
    "run_id",
    "run_ids",
    "comparison_job_id",
    "strategy_id",
    "revision_id",
    "manifest_hash",
    "dataset_id",
    "report_id",
    "artifact_id",
    "path",
    "scope",
    "available_scopes",
    "default_scope",
}
_RESULT_FIELDS = _IDENTITY_FIELDS | {
    "success",
    "is_error",
    "code",
    "error",
    "status",
    "next_tool",
    "required_parameter_paths",
    "display_metrics",
}


def _compact(value: Any, *, limit: int = 4000) -> Any:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if len(encoded) <= limit:
        return value
    return {
        "complete": False,
        "characters": len(encoded),
        "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "preview": encoded[:400],
    }


def comparison_failure_diagnostics(calls: list[ToolCall]) -> dict[str, Any]:
    """Retain every attempt, its small inputs and output identity, never source/data."""
    attempts = []
    for index, call in enumerate(calls, 1):
        result = call.result
        if isinstance(result, dict):
            summary = {
                key: _compact(result[key])
                for key in sorted(_RESULT_FIELDS & result.keys())
            }
            if isinstance(result.get("run"), dict):
                run = result["run"]
                summary["run"] = {
                    key: _compact(run[key])
                    for key in sorted(_IDENTITY_FIELDS & run.keys())
                }
            if isinstance(result.get("comparison"), dict):
                comparison = result["comparison"]
                summary["comparison"] = {
                    key: _compact(comparison[key])
                    for key in ("outcome", "counts", "candidates", "dataset_id")
                    if key in comparison
                }
        else:
            summary = {"value": _compact(result)}
        attempts.append(
            {
                "index": index,
                "tool": call.name,
                "arguments": {
                    key: [
                        {
                            field: _compact(
                                setting, limit=400 if field == "rationale" else 4000
                            )
                            for field, setting in candidate.items()
                        }
                        if isinstance(candidate, dict)
                        else _compact(candidate)
                        for candidate in value
                    ]
                    if key == "candidates" and isinstance(value, list)
                    else _compact(value)
                    for key, value in call.arguments.items()
                },
                "result_type": type(result).__name__,
                "result_keys": sorted(result) if isinstance(result, dict) else None,
                "result": summary,
            }
        )
    return {
        "attempt_count": len(calls),
        "all_attempts_retained": True,
        "detail_policy": (
            "Only result identity, status and errors are shown; result source, "
            "schemas, trades and observations stay out of exception text. Large "
            "individual values have an explicit length/hash/preview."
        ),
        "attempts": attempts,
    }
