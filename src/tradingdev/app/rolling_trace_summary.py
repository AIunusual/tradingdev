"""Keep growing rolling evidence in artifacts when projecting query metadata."""

from __future__ import annotations

from typing import Any


def query_metadata(
    metadata: dict[str, Any], *, run_id: str, scope_id: str
) -> dict[str, Any]:
    """Project traces, including nested fold metadata, without changing storage."""
    root = f"/scopes/{_escape(scope_id)}/metadata"
    return {
        key: _project(value, key, f"{root}/{_escape(key)}", run_id)
        for key, value in metadata.items()
    }


def _escape(component: str) -> str:
    return component.replace("~", "~0").replace("/", "~1")


def _project(value: Any, key: str, pointer: str, run_id: str) -> Any:
    if isinstance(value, dict):
        if key == "rolling_trace":
            return _summary(value, pointer, run_id)
        return {
            name: _project(item, name, f"{pointer}/{_escape(name)}", run_id)
            for name, item in value.items()
        }
    if isinstance(value, list):
        return [
            _project(item, "", f"{pointer}/{index}", run_id)
            for index, item in enumerate(value)
        ]
    return value


def _summary(trace: dict[str, Any], pointer: str, run_id: str) -> dict[str, Any]:
    rolls = trace.get("rolls")
    unscored = trace.get("unscored_bars")
    complete_training_flags = isinstance(rolls, list) and all(
        isinstance(roll, dict) and isinstance(roll.get("trained"), bool)
        for roll in rolls
    )
    result: dict[str, Any] = {
        "representation": "summary",
        "content_included": False,
        "artifact_id": f"{run_id}:performance_json",
        "json_pointer": pointer,
        "roll_count": len(rolls) if isinstance(rolls, list) else None,
        "trained_roll_count": sum(
            isinstance(roll, dict) and roll.get("trained") is True for roll in rolls
        )
        if complete_training_flags and isinstance(rolls, list)
        else None,
        "unscored_bar_count": len(unscored) if isinstance(unscored, list) else None,
    }
    for key in (
        "scheme",
        "rtype",
        "step",
        "window_size",
        "min_train_bars",
        "predict_lookback",
        "label_horizon",
        "initial_model_available",
        "bars",
        "bars_scored",
    ):
        value = trace.get(key)
        # Custom strategy hooks need not follow the built-in payload. Only
        # compact scalars are projected; their complete evidence stays readable.
        if key in trace and (
            value is None
            or isinstance(value, (bool, int, float))
            or isinstance(value, str)
            and len(value) <= 64
        ):
            result[key] = value
    return result
