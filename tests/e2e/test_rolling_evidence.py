"""Saved rolling evidence may be discovered through either supported read route."""

from __future__ import annotations

import hashlib
import json

import pytest

from tests.e2e.llm_client import ToolCall
from tests.e2e.test_llm_ml import verify_rolling_evidence

_TEXT = json.dumps(
    {
        "rolls": [
            {"index": 0, "n_train_rows": 52, "trained": False},
            {"index": 1, "n_train_rows": 57, "trained": True},
            {"index": 2, "n_train_rows": 62, "trained": True},
            {"index": 3, "n_train_rows": 67, "trained": True},
        ]
    }
)
_REPLY = (
    "rolling_saved 的完整證據保存在 performance_json；"
    "嘗試 4 次、成功 3 次，第 2 次有 57 筆。"
)


def _page(start: int, end: int) -> ToolCall:
    artifact_id = "rolling_saved:performance_json"
    return ToolCall(
        "get_artifact",
        {"artifact_id": artifact_id},
        {
            "success": True,
            "artifact": {"artifact_id": artifact_id},
            "content": _TEXT[start:end],
            "content_page": {
                "offset": start,
                "total_characters": len(_TEXT),
                "content_sha256": hashlib.sha256(_TEXT.encode()).hexdigest(),
            },
        },
    )


@pytest.mark.parametrize("route", ["full", "pages", "summary"])
def test_rolling_discovery_accepts_saved_artifact_or_summary_and_detail(
    route: str,
) -> None:
    calls = [ToolCall("get_ml_contract", {}, {"success": True})]
    if route == "full":
        calls.append(_page(0, len(_TEXT)))
    elif route == "pages":
        # A valid answer may span a page boundary inside the requested number.
        split = _TEXT.index("57") + 1
        calls.extend([_page(0, split), _page(split, len(_TEXT))])
    else:
        calls.extend(
            [
                ToolCall(
                    "get_run_metrics",
                    {"run_id": "rolling_saved"},
                    {
                        "success": True,
                        "scope": "fold/0/test",
                        "metadata": {
                            "rolling_trace": {
                                "roll_count": 4,
                                "trained_roll_count": 3,
                                "artifact_id": "rolling_saved:performance_json",
                            }
                        },
                    },
                ),
                _page(50, 120),
            ]
        )
    verify_rolling_evidence(calls, _REPLY, _TEXT)


@pytest.mark.parametrize(
    "damage",
    ["gap", "wrong_artifact", "changed_content", "missing_count", "no_contract"],
)
def test_rolling_discovery_rejects_incomplete_or_unbound_evidence(damage: str) -> None:
    page = _page(0, len(_TEXT))
    calls = [ToolCall("get_ml_contract", {}, {"success": True}), page]
    reply = _REPLY
    if damage == "gap":
        page.result["content"] = _TEXT[:150]
    elif damage == "wrong_artifact":
        page.arguments["artifact_id"] = "another:performance_json"
    elif damage == "changed_content":
        page.result["content"] = _TEXT.replace("57", "58")
    elif damage == "missing_count":
        reply = reply.replace("成功 3 次", "成功次數未查到")
    else:
        calls.pop(0)
    with pytest.raises(AssertionError):
        verify_rolling_evidence(calls, reply, _TEXT)
