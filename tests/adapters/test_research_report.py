"""Saved explanation display preserves provenance and treats prose as text."""

from __future__ import annotations

from typing import Any

import pytest

from tradingdev.adapters.reporting.research import (
    execution_assumptions_html,
    render_research_documents,
)


def _payload() -> dict[str, Any]:
    return {
        "runs": [{"identity": {"run_id": "run"}, "research_evidence_hash": "saved"}],
        "research_documents": {
            "saved": {
                "evidence_id": "record",
                "draft": {
                    "presentation": {
                        "title": "A recorded strategy",
                        "summary": "Original research intent",
                        "parameter_descriptions": {
                            "/threshold": {"label": "Threshold", "unit": "quote"},
                        },
                    },
                    "rules": [
                        {
                            "condition": "close < lower",
                            "action": "Enter after the completed bar",
                            "formula": "signal[t] = close[t] < lower[t]",
                            "variables": [
                                {
                                    "symbol": "t",
                                    "meaning": "Completed bar",
                                    "unit": "bar",
                                }
                            ],
                            "source_ids": ["reference"],
                        }
                    ],
                    "sources": [
                        {
                            "source_id": "reference",
                            "title": "Recorded reference",
                            "url": "https://example.invalid/reference?a=1&b=2",
                            "accessed_at": "2024-01-01T00:00:00Z",
                            "retrieval_kind": "user_supplied",
                            "support": "This reference explains a formula only.",
                            "claims": [],
                        }
                    ],
                    "limits": ["The chosen threshold has not been sourced."],
                },
                "parameters": [
                    {
                        "target_id": "base",
                        "parameter_path": "/threshold",
                        "value": 0,
                        "provenance": "model_proposal",
                        "reason": "User will review it",
                        "source_claims": [],
                    }
                ],
                "execution_assumptions": [
                    {
                        "policy_version": "saved-old-policy",
                        "items": [
                            {
                                "label": "Saved rule",
                                "value": "Original execution wording",
                            }
                        ],
                    }
                ],
            },
        },
    }


def test_formulas_values_and_unverified_origin_are_visible() -> None:
    html = render_research_documents(_payload())
    assert "signal[t] = close[t] &lt; lower[t]" in html
    assert "<td>0</td>" in html
    assert "模型提案（未經來源查證）" in html
    assert "使用者提供" in html
    assert "不證明來源真實、瀏覽已發生" in html
    assert "https://example.invalid/reference?a=1&amp;b=2" in html
    assert "Original execution wording" in html
    assert "saved-old-policy" in html


def test_source_and_formula_markup_cannot_become_executable_html() -> None:
    payload = _payload()
    draft = payload["research_documents"]["saved"]["draft"]
    malicious = "<img src=x onerror=alert(1)><script>alert(2)</script>"
    draft["rules"][0]["formula"] = malicious
    draft["sources"][0]["title"] = malicious
    draft["sources"][0]["support"] = malicious
    draft["limits"] = [malicious]
    html = render_research_documents(payload)
    assert "<img" not in html
    assert "<script>" not in html
    assert "&lt;img" in html


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "file:///tmp/reference",
        "https://user:secret@example.invalid/",
    ],
)
def test_only_safe_reference_links_are_clickable(url: str) -> None:
    payload = _payload()
    payload["research_documents"]["saved"]["draft"]["sources"][0]["url"] = url
    html = render_research_documents(payload)
    assert "<a href=" not in html
    assert "來源網址不可作為連結" in html


def test_missing_records_stay_distinct_from_a_record_with_no_references() -> None:
    absent = render_research_documents({"runs": [], "research_documents": {}})
    assert "未保存交易規則" in absent
    payload = _payload()
    payload["research_documents"]["saved"]["draft"]["sources"] = []
    payload["runs"].append({"identity": {"run_id": "legacy"}})
    html = render_research_documents(payload)
    assert "沒有保存外部參考來源" in html
    assert "以下執行未保存研究說明：legacy" in html
    assert "Original research intent" in html
    assert "未保存執行規則說明" in execution_assumptions_html(None)


def test_all_failed_comparison_still_renders_approved_rules_and_candidate_names() -> (
    None
):
    payload = _payload()
    payload["runs"] = []
    payload["fixed_comparison"] = {
        "candidates": [{"candidate_id": "candidate_1", "label": "First proposal"}]
    }
    payload["research_documents"]["saved"]["parameters"][0]["target_id"] = "candidate_1"
    html = render_research_documents(payload)
    assert "First proposal" in html
    assert "Original research intent" in html
    assert "signal[t]" in html
