"""The one approval document must disclose every fixed candidate and its checks."""

from __future__ import annotations

from html import unescape

import pytest
from tests.domain.test_confirmation_presentation import (
    _manifest,
    _presentation,
    _receipt,
)

from tradingdev.adapters.presentation.confirmation import (
    render_confirmation_html,
    render_confirmation_summary,
)
from tradingdev.domain.comparison import ComparisonCandidate, ComparisonManifest
from tradingdev.domain.preflight import (
    ComparisonCandidateReceipt,
    ComparisonPreflightReceipt,
)
from tradingdev.domain.presentation.comparison_confirmation import (
    build_comparison_confirmation,
)
from tradingdev.domain.presentation.confirmation import (
    ConfirmationPresentationError,
    ParameterDescription,
)


def comparison_fixture() -> tuple[ComparisonManifest, ComparisonPreflightReceipt]:
    manifest = ComparisonManifest.create(
        [
            ComparisonCandidate(
                candidate_id=f"candidate_{index}",
                label=f"週期 {period}",
                rationale=f"事先選定的 {period} 根示範，不宣稱最佳。",
                manifest=_manifest(parameters={"period": period}),
            )
            for index, period in enumerate((10, 20, 30), 1)
        ]
    )
    return manifest, ComparisonPreflightReceipt(
        manifest_hash=manifest.manifest_hash,
        elapsed_seconds=1.5,
        candidates=[
            ComparisonCandidateReceipt(
                candidate_id=c.candidate_id, receipt=_receipt(c.manifest)
            )
            for c in manifest.candidates
        ],
    )


def test_confirmation_shows_each_candidate_without_combining_capital_or_results() -> (
    None
):
    manifest, receipt = comparison_fixture()
    presentation = _presentation(manifest.candidates[0].manifest)
    document = build_comparison_confirmation(
        manifest, presentation, receipt, plan_id="review"
    )
    html = unescape(render_confirmation_html(document))
    summary = render_confirmation_summary(document, mode="signal")
    assert "固定參數比較（3 組獨立回測）" in summary
    assert "10000" in summary and "30000" not in summary
    assert document.identity.manifest_hash == manifest.manifest_hash
    assert "不進行自動搜尋" in html
    assert "失敗、取消或未執行時列出原因" in html
    parameters = next(
        section for section in document.sections if section.id == "parameters"
    )
    checks = next(section for section in document.sections if section.id == "preflight")
    for candidate in manifest.candidates:
        assert candidate.rationale in html
        assert candidate.manifest.manifest_hash in html
        assert any(
            field.label.startswith(candidate.label + "／") for field in checks.fields
        )
        assert any(
            field.label.startswith(candidate.label + "／")
            and field.value_text
            == str(candidate.manifest.strategy_execution.constructor_kwargs["period"])
            for field in parameters.fields
        )


@pytest.mark.parametrize(
    "failure", ["missing", "extra", "swapped_receipt", "lost_candidate"]
)
def test_incomplete_or_mismatched_candidate_presentation_is_rejected(
    failure: str,
) -> None:
    manifest, receipt = comparison_fixture()
    presentation = _presentation(manifest.candidates[0].manifest)
    if failure == "missing":
        presentation.parameter_descriptions.clear()
    elif failure == "extra":
        presentation.parameter_descriptions["/not_a_parameter"] = ParameterDescription(
            label="不存在", description="錯誤欄位"
        )
    elif failure == "swapped_receipt":
        receipt.candidates.reverse()
    else:
        receipt.candidates.pop()
    with pytest.raises(ValueError):
        build_comparison_confirmation(manifest, presentation, receipt)


def test_parameter_error_lists_union_of_candidate_constructor_paths() -> None:
    manifest, receipt = comparison_fixture()
    presentation = _presentation(manifest.candidates[0].manifest)
    presentation.parameter_descriptions.clear()
    with pytest.raises(ConfirmationPresentationError) as error:
        build_comparison_confirmation(manifest, presentation, receipt)
    assert error.value.required_parameter_paths == ["/period"]
