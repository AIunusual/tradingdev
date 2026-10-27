"""Offline evidence rejects invented execution coverage and stale requests."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from tests.domain.test_comparison import candidate

from tradingdev.domain.comparison import ComparisonManifest
from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.preflight import (
    ComparisonCandidateReceipt,
    ComparisonPreflightReceipt,
    PreflightReceipt,
    PreflightRequest,
    verify_program_receipt,
)
from tradingdev.domain.strategies.execution import StrategyExecution


def _receipt(**changed: Any) -> PreflightReceipt:
    return PreflightReceipt.model_validate(
        {
            "manifest_hash": "a" * 64,
            "elapsed_seconds": 1.0,
            "checked_paths": [
                "configuration",
                "execution_capabilities",
                "parameter_binding",
                "manifest_serialization",
                "signal_contract",
            ],
            "synthetic_fixture_rows": [80, 240],
            "signal_check": "synthetic_contract_passed",
            "training_requirement": "unknown",
            "not_checked": [
                "historical_data",
                "backtest_engine",
                "training",
                "performance",
            ],
            **changed,
        }
    )


def _manifest(*, bundled: bool = False) -> ExecutionManifest:
    return ExecutionManifest.create(
        kind="backtest",
        config={
            "strategy": {"id": "fixture", "source_hash": "b" * 64},
            "backtest": {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-02-01",
                "init_cash": 10000,
            },
        },
        strategy_execution=StrategyExecution(
            kind="bundled" if bundled else "generated",
            constructor_kwargs={},
        ),
    )


def test_request_needs_only_concrete_execution_arguments() -> None:
    request = PreflightRequest(kind="backtest", arguments={})
    assert request.model_dump() == {"kind": "backtest", "arguments": {}}


@pytest.mark.parametrize("field", ["minimum_history_bars", "sample_bars"])
def test_request_rejects_removed_history_controls(field: str) -> None:
    with pytest.raises(ValidationError):
        PreflightRequest.model_validate(
            {"kind": "backtest", "arguments": {}, field: 128}
        )


@pytest.mark.parametrize(
    "changed",
    [
        {"schema_version": 1},
        {"synthetic_fixture_rows": []},
        {"synthetic_fixture_rows": [240]},
        {"synthetic_fixture_rows": [True, 240]},
        {"synthetic_fixture_rows": [80, 240, 480]},
        {"signal_check": "deferred_until_training"},
        {"training_requirement": "required"},
        {"parameter_combinations_checked": 2},
        {"checked_paths": ["configuration"]},
        {"checked_paths": ["engine"]},
        {"not_checked": ["historical_data", "training", "performance"]},
        {"not_checked": ["historical_data"] * 4},
        {"trade_count": 0},
        {"data_origin": "market_data"},
    ],
)
def test_receipt_rejects_contradictory_or_historical_evidence(
    changed: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        _receipt(**changed)


def test_receipt_revalidates_mutable_evidence_before_reuse() -> None:
    manifest = _manifest()
    receipt = _receipt(manifest_hash=manifest.manifest_hash)
    receipt.verify_manifest(manifest)
    receipt.synthetic_fixture_rows[0] = 79
    with pytest.raises(ValueError, match="both contract fixtures"):
        receipt.verify_manifest(manifest)


@pytest.mark.parametrize("bundled", [False, True])
def test_receipt_is_bound_to_exact_manifest_and_strategy_kind(bundled: bool) -> None:
    manifest = _manifest(bundled=bundled)
    receipt = _receipt()
    with pytest.raises(ValueError, match="hash"):
        receipt.verify_manifest(manifest)
    receipt = _receipt(manifest_hash=manifest.manifest_hash)
    if bundled:
        with pytest.raises(ValueError, match="strategy execution kind"):
            receipt.verify_manifest(manifest)
    else:
        receipt.verify_manifest(manifest)


def test_receipt_rejects_search_coverage_for_non_search_manifest() -> None:
    manifest = _manifest()
    receipt = _receipt(manifest_hash=manifest.manifest_hash)
    fields = receipt.model_dump()
    fields["checked_paths"].append("candidate_binding")
    fields["parameter_combinations_checked"] = 2
    receipt = PreflightReceipt.model_validate(fields)
    with pytest.raises(ValueError, match="Candidate binding"):
        receipt.verify_manifest(manifest)


def _comparison_receipt(manifest: ComparisonManifest) -> ComparisonPreflightReceipt:
    return ComparisonPreflightReceipt(
        manifest_hash=manifest.manifest_hash,
        elapsed_seconds=1.0,
        candidates=[
            ComparisonCandidateReceipt(
                candidate_id=member.candidate_id,
                receipt=_receipt(manifest_hash=member.manifest.manifest_hash),
            )
            for member in manifest.candidates
        ],
    )


@pytest.mark.parametrize("version", [True, 1.0, "1"])
def test_comparison_receipt_schema_version_is_a_strict_integer(version: object) -> None:
    manifest = ComparisonManifest.create([candidate(1, 10)])
    fields = _comparison_receipt(manifest).model_dump()
    fields["schema_version"] = version
    with pytest.raises(ValueError, match="must be an integer"):
        ComparisonPreflightReceipt.model_validate(fields)


@pytest.mark.parametrize("version", [True, 2.0, "2"])
def test_single_receipt_schema_version_is_a_strict_integer(version: object) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        _receipt(schema_version=version)


@pytest.mark.parametrize(
    "invalid",
    ["bundle_hash", "child_hash", "missing", "duplicate", "order", "incomplete"],
)
def test_comparison_receipt_binds_complete_bundle_and_all_ordered_children(
    invalid: str,
) -> None:
    manifest = ComparisonManifest.create([candidate(1, 10), candidate(2, 20)])
    receipt = _comparison_receipt(manifest)
    verify_program_receipt(manifest, receipt)
    if invalid == "bundle_hash":
        receipt.manifest_hash = "0" * 64
    elif invalid == "child_hash":
        receipt.candidates[1].receipt.manifest_hash = receipt.candidates[
            0
        ].receipt.manifest_hash
    elif invalid == "missing":
        receipt.candidates.pop()
    elif invalid == "duplicate":
        receipt.candidates[1] = receipt.candidates[0]
    elif invalid == "order":
        receipt.candidates.reverse()
    else:
        receipt.candidates[1].receipt.synthetic_fixture_rows = [80]
    with pytest.raises(ValueError):
        verify_program_receipt(manifest, receipt)


def test_receipts_cannot_be_reused_between_single_and_comparison_targets() -> None:
    manifest = ComparisonManifest.create([candidate(1, 10)])
    receipt = _comparison_receipt(manifest)
    with pytest.raises(ValueError, match="per-candidate"):
        verify_program_receipt(manifest, receipt.candidates[0].receipt)
    with pytest.raises(ValueError, match="single program"):
        verify_program_receipt(manifest.candidates[0].manifest, receipt)


def test_comparison_child_requires_synthetic_evidence_even_for_bundled_strategy() -> (
    None
):
    manifest = _manifest(bundled=True)
    receipt = _receipt(
        manifest_hash=manifest.manifest_hash,
        training_requirement="optional",
    )
    receipt.verify_manifest(manifest, comparison_candidate=True)
    with pytest.raises(ValueError, match="strategy execution kind"):
        receipt.verify_manifest(manifest)
    fields = receipt.model_dump()
    fields["synthetic_fixture_rows"] = []
    fields["checked_paths"].remove("signal_contract")
    fields["signal_check"] = "bundled_not_executed"
    unexecuted = PreflightReceipt.model_validate(fields)
    with pytest.raises(ValueError, match="both contract fixtures"):
        unexecuted.verify_manifest(manifest, comparison_candidate=True)
