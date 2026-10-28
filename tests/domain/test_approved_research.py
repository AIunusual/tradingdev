"""Saved disclosure identity and exact search provenance remain verifiable."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from tests.domain.test_comparison_confirmation import comparison_fixture
from tests.domain.test_confirmation_presentation import _manifest
from tests.domain.test_research_evidence import evidence_draft

from tradingdev.domain.approved_research import ApprovedResearch
from tradingdev.domain.backtest.schemas import BacktestConfig
from tradingdev.domain.optimization.grid_search import parameter_grid
from tradingdev.domain.performance.artifacts import (
    PerformanceArtifacts,
    PerformanceScope,
    ScopeObservations,
    build_artifacts,
)
from tradingdev.domain.research_evidence import (
    ResearchEvidenceDraft,
    ResearchEvidenceRecord,
)
from tradingdev.domain.research_scope import (
    effective_scope_parameters,
    scope_parameter_sources,
    verify_research_scopes,
)

if TYPE_CHECKING:
    from tradingdev.domain.execution import ExecutionManifest


@pytest.mark.parametrize("kind", ["backtest", "optimization", "comparison"])
def test_approval_retains_one_policy_per_unique_executed_manifest(kind: Any) -> None:
    target = comparison_fixture()[0] if kind == "comparison" else _manifest(kind)
    evidence = ApprovedResearch.create(
        ResearchEvidenceRecord.create(evidence_draft(target)), target
    )
    restored = ApprovedResearch.model_validate_json(evidence.model_dump_json())
    restored.verify(target)
    assert restored == evidence
    assert len(evidence.execution_assumptions) == (3 if kind == "comparison" else 1)


@pytest.mark.parametrize("version", [True, 1.0, "1", 2])
def test_approval_version_is_strict(version: Any) -> None:
    target = _manifest()
    evidence = ApprovedResearch.create(
        ResearchEvidenceRecord.create(evidence_draft(target)), target
    )
    payload = evidence.model_dump(mode="json")
    payload["schema_version"] = version
    with pytest.raises(ValueError):
        ApprovedResearch.model_validate(payload)


def test_approval_detects_wrong_binding_and_unsafe_policy_mutation() -> None:
    target = _manifest()
    evidence = ApprovedResearch.create(
        ResearchEvidenceRecord.create(evidence_draft(target)), target
    )
    with pytest.raises(ValueError, match="another execution"):
        evidence.verify(_manifest(parameters={"period": 99}))
    object.__setattr__(evidence.execution_assumptions[0], "policy_version", "forged")
    with pytest.raises(ValueError, match="hash"):
        evidence.verify(target)


def optimization_fixture(
    *, bundled: bool = False
) -> tuple[ExecutionManifest, ApprovedResearch, PerformanceArtifacts]:
    base = {"period": 20, "nested": {"low": 5, "high": 80}, "unchanged": 0}
    target = _manifest(
        "optimization",
        bundled=bundled,
        parameters={"config": base, "fit_config": {"seed": 7}} if bundled else base,
        param_ranges={"period": [10, 10], "nested": [{"low": 2}, {}]},
    )
    draft = evidence_draft(target).model_dump(mode="json")
    for declaration in draft["parameter_provenance"]:
        if declaration["target_ids"] == ["search_2_2"]:
            declaration.update(provenance="user", reason="第二個相同值的來源仍保留。")
    evidence = ApprovedResearch.create(
        ResearchEvidenceRecord.create(ResearchEvidenceDraft.model_validate(draft)),
        target,
    )
    config = target.config_copy()
    search = target.optimization
    assert search is not None
    scopes = {}
    observations = {}
    combos = parameter_grid(search.param_ranges)
    for index in range(len(combos) + 1):
        test = index == len(combos)
        parameters = combos[-1] if test else combos[index]
        scope_id = "test" if test else f"trial/{index}/train"
        start, end = (
            (search.test_start, search.test_end)
            if test
            else (search.train_start, search.train_end)
        )
        backtest = BacktestConfig.model_validate(
            {
                **config["backtest"],
                "start_date": start.isoformat(),
                "end_date": f"{end.isoformat()}T23:59:59.999999",
            }
        ).model_dump(mode="json")
        scopes[scope_id] = PerformanceScope(
            mode="signal",
            values={"sharpe_ratio": float(index)},
            parameters=parameters,
            split="test" if test else "train",
            trial_index=None if test else index,
            metadata={
                "execution_context": {
                    **backtest,
                    "random_seed": config.get("random_seed"),
                },
                "strategy_parameters": {"fitted_threshold": 999},
            },
        )
        observations[scope_id] = ScopeObservations(
            init_cash=10000,
            equity_curve=[10000],
            returns=[0],
            timestamps=[start.isoformat() + "T00:00:00+00:00"],
            trades=[],
        )
    artifacts = build_artifacts(
        "job",
        target.manifest_hash,
        "test",
        scopes,
        observations,
        selected_train_scope="trial/3/train",
    )
    return target, evidence, artifacts


@pytest.mark.parametrize("bundled", [False, True])
def test_search_scope_merges_defaults_and_tracks_duplicate_options_by_position(
    bundled: bool,
) -> None:
    target, evidence, artifacts = optimization_fixture(bundled=bundled)
    verify_research_scopes(target, artifacts.performance, artifacts.observations)
    sources = scope_parameter_sources(evidence, target, artifacts.performance, "test")
    prefix = "/config" if bundled else ""
    indexed = {row.parameter_path: row for row in sources}
    assert indexed[prefix + "/period"].target_id == "search_2_2"
    assert indexed[prefix + "/period"].provenance == "user"
    assert indexed[prefix + "/nested/low"].target_id == "search_1_2"
    assert indexed[prefix + "/nested/low"].value == 5
    assert indexed[prefix + "/nested/high"].value == 80
    assert indexed[prefix + "/unchanged"].target_id == "base"
    if bundled:
        assert indexed["/fit_config/seed"].target_id == "base"
    assert (
        scope_parameter_sources(
            evidence, target, artifacts.performance, "trial/3/train"
        )
        == sources
    )
    effective = effective_scope_parameters(
        target, artifacts.performance.scopes["trial/0/train"]
    )
    assert effective == {"period": 10, "nested": {"low": 2, "high": 80}, "unchanged": 0}
    assert not any("fitted_threshold" in row.parameter_path for row in sources)


@pytest.mark.parametrize(
    "change",
    [
        "missing_trial",
        "extra_trial",
        "parameters",
        "trial_index",
        "split",
        "dates",
        "test_parameters",
        "winner",
        "context",
        "tie",
    ],
)
def test_forged_search_scopes_cannot_acquire_approved_parameter_sources(
    change: str,
) -> None:
    target, _evidence, artifacts = optimization_fixture()
    performance = artifacts.performance.model_copy(deep=True)
    observations = artifacts.observations.model_copy(deep=True)
    trial = performance.scopes["trial/0/train"]
    if change == "missing_trial":
        del performance.scopes["trial/0/train"]
    elif change == "extra_trial":
        performance.scopes["trial/4/train"] = trial
    elif change == "parameters":
        trial.parameters["period"] = 99
    elif change == "trial_index":
        object.__setattr__(trial, "trial_index", 1)
    elif change == "split":
        object.__setattr__(trial, "split", "test")
    elif change == "dates":
        object.__setattr__(
            observations.scopes["trial/0/train"],
            "timestamps",
            ["2024-07-01T00:00:00+00:00"],
        )
    elif change == "test_parameters":
        performance.scopes["test"].parameters["period"] = 99
    elif change == "winner":
        object.__setattr__(performance, "selected_train_scope", "trial/2/train")
    elif change == "context":
        trial.metadata["execution_context"] = {}
    else:
        trial.values["sharpe_ratio"] = 3.0
    with pytest.raises(ValueError, match="Research"):
        verify_research_scopes(target, performance, observations)
