"""Approved walk-forward evidence cannot relabel training data as test results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pytest
from tests.domain.test_confirmation_presentation import _manifest

from tradingdev.domain.performance.artifacts import (
    PerformanceArtifacts,
    PerformanceScope,
    ScopeObservations,
    build_artifacts,
)
from tradingdev.domain.research_scope import verify_research_scopes

if TYPE_CHECKING:
    from tradingdev.domain.execution import ExecutionManifest


def _artifacts(
    validation: dict[str, Any],
) -> tuple[ExecutionManifest, PerformanceArtifacts]:
    manifest = _manifest("walk_forward", validation=validation)
    config = manifest.config_copy()
    explicit = validation["mode"] == "explicit"
    count = 1 if explicit else validation["n_splits"]
    scopes = {}
    observations = {}
    parameters = manifest.strategy_execution.constructor_kwargs
    for index in range(count):
        splits: tuple[tuple[Literal["train", "test"], int], ...] = (
            ("train", 1 + index * 2),
            ("test", 7 + index),
        )
        for split, month in splits:
            name = f"fold/{index}/{split}"
            scopes[name] = PerformanceScope(
                mode="signal",
                split=split,
                fold_index=index,
                values={"total_return": 0.0},
                parameters=parameters,
                metadata={
                    "execution_context": {
                        **config["backtest"],
                        "random_seed": config["random_seed"],
                    }
                },
            )
            observations[name] = ScopeObservations(
                init_cash=10000,
                equity_curve=[10000],
                returns=[0],
                timestamps=[f"2024-{month:02}-01T00:00:00+00:00"],
                trades=[],
            )
    scopes["test_summary"] = PerformanceScope(
        kind="fold_summary",
        mode="signal",
        split="test",
        values={},
        parameters=parameters,
        metadata={"n_folds": count},
    )
    return manifest, build_artifacts(
        "job", manifest.manifest_hash, "test_summary", scopes, observations
    )


_EXPLICIT = {
    "mode": "explicit",
    "train_start": "2024-01-01",
    "train_end": "2024-06-30",
    "test_start": "2024-07-01",
    "test_end": "2024-12-31",
}


@pytest.mark.parametrize(
    "validation",
    [
        _EXPLICIT,
        {"mode": "automatic", "n_splits": 3},
        {"mode": "purged", "n_splits": 3, "label_horizon": 1},
    ],
)
def test_valid_fold_membership_and_active_windows(validation: dict[str, Any]) -> None:
    manifest, artifacts = _artifacts(validation)
    verify_research_scopes(manifest, artifacts.performance, artifacts.observations)


@pytest.mark.parametrize(
    "change",
    [
        "training_as_test",
        "test_as_train",
        "missing_pair",
        "extra_fold",
        "missing_summary",
        "wrong_default",
        "wrong_count",
        "summary_parameters",
        "test_overlap",
        "train_overlap",
    ],
)
def test_invalid_fold_identity_or_window_is_rejected(change: str) -> None:
    validation = (
        {"mode": "automatic", "n_splits": 3} if "overlap" in change else _EXPLICIT
    )
    manifest, artifacts = _artifacts(validation)
    performance, observations = artifacts.performance, artifacts.observations
    if change in {"training_as_test", "test_as_train"}:
        split = "test" if change == "training_as_test" else "train"
        value = "2024-06-01" if split == "test" else "2024-08-01"
        object.__setattr__(
            observations.scopes[f"fold/0/{split}"],
            "timestamps",
            [value + "T00:00:00+00:00"],
        )
    elif change == "missing_pair":
        del performance.scopes["fold/0/train"]
        del observations.scopes["fold/0/train"]
    elif change == "extra_fold":
        performance.scopes["fold/999/test"] = performance.scopes.pop("fold/0/test")
        observations.scopes["fold/999/test"] = observations.scopes.pop("fold/0/test")
    elif change == "missing_summary":
        del performance.scopes["test_summary"]
    elif change == "wrong_default":
        object.__setattr__(performance, "default_scope", "fold/0/test")
    elif change == "wrong_count":
        performance.scopes["test_summary"].metadata["n_folds"] = 99
    elif change == "summary_parameters":
        performance.scopes["test_summary"].parameters["period"] = 999
    else:
        name = "fold/1/test" if change == "test_overlap" else "fold/0/train"
        object.__setattr__(
            observations.scopes[name], "timestamps", ["2024-07-01T00:00:00+00:00"]
        )
    with pytest.raises(ValueError, match="Research"):
        verify_research_scopes(manifest, performance, observations)
