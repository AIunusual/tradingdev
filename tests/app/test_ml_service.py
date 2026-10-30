"""Unit tests for the ML application service (no MCP layer)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.ml_service import MLService
from tradingdev.domain.ml.report import ML_SAMPLE_PROVENANCE_NOTE

if TYPE_CHECKING:
    from pathlib import Path

_PERFECT = np.linspace(-1.0, 1.0, 120).tolist()


@pytest.fixture
def service(tmp_path: Path) -> MLService:
    return MLService(WorkspacePaths(tmp_path / "workspace"))


def test_get_contract_enumerates_the_full_vocabulary(service: MLService) -> None:
    contract = service.get_contract()
    assert contract["success"] is True
    names = {fs["name"] for fs in contract["feature_sets"]}
    assert "comprehensive" in names
    kind_names = {mk["name"] for mk in contract["model_kinds"]}
    assert {
        "xgboost_direction",
        "lightgbm_direction",
        "catboost_direction",
    } <= kind_names
    assert {vs["name"] for vs in contract["validation_schemes"]} == {
        "purged_walk_forward",
        "purged_kfold",
    }
    assert len(contract["processor_types"]) == 6
    assert contract["ic_methods"] == ["pearson", "spearman"]
    assert contract["n_quantiles_range"] == [2, 10]
    assert contract["schema_version"] == 1


def test_get_contract_model_defaults_are_json_safe(service: MLService) -> None:
    contract = service.get_contract()
    assert contract["model_kinds"]
    for kind in contract["model_kinds"]:
        params = kind["default_parameters"]
        assert isinstance(params, dict)
        assert "n_estimators" in params
        assert "learning_rate" in params
        assert json.loads(json.dumps(params, allow_nan=False)) == params


def test_report_guidance_retains_unknown_sample_provenance(service: MLService) -> None:
    guidance = service.get_contract()["guidance"]["reports"]
    assert ML_SAMPLE_PROVENANCE_NOTE in guidance
    assert "無法判定" in guidance and "樣本內或樣本外" in guidance


def test_evaluate_signal_on_a_perfectly_monotonic_signal(service: MLService) -> None:
    signal = _PERFECT
    forward = [s * 0.01 for s in signal]
    result = service.evaluate_signal(signal, forward)
    assert result["success"] is True
    assert result["n_observations"] == 120
    assert result["metrics"]["rank_ic_mean"] > 0.9
    assert (
        result["strata_monotonicity"] is not None
        and result["strata_monotonicity"] > 0.9
    )
    assert result["long_short_spread"] is not None and result["long_short_spread"] > 0
    assert result["simulated"]["long_short_mean"] is not None
    assert result["simulated"]["long_short_mean"] > 0
    assert len(result["strata"]) == 5
    assert len(result["simulated"]["quantile_means"]) == 5


def test_evaluate_signal_on_an_inverse_signal_is_negative(service: MLService) -> None:
    signal = _PERFECT
    forward = [-s * 0.01 for s in signal]
    result = service.evaluate_signal(signal, forward)
    assert result["metrics"]["rank_ic_mean"] is not None
    assert result["metrics"]["rank_ic_mean"] < -0.9
    assert result["simulated"]["long_short_mean"] < 0


def test_evaluate_signal_rejects_length_mismatch(service: MLService) -> None:
    with pytest.raises(ValueError, match="equal length"):
        service.evaluate_signal([1.0, 2.0], [1.0])


def test_evaluate_signal_rejects_too_few_finite_bars(service: MLService) -> None:
    with pytest.raises(ValueError, match="not enough finite observations"):
        service.evaluate_signal([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], window=60)


def test_evaluate_signal_drops_non_finite_then_fails(service: MLService) -> None:
    with pytest.raises(ValueError):
        service.evaluate_signal([1.0, float("nan"), 3.0], [0.1, 0.2, 0.3])


def test_evaluation_id_is_deterministic_for_equal_inputs(service: MLService) -> None:
    signal = _PERFECT
    forward = [s * 0.01 for s in signal]
    first = service.evaluate_signal(signal, forward)["evaluation_id"]
    second = service.evaluate_signal(signal, forward)["evaluation_id"]
    assert first == second
    assert first.startswith("eval_")
