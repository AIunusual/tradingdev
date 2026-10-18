"""ML tool contracts through in-process MCP dispatch with FastMCP."""

from __future__ import annotations

import asyncio

import numpy as np
from mcp.server.fastmcp import FastMCP

from tradingdev.app.contracts.ml import MlContractResponse, SignalEvaluationResponse
from tradingdev.app.ml_service import MLService
from tradingdev.mcp.tools import ml

_PERFECT = np.linspace(-1.0, 1.0, 120).tolist()


def _dispatch(
    mcp: FastMCP, name: str, arguments: dict[str, object]
) -> dict[str, object]:
    async def call() -> dict[str, object]:
        result = await mcp.call_tool(name, arguments)
        payload = result[1]
        assert isinstance(payload, dict)
        response = payload.get("result", payload)
        assert isinstance(response, dict)
        return response

    return asyncio.run(call())


def test_get_ml_contract_lists_the_enumerable_vocabulary() -> None:
    mcp = FastMCP("ml-contract-test")
    ml.register(mcp, MLService())
    contract = MlContractResponse.model_validate(_dispatch(mcp, "get_ml_contract", {}))
    names = {fs.name for fs in contract.feature_sets}
    assert "comprehensive" in names
    kind_names = {mk.name for mk in contract.model_kinds}
    assert {
        "xgboost_direction",
        "lightgbm_direction",
        "catboost_direction",
    } <= kind_names
    assert {vs.name for vs in contract.validation_schemes} == {
        "purged_walk_forward",
        "purged_kfold",
    }
    assert len(contract.processor_types) == 6
    assert contract.ic_methods == ["pearson", "spearman"]
    assert contract.n_quantiles_range == [2, 10]
    assert contract.schema_version == 1


def test_evaluate_signal_runs_the_signal_suite() -> None:
    mcp = FastMCP("ml-eval-test")
    ml.register(mcp, MLService())
    forward = [s * 0.01 for s in _PERFECT]
    response = SignalEvaluationResponse.model_validate(
        _dispatch(
            mcp,
            "evaluate_signal",
            {"signal": _PERFECT, "forward_returns": forward},
        )
    )
    assert response.n_observations == 120
    assert response.metrics["rank_ic_mean"] > 0.9
    assert (
        response.strata_monotonicity is not None and response.strata_monotonicity > 0.9
    )
    assert response.long_short_spread is not None and response.long_short_spread > 0
    assert response.simulated.long_short_mean is not None
    assert response.simulated.long_short_mean > 0
    assert len(response.strata) == 5
    assert len(response.simulated.quantile_means) == 5


def test_evaluate_signal_rejects_length_mismatch() -> None:
    mcp = FastMCP("ml-eval-err")
    ml.register(mcp, MLService())
    payload = _dispatch(
        mcp,
        "evaluate_signal",
        {"signal": [1.0, 2.0], "forward_returns": [1.0]},
    )
    assert payload["success"] is False
    assert payload["code"] == "invalid_signal"


def test_evaluate_signal_rejects_too_few_observations() -> None:
    mcp = FastMCP("ml-eval-short")
    ml.register(mcp, MLService())
    payload = _dispatch(
        mcp,
        "evaluate_signal",
        {
            "signal": [1.0, 2.0, 3.0],
            "forward_returns": [0.1, 0.2, 0.3],
            "window": 60,
        },
    )
    assert payload["success"] is False
    assert payload["code"] == "invalid_signal"
