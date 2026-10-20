"""ML tool contracts through in-process MCP dispatch with MCPServer."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
from mcp.server import MCPServer
from mcp_types import CallToolResult

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.contracts.ml import MlContractResponse, SignalEvaluationResponse
from tradingdev.app.ml_service import MLService
from tradingdev.mcp.tools import ml

if TYPE_CHECKING:
    from pathlib import Path

_PERFECT = np.linspace(-1.0, 1.0, 120).tolist()


def _dispatch(
    mcp: MCPServer, name: str, arguments: dict[str, object]
) -> dict[str, object]:
    async def call() -> dict[str, object]:
        result = await mcp.call_tool(name, arguments)
        assert isinstance(result, CallToolResult)
        assert result.structured_content is not None
        payload = result.structured_content
        assert isinstance(payload, dict)
        response = payload.get("result", payload)
        assert isinstance(response, dict)
        return response

    return asyncio.run(call())


def test_get_ml_contract_lists_the_enumerable_vocabulary(tmp_path: Path) -> None:
    mcp = MCPServer("ml-contract-test")
    ml.register(mcp, MLService(WorkspacePaths(tmp_path / "workspace")))
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


def test_evaluate_signal_runs_the_signal_suite(tmp_path: Path) -> None:
    mcp = MCPServer("ml-eval-test")
    ml.register(mcp, MLService(WorkspacePaths(tmp_path / "workspace")))
    forward = [s * 0.01 for s in _PERFECT]
    response = SignalEvaluationResponse.model_validate(
        _dispatch(
            mcp,
            "evaluate_signal",
            {"signal": _PERFECT, "forward_returns": forward},
        )
    )
    assert response.n_observations == 120
    rank_ic_mean = response.metrics["rank_ic_mean"]
    assert rank_ic_mean is not None and rank_ic_mean > 0.9
    assert (
        response.strata_monotonicity is not None and response.strata_monotonicity > 0.9
    )
    assert response.long_short_spread is not None and response.long_short_spread > 0
    assert response.simulated.long_short_mean is not None
    assert response.simulated.long_short_mean > 0
    assert len(response.strata) == 5
    assert len(response.simulated.quantile_means) == 5


def test_evaluate_signal_rejects_length_mismatch(tmp_path: Path) -> None:
    mcp = MCPServer("ml-eval-err")
    ml.register(mcp, MLService(WorkspacePaths(tmp_path / "workspace")))
    payload = _dispatch(
        mcp,
        "evaluate_signal",
        {"signal": [1.0, 2.0], "forward_returns": [1.0]},
    )
    assert payload["success"] is False
    assert payload["code"] == "invalid_signal"


def test_evaluate_signal_rejects_too_few_observations(tmp_path: Path) -> None:
    mcp = MCPServer("ml-eval-short")
    ml.register(mcp, MLService(WorkspacePaths(tmp_path / "workspace")))
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


def test_ml_annotations_disclose_replacement_and_cache_eviction(tmp_path: Path) -> None:
    mcp = MCPServer("ml-side-effects")
    ml.register(mcp, MLService(WorkspacePaths(tmp_path / "workspace")))
    entries = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}
    writers = {"build_dataset", "build_panel", "train_model", "evaluate_signal"}
    readers = {"get_ml_contract", "predict", "leaderboard"}
    assert set(entries) == writers | readers
    for name, tool in entries.items():
        annotations = tool.annotations
        assert annotations is not None
        assert annotations.read_only_hint is (name in readers)
        assert annotations.destructive_hint is (name in writers)
        assert annotations.idempotent_hint is True
        assert annotations.open_world_hint is False
