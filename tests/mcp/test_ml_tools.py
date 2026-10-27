"""ML tool contracts through in-process MCP dispatch with MCPServer."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import CallToolResult
from tests.mcp.ml_fixtures import model_pair, seed_bars

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.contracts.ml import MlContractResponse, PredictionEvaluationResponse
from tradingdev.app.ml_service import MLService
from tradingdev.mcp.strict_server import StrictMCPServer
from tradingdev.mcp.tools import ml

if TYPE_CHECKING:
    from pathlib import Path


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
    workspace = WorkspacePaths(tmp_path / "workspace")
    seed_bars(workspace)
    service = MLService(workspace)
    identity = model_pair(service)
    mcp = StrictMCPServer("ml-eval-test")
    ml.register(mcp, service)
    response = PredictionEvaluationResponse.model_validate(
        _dispatch(mcp, "evaluate_signal", {**identity})
    )
    complete = service.predict(**identity)["prediction"]
    expected = service.evaluate_signal(
        complete["signal"],
        complete["forward_returns"],
        model_id=identity["model_id"],
        dataset_id=identity["dataset_id"],
    )
    assert response.n_observations == complete["n_rows"] > 100
    assert response.evaluation_id == expected["evaluation_id"]
    assert response.metrics == expected["metrics"]
    assert response.n_bins == 5
    assert len(response.strata) == len(response.simulated.quantile_means) == 2


def test_evaluate_signal_rejects_an_unknown_model(tmp_path: Path) -> None:
    mcp = StrictMCPServer("ml-eval-err")
    ml.register(mcp, MLService(WorkspacePaths(tmp_path / "workspace")))
    payload = _dispatch(
        mcp,
        "evaluate_signal",
        {"model_id": "not-a-model", "dataset_id": "ds_" + "0" * 16},
    )
    assert payload["success"] is False
    assert payload["code"] == "invalid_signal"


@pytest.mark.parametrize("field", ["signal", "forward_returns", "horizon"])
def test_evaluate_signal_rejects_caller_supplied_data(
    tmp_path: Path, field: str
) -> None:
    mcp = StrictMCPServer("ml-eval-identity")
    ml.register(mcp, MLService(WorkspacePaths(tmp_path / "workspace")))
    with pytest.raises(ToolError, match="Unknown arguments"):
        _dispatch(
            mcp,
            "evaluate_signal",
            {"model_id": "0" * 16, "dataset_id": "ds_" + "0" * 16, field: []},
        )


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
