"""Real MCP optimization: confirmation, grid search, OOS, and persisted results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.integration.execution_fixtures import confirm_prepared, execution_options
from tradingdev.domain.execution import ExecutionManifest

if TYPE_CHECKING:
    from tests.integration.mcp_harness import MCPClient, MCPWorkspace

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

STRATEGY_ID = "optimization_direction"
STRATEGY_CODE = '''\
"""Hold a chosen direction after a configurable warmup."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tradingdev.domain.strategies.base import BaseStrategy

if TYPE_CHECKING:
    import pandas as pd


class DirectionStrategy(BaseStrategy):
    """Expose both searched and required fixed YAML parameters."""

    def __init__(self, direction: int, warmup: int = 2) -> None:
        self._direction = direction
        self._warmup = warmup

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["signal"] = 0
        result.loc[result.index[self._warmup :], "signal"] = self._direction
        return result

    def get_parameters(self) -> dict[str, Any]:
        return {"direction": self._direction, "warmup": self._warmup}
'''


def _opposite_trends() -> pd.DataFrame:
    # Train rises for three days; OOS falls for four days. Both periods have
    # equal open/close prices, so the fee-free hold return can be hand calculated.
    prices = np.concatenate((np.arange(100.0, 172.0), np.arange(200.0, 104.0, -1)))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01", periods=len(prices), freq="h", tz="UTC"
            ),
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": np.full(len(prices), 1000.0),
        }
    )


async def _save_runnable(
    client: MCPClient, *, walk_forward: bool = False
) -> dict[str, Any]:
    contract = await client.call("get_strategy_contract")
    config = yaml.safe_load(contract["example_yaml_config"])
    config["strategy"].update(
        class_name="DirectionStrategy", parameters={"direction": -1}
    )
    # The MCP request must override these without editing the saved strategy.
    config["backtest"].update(symbol="ETH/USDT", timeframe="4h", fees=0, slippage=0)
    if walk_forward:
        config["validation"] = {"n_splits": 2}
    saved = await client.call(
        "save_strategy",
        strategy_id=STRATEGY_ID,
        code=STRATEGY_CODE,
        yaml_config=yaml.safe_dump(config),
    )
    assert saved["success"], saved
    for tool in ("validate_strategy", "dry_run_strategy"):
        checked = await client.call(
            tool, strategy_id=STRATEGY_ID, revision_id=saved["revision_id"]
        )
        assert checked["success"], checked
    return dict(saved)


async def test_optimization_rejects_walk_forward_config_through_mcp(
    mcp_workspace: MCPWorkspace,
) -> None:
    async with mcp_workspace.connect() as client:
        saved = await _save_runnable(client, walk_forward=True)
        rejected = await client.call(
            "prepare_optimization",
            strategy_id=STRATEGY_ID,
            revision_id=saved["revision_id"],
            symbol="BTC/USDT",
            timeframe="1h",
            param_ranges={"direction": [-1, 1]},
            optimization_metric="total_return",
            train_start="2024-01-01",
            train_end="2024-01-03",
            test_start="2024-01-04",
            test_end="2024-01-07",
            **execution_options({"direction": -1, "warmup": 2}),
        )
        assert rejected["success"] is False
        assert rejected["code"] == "invalid_optimization_request"
        assert "must not contain validation settings" in rejected["error"]
        assert await client.call("list_jobs") == []


async def test_optimization_confirmation_search_and_persisted_oos(
    mcp_workspace: MCPWorkspace,
) -> None:
    mcp_workspace.seed_market(_opposite_trends())
    async with mcp_workspace.connect() as client:
        saved = await _save_runnable(client)
        assert not (
            await client.call(
                "submit_execution_decision", plan_id="missing", decision="approve"
            )
        )["success"]
        overlapping = await client.call(
            "prepare_optimization",
            strategy_id=STRATEGY_ID,
            revision_id=saved["revision_id"],
            symbol="BTC/USDT",
            timeframe="1h",
            param_ranges={"direction": [-1, 0, 1]},
            optimization_metric="total_return",
            train_start="2024-01-01",
            train_end="2024-01-03",
            test_start="2024-01-03",
            test_end="2024-01-07",
            **execution_options({"direction": -1, "warmup": 2}),
        )
        assert not overlapping["success"]
        assert "without overlap" in overlapping["error"]
        assert await client.call("list_jobs") == []
        prepared = await client.call(
            "prepare_optimization",
            strategy_id=STRATEGY_ID,
            revision_id=saved["revision_id"],
            symbol="BTC/USDT",
            timeframe="1h",
            param_ranges={"direction": [-1, 0, 1]},
            optimization_metric="total_return",
            train_start="2024-01-01",
            train_end="2024-01-03",
            test_start="2024-01-04",
            test_end="2024-01-07",
            **execution_options({"direction": -1, "warmup": 2}),
        )
        assert prepared["success"] and prepared["status"] == "ready", prepared
        manifest_hash = prepared["manifest_hash"]
        assert len(manifest_hash) == 64
        assert prepared["preflight"]
        retained = await client.call("get_execution_plan", plan_id=prepared["plan_id"])
        assert retained["manifest_hash"] == manifest_hash
        assert await client.call("list_jobs") == []
        assert await client.call("list_runs") == []
        assert not list(mcp_workspace.workspace.rglob("result.json"))
        replacement = await client.call(
            "save_strategy",
            strategy_id=STRATEGY_ID,
            code=STRATEGY_CODE.replace("= self._direction", "= -self._direction"),
            yaml_config=Path(saved["yaml_path"]).read_text(),
        )
        assert replacement["revision_id"] != saved["revision_id"]
        assert replacement["status"] == "draft"
        # A newer draft cannot change the previously prepared revision. Only the
        # explicit test decision submits the frozen plan for formal execution.
        confirmed = await confirm_prepared(client, prepared)
        job_id = confirmed["job_id"]
        completed = await client.wait_for_job(job_id, timeout=180)
        assert completed["status"] == "done", json.dumps(completed, indent=2)
        assert completed["revision_id"] == saved["revision_id"]
        assert completed["manifest_hash"] == manifest_hash
        assert completed["best_params"] == {"direction": 1}
        assert completed["optimization_metric"] == "total_return"
        assert completed["total_combinations"] == 3
        assert (await client.call("list_jobs"))[0]["completed"] == 3
        # warmup=2 is a captured constructor default, not a grid parameter.
        # The engine enters one
        # bar later (index 3), and training must include the entire last day.
        train_return = 171.0 / 103.0 - 1
        oos_return = 105.0 / 197.0 - 1
        for metrics, expected in (
            (completed["train_metrics"], train_return),
            (completed["test_metrics"], oos_return),
        ):
            assert metrics["total_return"] == pytest.approx(expected)
            assert metrics["total_pnl"] == pytest.approx(10000 * expected)
            # The strategy holds its final position; trade statistics use closed trades.
            assert metrics["total_trades"] == 0
        assert completed["train_metrics"]["profit_factor"] is None
        repeated = await client.call(
            "submit_execution_decision", plan_id=prepared["plan_id"], decision="approve"
        )
        assert repeated["success"] and repeated["job_id"] == job_id
        assert len(await client.call("list_jobs")) == 1
        run = (await client.call("get_run", run_id=completed["run_id"]))["run"]
        assert run["strategy_id"] == STRATEGY_ID
        assert run["revision_id"] == saved["revision_id"]
        assert run["manifest_hash"] == manifest_hash
        assert run["dataset_id"].startswith("BTC/USDT:1h:2024-01-01:2024-01-07:")
        assert run["details_available"] is True
        assert run["default_scope"] == "test"
        assert run["selected_train_scope"] == "trial/2/train"
        assert set(run["available_scopes"]) == {
            "trial/0/train",
            "trial/1/train",
            "trial/2/train",
            "test",
        }
        assert "daily_pnl_median" in run["available_metric_ids"]
        assert "daily_pnl_median" not in run["metrics"]
        assert run["metrics"]["total_return"] == pytest.approx(oos_return)
        train_detail = await client.call(
            "get_run_metrics", run_id=job_id, scope=run["selected_train_scope"]
        )
        test_detail = await client.call("get_run_metrics", run_id=job_id)
        assert train_detail["metrics"]["total_return"] == pytest.approx(train_return)
        assert test_detail["metrics"]["total_return"] == pytest.approx(oos_return)
        assert train_detail["metrics"]["open_trades"] == 1
        assert test_detail["metrics"]["open_trades"] == 1
        assert (
            train_detail["parameters"] == test_detail["parameters"] == {"direction": 1}
        )
        assert train_detail["metadata"]["strategy_parameters"] == {
            "direction": 1,
            "warmup": 2,
        }
        assert (
            train_detail["metadata"]["execution_context"]["end_date"]
            == "2024-01-03T23:59:59.999999"
        )
        assert train_detail["metadata"]["providers"]["quantstats"]
        assert (
            train_detail["definitions"]["max_drawdown"]["optimization_direction"]
            == "minimize"
        )
        # A losing and a flat trial remain discoverable without re-running them.
        losing = await client.call(
            "get_run_metrics",
            run_id=job_id,
            scope="trial/0/train",
            metric_ids=["total_return", "daily_pnl_median"],
        )
        assert losing["parameters"] == {"direction": -1}
        assert losing["metrics"]["total_return"] < 0
        assert losing["metrics"]["daily_pnl_median"] < 0
        flat = await client.call(
            "get_run_metrics",
            run_id=job_id,
            scope="trial/1/train",
            metric_ids=["total_return", "win_rate"],
        )
        assert flat["metrics"] == {"total_return": 0.0, "win_rate": None}
        assert flat["metadata"]["unavailable"]["win_rate"] == "no_trades"
        artifacts = {
            item["artifact_type"]: item
            for item in await client.call("list_artifacts", run_id=job_id)
        }
        specification = await client.call(
            "get_artifact",
            artifact_id=artifacts["execution_manifest"]["artifact_id"],
            include_content=True,
        )
        manifest = ExecutionManifest.model_validate_json(specification["content"])
        manifest.verify(manifest_hash)
        assert manifest.kind == "optimization"
        assert manifest.strategy_execution.constructor_kwargs == {
            "direction": -1,
            "warmup": 2,
        }
        search = manifest.optimization
        assert search is not None
        assert search.param_ranges == {"direction": [-1, 0, 1]}
        assert search.optimization_metric == "total_return"
        assert search.train_start.isoformat() == "2024-01-01"
        assert search.train_end.isoformat() == "2024-01-03"
        assert search.test_start.isoformat() == "2024-01-04"
        assert search.test_end.isoformat() == "2024-01-07"
        assert search.direction == "maximize"
        assert search.trial_timeout_seconds == 300
        assert "confirmation_timeout_seconds" not in type(search).model_fields
        assert "confirmation_poll_interval" not in type(search).model_fields
        stored = await client.call(
            "get_artifact",
            artifact_id=artifacts["result_json"]["artifact_id"],
            include_content=True,
        )
        assert "Infinity" not in stored["content"]
        assert "NaN" not in stored["content"]
        result = json.loads(stored["content"])
        assert result["best_params"] == completed["best_params"]
        assert result["train_metrics"] == train_detail["metrics"]
        assert result["test_metrics"] == test_detail["metrics"]
        assert result["best_train_metric_value"] == pytest.approx(train_return)
        assert result["best_oos_metric_value"] == pytest.approx(oos_return)
        original_observations = await client.call(
            "get_artifact",
            artifact_id=artifacts["observations_json"]["artifact_id"],
            include_content=True,
        )
        observations = json.loads(original_observations["content"])
        assert observations["manifest_hash"] == manifest_hash
        assert set(observations["scopes"]) == set(run["available_scopes"])
        for scope_id in ["trial/0/train", "trial/1/train", "trial/2/train"]:
            raw = observations["scopes"][scope_id]
            assert (
                len(raw["equity_curve"])
                == len(raw["returns"])
                == len(raw["timestamps"])
                == 72
            )
        assert observations["scopes"]["trial/1/train"]["trades"] == []
        assert observations["scopes"]["trial/2/train"]["trades"][0][
            "net_pnl"
        ] == pytest.approx(10000 * train_return)
        assert len(observations["scopes"]["test"]["equity_curve"]) == 96
        effective = yaml.safe_load(
            Path(artifacts["config_snapshot"]["path"]).read_text()
        )
        assert effective["strategy"]["revision_id"] == saved["revision_id"]
        assert Path(artifacts["strategy_source"]["path"]).read_text() == STRATEGY_CODE
        assert effective["backtest"]["symbol"] == "BTC/USDT"
        assert effective["backtest"]["timeframe"] == "1h"
        assert effective["backtest"]["end_date"] == "2024-01-07T23:59:59.999999"
        assert effective["data"]["requirements"]["market"]["symbol"] == "BTC/USDT"
        assert effective["strategy"]["parameters"] == {"direction": -1}
        assert effective == manifest.config_copy()
        original = yaml.safe_load(Path(saved["yaml_path"]).read_text())
        assert original["backtest"]["symbol"] == "ETH/USDT"
        assert original["backtest"]["timeframe"] == "4h"

    async with mcp_workspace.connect() as client:
        reloaded = await client.call("get_job_status", job_id=job_id)
        assert reloaded["status"] == "done"
        assert reloaded["manifest_hash"] == manifest_hash
        assert reloaded["best_params"] == {"direction": 1}
        assert reloaded["train_metrics"] == completed["train_metrics"]
        assert reloaded["test_metrics"] == completed["test_metrics"]
        reloaded_trial = await client.call(
            "get_run_metrics",
            run_id=job_id,
            scope="trial/0/train",
            metric_ids=["total_return", "daily_pnl_median"],
        )
        assert reloaded_trial == losing
        assert any(item["run_id"] == job_id for item in await client.call("list_runs"))
