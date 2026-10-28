"""Research omissions survive real MCP serialization and supervised preparation."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

from tests.e2e.conversation import links_to_file
from tests.integration.execution_fixtures import (
    execution_evidence_draft,
    research_input,
    save_execution_evidence,
)
from tradingdev.domain.strategies.bundled.kd_strategy.config import KDStrategyConfig

if TYPE_CHECKING:
    from tests.integration.mcp_harness import MCPWorkspace

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


async def test_omitted_finances_and_inherited_choices_survive_worker_round_trip(
    mcp_workspace: MCPWorkspace,
) -> None:
    research = {
        "kind": "new",
        "strategy_outline": "BTC 均線交叉做多，向下交叉平倉",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-02-01",
    }
    async with mcp_workspace.connect() as client:
        example = (await client.call("get_strategy_contract", section="example"))[
            "content"
        ]
        config = yaml.safe_load(example["example_yaml_config"])
        for field in (
            "symbol",
            "timeframe",
            "start_date",
            "end_date",
            "init_cash",
            "fees",
            "slippage",
            "periods_per_year",
        ):
            config["backtest"].pop(field, None)
        for field in ("symbol", "timeframe"):
            config["data"]["requirements"]["market"].pop(field, None)
        saved = await client.call(
            "save_strategy",
            code=example["example_strategy_code"],
            yaml_config=yaml.safe_dump(config),
            research=research,
        )
        assert saved["success"], saved
        identity = {
            "strategy_id": saved["strategy_id"],
            "revision_id": saved["revision_id"],
        }
        for tool in ("validate_strategy", "dry_run_strategy"):
            checked = await client.call(tool, **identity)
            assert checked["success"], checked
        for selection, source in (
            (research, "project_default"),
            ({"kind": "inherit_selected_revision"}, "selected_revision"),
        ):
            prepared = await client.call(
                "prepare_backtest",
                **identity,
                research=selection,
                evidence_id=await save_execution_evidence(
                    client, {"fast_period": 10, "slow_period": 30}
                ),
            )
            assert prepared["success"], prepared
            assert links_to_file(
                prepared["confirmation_link"], Path(prepared["html_path"])
            )
            assert prepared["preflight"]["signal_check"] == "synthetic_contract_passed"
            with closing(
                sqlite3.connect(mcp_workspace.workspace / "tradingdev.sqlite")
            ) as connection:
                row = connection.execute(
                    "SELECT payload FROM execution_plans WHERE plan_id = ?",
                    (prepared["plan_id"],),
                ).fetchone()
            assert row is not None
            effective = json.loads(row[0])["manifest"]["config"]
            assert effective["backtest"]["symbol"] == "BTC/USDT"
            assert effective["backtest"]["timeframe"] == "1h"
            assert effective["backtest"]["init_cash"] == 10000
            assert effective["backtest"]["fees"] == 0.0006
            assert effective["backtest"]["slippage"] == 0.0005
            origins = effective["research"]["field_sources"]
            assert all(
                origins[f"/backtest/{field}"] == source
                for field in ("init_cash", "fees", "slippage")
            )
            if source == "selected_revision":
                assert (
                    effective["research"]["selected_revision"]["revision_id"]
                    == (saved["revision_id"])
                )
                assert origins["/backtest/start_date"] == "selected_revision"
                assert origins["/backtest/end_date"] == "selected_revision"
        assert (await client.call("list_jobs"))["items"] == []
        assert (await client.call("list_runs"))["items"] == []
        assert not list(mcp_workspace.data_root.rglob("*.parquet"))


async def test_walk_forward_replaces_old_windows_without_changing_the_research(
    mcp_workspace: MCPWorkspace,
) -> None:
    arguments = {
        "strategy_id": "kd_crossover",
        "research": research_input(start_date="2026-01-01", end_date="2026-02-01"),
    }
    async with mcp_workspace.connect() as client:
        arguments["evidence_id"] = await save_execution_evidence(
            client, {"config": KDStrategyConfig().model_dump(), "fit_config": None}
        )
        rejected = await client.call("prepare_walk_forward", **arguments)
        assert rejected["success"] is False
        assert "validation_config" in rejected["error"]
        prepared = await client.call(
            "prepare_walk_forward",
            **arguments,
            validation_config={"mode": "automatic", "n_splits": 2, "train_ratio": 0.6},
        )
        # The selected walk-forward config has additional fit parameters. Repair
        # the saved record using the public exact-binding response.
        assert prepared["success"] is False
        assert prepared["required_parameter_paths"]
        assert prepared["code"] == "research_evidence_binding_invalid", prepared
        draft = execution_evidence_draft(
            {},
            target_paths={
                target["target_id"]: [
                    item["parameter_path"] for item in target["parameters"]
                ]
                for target in prepared["required_parameter_bindings"]
            },
        )
        draft["presentation"]["parameter_descriptions"] = {
            path: {
                "label": f"分段訓練設定 {path}",
                "description": "此設定控制已選定策略的訊號或訓練程序。",
                "unit": "策略單位",
            }
            for path in prepared["required_parameter_paths"]
        }
        saved = await client.call("save_research_evidence", draft=draft)
        assert saved["success"], saved
        arguments["evidence_id"] = saved["evidence_id"]
        prepared = await client.call(
            "prepare_walk_forward",
            **arguments,
            validation_config={"mode": "automatic", "n_splits": 2, "train_ratio": 0.6},
        )
        assert prepared["success"], prepared
        with closing(
            sqlite3.connect(mcp_workspace.workspace / "tradingdev.sqlite")
        ) as connection:
            row = connection.execute(
                "SELECT payload FROM execution_plans WHERE plan_id = ?",
                (prepared["plan_id"],),
            ).fetchone()
        assert row is not None
        effective = json.loads(row[0])["manifest"]["config"]
        assert effective["backtest"]["start_date"].startswith("2026-01-01")
        assert effective["backtest"]["end_date"].startswith("2026-02-01")
        assert effective["validation"]["n_splits"] == 2
        assert "train_start" not in effective["validation"]
        assert "test_end" not in effective["validation"]
        assert (await client.call("list_jobs"))["items"] == []
        assert (await client.call("list_runs"))["items"] == []
        assert not list(mcp_workspace.data_root.rglob("*.parquet"))
