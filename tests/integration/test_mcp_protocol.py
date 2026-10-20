"""Exercise the client-visible strategy workflow over real stdio MCP."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from tests.integration.execution_fixtures import confirm_prepared, execution_options

if TYPE_CHECKING:
    import pandas as pd

    from tests.integration.mcp_harness import MCPClient, MCPWorkspace

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

STRATEGY_ID = "prompt_sma_strategy"
PROMPT = "建立 BTC/USDT 一小時均線交叉策略，快線 3、慢線 8，完成驗證與回測。"
RUN_ARGUMENTS = {
    "strategy_id": STRATEGY_ID,
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "start_date": "2024-01-01",
    "end_date": "2024-01-08",
}

PREPARE_ARGUMENTS = RUN_ARGUMENTS | execution_options(
    {"fast_period": 3, "slow_period": 8}
)


async def save_example(client: MCPClient) -> tuple[str, str, dict[str, Any]]:
    """Replay deterministic LLM output using the advertised contract."""
    contract = await client.call("get_strategy_contract")
    code = contract["example_strategy_code"]
    # Required YAML parameters catch workers silently using constructor defaults.
    code = code.replace(
        "        backtest_engine: BaseBacktestEngine | None = None,\n", ""
    )
    code = code.replace("fast_period: int = 10", "fast_period: int")
    code = code.replace(
        "slow_period: int = 30,",
        "slow_period: int,\n        backtest_engine: BaseBacktestEngine | None = None,",
    )
    config = yaml.safe_load(contract["example_yaml_config"])
    config["strategy"]["parameters"] = {"fast_period": 3, "slow_period": 8}
    config["backtest"]["symbol"] = "ETH/USDT"
    config["random_seed"] = 42
    config_text = yaml.safe_dump(config)
    saved = await client.call(
        "save_strategy",
        strategy_id=STRATEGY_ID,
        code=code,
        yaml_config=config_text,
        request_summary=PROMPT,
    )
    assert saved["success"], saved
    assert saved["status"] == "draft"
    return code, config_text, dict(saved)


async def make_runnable(client: MCPClient, revision_id: str | None = None) -> None:
    validated = await client.call(
        "validate_strategy", strategy_id=STRATEGY_ID, revision_id=revision_id
    )
    assert validated["success"], json.dumps(validated, indent=2)
    assert validated["status"] == "validated"
    dry_run = await client.call(
        "dry_run_strategy", strategy_id=STRATEGY_ID, revision_id=revision_id
    )
    assert dry_run["success"], json.dumps(dry_run, indent=2)
    assert dry_run["status"] == "runnable"
    assert dry_run["signal_analysis"]["rows"] == 240
    assert dry_run["signal_analysis"]["nan_count"] == 0


async def test_generated_strategy_full_mcp_workflow(
    mcp_workspace: MCPWorkspace,
    sample_ohlcv_with_kd: pd.DataFrame,
) -> None:
    cache_path = mcp_workspace.seed_market(sample_ohlcv_with_kd)
    async with mcp_workspace.connect() as client:
        tools = {tool.name: tool for tool in (await client.session.list_tools()).tools}
        assert {
            "get_strategy_contract",
            "save_strategy",
            "validate_strategy",
            "dry_run_strategy",
            "prepare_backtest",
            "get_job_status",
            "get_artifact",
        } <= tools.keys()
        assert {"strategy_id", "code", "yaml_config"} <= set(
            tools["save_strategy"].input_schema["required"]
        )
        assert all(
            item["kind"] == "bundled" for item in await client.call("list_strategies")
        )
        code, config_text, saved = await save_example(client)
        for key in ("py_path", "yaml_path"):
            assert Path(saved[key]).is_relative_to(mcp_workspace.workspace)
        assert Path(saved["py_path"]).read_text(encoding="utf-8") == code
        draft = await client.call("get_strategy", strategy_id=STRATEGY_ID)
        assert draft["metadata"]["request_summary"] == PROMPT
        assert draft["source_code"] == code
        assert not (await client.call("dry_run_strategy", strategy_id=STRATEGY_ID))[
            "success"
        ]
        assert not (await client.call("promote_strategy", strategy_id=STRATEGY_ID))[
            "success"
        ]
        assert not (await client.call("prepare_backtest", **PREPARE_ARGUMENTS))[
            "success"
        ]
        assert await client.call("list_jobs") == []
        await make_runnable(client, saved["revision_id"])
        promoted = await client.call(
            "promote_strategy",
            strategy_id=STRATEGY_ID,
            revision_id=saved["revision_id"],
        )
        assert promoted["success"] and promoted["status"] == "promoted"
        assert "binance_vision" in await client.call("list_data_sources")
        data = await client.call(
            "ensure_data",
            **{k: v for k, v in RUN_ARGUMENTS.items() if k != "strategy_id"},
        )
        assert data["success"] and data["rows"] > 0
        assert data["processed_path"] == str(cache_path)
        prepared = await client.call(
            "prepare_backtest", **PREPARE_ARGUMENTS, revision_id=saved["revision_id"]
        )
        started = await confirm_prepared(client, prepared)
        assert len(started["manifest_hash"]) == 64
        assert started["job_id"], started
        completed = await client.wait_for_job(started["job_id"])
        assert completed["status"] == "done", completed
        assert completed["revision_id"] == saved["revision_id"]
        assert completed["manifest_hash"] == started["manifest_hash"]
        assert completed["metrics"]["total_trades"] > 0
        run_id = completed["run_id"]
        run = (await client.call("get_run", run_id=run_id))["run"]
        assert run["strategy_id"] == STRATEGY_ID
        assert run["revision_id"] == saved["revision_id"]
        assert run["manifest_hash"] == started["manifest_hash"]
        assert run["metrics"] == completed["metrics"]
        assert run["details_available"] is True
        assert run["default_scope"] == "full"
        assert "daily_pnl_mean" not in run["metrics"]
        assert "daily_pnl_mean" in run["available_metric_ids"]
        detail = await client.call(
            "get_run_metrics",
            run_id=run_id,
            metric_ids=["daily_pnl_mean", "total_volume", "n_days"],
        )
        assert detail["success"] and detail["scope"] == "full"
        assert set(detail["metrics"]) == {"daily_pnl_mean", "total_volume", "n_days"}
        assert detail["metadata"]["providers"]["quantstats"]
        artifacts = await client.call("list_artifacts", run_id=run_id)
        by_type = {item["artifact_type"]: item for item in artifacts}
        assert {
            "result_json",
            "config_snapshot",
            "strategy_source",
            "dataset_fingerprint",
            "pipeline_result",
            "execution_manifest",
            "performance_json",
            "observations_json",
        } <= by_type.keys()
        performance = json.loads(Path(by_type["performance_json"]["path"]).read_text())
        values = performance["scopes"]["full"]["values"]
        assert detail["metrics"] == {key: values[key] for key in detail["metrics"]}
        assert json.loads(Path(by_type["result_json"]["path"]).read_text()) == values
        raw = await client.call(
            "get_artifact",
            artifact_id=by_type["observations_json"]["artifact_id"],
            include_content=True,
        )
        observations = json.loads(raw["content"])["scopes"]["full"]
        assert len(observations["equity_curve"]) == len(observations["returns"])
        assert len(observations["returns"]) == len(observations["timestamps"])
        for artifact in artifacts:
            path = Path(artifact["path"])
            assert path.is_relative_to(mcp_workspace.workspace / "runs")
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
        discovered = await client.call(
            "find_runs", strategy_id=STRATEGY_ID, parameters={"fast_period": 3}
        )
        assert discovered["complete"] and discovered["matched"] == 1
        assert discovered["runs"][0]["run_id"] == run_id
        assert discovered["runs"][0]["parameters"] == {
            "fast_period": 3,
            "slow_period": 8,
        }
        records: list[dict[str, Any]] = []
        offset: int | None = 0
        while offset is not None:
            page = await client.call(
                "get_run_trades", run_id=run_id, limit=2, offset=offset
            )
            assert page["success"] and page["scope"] == "full"
            records.extend(trade["record"] for trade in page["trades"])
            offset = page["next_offset"]
        assert records == observations["trades"]
        equity = await client.call("get_run_equity", run_id=run_id, limit=2)
        assert [point["equity"] for point in equity["points"]] == observations[
            "equity_curve"
        ][:2]
        catalog = await client.call("get_report_sections")
        assert "trades" in catalog["templates"]["standard"]
        executions = await client.call("get_run_executions", run_id=run_id, limit=2)
        assert executions["availability"] == "available"
        assert executions["records"] == observations["execution_records"][:2]
        assert executions["valuation_basis"] == "requested_price_before_slippage"
        accounts = await client.call("get_run_account_history", run_id=run_id, limit=2)
        assert accounts["availability"] == "available"
        assert accounts["states"] == observations["account_history"][:2]
        assert accounts["states"][0]["equity"] == observations["equity_curve"][0]
        assert "ledger" in catalog["templates"]["standard"]
        assert not {"executions", "account_history"} & set(
            catalog["templates"]["standard"]
        )
        report = await client.call(
            "generate_report",
            run_ids=[run_id],
            sections=["overview", "metrics", "ledger", "trades"],
            commentary=[{"title": "評語", "text": "這是模擬結果。"}],
        )
        assert report["success"], report
        html = await client.call(
            "get_artifact", artifact_id=report["artifact_id"], include_content=True
        )
        assert "這是模擬結果。" in html["content"]
        assert hashlib.sha256(html["content"].encode()).hexdigest() == report["sha256"]
        report_manifest = await client.call(
            "get_artifact",
            artifact_id=report["manifest_artifact_id"],
            include_content=True,
        )
        assert json.loads(report_manifest["content"])["sections"] == [
            "overview",
            "metrics",
            "ledger",
            "trades",
        ]
        assert json.loads(report_manifest["content"])["schema_version"] == 2
        source = await client.call(
            "get_artifact",
            artifact_id=by_type["strategy_source"]["artifact_id"],
            include_content=True,
        )
        assert source["content"] == code
        config_artifact = await client.call(
            "get_artifact",
            artifact_id=by_type["config_snapshot"]["artifact_id"],
            include_content=True,
        )
        effective = yaml.safe_load(config_artifact["content"])
        assert effective["strategy"]["revision_id"] == saved["revision_id"]
        assert effective["strategy"]["parameters"] == {
            "fast_period": 3,
            "slow_period": 8,
        }
        assert effective["backtest"]["symbol"] == "BTC/USDT"
        assert effective["backtest"]["end_date"].startswith("2024-01-08")
        assert effective["data"]["requirements"]["market"]["symbol"] == "BTC/USDT"
        manifest_artifact = await client.call(
            "get_artifact",
            artifact_id=by_type["execution_manifest"]["artifact_id"],
            include_content=True,
        )
        manifest = json.loads(manifest_artifact["content"])
        assert manifest["schema_version"] == 5
        assert manifest["strategy_execution"]["constructor_kwargs"] == {
            "fast_period": 3,
            "slow_period": 8,
        }
        assert manifest["kind"] == "backtest"
        assert manifest["manifest_hash"] == started["manifest_hash"]
        assert manifest["config"] == effective
        assert (
            yaml.safe_load(Path(saved["yaml_path"]).read_text())["backtest"]["symbol"]
            == "ETH/USDT"
        )
        inspection = await client.call(
            "inspect_dataset", config_path=by_type["config_snapshot"]["path"]
        )
        assert inspection["data_root"] == str(mcp_workspace.data_root)
        assert inspection["market_available"]
        assert inspection["market"]["rows"] == len(sample_ohlcv_with_kd)

    # Persist across MCP sessions; a new draft does not inherit old approval.
    async with mcp_workspace.connect() as client:
        assert (await client.call("get_job_status", job_id=started["job_id"]))[
            "status"
        ] == "done"
        assert any(item["run_id"] == run_id for item in await client.call("list_runs"))
        assert (await client.call("get_strategy", strategy_id=STRATEGY_ID))["metadata"][
            "status"
        ] == "promoted"
        revision = await client.call(
            "save_strategy",
            strategy_id=STRATEGY_ID,
            code=code + "\n# Revised draft\n",
            yaml_config=config_text,
        )
        assert revision["status"] == "draft"
        assert revision["revision_id"] != saved["revision_id"]
        old = await client.call(
            "get_strategy", strategy_id=STRATEGY_ID, revision_id=saved["revision_id"]
        )
        assert old["metadata"]["status"] == "promoted"
        assert old["source_code"] == code
        rerun_prepared = await client.call(
            "prepare_backtest", **PREPARE_ARGUMENTS, revision_id=saved["revision_id"]
        )
        rerun = await confirm_prepared(client, rerun_prepared)
        rerun_done = await client.wait_for_job(rerun["job_id"])
        assert rerun_done["status"] == "done"
        assert rerun_done["revision_id"] == saved["revision_id"]
        assert rerun_done["metrics"] == completed["metrics"]
        assert not (await client.call("prepare_backtest", **PREPARE_ARGUMENTS))[
            "success"
        ]
        assert Path(by_type["strategy_source"]["path"]).read_text() == code


async def test_rejected_drafts_diagnostics_and_repair(
    mcp_workspace: MCPWorkspace,
) -> None:
    async with mcp_workspace.connect() as client:
        code, config_text, saved = await save_example(client)
        before = {
            path: path.read_bytes()
            for path in mcp_workspace.workspace.rglob("*")
            if path.is_file()
        }
        for args in (
            {"strategy_id": "../escaped", "code": code, "yaml_config": config_text},
            {
                "strategy_id": STRATEGY_ID,
                "code": "def broken(:",
                "yaml_config": config_text,
            },
            {"strategy_id": STRATEGY_ID, "code": code, "yaml_config": "strategy: ["},
        ):
            assert not (await client.call("save_strategy", **args))["success"]
        assert {path: path.read_bytes() for path in before} == before
        assert not (mcp_workspace.root / "escaped.py").exists()
        marker = mcp_workspace.root / "must-not-execute.txt"
        forbidden = (
            f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
        )
        await client.call(
            "save_strategy",
            strategy_id=STRATEGY_ID,
            code=forbidden,
            yaml_config=config_text,
        )
        rejected = await client.call("validate_strategy", strategy_id=STRATEGY_ID)
        assert not rejected["success"] and rejected["status"] == "draft"
        assert "banned_import" in {item["code"] for item in rejected["diagnostics"]}
        assert not marker.exists(), "Rejected code must never be imported"
        removed_import = code.replace(
            "from __future__ import annotations",
            "from __future__ import annotations\n\nimport pandas_ta",
        )
        await client.call(
            "save_strategy",
            strategy_id=STRATEGY_ID,
            code=removed_import,
            yaml_config=config_text,
        )
        rejected = await client.call("validate_strategy", strategy_id=STRATEGY_ID)
        assert not rejected["success"] and rejected["status"] == "draft"
        import_diagnostic = next(
            item
            for item in rejected["diagnostics"]
            if item["code"] == "import_not_allowed"
        )
        assert import_diagnostic["message"] == "import not allowed: pandas_ta"
        assert "talib" in import_diagnostic["fix"]
        for root in (
            "__future__",
            "collections",
            "dataclasses",
            "enum",
            "statistics",
            "typing_extensions",
        ):
            assert root in import_diagnostic["fix"]
        bad_signal = code.replace('result["signal"] = 0', 'result["signal"] = 7')
        await client.call(
            "save_strategy",
            strategy_id=STRATEGY_ID,
            code=bad_signal,
            yaml_config=config_text,
        )
        rejected = await client.call("validate_strategy", strategy_id=STRATEGY_ID)
        assert not rejected["success"]
        assert "invalid_signal_values" in {
            item["code"] for item in rejected["diagnostics"]
        }
        assert all(item.get("fix") for item in rejected["diagnostics"])
        repaired = await client.call(
            "save_strategy", strategy_id=STRATEGY_ID, code=code, yaml_config=config_text
        )
        await make_runnable(client, repaired["revision_id"])
        metadata_path = Path(repaired["py_path"]).parent / "metadata.json"
        assert json.loads(metadata_path.read_text())["status"] == "runnable"
        assert (
            json.loads((Path(saved["py_path"]).parent / "metadata.json").read_text())[
                "status"
            ]
            == "draft"
        )


async def test_worker_failure_is_reported_and_cancelled_job_stops(
    mcp_workspace: MCPWorkspace,
    sample_ohlcv_with_kd: pd.DataFrame,
) -> None:
    cache = mcp_workspace.seed_market(sample_ohlcv_with_kd)
    async with mcp_workspace.connect() as client:
        await save_example(client)
        await make_runnable(client)
        prepared = await client.call("prepare_backtest", **PREPARE_ARGUMENTS)
        assert prepared["success"], prepared
        cache.write_bytes(b"invalid parquet")
        started = await confirm_prepared(client, prepared)
        failed = await client.wait_for_job(started["job_id"])
        assert failed["status"] == "failed", failed
        assert failed["error"]
        assert await client.call("list_runs") == []
        mcp_workspace.seed_market(sample_ohlcv_with_kd)
        prepared = await client.call("prepare_backtest", **PREPARE_ARGUMENTS)
        started = await confirm_prepared(client, prepared)
        cancelled = await client.call("cancel_job", job_id=started["job_id"])
        assert cancelled["success"] and cancelled["process_terminated"], cancelled
        assert (await client.call("get_job_status", job_id=started["job_id"]))[
            "status"
        ] == "cancelled"
        assert not (await client.call("cancel_job", job_id=started["job_id"]))[
            "success"
        ]


async def test_real_quality_gates_reject_invalid_llm_output(
    mcp_workspace: MCPWorkspace,
) -> None:
    async with mcp_workspace.connect() as client:
        code, config_text, _ = await save_example(client)
        await client.call(
            "save_strategy",
            strategy_id=STRATEGY_ID,
            code=code + "\nUNKNOWN = undefined_variable\n",
            yaml_config=config_text,
        )
        rejected = await client.call("validate_strategy", strategy_id=STRATEGY_ID)
        assert not rejected["success"]
        codes = {item["code"] for item in rejected["diagnostics"]}
        assert {"ruff_failed", "mypy_failed"} <= codes
        assert "contract_execution_error" not in codes
        assert not (await client.call("prepare_backtest", **PREPARE_ARGUMENTS))[
            "success"
        ]
        malformed = await client.session.call_tool(
            "save_strategy", {"strategy_id": STRATEGY_ID}
        )
        assert malformed.is_error
        assert (await client.call("get_strategy", strategy_id=STRATEGY_ID))["metadata"][
            "status"
        ] == "draft"


@pytest.mark.parametrize(
    ("validation", "fold_count"),
    [
        ({"n_splits": 2, "train_ratio": 0.6}, 2),
        (
            {
                "train_start": "2024-01-02T03:00:00+08:00",
                "train_end": "2024-01-03T03:00:00+08:00",
                "test_start": "2024-01-04T03:00:00+08:00",
                "test_end": "2024-01-05T03:00:00+08:00",
            },
            1,
        ),
        (
            {
                "n_splits": 2,
                "purged_split": True,
                "label_horizon": 5,
                "train_start": "2024-01-08",
            },
            2,
        ),
    ],
    ids=["automatic", "explicit_offset_dates", "purged_partial_dates"],
)
async def test_walk_forward_runs_through_real_worker(
    mcp_workspace: MCPWorkspace,
    sample_ohlcv_with_kd: pd.DataFrame,
    validation: dict[str, Any],
    fold_count: int,
) -> None:
    mcp_workspace.seed_market(sample_ohlcv_with_kd)
    async with mcp_workspace.connect() as client:
        code, config_text, _ = await save_example(client)
        config = yaml.safe_load(config_text)
        config["validation"] = validation
        await client.call(
            "save_strategy",
            strategy_id=STRATEGY_ID,
            code=code,
            yaml_config=yaml.safe_dump(config),
        )
        await make_runnable(client)
        wrong_mode = await client.call("prepare_backtest", **PREPARE_ARGUMENTS)
        assert not wrong_mode["success"]
        assert "prepare_walk_forward" in wrong_mode["error"]
        prepared = await client.call("prepare_walk_forward", **PREPARE_ARGUMENTS)
        assert prepared["success"], prepared
        if fold_count == 1:
            confirmation = (
                Path(prepared["html_path"]).with_name("confirmation.txt").read_text()
            )
            assert "訓練開始：2024-01-01 00:00:00 UTC（含）" in confirmation
            assert "測試結束：2024-01-05 00:00:00 UTC（不含）" in confirmation
        started = await confirm_prepared(client, prepared)
        completed = await client.wait_for_job(started["job_id"])
        assert completed["status"] == "done", completed
        assert completed["default_scope"] == "test_summary"
        assert set(completed["available_scopes"]) == {"test_summary"} | {
            f"fold/{index}/{role}"
            for index in range(fold_count)
            for role in ("train", "test")
        }
        summary = await client.call("get_run_metrics", run_id=completed["run_id"])
        assert summary["metadata"]["n_folds"] == fold_count
        assert summary["kind"] == "fold_summary"
        assert summary["metrics"]["total_pnl"]["valid_count"] == fold_count
        fold = await client.call(
            "get_run_metrics", run_id=completed["run_id"], scope="fold/0/test"
        )
        assert fold["kind"] == "backtest" and fold["split"] == "test"
        assert isinstance(fold["metrics"]["total_pnl"], float)
        if fold_count == 1:
            assert fold["metadata"]["observations"]["bar_count"] == 48
            assert fold["metadata"]["observations"]["start_date"].startswith(
                "2024-01-03T00:00:00"
            )
            assert fold["metadata"]["observations"]["end_date"].startswith(
                "2024-01-04T23:00:00"
            )
        artifacts = await client.call("list_artifacts", run_id=completed["run_id"])
        assert any(item["artifact_type"] == "pipeline_result" for item in artifacts)
