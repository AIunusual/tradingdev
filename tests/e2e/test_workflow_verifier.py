"""Offline regression checks for model-workflow evidence and semantic verifiers.

Each strategy runs once through real MCP and a real backtest worker. Subsequent
checks corrupt one piece of evidence at a time without launching another job.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from tests.e2e.codex_harness import verify_generated_strategy
from tests.e2e.llm_client import ToolCall
from tests.e2e.strategy_scenarios import SCENARIOS, Scenario, market_frame
from tests.e2e.test_llm_workflows import assert_workflow
from tests.integration.execution_fixtures import execution_options
from tests.integration.mcp_harness import (
    MCPWorkspace,
    temporary_mcp_workspace,
)
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.report_service import ReportService

if TYPE_CHECKING:
    from collections.abc import Iterator

_SMA_CODE = '''\
"""Deterministic SMA direction fixture for workflow verification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tradingdev.domain import indicators
from tradingdev.domain.randomness import get_seed
from tradingdev.domain.strategies.base import BaseStrategy

if TYPE_CHECKING:
    import pandas as pd


class SmaFixture(BaseStrategy):
    def __init__(self, fast_period: int, slow_period: int) -> None:
        self.run_seed = get_seed()
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        fast = indicators.sma(result["close"], self.fast_period)
        slow = indicators.sma(result["close"], self.slow_period)
        ready = np.isfinite(fast) & np.isfinite(slow)
        result["signal"] = 0
        result.loc[ready & (fast > slow), "signal"] = 1
        result.loc[ready & (fast < slow), "signal"] = -1
        return result

    def get_parameters(self) -> dict[str, Any]:
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
        }
'''

_MOMENTUM_CODE = '''\
"""Deterministic momentum fixture for workflow verification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tradingdev.domain.strategies.base import BaseStrategy

if TYPE_CHECKING:
    import pandas as pd


class MomentumFixture(BaseStrategy):
    def __init__(self, lookback: int) -> None:
        self.lookback = lookback

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        previous = result["close"].shift(self.lookback)
        result["signal"] = 0
        result.loc[result["close"] > previous, "signal"] = 1
        result.loc[result["close"] < previous, "signal"] = -1
        return result

    def get_parameters(self) -> dict[str, Any]:
        return {"lookback": self.lookback}
'''


@dataclass
class CompletedWorkflow:
    workspace: MCPWorkspace
    scenario: Scenario
    calls: list[ToolCall]
    source_path: Path
    config_path: Path
    run_id: str

    def verify(self) -> None:
        verify_generated_strategy(self.workspace.root, scenario=self.scenario.name)


async def _complete_workflow(
    workspace: MCPWorkspace, scenario: Scenario
) -> CompletedWorkflow:
    calls: list[ToolCall] = []
    async with workspace.connect() as client:

        async def call(name: str, **arguments: Any) -> Any:
            result = await client.call(name, **arguments)
            calls.append(ToolCall(name, arguments, result))
            return result

        contract = await call("get_strategy_contract")
        config = yaml.safe_load(contract["example_yaml_config"])
        if scenario.name == "momentum":
            code = _MOMENTUM_CODE
            config["strategy"]["class_name"] = "MomentumFixture"
        else:
            code = _SMA_CODE
            config["strategy"]["class_name"] = "SmaFixture"
        config["strategy"]["id"] = scenario.strategy_id
        config["strategy"]["parameters"] = scenario.parameters
        config["backtest"].update(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-08",
            init_cash=10000,
            fees=0,
            slippage=0,
            mode="signal",
        )
        config["random_seed"] = 42
        saved = await call(
            "save_strategy",
            strategy_id=scenario.strategy_id,
            code=code,
            yaml_config=yaml.safe_dump(config),
        )
        assert saved["success"], saved
        for name in ("validate_strategy", "dry_run_strategy"):
            result = await call(
                name, strategy_id=scenario.strategy_id, revision_id=saved["revision_id"]
            )
            assert result["success"], result
        prepared = await call(
            "prepare_backtest",
            strategy_id=scenario.strategy_id,
            revision_id=saved["revision_id"],
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-08",
            **execution_options(scenario.parameters),
            **({"parameters": scenario.overrides} if scenario.experiment else {}),
        )
        assert prepared["success"], prepared
        started = await call(
            "submit_execution_decision", plan_id=prepared["plan_id"], decision="approve"
        )
        assert started["job_id"], started
        finished = await client.wait_for_job(started["job_id"])
        assert finished["status"] == "done", finished
        await call("get_job_status", job_id=started["job_id"])
        run_id = finished["run_id"]
        await call("get_run", run_id=run_id)
        await call("list_artifacts", run_id=run_id)
        await call("get_metric_catalog", mode="signal")
        await call(
            "get_run_metrics",
            run_id=run_id,
            metric_ids=["daily_pnl_mean", "total_volume", "n_days"],
        )
    return CompletedWorkflow(
        workspace,
        scenario,
        calls,
        Path(saved["py_path"]),
        Path(saved["yaml_path"]),
        run_id,
    )


@pytest.fixture(scope="module", params=["sma", "momentum", "experiment"])
def completed_workflow(request: pytest.FixtureRequest) -> Iterator[CompletedWorkflow]:
    scenario = SCENARIOS[request.param]
    with temporary_mcp_workspace() as workspace:
        workspace.seed_market(market_frame())
        yield asyncio.run(_complete_workflow(workspace, scenario))
    assert not workspace.root.exists()


def test_workflow_verifier_accepts_completed_backtest(
    completed_workflow: CompletedWorkflow,
) -> None:
    assert_workflow(completed_workflow.calls, completed_workflow.scenario)
    completed_workflow.verify()


@pytest.fixture
def workflow_report(
    completed_workflow: CompletedWorkflow,
) -> Iterator[tuple[dict[str, Any], SQLiteStore]]:
    """Add a report temporarily without changing the shared completed run."""
    workspace = WorkspacePaths(completed_workflow.workspace.workspace)
    store = SQLiteStore(workspace)
    original_artifacts = store.list_artifacts()
    original_ids = {item["artifact_id"] for item in original_artifacts}
    reports = workspace.root / "reports"
    reports_existed = reports.exists()
    original_files = set(reports.rglob("*"))
    result = ReportService(workspace=workspace, store=store).generate_report(
        [completed_workflow.run_id],
        sections=["trades", "metrics"],
        commentary=[{"title": "Workflow review", "text": "Review saved observations."}],
    )
    assert result["success"], result
    report_ids = [result["artifact_id"], result["manifest_artifact_id"]]
    new_ids = [
        artifact_id for artifact_id in report_ids if artifact_id not in original_ids
    ]
    try:
        yield result, store
    finally:
        with store.connect() as connection:
            connection.executemany(
                "DELETE FROM artifacts WHERE artifact_id = ?",
                [(artifact_id,) for artifact_id in new_ids],
            )
        directory = reports / result["report_id"]
        for filename in ("report.html", "manifest.json"):
            path = directory / filename
            if path not in original_files:
                path.unlink(missing_ok=True)
        if directory not in original_files:
            directory.rmdir()
        if not reports_existed:
            reports.rmdir()
        assert store.list_artifacts() == original_artifacts
        assert set(reports.rglob("*")) == original_files


def test_workflow_verifier_accepts_saved_custom_report(
    completed_workflow: CompletedWorkflow,
    workflow_report: tuple[dict[str, Any], SQLiteStore],
) -> None:
    result, _ = workflow_report
    assert Path(result["path"]).is_relative_to(
        completed_workflow.workspace.workspace / "reports"
    )
    completed_workflow.verify()


@pytest.mark.parametrize("field", ["path", "report_id"])
def test_workflow_verifier_rejects_report_registration_corruption(
    completed_workflow: CompletedWorkflow,
    workflow_report: tuple[dict[str, Any], SQLiteStore],
    field: str,
) -> None:
    result, store = workflow_report
    artifact_id = result["artifact_id"]
    artifact = store.get_artifact(artifact_id)
    assert artifact is not None
    unrelated = completed_workflow.workspace.workspace / "unrelated_report.html"
    assert not unrelated.exists()
    try:
        with store.connect() as connection:
            if field == "path":
                unrelated.write_bytes(Path(result["path"]).read_bytes())
                connection.execute(
                    "UPDATE artifacts SET path = ? WHERE artifact_id = ?",
                    (str(unrelated), artifact_id),
                )
            else:
                metadata = {**artifact["metadata"], "report_id": "../runs"}
                connection.execute(
                    "UPDATE artifacts SET metadata = ? WHERE artifact_id = ?",
                    (json.dumps(metadata), artifact_id),
                )
        with pytest.raises(AssertionError, match="Report (path|ID)"):
            completed_workflow.verify()
    finally:
        with store.connect() as connection:
            connection.execute(
                "UPDATE artifacts SET path = ?, metadata = ? WHERE artifact_id = ?",
                (artifact["path"], json.dumps(artifact["metadata"]), artifact_id),
            )
        unrelated.unlink(missing_ok=True)


@pytest.mark.parametrize("field", ["run_ids", "available_scopes", "html_sha256"])
def test_workflow_verifier_rejects_report_manifest_corruption(
    completed_workflow: CompletedWorkflow,
    workflow_report: tuple[dict[str, Any], SQLiteStore],
    field: str,
) -> None:
    result, store = workflow_report
    artifact_id = result["manifest_artifact_id"]
    artifact = store.get_artifact(artifact_id)
    assert artifact is not None
    path = Path(artifact["path"])
    original = path.read_bytes()
    manifest = json.loads(original)
    replacements: dict[str, Any] = {
        "run_ids": ["another-run"],
        "available_scopes": {completed_workflow.run_id: ["another-scope"]},
        "html_sha256": "0" * 64,
    }
    manifest[field] = replacements[field]
    try:
        path.write_text(json.dumps(manifest), encoding="utf-8")
        # Keep the registry SHA valid: semantic linkage must independently fail.
        with store.connect() as connection:
            connection.execute(
                "UPDATE artifacts SET sha256 = ? WHERE artifact_id = ?",
                (hashlib.sha256(path.read_bytes()).hexdigest(), artifact_id),
            )
        with pytest.raises(AssertionError, match="Report (run IDs|scopes|HTML digest)"):
            completed_workflow.verify()
    finally:
        path.write_bytes(original)
        with store.connect() as connection:
            connection.execute(
                "UPDATE artifacts SET sha256 = ? WHERE artifact_id = ?",
                (artifact["sha256"], artifact_id),
            )


def test_workflow_verifier_rejects_incorrect_signals(
    completed_workflow: CompletedWorkflow,
) -> None:
    source = completed_workflow.source_path
    original = source.read_text(encoding="utf-8")
    corrupted = original.replace(
        "        return result\n",
        '        result["signal"] = 0\n        return result\n',
    )
    assert corrupted != original
    try:
        source.write_text(corrupted, encoding="utf-8")
        with pytest.raises(AssertionError, match="content mismatch"):
            completed_workflow.verify()
    finally:
        source.write_text(original, encoding="utf-8")


def test_workflow_verifier_rejects_incorrect_yaml_parameters(
    completed_workflow: CompletedWorkflow,
) -> None:
    config_path = completed_workflow.config_path
    original = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(original)
    first_parameter = next(iter(config["strategy"]["parameters"]))
    config["strategy"]["parameters"][first_parameter] += 1
    try:
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with pytest.raises(AssertionError, match="content mismatch"):
            completed_workflow.verify()
    finally:
        config_path.write_text(original, encoding="utf-8")


def test_workflow_verifier_rejects_incorrect_stored_metrics(
    completed_workflow: CompletedWorkflow,
) -> None:
    database = completed_workflow.workspace.workspace / "tradingdev.sqlite"
    with closing(sqlite3.connect(database)) as connection:
        row = connection.execute(
            "SELECT metrics FROM runs WHERE run_id = ?", (completed_workflow.run_id,)
        ).fetchone()
        original = row[0]
        metrics = json.loads(original)
        metrics["total_return"] += 1.0
        try:
            with connection:
                connection.execute(
                    "UPDATE runs SET metrics = ? WHERE run_id = ?",
                    (json.dumps(metrics), completed_workflow.run_id),
                )
            with pytest.raises(AssertionError, match="total_return"):
                completed_workflow.verify()
        finally:
            with connection:
                connection.execute(
                    "UPDATE runs SET metrics = ? WHERE run_id = ?",
                    (original, completed_workflow.run_id),
                )


def test_workflow_evidence_requires_model_to_query_result(
    completed_workflow: CompletedWorkflow,
) -> None:
    incomplete = [call for call in completed_workflow.calls if call.name != "get_run"]
    with pytest.raises((AssertionError, StopIteration, ValueError)):
        assert_workflow(incomplete, completed_workflow.scenario)
