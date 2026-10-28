"""Real MCP, preparation supervisor and comparison worker on synthetic cached prices."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from contextlib import closing
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.e2e.conversation import ModelTurn, links_to_file, visible_text
from tests.e2e.conversation_user import ObservedPlanUser, review_text, reviewed_config
from tests.e2e.llm_client import ToolCall
from tests.e2e.strategy_scenarios import market_frame
from tests.e2e.workflow_arguments import effective_workflow_calls
from tests.integration.execution_fixtures import (
    confirmation_presentation,
    execution_evidence_draft,
    research_input,
)
from tests.integration.mcp_harness import temporary_mcp_workspace
from tradingdev.adapters.storage.comparisons import ComparisonStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.report_service import ReportService
from tradingdev.domain.comparison import ComparisonManifest
from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.preflight import ComparisonPreflightReceipt
from tradingdev.domain.presentation.reports import REPORT_RECIPES
from tradingdev.domain.strategies.bundled.kd_strategy.config import KDStrategyConfig

if TYPE_CHECKING:
    from tests.integration.mcp_harness import MCPClient, MCPWorkspace


def comparison_arguments() -> dict[str, Any]:
    """Explicit fixture inputs, never instructions supplied to the live model."""
    return {
        "strategy_id": "kd_crossover",
        "research": research_input(strategy_outline="沿用 KD 交叉規則比較兩種 K 週期"),
        "candidates": [
            {
                "label": f"K 週期 {period}",
                "rationale": "事先指定的固定比較範圍。",
                "parameters": {"k_period": period},
            }
            for period in (9, 14)
        ],
    }


class ComparisonUser(ObservedPlanUser):
    """Inspect every tool attempt before issuing a separate, simulated approval."""

    async def __call__(self, turn: ModelTurn) -> str | None:
        if not self.replies:
            assert not any(
                call.name
                in {
                    "ensure_data",
                    "train_model",
                    "submit_execution_decision",
                    "prepare_optimization",
                    "prepare_backtest",
                    "prepare_walk_forward",
                }
                for call in turn.calls
            ), (
                "Fixed comparison must not launch, fetch history, train or switch "
                "workflows before review"
            )
        reply = await super().__call__(turn)
        if len(self.turns) == 1:
            assert len(self.plans) == 1
            plan = self.plans[0]
            assert isinstance(plan.manifest, ComparisonManifest)
            assert isinstance(plan.preflight, ComparisonPreflightReceipt)
            assert len(plan.manifest.candidates) == 2
            periods: list[int] = []
            for candidate in plan.manifest.candidates:
                constructor = candidate.manifest.strategy_execution.constructor_kwargs
                parameters = constructor["config"]
                assert isinstance(parameters, dict)
                period = parameters["k_period"]
                assert isinstance(period, int)
                expected = KDStrategyConfig(k_period=period).model_dump(mode="json")
                assert parameters == expected, "Other KD settings must remain unchanged"
                assert constructor["fit_config"] is None
                periods.append(period)
            assert sorted(periods) == [9, 14]
            config = reviewed_config(plan)
            assert config["strategy"]["id"] == "kd_crossover"
            assert config["backtest"]["symbol"] == "BTC/USDT"
            assert config["backtest"]["timeframe"] == "1h"
            assert datetime.fromisoformat(config["backtest"]["start_date"]) == datetime(
                2024, 1, 1, tzinfo=UTC
            )
            assert datetime.fromisoformat(config["backtest"]["end_date"]) == datetime(
                2024, 1, 8, tzinfo=UTC
            )
            for checked in plan.preflight.candidates:
                assert checked.receipt.synthetic_fixture_rows == [80, 240]
                assert checked.receipt.signal_check == "synthetic_contract_passed"
        return reply


def verify_comparison_delivery(
    workspace: MCPWorkspace,
    user: ObservedPlanUser,
    calls: list[ToolCall],
) -> None:
    """Check exact approved membership, child provenance and a usable final report."""
    user.assert_complete()
    plan = user.plans[-1]
    assert isinstance(plan.manifest, ComparisonManifest)
    calls = effective_workflow_calls(calls)
    approvals = [call for call in calls if call.name == "submit_execution_decision"]
    assert len(approvals) == 1
    assert approvals[0].arguments == {"plan_id": plan.plan_id, "decision": "approve"}
    assert approvals[0].result.get("success") is True
    job_id = approvals[0].result["job_id"]
    assert job_id == plan.plan_id
    completed = [
        call.result
        for call in calls
        if call.name == "get_job_status"
        and call.arguments.get("job_id") == job_id
        and call.result.get("status") == "done"
    ]
    assert completed, "Completion must be observed before delivery"
    status = completed[-1]
    assert status["run_id"] is None and status["metrics"] is None
    assert status["comparison"]["outcome"] == "all_succeeded"
    assert status["comparison"]["counts"]["succeeded"] == len(plan.manifest.candidates)
    paths = WorkspacePaths(workspace.workspace)
    store = SQLiteStore(paths)
    with closing(store.connect()) as connection:
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    comparisons = ComparisonStore(paths)
    saved = comparisons.load_manifest(job_id, expected_hash=plan.manifest.manifest_hash)
    assert saved == plan.manifest
    ledger = comparisons.load_ledger(job_id, expected_hash=saved.manifest_hash)
    receipt, snapshot = comparisons.load_snapshot(
        job_id, expected_hash=saved.manifest_hash
    )
    dataset_id = "sha256:" + hashlib.sha256(snapshot).hexdigest()
    assert status["comparison"]["dataset_id"] == dataset_id
    run_ids = [row.run_id for row in ledger.candidates]
    assert {run["run_id"] for run in store.list_runs()} == set(run_ids)
    for candidate, row in zip(saved.candidates, ledger.candidates, strict=True):
        assert row.status == "succeeded"
        queried = [
            call
            for call in calls
            if call.name in {"get_run", "get_run_metrics"}
            and call.arguments.get("run_id") == row.run_id
            and call.result.get("success") is True
        ]
        assert queried, f"Saved evidence must be queried for {candidate.label}"
        run = store.get_run(row.run_id)
        assert run is not None
        assert run["job_id"] == job_id and run["dataset_id"] == dataset_id
        assert run["manifest_hash"] == candidate.manifest.manifest_hash
        child = ExecutionManifest.model_validate_json(
            (paths.runs / row.run_id / "manifest.json").read_text()
        )
        assert child == candidate.manifest
    final = user.turns[-1]
    report_calls = {
        call.result["report_id"]: call
        for call in effective_workflow_calls(final.calls)
        if call.name == "generate_report"
        and call.arguments.get("comparison_job_id") == job_id
        and call.result.get("success") is True
        and links_to_file(final.text, Path(call.result["path"]))
    }
    assert len(report_calls) == 1, (
        "Final reply must deliver the complete parent comparison report"
    )
    report_call = next(iter(report_calls.values()))
    assert report_call.arguments.get("run_ids") is None
    report = report_call.result
    assert report["comparison_job_id"] == job_id and report["run_ids"] == run_ids
    assert report["navigation_instructions"] in visible_text(final.text), (
        "Final delivery must retain the report's detail navigation guide"
    )
    assert_comparison_references(
        final.text,
        job_id=job_id,
        run_ids=run_ids,
    )
    assert report_call.arguments.get("commentary"), (
        "Report requires explanatory commentary"
    )
    selected = report_call.arguments.get("sections")
    assert selected is None or selected == list(REPORT_RECIPES["standard"])
    service = ReportService(workspace=paths, store=store)
    downloaded = service.get_report_download(
        report["report_id"], expected_sha256=report["sha256"]
    )
    assert downloaded.get("success") is True
    html = downloaded["content"].decode()
    raw = html.split("<script id='report-data' type='application/json'>", 1)[1].split(
        "</script>", 1
    )[0]
    payload = json.loads(raw)
    assert payload["sections"] == list(REPORT_RECIPES["standard"])
    evidence = payload["fixed_comparison"]
    assert evidence["manifest"] == saved.model_dump(mode="json")
    assert evidence["receipt"] == receipt.model_dump(mode="json")
    assert evidence["ledger"] == ledger.model_dump(mode="json")
    assert len(payload["comparison_table"]["rows"]) == len(saved.candidates)
    for index, (candidate, row) in enumerate(
        zip(saved.candidates, payload["comparison_table"]["rows"], strict=True)
    ):
        assert row["candidate"] == candidate.label
        assert row["detail"] == f"#result-{index}-0"
        assert f"href='#result-{index}-0'" in html and f"id='result-{index}-0'" in html
        assert payload["runs"][index]["identity"]["run_id"] == run_ids[index]
        assert [scope["scope_id"] for scope in payload["runs"][index]["scopes"]] == [
            "full"
        ]
    report_path = Path(report["path"])
    report_manifest = json.loads(report_path.with_name("manifest.json").read_text())
    assert report_manifest["fixed_comparison"] == evidence
    registered = store.get_artifact(report["artifact_id"])
    assert registered is not None and registered["path"] == str(report_path)
    assert registered["sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
    print(
        "[comparison] Manual semantic review required: verify every candidate's "
        "metric attribution, success/failure wording, limitations and detail "
        "navigation against the saved evidence. Identity checks do not prove prose "
        "correctness.\nFinal delivery:\n" + final.text,
        flush=True,
    )


def assert_comparison_references(text: str, *, job_id: str, run_ids: list[str]) -> None:
    """Locate saved results by all child IDs or their unique comparison parent.

    Candidate wording may be paraphrased. Its correspondence to metrics, outcome
    states, natural-language scope and navigation instructions requires semantic
    review of the final prose. The report itself must retain exact saved scope IDs;
    users need not see an internal English scope label in their conversation.
    """
    visible = visible_text(text)

    def mentions(value: str) -> bool:
        return (
            re.search(
                rf"(?<![\w-]){re.escape(value)}(?![\w-])", visible, flags=re.ASCII
            )
            is not None
        )

    child_references = all(mentions(run_id) for run_id in run_ids)
    parent_references = mentions(job_id)
    assert child_references or parent_references, (
        "Final reply must identify every result using all child run IDs, or the "
        "exact comparison job ID"
    )


async def direct_comparison(
    client: MCPClient,
    workspace: MCPWorkspace,
    user: ComparisonUser,
) -> list[ToolCall]:
    calls: list[ToolCall] = []

    async def call(name: str, **arguments: Any) -> Any:
        result = await client.call(name, **arguments)
        calls.append(ToolCall(name, arguments, result))
        return result

    await call("get_strategy", strategy_id="kd_crossover")
    parameters = {"config": KDStrategyConfig().model_dump(), "fit_config": None}
    paths = list(confirmation_presentation(parameters)["parameter_descriptions"])
    evidence = await call(
        "save_research_evidence",
        draft=execution_evidence_draft(
            parameters,
            target_paths={
                "candidate_1": paths,
                "candidate_2": paths,
            },
        ),
    )
    assert evidence["success"], evidence
    prepared = await call(
        "prepare_comparison",
        **comparison_arguments(),
        evidence_id=evidence["evidence_id"],
    )
    assert prepared["success"], prepared
    assert not any(path.is_file() for path in workspace.data_root.rglob("*"))
    assert not list((workspace.workspace / "comparisons").rglob("snapshot.parquet"))
    assert await user(ModelTurn(1, review_text(prepared), calls.copy()))
    boundary = len(calls)
    workspace.seed_market(market_frame())
    accepted = await call(
        "submit_execution_decision", plan_id=prepared["plan_id"], decision="approve"
    )
    assert accepted["success"], accepted
    done = await client.wait_for_job(accepted["job_id"], timeout=180)
    calls.append(ToolCall("get_job_status", {"job_id": accepted["job_id"]}, done))
    assert done["status"] == "done", done
    run_ids = [candidate["run_id"] for candidate in done["comparison"]["candidates"]]
    for run_id in run_ids:
        await call("get_run", run_id=run_id)
        await call("get_run_metrics", run_id=run_id)
    await call("get_report_sections")
    report = await call(
        "generate_report",
        comparison_job_id=accepted["job_id"],
        commentary=[
            {
                "title": "比較限制",
                "text": "固定兩組參數使用相同合成行情；結果不代表真實交易績效。",
            }
        ],
    )
    assert report["success"], report
    final = (
        "已完成兩組比較："
        + "、".join(run_ids)
        + f"。兩組範圍皆為 full。\n[閱讀完整比較報告](<{report['path']}>)"
        + "\n"
        + report["navigation_instructions"]
    )
    assert await user(ModelTurn(2, final, calls[boundary:])) is None
    return calls


@pytest.mark.integration
def test_fixed_comparison_approval_worker_and_complete_report() -> None:
    with temporary_mcp_workspace() as workspace:
        user = ComparisonUser(workspace)

        async def run() -> list[ToolCall]:
            async with workspace.connect() as client:
                return await direct_comparison(client, workspace, user)

        calls = asyncio.run(run())
        verify_comparison_delivery(workspace, user, calls)
        plan = user.plans[-1]
        assert isinstance(plan.manifest, ComparisonManifest)
        parent_text = (
            f"比較工作：{plan.plan_id}；結果範圍為 full。\n"
            + "、".join(candidate.label for candidate in plan.manifest.candidates)
            + "\n"
            + user.turns[-1].text.split("\n", 1)[1]
        )
        parent_user = replace(
            user, turns=[user.turns[0], replace(user.turns[-1], text=parent_text)]
        )
        verify_comparison_delivery(workspace, parent_user, calls)
        # Reject misleading delivery evidence without replaying the workers.
        for failure in (
            "subset_report",
            "wrong_parent",
            "missing_link",
            "unqueried_child",
        ):
            broken = deepcopy(calls)
            final_calls = deepcopy(user.turns[-1].calls)
            final_text = user.turns[-1].text
            report_call = next(
                call for call in final_calls if call.name == "generate_report"
            )
            if failure == "subset_report":
                report_call.result["run_ids"].pop()
            elif failure == "wrong_parent":
                report_call.arguments["comparison_job_id"] = "another_parent"
            elif failure == "missing_link":
                final_text = "完成，但未交付報告連結。"
            else:
                hidden = report_call.result["run_ids"][-1]
                broken = [
                    call
                    for call in broken
                    if not (
                        call.name in {"get_run", "get_run_metrics"}
                        and call.arguments.get("run_id") == hidden
                    )
                ]
            changed = replace(
                user, turns=[user.turns[0], ModelTurn(2, final_text, final_calls)]
            )
            with pytest.raises(AssertionError):
                verify_comparison_delivery(workspace, changed, broken)


@pytest.mark.parametrize(
    "text",
    [
        "結果：parent_candidate_1、parent_candidate_2，兩組都是 full scope。",
        "比較工作：`parent`，K=9 與 K=14 的結果皆為 `full` scope。",
        "比較工作：parent，full scope；較短與較長週期分別列在報告中。",
        "比較工作：parent，full scope；完整報告保留各組設定及結果。",
        "比較工作：parent，UTC 2024-01-01 至 2024-01-08，兩組完整期間比較。",
        "| 比較工作 | 候選 | 範圍 |\n|---|---|---|\n"
        "| parent | K=9 | full |\n| parent | K=14 | full |",
    ],
)
def test_comparison_references_accept_child_ids_or_parent_without_exact_labels(
    text: str,
) -> None:
    assert_comparison_references(
        text,
        job_id="parent",
        run_ids=["parent_candidate_1", "parent_candidate_2"],
    )


@pytest.mark.parametrize(
    "text",
    [
        "比較工作：another_parent，K=9、K=14，full scope。",
        "結果：parent_candidate_1，full scope。",  # Partial child selection.
        "比較工作：parent_other，K=9、K=14，full scope。",
        "[閱讀報告](</parent/K=9/K=14/full>)",  # Hidden URL values are not prose.
    ],
)
def test_comparison_references_reject_missing_wrong_or_hidden_identity(
    text: str,
) -> None:
    with pytest.raises(AssertionError):
        assert_comparison_references(
            text,
            job_id="parent",
            run_ids=["parent_candidate_1", "parent_candidate_2"],
        )
