"""Model-driven research plans with an explicit simulated user approval.

Live cases retain one conversation and inspect each completed model turn before
injecting a simulated user reply into the same messages or Codex thread.
Offline cases use the same real MCP server, workers, fixture and evidence checks
without calling a model. Cached prices are synthetic; these tests do not validate
external market-data access or a real human interface.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import pytest
import yaml

from tests.e2e.codex_harness import run_codex
from tests.e2e.conversation import ModelTurn
from tests.e2e.conversation_user import ObservedPlanUser, review_text
from tests.e2e.llm_client import ToolCall, codex_calls, run_local_model
from tests.e2e.research_evidence import assert_saved_research
from tests.e2e.strategy_scenarios import market_frame
from tests.e2e.workflow_arguments import effective_workflow_calls
from tests.e2e.workflow_research import (
    assert_research_matches_config,
    research_values,
)
from tests.integration.execution_fixtures import (
    confirmation_presentation,
    execution_evidence_draft,
)
from tests.integration.mcp_harness import temporary_mcp_workspace
from tests.research_fixtures import research_for_config
from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.preflight import PreflightReceipt
from tradingdev.domain.research import NewResearchSpec, research_payload
from tradingdev.domain.strategies.bundled.kd_strategy.config import KDStrategyConfig
from tradingdev.domain.validation.config import parse_walk_forward_config

if TYPE_CHECKING:
    from tests.integration.mcp_harness import MCPClient, MCPWorkspace

type PlanKind = Literal["optimization", "walk_forward"]
_KINDS: tuple[PlanKind, ...] = ("optimization", "walk_forward")
_WF_STRATEGY = "llm_walk_forward_sma"
_MARKET = {"symbol": "BTC/USDT", "timeframe": "1h"}
_COSTS = {"init_cash": 10000, "fees": 0, "slippage": 0, "periods_per_year": 365}
_WF_VALIDATION = {
    "n_splits": 2,
    "label_horizon": 5,
    "embargo_bars": 2,
    "mode": "purged",
}


def _strategy_id(kind: PlanKind) -> str:
    return "kd_crossover" if kind == "optimization" else _WF_STRATEGY


def _arguments(kind: PlanKind, revision_id: str | None) -> dict[str, Any]:
    args: dict[str, Any] = {
        "strategy_id": _strategy_id(kind),
        "revision_id": revision_id,
        "research": {"kind": "inherit_selected_revision"},
    }
    if kind == "optimization":
        args["research"] = research_payload(
            NewResearchSpec.model_validate(
                {
                    "kind": "new",
                    "strategy_outline": "比較 KD 指標不同週期的買賣規則",
                    **_MARKET,
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-08T23:59:59.999999Z",
                    "init_cash": 10000,
                    "fees": 0,
                    "slippage": 0,
                    "annualization": 365,
                }
            )
        )
        args.update(
            param_ranges={"k_period": [9, 14]},
            optimization_metric="total_return",
            train_start="2024-01-01",
            train_end="2024-01-03",
            test_start="2024-01-04",
            test_end="2024-01-08",
        )
    return args


async def _seed_walk_forward(
    client: MCPClient, *, validation: dict[str, Any] | None = None
) -> str:
    """Persist a genuinely validated/runnable fixture before involving the model."""
    contract = (await client.call("get_strategy_contract", section="example"))[
        "content"
    ]
    config = yaml.safe_load(contract["example_yaml_config"])
    config["strategy"]["parameters"] = {"fast_period": 3, "slow_period": 8}
    config["backtest"].update(
        **_MARKET, **_COSTS, start_date="2024-01-01", end_date="2024-01-08"
    )
    config["validation"] = (
        _WF_VALIDATION.copy() if validation is None else validation.copy()
    )
    config["random_seed"] = 42
    saved = await client.call(
        "save_strategy",
        strategy_id=_WF_STRATEGY,
        code=contract["example_strategy_code"],
        yaml_config=yaml.safe_dump(config),
        research=research_payload(
            research_for_config(
                config, strategy_outline="已驗證的雙分段 SMA 測試策略。"
            )
        ),
    )
    assert saved["success"], saved
    revision_id = str(saved["revision_id"])
    for tool, status in (
        ("validate_strategy", "validated"),
        ("dry_run_strategy", "runnable"),
    ):
        checked = await client.call(
            tool, strategy_id=_WF_STRATEGY, revision_id=revision_id
        )
        assert checked["success"] and checked["status"] == status, checked
        assert checked["revision_id"] == revision_id
    return revision_id


def _prompt(kind: PlanKind, revision_id: str | None) -> str:
    description = (
        "內建 KD 策略的參數最佳化。候選只有 k_period=9、14，"
        "使用 total_return 最大化，訓練 2024-01-01 至 2024-01-03，"
        "樣本外測試 2024-01-04 至 2024-01-08。"
        if kind == "optimization"
        else "已完成驗證與 dry-run 的生成 SMA 策略，執行 walk-forward。"
        "保持現有 validation：mode=purged、n_splits=2、label_horizon=5、"
        "embargo_bars=2；"
        "策略參數與 random_seed=42 也保持原值，"
        "整體期間 2024-01-01 至 2024-01-08。"
    )
    queries = (
        "get_run 回覆後，以 selected_train_scope 取得最佳候選的訓練結果，"
        "並以 scope=test 取得樣本外測試結果，兩次都用 get_run_metrics。"
        if kind == "optimization"
        else "get_run 回覆後，以 get_run_metrics 分別取得 scope=fold/0/train、"
        "fold/0/test、fold/1/test、test_summary。"
    )
    return f"""透過 TradingDev MCP 完成{description}
策略 strategy_id={_strategy_id(kind)}，revision_id={revision_id!r}。
先 get_strategy 讀取目前策略原始碼與 YAML；不要 save 或修改策略。
使用 prepare_{kind}，research如下：
{json.dumps(_arguments(kind, revision_id)["research"], ensure_ascii=False)}
成本只透過 research 設定，不放進 backtest_overrides；修改費用時也在 research 明示。
確認前只做離線程式驗證；正式執行所需行情已在快取，不需下載。
先依 evidence 契約保存研究說明，再以 evidence_id 準備計畫。
研究說明中的 presentation 提供繁體中文 title、summary、parameter_descriptions。
每個有效建構子參數含預設值都要白話 label、description、unit。
參數鍵為 JSON pointer：內建 KD 用 /config/k_period 等 /config/*；
生成 SMA 用 /fast_period、/slow_period。請依讀到的設定自行完整撰寫說明。
若工具指出 required_parameter_paths，補齊後重新 prepare，不略過預設值。
取得 success=true/status=ready 後，將 confirmation_summary 逐行原文引用，
建議每行加 > 作為 Markdown 引用；只可調整排版，
不可改寫、刪除欄名、值、單位或訓練／測試角色。
摘要已由後端完成，不要再摘要。引用外另行逐字貼上工具回覆的 confirmation_link，
核對連結目的地為 html_path；不要自行組合或改寫識別碼與路徑。
連結放在一般 Markdown 文字行，不放 code block，不加 file:// 或自行創造其他 URI 前綴。
然後結束本回合等待我回覆。一般解釋及回答問題可用自己的表達方式。
目前尚未同意正式執行，不可在第一回合呼叫 submit_execution_decision。
只有下一回合我的明確同意，才用 plan_id、decision=approve 提交。
拒絕用 decline；修改用 revise，依修改內容重新 prepare、呈現新摘要與HTML，
再次結束回合等待確認。模糊回覆須澄清，不可當成同意。
持續 get_job_status 直到 done，再 get_run 查同一 run_id。
{queries}
最後以同一 run_id 呼叫 list_artifacts(run_id=該結果的 run_id)，確認該結果的產物。
必須取得全部結果才結束，只回覆「完成」與 run_id。
只用 MCP 工具，不用 shell、不直接改檔。不建立第二次正式執行。
"""


def _successful(calls: list[ToolCall], name: str) -> list[tuple[int, ToolCall]]:
    return [
        (index, call)
        for index, call in enumerate(calls)
        if call.name == name
        and isinstance(call.result, dict)
        and call.result.get("success") is True
    ]


def _assert_selected_research(
    calls: list[ToolCall],
    prepare_index: int,
    approval: ObservedPlanUser,
    revision_id: str | None,
) -> None:
    """Bind reused values to the actual fixture revision, not the prepared request."""
    assert isinstance(revision_id, str) and revision_id
    directory = (
        approval.workspace.workspace
        / "generated_strategies"
        / _WF_STRATEGY
        / "revisions"
        / revision_id
    )
    config_text = (directory / "config.yaml").read_text()
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["strategy_id"] == _WF_STRATEGY
    assert metadata["revision_id"] == revision_id
    assert hashlib.sha256(config_text.encode()).hexdigest() == metadata["config_hash"]
    assert any(
        index < prepare_index
        and call.arguments.get("strategy_id") == _WF_STRATEGY
        and call.result.get("strategy_id") == _WF_STRATEGY
        and call.result.get("revision_id") == revision_id
        and call.result.get("yaml_config") == config_text
        for index, call in _successful(calls, "get_strategy")
    ), "The model must read the selected revision's exact saved configuration"
    selected_config = yaml.safe_load(config_text)
    expected = research_values(selected_config, context="Selected revision YAML")
    for phase in ("validation", "dry_run"):
        checked = metadata[phase]
        assert checked["success"] and checked["revision_id"] == revision_id
        assert (
            research_values(checked.get("effective_config"), context=phase) == expected
        ), "Selected source conflicts with its saved validation evidence"
    if len(approval.plans) > 1:
        selected_config["backtest"]["fees"] = approval.revised_fees
        assert calls[prepare_index].arguments["research"].get("fees") == (
            approval.revised_fees
        ), "The revised request must explicitly include the authorized fee change"
        expected["fees"] = approval.revised_fees
    assert_research_matches_config(
        calls[prepare_index].arguments.get("research"), selected_config
    )
    selected_manifest = approval.plans[-1].manifest
    assert isinstance(selected_manifest, ExecutionManifest)
    config = selected_manifest.config_copy()
    assert research_values(config, context="Prepared manifest") == expected, (
        "Prepared research settings differ from the selected source and user decision"
    )
    assert config["research"]["selected_revision"]["revision_id"] == revision_id
    assert (
        config["research"]["selected_revision"]["config_hash"]
        == (metadata["config_hash"])
    )


def _assert_evidence(
    calls: list[ToolCall],
    kind: PlanKind,
    approval: ObservedPlanUser,
    *,
    selected_revision_id: str | None,
) -> None:
    calls = effective_workflow_calls(calls)
    approval.assert_complete()
    plan = approval.plans[-1]
    manifest = plan.manifest
    assert isinstance(manifest, ExecutionManifest)
    assert isinstance(plan.preflight, PreflightReceipt)
    prepared = _successful(calls, f"prepare_{kind}")
    assert len(prepared) == len(approval.plans), prepared
    prepare_index, prepare_call = prepared[-1]
    assert any(
        index < prepare_index
        and call.arguments.get("strategy_id") == _strategy_id(kind)
        for index, call in _successful(calls, "get_strategy")
    ), "The model must read the existing strategy before preparing it"
    assert not {"save_strategy", "prepare_backtest"}.intersection(
        call.name for call in calls
    )
    assert prepare_call.result["plan_id"] == plan.plan_id
    assert prepare_call.result["status"] == "ready"
    assert not prepare_call.result.get("job_id")
    assert prepare_call.result["manifest_hash"] == manifest.manifest_hash
    assert prepare_call.result["preflight"] == plan.preflight.model_dump(mode="json")
    assert Path(prepare_call.result["html_path"]).is_file()
    assert prepare_call.result["confirmation_summary"]
    assert prepare_call.result["artifact_id"]
    revision_id = manifest.config_copy()["strategy"].get("revision_id")
    assert revision_id == selected_revision_id, (
        "Execution must use the fixture's selected immutable revision"
    )
    expected_arguments = _arguments(kind, revision_id)
    if len(approval.plans) > 1:
        expected_arguments["research"]["fees"] = approval.revised_fees
    for name, value in expected_arguments.items():
        if name == "research" and kind == "walk_forward":
            continue
        assert prepare_call.arguments.get(name) == value
    if kind == "walk_forward":
        _assert_selected_research(calls, prepare_index, approval, selected_revision_id)
    assert plan.preflight.schema_version == 2
    assert {
        "configuration",
        "execution_capabilities",
        "parameter_binding",
        "manifest_serialization",
    } <= set(plan.preflight.checked_paths)
    assert set(plan.preflight.not_checked) == {
        "historical_data",
        "backtest_engine",
        "training",
        "performance",
    }
    draft = assert_saved_research(calls, prepare_index)
    assert draft == plan.research_evidence.research.record.draft
    assert (
        prepare_call.arguments["evidence_id"]
        == plan.research_evidence.research.record.evidence_id
    )
    descriptions = draft.presentation.model_dump(mode="json")["parameter_descriptions"]
    expected_parameters: dict[str, Any] = (
        {"config": KDStrategyConfig().model_dump(), "fit_config": None}
        if kind == "optimization"
        else {"fast_period": 3, "slow_period": 8}
    )
    assert manifest.strategy_execution.constructor_kwargs == expected_parameters
    expected_paths = (
        {"/config/" + key for key in KDStrategyConfig.model_fields} | {"/fit_config"}
        if kind == "optimization"
        else {"/fast_period", "/slow_period"}
    )
    assert set(descriptions) == expected_paths
    for pointer, description in descriptions.items():
        assert description["label"] != pointer.rsplit("/", 1)[-1]
        assert description["description"]
        assert description.get("unit") is None or description["unit"]
    expected_costs = {
        **_COSTS,
        "fees": approval.revised_fees if len(approval.plans) > 1 else 0,
    }
    for name, value in expected_costs.items():
        assert manifest.config_copy()["backtest"][name] == value
    confirmations = [
        (index, call)
        for index, call in enumerate(calls)
        if call.name == "submit_execution_decision"
    ]
    assert len(confirmations) == len(approval.plans)
    confirm_index, confirmed = confirmations[-1]
    assert prepare_index < confirm_index
    assert confirmed.arguments == {"plan_id": plan.plan_id, "decision": "approve"}
    assert confirmed.result["success"]
    assert confirmed.result["manifest_hash"] == manifest.manifest_hash
    job_id = confirmed.result["job_id"]
    done = [
        (index, call)
        for index, call in enumerate(calls)
        if index > confirm_index
        and call.name == "get_job_status"
        and call.arguments.get("job_id") == job_id
        and call.result.get("status") == "done"
    ]
    assert done, "The model must wait for formal execution to complete"
    done_index, completed = done[0]
    assert completed.result["manifest_hash"] == manifest.manifest_hash
    assert completed.result["revision_id"] == revision_id
    run_id = completed.result["run_id"]
    run_index, queried = next(
        (index, call)
        for index, call in _successful(calls, "get_run")
        if index > done_index and call.arguments.get("run_id") == run_id
    )
    run = queried.result["run"]
    assert run["manifest_hash"] == manifest.manifest_hash
    assert run["strategy_id"] == _strategy_id(kind)
    assert run["revision_id"] == revision_id
    if kind == "optimization":
        assert plan.preflight.parameter_combinations_checked == 2
        assert "candidate_binding" in plan.preflight.checked_paths
        assert plan.preflight.signal_check == "bundled_not_executed"
        assert plan.preflight.synthetic_fixture_rows == []
        assert manifest.optimization is not None
        assert manifest.optimization.param_ranges == {"k_period": [9, 14]}
        assert manifest.optimization.optimization_metric == "total_return"
        assert completed.result["total_combinations"] == 2
        assert completed.result["best_params"]["k_period"] in {9, 14}
        assert set(run["available_scopes"]) == {
            "trial/0/train",
            "trial/1/train",
            "test",
        }
        scopes = {run["selected_train_scope"], "test"}
        assert run["default_scope"] == "test"
    else:
        assert plan.preflight.parameter_combinations_checked is None
        assert plan.preflight.signal_check == "synthetic_contract_passed"
        assert plan.preflight.synthetic_fixture_rows == [80, 240]
        assert "signal_contract" in plan.preflight.checked_paths
        assert manifest.config_copy()["random_seed"] == 42
        assert manifest.config_copy()["validation"] == parse_walk_forward_config(
            _WF_VALIDATION
        ).model_dump(mode="json")
        for _, read in _successful(calls, "get_strategy"):
            source_config = yaml.safe_load(read.result["yaml_config"])
            assert parse_walk_forward_config(
                source_config["validation"]
            ) == parse_walk_forward_config(_WF_VALIDATION)
            assert source_config["random_seed"] == 42
        confirmation = (
            Path(prepare_call.result["html_path"])
            .with_name("confirmation.txt")
            .read_text()
        )
        assert "已試跑的驗證分段" not in confirmation
        assert "固定起點、逐次擴大訓練區間" in confirmation
        section = next(
            item for item in plan.document.sections if item.id == "validation"
        )
        assert {field.label for field in section.fields} == {
            "分段方式",
            "預定分段組數",
            "訓練區間逐次擴大",
            "額外隔離根數",
            "驗證紀錄重點指標",
            "標籤前視根數",
        }
        assert "預定分段組數：2" in confirmation
        assert "標籤前視根數：5 根 K 線" in confirmation
        assert "額外隔離根數：2 根 K 線" in confirmation
        assert set(run["available_scopes"]) == {
            "fold/0/train",
            "fold/0/test",
            "fold/1/train",
            "fold/1/test",
            "test_summary",
        }
        scopes = {"fold/0/train", "fold/0/test", "fold/1/test", "test_summary"}
        assert run["default_scope"] == "test_summary"
    details = {
        call.result["scope"]: call.result
        for index, call in _successful(calls, "get_run_metrics")
        if index > run_index and call.arguments.get("run_id") == run_id
    }
    assert set(details) >= scopes
    run_directory = approval.workspace.workspace / "runs" / run_id
    saved_performance = json.loads((run_directory / "performance.json").read_text())
    if kind == "walk_forward":
        # The inclusive Jan 1--Jan 8 bounds contain 169 hourly bars. Anchored
        # thirds start tests at 57 and 113; each excludes a 5+2 bar gap.
        origin = pd.Timestamp("2024-01-01", tz="UTC")
        for index, windows in enumerate((((0, 50), (57, 113)), ((0, 106), (113, 169)))):
            for role, (start, end) in zip(("train", "test"), windows, strict=True):
                observations = saved_performance["scopes"][f"fold/{index}/{role}"][
                    "metadata"
                ]["observations"]
                assert observations["bar_count"] == end - start
                assert pd.Timestamp(
                    observations["start_date"]
                ) == origin + pd.Timedelta(hours=start)
                assert pd.Timestamp(observations["end_date"]) == origin + pd.Timedelta(
                    hours=end - 1
                )
    for scope in scopes:
        assert details[scope]["scope"] == scope
        assert details[scope]["metrics"]
        for key, value in details[scope]["metrics"].items():
            assert saved_performance["scopes"][scope]["values"][key] == value
    saved_manifest = ExecutionManifest.model_validate_json(
        (run_directory / "manifest.json").read_text()
    )
    assert saved_manifest == manifest, "Formal execution must use the approved manifest"
    assert any(
        index > done_index
        and call.name == "list_artifacts"
        and call.arguments.get("run_id") == run_id
        and isinstance(call.result, dict)
        and call.result.get("items")
        for index, call in enumerate(calls)
    ), json.dumps(
        [
            {"arguments": call.arguments, "result": call.result}
            for call in calls
            if call.name == "list_artifacts"
        ],
        ensure_ascii=False,
    )


async def _direct_workflow(
    client: MCPClient, kind: PlanKind, revision_id: str | None, user: ObservedPlanUser
) -> list[ToolCall]:
    """Exercise the live fixture/checker offline with deterministic MCP requests."""
    calls: list[ToolCall] = []

    async def call(name: str, **arguments: Any) -> Any:
        result = await client.call(name, **arguments)
        calls.append(ToolCall(name, arguments, result))
        return result

    await call("get_strategy", strategy_id=_strategy_id(kind), revision_id=revision_id)
    parameters = (
        {"config": KDStrategyConfig().model_dump(), "fit_config": None}
        if kind == "optimization"
        else {"fast_period": 3, "slow_period": 8}
    )
    paths = list(confirmation_presentation(parameters)["parameter_descriptions"])
    target_paths = {"base": paths}
    if kind == "optimization":
        target_paths.update(
            {
                "search_1_1": ["/config/k_period"],
                "search_1_2": ["/config/k_period"],
            }
        )
    saved = await call(
        "save_research_evidence",
        draft=json.dumps(
            execution_evidence_draft(parameters, target_paths=target_paths)
        ),
    )
    assert saved["success"], saved
    prepared = await call(
        f"prepare_{kind}",
        **{
            name: json.dumps(value) if isinstance(value, dict | int) else value
            for name, value in (
                _arguments(kind, revision_id) | {"evidence_id": saved["evidence_id"]}
            ).items()
        },
    )
    assert prepared["success"], prepared
    assert await user(ModelTurn(1, review_text(prepared), calls.copy()))
    boundary = len(calls)
    confirmed = await call(
        "submit_execution_decision", plan_id=prepared["plan_id"], decision="approve"
    )
    assert confirmed["success"], confirmed
    completed = await client.wait_for_job(confirmed["job_id"], timeout=180)
    calls.append(ToolCall("get_job_status", {"job_id": confirmed["job_id"]}, completed))
    assert completed["status"] == "done", completed
    run_id = completed["run_id"]
    run = (await call("get_run", run_id=run_id))["run"]
    scopes = (
        [run["selected_train_scope"], "test"]
        if kind == "optimization"
        else ["fold/0/train", "fold/0/test", "fold/1/test", "test_summary"]
    )
    for scope in scopes:
        await call("get_run_metrics", run_id=run_id, scope=scope)
    await call("list_artifacts", run_id=run_id)
    assert await user(ModelTurn(2, "完成", calls[boundary:])) is None
    return calls


@pytest.mark.integration
@pytest.mark.parametrize("kind", _KINDS)
def test_execution_plan_live_fixtures_without_model(kind: PlanKind) -> None:
    with temporary_mcp_workspace() as workspace:
        workspace.seed_market(market_frame())
        approval = ObservedPlanUser(workspace)

        async def run() -> tuple[list[ToolCall], str | None]:
            async with workspace.connect() as client:
                revision_id = (
                    await _seed_walk_forward(client) if kind == "walk_forward" else None
                )
                calls = await _direct_workflow(client, kind, revision_id, approval)
                return calls, revision_id

        calls, revision_id = asyncio.run(run())
        original = deepcopy(calls)
        _assert_evidence(calls, kind, approval, selected_revision_id=revision_id)
        _assert_evidence(
            effective_workflow_calls(calls),
            kind,
            approval,
            selected_revision_id=revision_id,
        )
        assert calls == original, "The verifier must retain raw model arguments"
        prepare_index = next(
            index for index, call in enumerate(calls) if call.name == f"prepare_{kind}"
        )
        for invalid in ("not JSON", "[]", "null", "{}"):
            broken = deepcopy(calls)
            saved_call = next(
                call for call in broken if call.name == "save_research_evidence"
            )
            saved_call.arguments["draft"] = invalid
            with pytest.raises(AssertionError, match="draft"):
                _assert_evidence(
                    broken, kind, approval, selected_revision_id=revision_id
                )
        if kind == "walk_forward":
            _check_reused_research_regressions(
                calls, prepare_index, approval, revision_id
            )


def _check_reused_research_regressions(
    calls: list[ToolCall],
    prepare_index: int,
    approval: ObservedPlanUser,
    revision_id: str | None,
) -> None:
    """Corrupt evidence after one real execution instead of repeating the worker."""
    normalized = effective_workflow_calls(calls)
    repeated = {
        "kind": "inherit_selected_revision",
        "start_date": "2024-01-01T08:00:00+08:00",
        "end_date": "2024-01-08T00:00:00Z",
        "fees": 0,
        "slippage": 0,
    }
    for research in (
        repeated,
        {
            **repeated,
            "strategy_outline": "已驗證的雙分段 SMA 測試策略。",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "init_cash": 10000,
            "annualization": 365,
        },
    ):
        accepted = deepcopy(normalized)
        accepted[prepare_index].arguments["research"] = json.dumps(research)
        original = deepcopy(accepted)
        _assert_evidence(
            accepted, "walk_forward", approval, selected_revision_id=revision_id
        )
        assert accepted == original, "Repeated research must preserve raw evidence"
    for field, value in (
        ("start_date", "2024-01-01T01:00:00Z"),
        ("end_date", "2024-01-08T01:00:00Z"),
        ("fees", 0.0006),
        ("slippage", 0.0005),
        ("strategy_outline", "Different rules"),
        ("annualization", "market"),
    ):
        broken = deepcopy(normalized)
        broken[prepare_index].arguments["research"] = {**repeated, field: value}
        with pytest.raises(AssertionError, match="research changes|reselect"):
            _assert_evidence(
                broken, "walk_forward", approval, selected_revision_id=revision_id
            )
    broken = deepcopy(normalized)
    broken[prepare_index].arguments["revision_id"] = "another_revision"
    with pytest.raises(AssertionError):
        _assert_evidence(
            broken, "walk_forward", approval, selected_revision_id=revision_id
        )
    with pytest.raises(AssertionError, match="selected immutable revision"):
        _assert_evidence(
            normalized, "walk_forward", approval, selected_revision_id="other"
        )
    read_index = next(
        index for index, call in enumerate(normalized) if call.name == "get_strategy"
    )
    for field, value in (
        ("revision_id", "another_revision"),
        ("yaml_config", "backtest: {fees: 0.001}"),
    ):
        broken = deepcopy(normalized)
        broken[read_index].result[field] = value
        with pytest.raises(AssertionError, match="exact saved configuration"):
            _assert_evidence(
                broken, "walk_forward", approval, selected_revision_id=revision_id
            )
    assert isinstance(revision_id, str)
    metadata_path = (
        approval.workspace.workspace
        / "generated_strategies"
        / _WF_STRATEGY
        / "revisions"
        / revision_id
        / "metadata.json"
    )
    original_metadata = metadata_path.read_bytes()
    for phase in ("validation", "dry_run"):
        metadata = json.loads(original_metadata)
        metadata[phase]["effective_config"]["backtest"]["fees"] = 0.001
        try:
            metadata_path.write_text(json.dumps(metadata))
            with pytest.raises(AssertionError, match="conflicts.*validation evidence"):
                _assert_evidence(
                    normalized,
                    "walk_forward",
                    approval,
                    selected_revision_id=revision_id,
                )
        finally:
            metadata_path.write_bytes(original_metadata)


async def _model_workflow(
    workspace: MCPWorkspace,
    pytestconfig: pytest.Config,
    kind: PlanKind,
    approval: ObservedPlanUser,
) -> tuple[list[ToolCall], str | None]:
    provider = pytestconfig.getoption("llm_provider")
    assert provider in {"codex", "local"}, "Use scripts/check-llm.sh codex|local"
    if provider == "codex":
        revision_id = None
        if kind == "walk_forward":
            async with workspace.connect() as client:
                revision_id = await _seed_walk_forward(client)
        calls = codex_calls(
            await run_codex(
                workspace.root,
                _prompt(kind, revision_id),
                user_reply=approval,
                model=pytestconfig.getoption("llm_model"),
                timeout_seconds=pytestconfig.getoption("llm_timeout"),
            )
        )
        return calls, revision_id
    async with workspace.connect() as client:
        revision_id = (
            await _seed_walk_forward(client) if kind == "walk_forward" else None
        )
        calls = await run_local_model(
            client,
            _prompt(kind, revision_id),
            model=pytestconfig.getoption("llm_model"),
            base_url=pytestconfig.getoption("llm_base_url"),
            timeout_seconds=pytestconfig.getoption("llm_timeout"),
            temperature=pytestconfig.getoption("llm_temperature"),
            reasoning_effort=pytestconfig.getoption("llm_reasoning_effort"),
            user_reply=approval,
        )
        return calls, revision_id


@pytest.mark.live_llm
@pytest.mark.parametrize("kind", _KINDS)
def test_llm_prepares_confirms_and_queries_execution_plan(
    pytestconfig: pytest.Config, kind: PlanKind
) -> None:
    with temporary_mcp_workspace() as workspace:
        workspace.seed_market(market_frame())
        approval = ObservedPlanUser(workspace)
        calls, revision_id = asyncio.run(
            _model_workflow(workspace, pytestconfig, kind, approval)
        )
        _assert_evidence(calls, kind, approval, selected_revision_id=revision_id)


@pytest.mark.live_llm
def test_llm_revises_plan_then_waits_for_explicit_user_approval(
    pytestconfig: pytest.Config,
) -> None:
    with temporary_mcp_workspace() as workspace:
        workspace.seed_market(market_frame())
        approval = ObservedPlanUser(workspace, decisions=("revise", "approve"))
        calls, revision_id = asyncio.run(
            _model_workflow(workspace, pytestconfig, "optimization", approval)
        )
        _assert_evidence(
            calls, "optimization", approval, selected_revision_id=revision_id
        )
        assert len(approval.turns) == 3
        for plan, expected_fees in zip(approval.plans, (0, 0.001), strict=True):
            assert isinstance(plan.manifest, ExecutionManifest)
            assert plan.manifest.config_copy()["backtest"]["fees"] == expected_fees


@pytest.mark.live_llm
def test_llm_clarifies_ambiguous_reply_then_honors_decline(
    pytestconfig: pytest.Config,
) -> None:
    with temporary_mcp_workspace() as workspace:
        workspace.seed_market(market_frame())
        user = ObservedPlanUser(workspace, decisions=("ambiguous", "decline"))
        calls, _revision_id = asyncio.run(
            _model_workflow(workspace, pytestconfig, "optimization", user)
        )
        user.assert_complete()
        assert len(user.turns) == 3 and len(user.plans) == 1
        submitted = [call for call in calls if call.name == "submit_execution_decision"]
        assert len(submitted) == 1
        assert submitted[0].arguments == {
            "plan_id": user.plans[0].plan_id,
            "decision": "decline",
        }
        assert submitted[0].result["code"] == "confirmation_declined"
