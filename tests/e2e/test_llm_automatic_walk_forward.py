"""Natural automatic walk-forward intent reaches the final cached observation."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pandas as pd
import pytest

from tests.e2e.codex_harness import run_codex
from tests.e2e.conversation_user import ObservedPlanUser
from tests.e2e.llm_client import ToolCall, codex_calls, run_local_model
from tests.e2e.strategy_scenarios import market_frame
from tests.e2e.test_llm_execution_plans import _WF_STRATEGY, _seed_walk_forward
from tests.integration.mcp_harness import temporary_mcp_workspace
from tradingdev.domain.execution import ExecutionManifest


@pytest.mark.live_llm
def test_natural_automatic_walk_forward_covers_final_observation(
    pytestconfig: pytest.Config,
) -> None:
    with temporary_mcp_workspace() as workspace:
        workspace.seed_market(market_frame())
        user = ObservedPlanUser(workspace)

        async def run() -> list[ToolCall]:
            options: dict[str, Any] = {
                "model": pytestconfig.getoption("llm_model"),
                "timeout_seconds": pytestconfig.getoption("llm_timeout"),
                "user_reply": user,
            }
            async with workspace.connect() as client:
                revision = await _seed_walk_forward(
                    client,
                    validation={
                        "mode": "automatic",
                        "n_splits": 2,
                        "train_ratio": 0.8,
                        "expanding": False,
                    },
                )
                prompt = (
                    f"請沿用既有 SMA 研究 {_WF_STRATEGY} 的版本 {revision}，"
                    "依它保存的設定執行自動 walk-forward，維持兩段、訓練比例 0.8"
                    "及固定長度滾動訓練。標的、期間、成本與策略參數全部沿用，"
                    "不要建立或修訂策略。本機已有這段期間的合成測試行情。"
                    "請先讓我檢視執行設定，等我同意才執行。完成後查詢最後一段"
                    "樣本外測試的起訖時間與績效，說明是否涵蓋研究期間最後一根資料，"
                    "並提供這次結果的識別碼與可開啟的完整報告。"
                )
                names = [
                    tool.name for tool in (await client.session.list_tools()).tools
                ]
                assert all(name not in prompt for name in names)
                options["tool_names"] = names
                if pytestconfig.getoption("llm_provider") == "local":
                    return await run_local_model(
                        client,
                        prompt,
                        base_url=pytestconfig.getoption("llm_base_url"),
                        temperature=pytestconfig.getoption("llm_temperature"),
                        reasoning_effort=pytestconfig.getoption("llm_reasoning_effort"),
                        **options,
                    )
            assert pytestconfig.getoption("llm_provider") == "codex"
            return codex_calls(await run_codex(workspace.root, prompt, **options))

        calls = asyncio.run(run())
        user.assert_complete()
        assert len(user.plans) == 1
        manifest = user.plans[0].manifest
        assert isinstance(manifest, ExecutionManifest)
        validation = manifest.config_copy()["validation"]
        assert validation["mode"] == "automatic"
        assert validation["n_splits"] == 2
        assert validation["train_ratio"] == 0.8
        assert validation["expanding"] is False
        assert not any(call.name == "save_strategy" for call in calls)
        details = [
            call.result
            for call in calls
            if call.name == "get_run_metrics"
            and isinstance(call.result, dict)
            and call.result.get("scope") == "fold/1/test"
        ]
        assert details, "The model must query the final test fold"
        detail = details[-1]
        run_id = detail["run_id"]
        observations = detail["metadata"]["observations"]
        assert observations["bar_count"] == 29
        assert pd.Timestamp(observations["start_date"]) == pd.Timestamp(
            "2024-01-06T20:00:00Z"
        )
        assert pd.Timestamp(observations["end_date"]) == pd.Timestamp(
            "2024-01-08T00:00:00Z"
        )
        saved = json.loads(
            (workspace.workspace / "runs" / run_id / "performance.json").read_text()
        )
        assert (
            saved["scopes"]["fold/1/test"]["metadata"]["observations"] == observations
        )
        assert run_id in user.turns[-1].text
        from pathlib import Path

        from tests.e2e.conversation import links_to_file

        reports = [
            call.result
            for call in calls
            if call.name == "generate_report" and call.result.get("success")
        ]
        assert reports
        report = Path(reports[-1]["path"])
        assert report.is_file() and report.read_text()
        assert links_to_file(user.turns[-1].text, report)
        print(
            "[automatic-walk-forward] Final model reply:\n" + user.turns[-1].text,
            flush=True,
        )
