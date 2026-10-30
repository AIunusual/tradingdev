"""Opt-in product discovery with natural intent and the complete tool surface."""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from tests.data_cache import seed_market_cache
from tests.e2e.codex_harness import run_codex, verify_generated_strategy
from tests.e2e.conversation_user import ObservedPlanUser
from tests.e2e.discovery_workflow import (
    NATURAL_REQUEST,
    verify_discovery_outcome,
    verify_dvol_query_reply,
    verify_market_cache_reply,
)
from tests.e2e.llm_client import ToolCall, codex_calls, run_local_model
from tests.e2e.strategy_scenarios import SCENARIOS, market_frame
from tests.e2e.workflow_diagnostics import workflow_diagnostics
from tests.integration.mcp_harness import temporary_mcp_workspace
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.job_store import JobStore

pytestmark = pytest.mark.live_llm

if TYPE_CHECKING:
    from tests.e2e.conversation import ModelTurn


def test_natural_request_discovers_strategy_and_report(
    pytestconfig: pytest.Config,
) -> None:
    provider = pytestconfig.getoption("llm_provider")
    assert provider in {"codex", "local"}
    with (
        temporary_mcp_workspace() as workspace,
        workflow_diagnostics(workspace, SCENARIOS["sma"]),
    ):
        workspace.seed_market(market_frame())
        user = ObservedPlanUser(workspace)

        async def run() -> list[ToolCall]:
            options: dict[str, Any] = {
                "timeout_seconds": pytestconfig.getoption("llm_timeout"),
                "model": pytestconfig.getoption("llm_model"),
                "user_reply": user,
            }
            async with workspace.connect() as client:
                tools = (await client.session.list_tools()).tools
                names = [tool.name for tool in tools]
                assert all(name not in NATURAL_REQUEST for name in names), (
                    "Natural user intent must not supply tool names"
                )
                print(
                    f"[discovery] Complete product tool set: {len(names)}", flush=True
                )
                options["tool_names"] = names
                if provider == "local":
                    return await run_local_model(
                        client,
                        NATURAL_REQUEST,
                        base_url=pytestconfig.getoption("llm_base_url"),
                        temperature=pytestconfig.getoption("llm_temperature"),
                        reasoning_effort=pytestconfig.getoption("llm_reasoning_effort"),
                        **options,
                    )
            return codex_calls(
                await run_codex(workspace.root, NATURAL_REQUEST, **options)
            )

        calls = asyncio.run(run())
        user.assert_complete()
        verify_discovery_outcome(workspace, calls, user.turns[-1])
        print(
            "[discovery] Final model reply (test evidence; links expire at cleanup):\n"
            + user.turns[-1].text,
            flush=True,
        )
        verify_generated_strategy(workspace.root, scenario="sma")


def test_natural_request_reads_historical_job_duration(
    pytestconfig: pytest.Config,
) -> None:
    with temporary_mcp_workspace() as workspace:
        store = JobStore(workspace=WorkspacePaths(workspace.workspace))
        for job_id, end in (
            ("timed-job", "2024-01-01T00:00:10+00:00"),
            ("legacy-job", None),
        ):
            record = store.create_job(job_id=job_id)
            store.store.upsert_job(
                {
                    **record,
                    "status": "failed",
                    "created_at": "2024-01-01T00:00:00+00:00",
                    "ended_at": end,
                    "error": "Historical fixture failure",
                }
            )
        request = (
            "請查我先前的 timed-job 和 legacy-job，各自失敗前經過幾秒？"
            "請每個工作各用一行回答。"
            "請區分有證據的耗時與無法判定的部分，缺少資料不要猜測。只查歷史，不要重新執行工作。"
        )
        turns: list[ModelTurn] = []

        async def user(turn: ModelTurn) -> None:
            turns.append(turn)

        async def run() -> list[ToolCall]:
            options: dict[str, Any] = {
                "timeout_seconds": pytestconfig.getoption("llm_timeout"),
                "model": pytestconfig.getoption("llm_model"),
                "user_reply": user,
            }
            async with workspace.connect() as client:
                names = [
                    tool.name for tool in (await client.session.list_tools()).tools
                ]
                assert all(name not in request for name in names)
                options["tool_names"] = names
                if pytestconfig.getoption("llm_provider") == "local":
                    return await run_local_model(
                        client,
                        request,
                        base_url=pytestconfig.getoption("llm_base_url"),
                        temperature=pytestconfig.getoption("llm_temperature"),
                        reasoning_effort=pytestconfig.getoption("llm_reasoning_effort"),
                        **options,
                    )
            return codex_calls(await run_codex(workspace.root, request, **options))

        calls = asyncio.run(run())
        records: dict[str, dict[str, Any]] = {}
        for call in calls:
            assert call.name in {"list_jobs", "get_job_status"}, (
                "History request must not execute work"
            )
            if call.name == "list_jobs":
                records.update(
                    {row["job_id"]: row for row in call.result.get("items", [])}
                )
            else:
                records[str(call.arguments["job_id"])] = call.result
        assert records["timed-job"]["elapsed_seconds"] == 10.0
        assert records["legacy-job"]["elapsed_seconds"] is None
        assert (
            records["legacy-job"]["elapsed_unavailable_reason"]
            == "missing_or_invalid_timestamps"
        )
        assert turns
        known_lines = " ".join(
            line for line in turns[-1].text.splitlines() if "timed-job" in line
        )
        assert re.search(r"(?<!\d)10(?:\.0)?(?!\d)", known_lines)
        unknown_lines = " ".join(
            line for line in turns[-1].text.splitlines() if "legacy-job" in line
        )
        assert re.search(
            "無法|未知|缺|不足|不明|不能|unknown|unavailable",
            unknown_lines,
            re.IGNORECASE,
        )
        print("[historical-duration] Final model reply:\n" + turns[-1].text, flush=True)


@pytest.mark.parametrize("settled", [True, False], ids=["settled", "early-query"])
def test_natural_request_inspects_separate_market_caches(
    pytestconfig: pytest.Config,
    settled: bool,
) -> None:
    with temporary_mcp_workspace() as workspace:
        paths = []
        for market, count in (("spot", 3), ("futures/um", 5)):
            cached = seed_market_cache(
                workspace.data_root / "processed",
                market_frame().head(count),
                market_type=market,
            )
            if not settled:
                # A query acquired before settlement does not become reusable
                # merely because time passes. Keep real files and the same bars.
                receipt_path = cached.with_name("2024.json")
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                receipt.update(
                    query_end="2024-01-02T00:00:00Z",
                    fetched_at="2024-01-03T00:00:00Z",
                )
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            path = workspace.root / f"{market.replace('/', '-')}.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "data": {"source": "binance_vision", "market_type": market},
                        "backtest": {
                            "symbol": "BTC/USDT",
                            "timeframe": "1h",
                            "start_date": "2024-01-01",
                            "end_date": "2024-01-02",
                            "init_cash": 10000,
                        },
                    }
                ),
                encoding="utf-8",
            )
            paths.append(str(path))
        request = (
            f"我有兩份研究設定：{paths[0]} 與 {paths[1]}。"
            "請只查本機行情，分別說明來源、市場、實際保存幾筆，以及要求期間的查詢紀錄。"
            "也請分別判斷現有快取是否能直接重用，並說明判斷理由。"
            "請每個市場各用一行，明確標示『可直接重用：是或否』與"
            "『須重新取得才符合重用政策：是或否』，筆數以『N筆』表示，再給理由。"
            "若理由接在決策同一行，請使用『理由：…』欄位。"
            "並說明查詢成功是否足以證明行情完整。"
            "不要下載、改資料或執行研究。"
        )
        turns: list[ModelTurn] = []

        async def user(turn: ModelTurn) -> None:
            turns.append(turn)

        async def run() -> list[ToolCall]:
            options: dict[str, Any] = {
                "timeout_seconds": pytestconfig.getoption("llm_timeout"),
                "model": pytestconfig.getoption("llm_model"),
                "user_reply": user,
            }
            async with workspace.connect() as client:
                names = [
                    tool.name for tool in (await client.session.list_tools()).tools
                ]
                assert all(name not in request for name in names)
                options["tool_names"] = names
                if pytestconfig.getoption("llm_provider") == "local":
                    return await run_local_model(
                        client,
                        request,
                        base_url=pytestconfig.getoption("llm_base_url"),
                        temperature=pytestconfig.getoption("llm_temperature"),
                        reasoning_effort=pytestconfig.getoption("llm_reasoning_effort"),
                        **options,
                    )
            return codex_calls(await run_codex(workspace.root, request, **options))

        calls = asyncio.run(run())
        assert turns
        reply = turns[-1].text
        print("[market-cache-inspection] Final model reply:\n" + reply, flush=True)
        observed = {}
        for call in calls:
            assert call.name in {
                "inspect_dataset",
                "list_available_data",
                "list_data_sources",
                "get_strategy_contract",
                "get_ml_contract",
            }
            if call.name == "inspect_dataset" and call.result.get("market"):
                market = call.result["market"]
                assert market["source"] == "binance_vision"
                assert market["queries"]
                assert call.result["market_available"] is settled
                observed[market["market_type"]] = market["rows"]
        assert observed == {"spot": 3, "futures/um": 5}
        verify_market_cache_reply(reply, settled=settled)


def test_natural_request_distinguishes_queried_and_unqueried_dvol(
    pytestconfig: pytest.Config,
) -> None:
    from datetime import UTC, datetime

    import pandas as pd

    from tradingdev.app.dvol_data import DVOLData
    from tradingdev.domain.data.schemas import DataConfig, MarketDataRequest

    with temporary_mcp_workspace() as workspace:
        workspace.seed_market(market_frame())
        request = MarketDataRequest(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-02",
        )
        managed = DVOLData(
            DataConfig(
                raw_dir=str(workspace.data_root / "raw"),
                processed_dir=str(workspace.data_root / "processed"),
            ),
            request,
        )
        feature = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
                "dvol_close": [10.0, 20.0],
            }
        )
        with managed.cache.lock(managed.key):
            managed.cache.publish(
                managed.key,
                raw=feature,
                frame=feature,
                query_start=request.start_date,
                query_end=request.end_date,
                fetched_at=datetime(2024, 1, 10, tzinfo=UTC),
                save_raw=lambda frame, path: frame.to_csv(path, index=False),
            )
        paths = []
        for label, end in (("short", "2024-01-02"), ("extended", "2024-01-03")):
            path = workspace.root / f"{label}.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "backtest": {
                            "symbol": "BTC/USDT",
                            "timeframe": "1h",
                            "start_date": "2024-01-01",
                            "end_date": end,
                            "init_cash": 10000,
                        },
                        "data": {
                            "requirements": {
                                "market": {"symbol": "BTC/USDT", "timeframe": "1h"},
                                "features": [
                                    {
                                        "type": "dvol",
                                        "source": "deribit",
                                        "column": "dvol",
                                    }
                                ],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            paths.append(str(path))
        prompt = (
            f"請比較這兩份設定的 DVOL：{paths[0]} 與 {paths[1]}。"
            "各自是否有對應期間的成功查詢紀錄、實存幾筆、可否重用？"
            "請每份設定各用一行，列出『查詢紀錄：有或無』『DVOL筆數：整數或未知』"
            "及『可直接重用：是或否』三欄，再說明理由。"
            "若理由接在決策同一行，請使用『理由：…』欄位。"
            "請區分未取得與零筆資料，不要用舊值猜測新增期間。"
            "只檢查本機，不下載、不執行研究。"
        )
        turns: list[ModelTurn] = []

        async def user(turn: ModelTurn) -> None:
            turns.append(turn)

        async def run() -> list[ToolCall]:
            options: dict[str, Any] = {
                "timeout_seconds": pytestconfig.getoption("llm_timeout"),
                "model": pytestconfig.getoption("llm_model"),
                "user_reply": user,
            }
            async with workspace.connect() as client:
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
            return codex_calls(await run_codex(workspace.root, prompt, **options))

        calls = asyncio.run(run())
        assert turns
        reply = turns[-1].text
        print("[dvol-query-inspection] Final model reply:\n" + reply, flush=True)
        observations = {}
        for call in calls:
            assert call.name in {
                "inspect_dataset",
                "list_available_data",
                "list_data_sources",
                "get_strategy_contract",
                "get_ml_contract",
            }
            if (
                call.name == "inspect_dataset"
                and call.arguments.get("config_path") in paths
            ):
                observations[call.arguments["config_path"]] = call.result["features"][0]
        short, extended = (observations[path] for path in paths)
        assert short["query"] and short["query_reusable"] is True and short["rows"] == 2
        assert (
            extended["query"] is None
            and extended["query_reusable"] is False
            and extended["rows"] is None
        )
        verify_dvol_query_reply(reply)
