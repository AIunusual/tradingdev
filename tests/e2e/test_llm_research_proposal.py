"""An ordinary vague strategy request discovers unsourced parameter proposals."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.e2e.comparison_diagnostics import comparison_failure_diagnostics
from tests.e2e.conversation_user import ObservedPlanUser, reviewed_config
from tests.e2e.research_evidence import assert_saved_research
from tests.e2e.strategy_scenarios import market_frame
from tests.e2e.test_llm_comparison import _run
from tests.e2e.workflow_arguments import effective_workflow_calls
from tests.integration.mcp_harness import temporary_mcp_workspace
from tests.integration.test_comparison_workflow import verify_comparison_delivery
from tradingdev.domain.comparison import ComparisonManifest

if TYPE_CHECKING:
    from tests.e2e.conversation import ModelTurn
    from tests.e2e.llm_client import ToolCall

REQUEST = (
    "幫我測試布林通道策略，價格低的時候買入，回到平均附近就賣出，先只做多。"
    "標的是 BTC/USDT 永續合約，一小時一根 K 棒，"
    "期間是 UTC 2024 年 1 月 1 日 00:00 到 1 月 8 日 00:00。"
    "我不懂指標參數，請建議幾組設定讓我確認。"
)


class ProposalUser(ObservedPlanUser):
    """Consent follows the actual saved proposal; the request supplies no tool plan."""

    async def __call__(self, turn: ModelTurn) -> str | None:
        if not self.replies:
            assert not any(
                call.name
                in {
                    "ensure_data",
                    "train_model",
                    "submit_execution_decision",
                    "prepare_optimization",
                }
                for call in turn.calls
            ), "No real-data execution or automatic search before consent"
        reply = await super().__call__(turn)
        if len(self.turns) == 1:
            plan = self.plans[-1]
            assert isinstance(plan.manifest, ComparisonManifest)
            assert 2 <= len(plan.manifest.candidates) <= 8
            assert all(
                candidate.manifest.strategy_execution.kind == "generated"
                for candidate in plan.manifest.candidates
            )
            config = reviewed_config(plan)
            assert config["backtest"]["symbol"] == "BTC/USDT"
            assert config["backtest"]["timeframe"] == "1h"
            assert config["backtest"]["init_cash"] == 10000
            assert config["backtest"]["fees"] == 0.0006
            assert config["backtest"]["slippage"] == 0.0005
            calls = effective_workflow_calls(turn.calls)
            first_save = next(
                index
                for index, call in enumerate(calls)
                if call.name == "save_strategy"
            )
            assert any(
                call.name == "get_strategy_contract"
                and call.arguments.get("section") == "indicators"
                and call.arguments.get("indicator") == "bollinger_bands"
                and call.result.get("content", {}).get("selected") is not None
                for call in calls[:first_save]
            ), "Discover the installed Bollinger API before authoring"
            prepare_index = next(
                index
                for index, call in enumerate(calls)
                if call.name == "prepare_comparison" and call.result.get("success")
            )
            draft = assert_saved_research(calls, prepare_index)
            assert draft == plan.research_evidence.research.record.draft
            parameters = plan.research_evidence.research.parameters
            variants: dict[str, set[str]] = {}
            for item in parameters:
                variants.setdefault(item.parameter_path, set()).add(
                    json.dumps(item.value, sort_keys=True)
                )
            varied = {path for path, values in variants.items() if len(values) > 1}
            assert varied, "Proposals must actually vary an unspecified parameter"
            assert all(
                item.provenance == "model_proposal"
                for item in parameters
                if item.parameter_path in varied
            )
        return reply


@pytest.mark.live_llm
def test_natural_unspecified_parameters_with_no_web(
    pytestconfig: pytest.Config,
) -> None:
    with temporary_mcp_workspace() as workspace:
        workspace.seed_market(market_frame())
        user = ProposalUser(workspace)
        calls: list[ToolCall] = []
        try:
            asyncio.run(_run(workspace, pytestconfig, user, calls, request=REQUEST))
            verify_comparison_delivery(workspace, user, calls)
            plan = user.plans[-1]
            assert isinstance(plan.manifest, ComparisonManifest)
            report = next(
                call
                for call in reversed(calls)
                if call.name == "generate_report" and call.result.get("success")
            )
            html = Path(report.result["path"]).read_text()
            payload = json.loads(
                html.split("<script id='report-data' type='application/json'>", 1)[
                    1
                ].split("</script>", 1)[0]
            )
            snapshot = plan.research_evidence
            assert payload["research_documents"][snapshot.content_hash]["draft"] == (
                snapshot.research.record.draft.model_dump(mode="json")
            )
            assert all(
                run["research_evidence_hash"] == snapshot.content_hash
                for run in payload["runs"]
            )
            source_path = plan.manifest.candidates[0].manifest.config_copy()[
                "strategy"
            ]["source_path"]
            print(
                "Proposal source for semantic review:\n" + Path(source_path).read_text()
            )
            print(
                "Saved proposal for semantic review:\n"
                + snapshot.model_dump_json(indent=2)
            )
        except BaseException as exc:
            exc.add_note(
                json.dumps(comparison_failure_diagnostics(calls), ensure_ascii=False)
            )
            raise
        finally:
            for turn in user.turns:
                print(f"Completed proposal turn {turn.number}:\n{turn.text}")
