"""Natural research intake without hidden choices or a tool-by-tool answer script."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tests.e2e.conversation_user import assert_no_execution, read_presented_plan
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.domain.execution import ExecutionManifest

if TYPE_CHECKING:
    from tests.e2e.conversation import ModelTurn
    from tests.integration.mcp_harness import MCPWorkspace
    from tradingdev.domain.execution_plan import ExecutionPlan


MISSING_REQUESTS = {
    "strategy_idea": (
        "我想回測 BTC/USDT 永續合約，用一小時 K 棒，"
        "期間是 UTC 2024 年 1 月 1 日到 1 月 8 日。"
    ),
    "instrument": (
        "用 10 根和 30 根均線，短均線比較高時做多，比較低時平倉。"
        "一小時 K 棒，期間是 UTC 2024 年 1 月 1 日到 1 月 8 日。"
    ),
    "interval": (
        "幫我測 BTC/USDT 永續合約，用一小時 K 棒，"
        "10 根均線比 30 根高時做多，比它低時平倉。"
    ),
    "timeframe": (
        "幫我測 BTC/USDT 永續合約，10 根均線比 30 根高時做多，"
        "比它低時平倉。期間是 UTC 2024 年 1 月 1 日到 1 月 8 日。"
    ),
}

INTAKE_REQUEST = (
    "幫我測 BTC/USDT 永續合約，用兩條均線，10 根平均比 30 根高時買入，"
    "比較低或相等時賣掉，只做多。這次先測一組。"
)
INTAKE_REPLY = (
    "用一小時 K 棒，期間就用 UTC 2024 年 1 月 1 日 00:00 到 1 月 8 日 00:00。"
)


def assert_missing_input_turn(
    workspace: MCPWorkspace, turn: ModelTurn, *, read_only_tools: set[str]
) -> None:
    """Check attempts and effects; the actual question still needs semantic review."""
    assert turn.text.strip(), "The user must receive a visible clarification"
    attempted = [call.name for call in turn.calls if call.name not in read_only_tools]
    assert not attempted, (
        "Missing research choices must be clarified before any mutating tool attempt, "
        f"including rejected attempts: {attempted}"
    )
    assert_no_execution(workspace)
    paths = WorkspacePaths(workspace.workspace)
    for directory in (
        paths.generated_strategies,
        paths.configs,
        paths.runs,
        paths.models,
        paths.ml_datasets,
        paths.ml_feature_cache,
        paths.feature_requests,
        paths.root / "execution_plans",
    ):
        assert not any(path.is_file() for path in directory.rglob("*")), (
            f"Research artifacts appeared before the missing choices: {directory}"
        )
    database = paths.root / "tradingdev.sqlite"
    with closing(
        sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
    ) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM execution_plans").fetchone()[0]
            == 0
        )


@dataclass
class MissingInputUser:
    """Stop after one clarification and preserve its text for human review."""

    workspace: MCPWorkspace
    read_only_tools: set[str]
    turns: list[ModelTurn] = field(default_factory=list)

    async def __call__(self, turn: ModelTurn) -> None:
        self.turns.append(turn)
        assert len(self.turns) == 1 and turn.number == 1
        assert_missing_input_turn(
            self.workspace, turn, read_only_tools=self.read_only_tools
        )


@dataclass
class ObservedIntakeUser:
    """Supply only missing choices, then approve the actual published proposal."""

    workspace: MCPWorkspace
    read_only_tools: set[str]
    turns: list[ModelTurn] = field(default_factory=list)
    plan: ExecutionPlan | None = None
    replies: list[str] = field(default_factory=list)

    async def __call__(self, turn: ModelTurn) -> str | None:
        assert turn.number == len(self.turns) + 1
        self.turns.append(turn)
        print(f"[intake] Model turn {turn.number}:\n{turn.text}", flush=True)
        if len(self.turns) == 1:
            assert_missing_input_turn(
                self.workspace, turn, read_only_tools=self.read_only_tools
            )
            self.replies.append(INTAKE_REPLY)
            return INTAKE_REPLY
        decisions = [
            call for call in turn.calls if call.name == "submit_execution_decision"
        ]
        if self.plan is not None:
            assert len(decisions) == 1
            assert decisions[0].arguments == {
                "plan_id": self.plan.plan_id,
                "decision": "approve",
            }
            assert decisions[0].result.get("success") is True
            return None
        assert not decisions, "No approval exists before the actual plan is reviewed"
        assert_no_execution(self.workspace)
        assert not any(
            call.name in {"ensure_data", "train_model"} for call in turn.calls
        ), "Preparing the proposal must not download history or train a model"
        presented = any(
            call.name.startswith("prepare_") and call.result.get("success") is True
            for call in turn.calls
        )
        if not presented:
            assert len(self.replies) == 1, (
                "The intake did not produce a reviewable plan"
            )
            reply = "我沒有其他設定偏好，請先依你建議的模擬設定提出具體方案，讓我確認。"
            self.replies.append(reply)
            return reply
        self.plan = read_presented_plan(self.workspace, turn)
        assert isinstance(self.plan.manifest, ExecutionManifest)
        backtest = self.plan.manifest.config_copy()["backtest"]
        assert backtest["symbol"] == "BTC/USDT"
        assert backtest["timeframe"] == "1h"
        assert datetime.fromisoformat(backtest["start_date"]) == datetime(
            2024, 1, 1, tzinfo=UTC
        )
        assert datetime.fromisoformat(backtest["end_date"]) == datetime(
            2024, 1, 8, tzinfo=UTC
        )
        assert backtest["init_cash"] == 10000.0
        assert backtest["fees"] == 0.0006
        assert backtest["slippage"] == 0.0005
        assert backtest["periods_per_year"] == 365.0
        reply = "我已閱讀這份確認單，同意按照這份方案正式回測。"
        self.replies.append(reply)
        return reply

    def assert_complete(self) -> None:
        assert self.plan is not None, "No published plan was reviewed"
        assert len(self.turns) == len(self.replies) + 1
