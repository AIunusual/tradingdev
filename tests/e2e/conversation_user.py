"""A simulated user reads a saved plan only after the model ends its turn."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from tests.e2e.conversation import links_to_file, visible_text
from tradingdev.domain.comparison import ComparisonManifest
from tradingdev.domain.execution_plan import ExecutionPlan
from tradingdev.domain.preflight import ComparisonPreflightReceipt

if TYPE_CHECKING:
    from tests.e2e.conversation import ModelTurn
    from tests.e2e.llm_client import ToolCall
    from tests.integration.mcp_harness import MCPWorkspace

type TestDecision = Literal["approve", "decline", "revise", "ambiguous"]


def _content_key(text: str) -> str:
    """Compare saved wording across whitespace, emphasis and table formatting."""
    return re.sub(r"[\W_]", "", text)


def _assert_summary(summary: str, text: str) -> None:
    expected = visible_text(summary)
    assert expected and expected in visible_text(text), (
        "The completed turn must preserve the backend summary wording.\n"
        f"Expected backend summary (complete):\n{summary}"
    )


def _assert_short_review(text: str, html_path: Path, plan: ExecutionPlan) -> None:
    """Reject the actual full document/settings without an arbitrary length limit."""
    content = _content_key(text)
    full_text = html_path.with_name("confirmation.txt").read_text(encoding="utf-8")
    assert _content_key(full_text) not in content, (
        "Keep the full confirmation document in HTML instead of repeating it in chat"
    )
    fields = [
        _content_key(field.label + field.value_text)
        for section in plan.document.sections
        for field in section.fields
    ]
    assert fields and not all(field in content for field in fields), (
        "Present the short summary, not the complete confirmation settings"
    )


def _database(workspace: MCPWorkspace) -> sqlite3.Connection:
    database = workspace.workspace / "tradingdev.sqlite"
    return sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)


def assert_no_execution(workspace: MCPWorkspace) -> None:
    """Verify no formal job or result exists before the simulated user agrees."""
    with closing(_database(workspace)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert not list(workspace.workspace.glob("runs/*/manifest.json"))


def reviewed_config(plan: ExecutionPlan) -> dict[str, Any]:
    """Read shared research without mistaking the first candidate for the bundle."""
    return (
        plan.manifest.common_config()
        if isinstance(plan.manifest, ComparisonManifest)
        else plan.manifest.config_copy()
    )


def read_presented_plan(workspace: MCPWorkspace, turn: ModelTurn) -> ExecutionPlan:
    """Verify the plan the user can read, without making a consent decision."""
    preparations = [
        call
        for call in turn.calls
        if call.name.startswith("prepare_")
        and isinstance(call.result, dict)
        and call.result.get("success") is True
    ]
    assert len(preparations) == 1, "Present exactly one newly prepared plan"
    prepared = preparations[0].result
    assert prepared["status"] == "ready" and not prepared.get("job_id")
    assert "confirmation_text" not in prepared
    summary = prepared["confirmation_summary"]
    assert isinstance(summary, str) and summary.strip()

    _assert_summary(summary, turn.text)
    html_path = Path(prepared["html_path"])
    assert links_to_file(turn.text, html_path), (
        "The user must receive a clickable Markdown link to this plan HTML.\n"
        f"Expected plan HTML path: {html_path}"
    )
    with closing(_database(workspace)) as conn:
        rows = conn.execute(
            "SELECT payload, html_hash FROM execution_plans WHERE state='ready'"
        ).fetchall()
    assert len(rows) == 1, "Exactly one saved plan must await this user"
    plan = ExecutionPlan.model_validate_json(rows[0][0])
    assert prepared["plan_id"] == plan.plan_id
    assert prepared["manifest_hash"] == plan.manifest.manifest_hash
    assert prepared["preflight"] == plan.preflight.model_dump(mode="json"), (
        "The shown offline checks must match this saved plan"
    )
    receipts = (
        [candidate.receipt for candidate in plan.preflight.candidates]
        if isinstance(plan.preflight, ComparisonPreflightReceipt)
        else [plan.preflight]
    )
    for receipt in receipts:
        assert set(receipt.not_checked) == {
            "historical_data",
            "backtest_engine",
            "training",
            "performance",
        }
    expected_path = (
        workspace.workspace / "execution_plans" / plan.plan_id / "confirmation.html"
    )
    assert html_path == expected_path, f"Expected saved plan HTML path: {expected_path}"
    html = html_path.read_bytes()
    assert hashlib.sha256(html).hexdigest() == rows[0][1]
    assert (
        plan.plan_id in html.decode() and plan.manifest.manifest_hash in html.decode()
    )
    _assert_short_review(turn.text, html_path, plan)
    return plan


@dataclass
class ObservedPlanUser:
    """Review a new execution in a workspace without previous jobs or runs.

    History-only conversations and fixtures that already executed a job must not
    use this observer: its first review proves that no execution has started.
    """

    workspace: MCPWorkspace
    decisions: tuple[TestDecision, ...] = ("approve",)
    revised_fees: float = 0.001
    turns: list[ModelTurn] = field(default_factory=list)
    plans: list[ExecutionPlan] = field(default_factory=list)
    replies: list[str] = field(default_factory=list)

    def _database(self) -> sqlite3.Connection:
        return _database(self.workspace)

    def _no_execution(self) -> None:
        assert_no_execution(self.workspace)

    def _cancelled(self, plan_id: str) -> None:
        with closing(self._database()) as conn:
            state = conn.execute(
                "SELECT state FROM execution_plans WHERE plan_id=?", (plan_id,)
            ).fetchone()
        assert state == ("cancelled",), "The previous plan must be cancelled"

    def _read_plan(self, turn: ModelTurn) -> ExecutionPlan:
        plan = read_presented_plan(self.workspace, turn)
        self.plans.append(plan)
        return plan

    @staticmethod
    def _decisions(turn: ModelTurn) -> list[ToolCall]:
        return [call for call in turn.calls if call.name == "submit_execution_decision"]

    async def __call__(self, turn: ModelTurn) -> str | None:
        try:
            return await self._reply(turn)
        except AssertionError as exc:
            exc.add_note(f"Completed model turn {turn.number} (complete):\n{turn.text}")
            raise

    async def _reply(self, turn: ModelTurn) -> str | None:
        assert turn.number == len(self.turns) + 1, (
            "Conversation turns must be consecutive"
        )
        self.turns.append(turn)
        prior = self.decisions[len(self.replies) - 1] if self.replies else None
        submissions = self._decisions(turn)
        if prior in {"approve", "decline", "revise"}:
            assert len(submissions) == 1, "Relay exactly the user's previous decision"
            assert submissions[0].arguments == {
                "plan_id": self.plans[-1].plan_id,
                "decision": prior,
            }
            if prior == "approve":
                assert submissions[0].result.get("success") is True
                return None
            assert submissions[0].result.get("success") is False
            assert submissions[0].result.get("code") == (
                "confirmation_declined"
                if prior == "decline"
                else "confirmation_adjustment_requested"
            )
            self._cancelled(self.plans[-1].plan_id)
        else:
            assert not submissions, "No explicit user decision exists in this turn"
        self._no_execution()
        if prior == "decline":
            return None
        if prior != "ambiguous":
            plan = self._read_plan(turn)
            if prior == "revise":
                assert plan.plan_id != self.plans[-2].plan_id
                assert reviewed_config(plan)["backtest"]["fees"] == self.revised_fees
        else:
            assert not any(call.name.startswith("prepare_") for call in turn.calls), (
                "An ambiguous reply must not silently replace the reviewed plan"
            )
            assert turn.text.strip(), "Ask the undecided user for an explicit decision"
        assert len(self.replies) < len(self.decisions), "Unexpected extra model turn"
        decision = self.decisions[len(self.replies)]
        replies = {
            "approve": (
                "我已閱讀這份 HTML 確認單，同意依這份設定正式執行；"
                "完成後查詢原先要求的結果。"
            ),
            "decline": "我不同意執行這份計畫，請取消，這次不要啟動任何工作。",
            "revise": (
                f"請把單邊手續費改成成交金額的 {self.revised_fees * 100:g}%。"
                "現在不要執行；取消原計畫，重新準備並提供新 HTML 與後端短摘要，"
                "然後結束回合等待我確認。不要修改策略 revision。"
            ),
            "ambiguous": "看起來大致可以，但我還在考慮。請再說明手續費的影響。",
        }
        reply = replies[decision]
        self.replies.append(reply)
        return reply

    def assert_complete(self) -> None:
        try:
            assert len(self.replies) == len(self.decisions)
            assert len(self.turns) == len(self.replies) + 1
        except AssertionError as exc:
            if self.turns:
                turn = self.turns[-1]
                exc.add_note(
                    f"Completed model turn {turn.number} (complete):\n{turn.text}"
                )
            else:
                exc.add_note("No completed model turn was observed.")
            raise


def review_text(prepared: dict[str, Any]) -> str:
    """Supply a deterministic presentation only for offline fixture verification."""
    return f"{prepared['confirmation_summary']}\n{prepared['confirmation_link']}"
