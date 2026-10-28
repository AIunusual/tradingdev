"""Verify the simulated user's evidence checks without invoking a model or worker."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import pytest

from tests.e2e.conversation import ModelTurn
from tests.e2e.conversation_user import ObservedPlanUser, _assert_summary, review_text
from tests.e2e.llm_client import ToolCall
from tests.integration.mcp_harness import MCPWorkspace
from tests.mcp.execution_fixtures import PlanContext, plan_context
from tradingdev.app.contracts.execution_requests import BacktestInput
from tradingdev.app.contracts.plans import PlanResponse
from tradingdev.app.data_service import DataService
from tradingdev.app.prepared_execution import PreparedExecution
from tradingdev.app.research_preparation import ResearchPreparationService
from tradingdev.app.strategy_service import StrategyService
from tradingdev.domain.preflight import PreflightRequest
from tradingdev.domain.strategies.loader import StrategyLoader

if TYPE_CHECKING:
    from tradingdev.app.contracts.plans import ExecutionDecision


_SUMMARY = (
    "KD 參數最佳化\n"
    "回測 K 線週期：每根 K 線 1 小時\n"
    "手續費率：0.04%；滑價率：0%\n"
    "訓練開始日：2024-01-01；保留測試開始日：2025-01-01"
)


@pytest.mark.parametrize("style", ["bullets", "quote", "emphasis", "line_breaks"])
def test_summary_accepts_visible_original_wording_with_markdown_formatting(
    style: str,
) -> None:
    lines = _SUMMARY.splitlines()
    actual = {
        "bullets": "\n".join("- " + line for line in lines),
        "quote": "\n".join("> " + line for line in lines),
        "emphasis": "\n".join("**" + line + "**" for line in lines),
        "line_breaks": "  \n".join(lines),
    }[style]
    _assert_summary(_SUMMARY, actual)


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ("回測 K 線週期", "K 線週期"),
        ("每根 K 線 1 小時", "每根 1 小時"),
        ("0.04%", "0.4%"),
        ("保留測試開始日", "訓練開始日"),
        ("；滑價率：0%", ""),
        ("手續費率：", "手續費率？"),
    ],
)
def test_summary_rejects_rewritten_wording_values_roles_or_punctuation(
    original: str, replacement: str
) -> None:
    actual = "\n".join(
        "- " + line for line in _SUMMARY.replace(original, replacement).splitlines()
    )
    with pytest.raises(AssertionError, match="preserve the backend summary wording"):
        _assert_summary(_SUMMARY, actual)


@pytest.mark.parametrize("hidden", ["link_target", "html_comment", "hidden_span"])
def test_summary_cannot_be_supplied_only_in_nonvisible_markdown_content(
    hidden: str,
) -> None:
    single_line = " ".join(_SUMMARY.splitlines())
    actual = {
        "link_target": f"[摘要](<{single_line}>)",
        "html_comment": f"<!-- {single_line} -->",
        "hidden_span": f"<span hidden>{single_line}</span>",
    }[hidden]
    with pytest.raises(AssertionError, match="preserve the backend summary wording"):
        _assert_summary(_SUMMARY, actual)


def test_summary_mismatch_preserves_complete_expected_and_actual_diagnostics(
    tmp_path: Path,
) -> None:
    context = plan_context(tmp_path)
    user = ObservedPlanUser(MCPWorkspace(tmp_path))
    prepared = _prepare(context)
    actual = "這是一般說明。" * 80 + "完整回覆的尾端"
    with pytest.raises(AssertionError) as error:
        asyncio.run(user(ModelTurn(1, actual, [prepared])))
    assert prepared.result["confirmation_summary"] in str(error.value)
    assert actual in "\n".join(error.value.__notes__)


def _prepare(context: PlanContext) -> ToolCall:
    arguments = dict(context.arguments)
    evidence_id = arguments.pop("evidence_id")
    result = context.service.prepare(
        PreflightRequest(
            kind="backtest",
            arguments=arguments,
        ),
        evidence_id,
    )
    assert isinstance(result, PlanResponse), result
    return ToolCall(
        "prepare_backtest", context.arguments, result.model_dump(mode="json")
    )


def _present(call: ToolCall, number: int = 1) -> ModelTurn:
    return ModelTurn(number, review_text(call.result), [call])


def test_review_rejects_preflight_reply_that_differs_from_saved_plan(
    tmp_path: Path,
) -> None:
    context = plan_context(tmp_path)
    user = ObservedPlanUser(MCPWorkspace(tmp_path))
    prepared = _prepare(context)
    prepared.result["preflight"]["elapsed_seconds"] = 9.0
    with pytest.raises(AssertionError, match="offline checks must match"):
        asyncio.run(user(_present(prepared)))
    assert context.runner.calls == []


def _decision(
    context: PlanContext, plan_id: str, decision: ExecutionDecision
) -> ToolCall:
    result = context.service.decide(plan_id, decision=decision)
    return ToolCall(
        "submit_execution_decision",
        {"plan_id": plan_id, "decision": decision},
        result.model_dump(mode="json"),
    )


def test_user_reviews_saved_html_then_approves_next_turn(tmp_path: Path) -> None:
    context = plan_context(tmp_path)
    user = ObservedPlanUser(MCPWorkspace(tmp_path))
    prepared = _prepare(context)
    turn = _present(prepared)
    # Natural Markdown emphasis does not erase the backend's factual summary.
    turn = ModelTurn(1, turn.text.replace("策略", "**策略**"), turn.calls)
    assert "同意" in str(asyncio.run(user(turn)))
    assert not context.runner.calls
    submitted = _decision(context, prepared.result["plan_id"], "approve")
    assert asyncio.run(user(ModelTurn(2, "完成", [submitted]))) is None
    user.assert_complete()
    assert len(context.runner.calls) == 1


@pytest.mark.parametrize(
    ("style", "spaces"),
    [
        ("inline", False),
        ("angle", False),
        ("angle", True),
        ("encoded", True),
        ("reference", True),
    ],
)
def test_user_accepts_rendered_links_to_the_saved_artifact(
    tmp_path: Path, style: str, spaces: bool
) -> None:
    root = tmp_path / "directory with spaces" if spaces else tmp_path
    context = plan_context(root)
    user = ObservedPlanUser(MCPWorkspace(root))
    prepared = _prepare(context)
    path = prepared.result["html_path"]
    link = {
        "inline": f"[閱讀確認單]({path})",
        "angle": f"[閱讀確認單](<{path}>)",
        "encoded": f"[閱讀確認單]({quote(path)})",
        "reference": f"[閱讀確認單][plan]\n\n[plan]: <{path}>",
    }[style]
    text = prepared.result["confirmation_summary"] + "\n\n" + link
    assert "同意" in str(asyncio.run(user(ModelTurn(1, text, [prepared]))))
    assert not context.runner.calls


@pytest.mark.parametrize(
    "directory",
    ["with spaces", "繁體中文", "literal%20space", "nested (a(b))", "query?#part"],
)
def test_backend_confirmation_link_is_directly_presentable_for_special_paths(
    tmp_path: Path, directory: str
) -> None:
    root = tmp_path / directory
    context = plan_context(root)
    user = ObservedPlanUser(MCPWorkspace(root))
    prepared = _prepare(context)
    assert prepared.result["confirmation_link"] in review_text(prepared.result)
    assert "同意" in str(asyncio.run(user(_present(prepared))))
    assert not context.runner.calls


@pytest.mark.parametrize(
    "style",
    [
        "bare",
        "inline_code",
        "fenced",
        "tilde_fenced",
        "indented",
        "escaped",
        "image",
        "empty_label",
        "wrong_destination",
        "truncated_plan_id",
        "unescaped_space",
    ],
)
def test_user_requires_a_clickable_markdown_link_to_this_html(
    tmp_path: Path, style: str
) -> None:
    root = (
        tmp_path / "directory with spaces" if style == "unescaped_space" else tmp_path
    )
    context = plan_context(root)
    user = ObservedPlanUser(MCPWorkspace(root))
    prepared = _prepare(context)
    path = prepared.result["html_path"]
    link = f"[閱讀確認單](<{path}>)"
    invalid = {
        "bare": path,
        "inline_code": f"`{link}`",
        "fenced": f"```markdown\n{link}\n```",
        "tilde_fenced": f"~~~markdown\n{link}\n~~~",
        "indented": f"    {link}",
        "escaped": rf"\[閱讀確認單](<{path}>)",
        "image": f"!{link}",
        "empty_label": f"[](<{path}>)",
        "wrong_destination": f"{path}\n[閱讀確認單](</other/confirmation.html>)",
        "truncated_plan_id": link.replace(
            prepared.result["plan_id"], prepared.result["plan_id"][:-1]
        ),
        "unescaped_space": f"[閱讀確認單]({path})",
    }[style]
    text = prepared.result["confirmation_summary"] + "\n\n" + invalid
    with pytest.raises(AssertionError, match="clickable Markdown link") as error:
        asyncio.run(user(ModelTurn(1, text, [prepared])))
    assert text in "\n".join(error.value.__notes__)
    assert f"Expected plan HTML path: {path}" in str(error.value)
    assert not user.replies and not context.runner.calls


@pytest.mark.parametrize("repetition", ["document", "document_code", "settings_table"])
def test_user_rejects_full_confirmation_content_even_with_a_valid_link(
    tmp_path: Path, repetition: str
) -> None:
    context = plan_context(tmp_path)
    user = ObservedPlanUser(MCPWorkspace(tmp_path))
    prepared = _prepare(context)
    full_text = (
        Path(prepared.result["html_path"]).with_name("confirmation.txt").read_text()
    )
    if repetition == "document_code":
        repeated = f"```text\n{full_text}\n```"
    elif repetition == "settings_table":
        # Omitting document headings/descriptions must not permit all settings
        # to be pasted into chat as a reformatted table.
        plan, _record = context.service.plans.get(prepared.result["plan_id"])
        repeated = "| 設定 | 內容 |\n| --- | --- |\n" + "\n".join(
            f"| {field.label} | {field.value_text} |"
            for section in plan.document.sections
            for field in section.fields
        )
    else:
        repeated = full_text
    text = review_text(prepared.result) + "\n\n" + repeated
    with pytest.raises(AssertionError, match="full confirmation|complete confirmation"):
        asyncio.run(user(ModelTurn(1, text, [prepared])))
    assert not user.replies and not context.runner.calls


@pytest.mark.parametrize("omission", ["summary", "html", "tamper"])
def test_user_rejects_missing_or_changed_review_evidence(
    tmp_path: Path, omission: str
) -> None:
    context = plan_context(tmp_path)
    user = ObservedPlanUser(MCPWorkspace(tmp_path))
    prepared = _prepare(context)
    text = review_text(prepared.result)
    if omission == "summary":
        text = prepared.result["html_path"]
    elif omission == "html":
        text = prepared.result["confirmation_summary"]
    else:
        Path(prepared.result["html_path"]).write_text("changed document")
    with pytest.raises(AssertionError):
        asyncio.run(user(ModelTurn(1, text, [prepared])))
    assert not user.replies and not context.runner.calls


def test_user_rejects_execution_in_initial_model_turn(tmp_path: Path) -> None:
    context = plan_context(tmp_path)
    user = ObservedPlanUser(MCPWorkspace(tmp_path))
    prepared = _prepare(context)
    submitted = _decision(context, prepared.result["plan_id"], "approve")
    with pytest.raises(AssertionError, match="No explicit user decision") as error:
        asyncio.run(
            user(ModelTurn(1, review_text(prepared.result), [prepared, submitted]))
        )
    assert review_text(prepared.result) in "\n".join(error.value.__notes__)
    assert not user.replies


def test_ambiguous_user_reply_requires_another_turn_before_decline(
    tmp_path: Path,
) -> None:
    context = plan_context(tmp_path)
    user = ObservedPlanUser(MCPWorkspace(tmp_path), decisions=("ambiguous", "decline"))
    prepared = _prepare(context)
    assert "考慮" in str(asyncio.run(user(_present(prepared))))
    assert "不同意" in str(
        asyncio.run(user(ModelTurn(2, "手續費會減少淨利，是否同意？", [])))
    )
    submitted = _decision(context, prepared.result["plan_id"], "decline")
    assert asyncio.run(user(ModelTurn(3, "已取消", [submitted]))) is None
    user.assert_complete()
    assert not context.runner.calls


@pytest.mark.parametrize("replacement", [False, True])
def test_ambiguous_reply_neither_approves_nor_replaces_plan(
    tmp_path: Path, replacement: bool
) -> None:
    context = plan_context(tmp_path)
    user = ObservedPlanUser(MCPWorkspace(tmp_path), decisions=("ambiguous", "approve"))
    prepared = _prepare(context)
    asyncio.run(user(_present(prepared)))
    call = (
        _prepare(context)
        if replacement
        else _decision(context, prepared.result["plan_id"], "approve")
    )
    with pytest.raises(AssertionError):
        asyncio.run(user(ModelTurn(2, "已處理", [call])))
    assert len(user.replies) == 1


def test_revision_requires_new_saved_plan_with_requested_settings(
    tmp_path: Path,
) -> None:
    context = plan_context(tmp_path)
    user = ObservedPlanUser(MCPWorkspace(tmp_path), decisions=("revise", "approve"))
    first = _prepare(context)
    assert "不要執行" in str(asyncio.run(user(_present(first))))
    rejected = _decision(context, first.result["plan_id"], "revise")
    context.arguments["research"]["fees"] = user.revised_fees
    args: dict[str, Any] = {
        key: value for key, value in context.arguments.items() if key != "evidence_id"
    }
    workspace = context.service.workspace
    changed = ResearchPreparationService(
        strategy_service=StrategyService(workspace),
        strategy_loader=StrategyLoader(workspace_root=workspace.root),
        data_service=DataService(workspace),
    ).prepare("backtest", BacktestInput.model_validate(args))
    assert isinstance(changed, PreparedExecution)
    context.preflight.prepared = changed
    second = _prepare(context)
    assert "同意" in str(
        asyncio.run(user(ModelTurn(2, review_text(second.result), [rejected, second])))
    )
    assert not context.runner.calls
    submitted = _decision(context, second.result["plan_id"], "approve")
    assert asyncio.run(user(ModelTurn(3, "完成", [submitted]))) is None
    user.assert_complete()
    assert first.result["plan_id"] != second.result["plan_id"]
    assert len(context.runner.calls) == 1
