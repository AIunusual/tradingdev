"""Explicit presentation metadata for test execution plans."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tests.research_fixtures import research_for_config
from tradingdev.domain.preflight import PreflightReceipt
from tradingdev.domain.research_evidence import ResearchEvidenceDraft

if TYPE_CHECKING:
    from tests.integration.mcp_harness import MCPClient


def research_input(
    *,
    strategy_outline: str = "以指標變化決定買賣的離線測試策略",
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    start_date: str = "2024-01-01",
    end_date: str = "2024-01-08",
    **settings: Any,
) -> dict[str, Any]:
    """Explicit research choices for scripted tests, never natural-user fixtures."""
    return {
        "kind": "new",
        "strategy_outline": strategy_outline,
        "symbol": symbol,
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "annualization": 365,
        **settings,
    }


def research_from_config(
    config: dict[str, Any], *, outline: str = "離線測試策略"
) -> dict[str, Any]:
    """Declare scripted authoring choices consistent with a fixture's full YAML."""
    return research_for_config(config, strategy_outline=outline).model_dump(
        mode="json", exclude_unset=True
    )


def research_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Convert controlled legacy fixture values to the new explicit request shape."""
    result = dict(arguments)
    fields: dict[str, Any] = {
        key: result.pop(key)
        for key in ("symbol", "timeframe", "start_date", "end_date")
        if key in result
    }
    if "train_start" in result:
        fields.update(
            start_date=result["train_start"],
            end_date=f"{result['test_end']}T23:59:59.999999",
        )
    result["research"] = research_input(**fields)
    return result


def confirmation_presentation(parameters: dict[str, Any]) -> dict[str, Any]:
    """Describe each captured constructor leaf, including nested parameter maps."""
    descriptions: dict[str, dict[str, str]] = {}

    def visit(value: Any, pointer: str) -> None:
        if isinstance(value, dict) and value:
            for key, child in value.items():
                escaped = str(key).replace("~", "~0").replace("/", "~1")
                visit(child, f"{pointer}/{escaped}")
        elif isinstance(value, list) and value:
            for index, child in enumerate(value):
                visit(child, f"{pointer}/{index}")
        else:
            descriptions[pointer] = {
                "label": f"研究參數 {pointer.rsplit('/', 1)[-1]}",
                "description": "此值控制本次策略的訊號規則，僅套用於這次歷史模擬。",
                "unit": "策略單位",
            }

    for key, value in parameters.items():
        visit(value, "/" + key.replace("~", "~0").replace("/", "~1"))
    return {
        "title": "歷史策略研究設定確認",
        "summary": "核對策略參數與行情期間，批准後執行歷史模擬。",
        "parameter_descriptions": descriptions,
    }


def execution_evidence_draft(
    parameters: dict[str, Any], *, target_paths: dict[str, list[str]] | None = None
) -> dict[str, Any]:
    """Authored evidence for scripted protocol tests; no browser is simulated."""
    presentation = confirmation_presentation(parameters)
    targets = target_paths or {"base": list(presentation["parameter_descriptions"])}
    draft = ResearchEvidenceDraft.model_validate(
        {
            "presentation": presentation,
            "rules": [
                {
                    "rule_id": "fixture",
                    "condition": "合成策略條件成立",
                    "action": "依測試策略的持倉訊號執行",
                    "formula": "signal[t] = fixture_rule(history[:t+1], parameters)",
                }
            ],
            "limits": ["離線合成測資的程式測試；未宣稱參數有外部來源。"],
            "parameter_provenance": [
                {
                    "target_ids": [target_id],
                    "parameter_paths": paths,
                    "provenance": "model_proposal",
                    "reason": "明確固定的測試提案。",
                }
                for target_id, paths in targets.items()
                if paths
            ],
        }
    )
    return draft.model_dump(mode="json")


async def save_execution_evidence(
    client: MCPClient,
    parameters: dict[str, Any],
    *,
    target_paths: dict[str, list[str]] | None = None,
) -> str:
    """Use the real public save tool before a scripted prepare call."""
    saved = await client.call(
        "save_research_evidence",
        draft=execution_evidence_draft(parameters, target_paths=target_paths),
    )
    assert saved["success"] is True, saved
    return str(saved["evidence_id"])


def synthetic_preflight_receipt(manifest_hash: str) -> dict[str, Any]:
    """A substituted receipt for generated-code evidence verifier unit tests."""
    return PreflightReceipt(
        manifest_hash=manifest_hash,
        elapsed_seconds=0.5,
        checked_paths=[
            "configuration",
            "execution_capabilities",
            "parameter_binding",
            "manifest_serialization",
            "signal_contract",
        ],
        synthetic_fixture_rows=[80, 240],
        signal_check="synthetic_contract_passed",
        training_requirement="unknown",
        not_checked=["historical_data", "backtest_engine", "training", "performance"],
    ).model_dump(mode="json")


async def confirm_prepared(
    client: MCPClient, prepared: dict[str, Any]
) -> dict[str, Any]:
    """Submit the test user's explicit approval for one ready plan."""
    assert prepared["success"] is True, prepared
    assert prepared["status"] == "ready", prepared
    assert not prepared.get("job_id"), prepared
    confirmed: dict[str, Any] = await client.call(
        "submit_execution_decision", plan_id=prepared["plan_id"], decision="approve"
    )
    assert confirmed["success"] is True, confirmed
    assert confirmed["plan_id"] == prepared["plan_id"]
    assert confirmed["manifest_hash"] == prepared["manifest_hash"]
    assert confirmed["job_id"]
    return confirmed
