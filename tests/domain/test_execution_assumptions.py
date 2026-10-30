"""Saved simulator explanations bind identity without claiming strategy proof."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from tradingdev.domain import execution_assumptions as assumptions_module
from tradingdev.domain.execution import ExecutionManifest, ManifestError
from tradingdev.domain.execution_assumptions import (
    ExecutionAssumptions,
    resolve_execution_assumptions,
)
from tradingdev.domain.strategies.execution import StrategyExecution


def _manifest(**backtest: Any) -> ExecutionManifest:
    return ExecutionManifest.create(
        kind="backtest",
        config={
            "strategy": {"id": "fixture"},
            "backtest": {
                "symbol": "BTCUSDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-02-01",
                "init_cash": 1000,
                "fees": 0,
                "slippage": 0,
                **backtest,
            },
        },
        strategy_execution=StrategyExecution(kind="bundled", constructor_kwargs={}),
    )


def _items(assumptions: ExecutionAssumptions) -> dict[str, str]:
    return {item.key: item.value for item in assumptions.items}


def test_assumptions_bind_manifest_and_do_not_claim_checks_passed() -> None:
    manifest = _manifest()
    result = resolve_execution_assumptions(manifest)
    assert result.manifest_hash == manifest.manifest_hash
    assert result.mode == "signal"
    items = _items(result)
    assert "不是固定數量" in items["sizing"]
    assert "stoplimit" in items["costs"]
    assert "手續費率 0%" in items["costs"]
    assert "滑價率 0%" in items["costs"]
    assert "未設定每年日數" in items["annualization"]
    assert "不證明合成檢查已通過" in items["limitations"]
    assert "不使用行情 high/low" in items["stops"]
    assert "零值是零距離" in items["stops"]
    assert "不是只賣出平多倉" in items["signals"]
    assert "沒有槓桿" in items["leverage"]


@pytest.mark.parametrize("size", [None, 0, 200, 450])
def test_volume_sizing_and_signal_flags_are_resolved(size: float | None) -> None:
    manifest = _manifest(
        mode="volume",
        init_cash=None,
        position_size=size,
        signal_as_position=False,
        re_entry_after_sl=False,
        periods_per_year=365,
    )
    items = _items(resolve_execution_assumptions(manifest))
    if size == 0:
        assert "不開新倉" in items["sizing"]
    else:
        assert str(size if size is not None else 200) in items["sizing"]
    assert "保留既有倉位" in items["signals"]
    assert "持有期間不重平衡" in items["weights"]
    assert "跨月不自行解除" in items["stop_reentry"]
    assert "零值停用" in items["stops"]
    assert "帳戶報酬及其年化指標不適用" in items["annualization"]
    assert "365" in items["annualization"]
    assert "強制平倉" in items["end_of_data"]
    assert "UTC 月份" in items["monthly_loss"]
    assert "上個有行情月份最後收盤權益" in items["monthly_loss"]
    assert "首月基準為零" in items["monthly_loss"]
    assert "不重計前月損益" in items["monthly_loss"]


def test_alternative_volume_flags_do_not_reuse_default_explanations() -> None:
    items = _items(
        resolve_execution_assumptions(
            _manifest(mode="volume", signal_as_position=True, re_entry_after_sl=True)
        )
    )
    assert "0 表示平倉" in items["signals"]
    assert "當棒 close 重新進場" in items["stop_reentry"]


def test_saved_assumptions_remain_frozen_when_policy_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    saved = resolve_execution_assumptions(manifest)
    encoded = saved.model_dump_json()
    monkeypatch.setattr(assumptions_module, "EXECUTION_POLICY_VERSION", "future")
    assert resolve_execution_assumptions(manifest).policy_version == "future"
    assert ExecutionAssumptions.model_validate_json(encoded) == saved
    with pytest.raises(ValidationError, match="frozen"):
        saved.__setattr__("policy_version", "changed")
    with pytest.raises(ValidationError, match="frozen"):
        saved.items[0].__setattr__("value", "changed")


def test_mutated_manifest_is_rejected_before_explaining_execution() -> None:
    manifest = _manifest()
    raw = manifest.config["backtest"]
    assert isinstance(raw, dict)
    raw["position_size"] = 0
    with pytest.raises(ManifestError, match="hash"):
        resolve_execution_assumptions(manifest)


def test_saved_snapshot_cannot_silently_omit_a_rule() -> None:
    saved = resolve_execution_assumptions(_manifest())
    data = saved.model_dump()
    data["items"] = data["items"][:-1]
    with pytest.raises(ValidationError, match="every policy item"):
        ExecutionAssumptions.model_validate(data)


@pytest.mark.parametrize("version", [True, 1.0, "1"])
def test_saved_schema_version_requires_an_integer(version: object) -> None:
    data = resolve_execution_assumptions(_manifest()).model_dump()
    data["schema_version"] = version
    with pytest.raises(ValidationError, match="schema_version must be an integer"):
        ExecutionAssumptions.model_validate(data)
