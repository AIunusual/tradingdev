"""Provider boundaries and saved-report interpretation, including unavailable data."""

import json
from pathlib import Path
from typing import Any

import pytest
from tests.app.test_report_service import _payload, _save, _scope

from tradingdev.adapters.reporting.maverick_diagnostics import MaverickDiagnostics
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.diagnostic_service import DiagnosticService
from tradingdev.app.report_service import ReportService
from tradingdev.app.trade_history_service import TradeHistoryService
from tradingdev.domain.diagnostics import DiagnosticContext


def analysis(run_id: str = "run", scope: str = "full") -> dict[str, Any]:
    return {
        "summary": "本次研究只適用於保存期間。",
        "observations": "<script>alert(1)</script> 保存回撤指標是觀察，非原因。",
        "hypotheses_and_limits": "尚未驗證高波動情境；不可推定因果。",
        "next_steps": "先檢查回撤期間的交易，再比較相同成本的出場變體。",
        "references": [{"run_id": run_id, "scope": scope}],
    }


@pytest.mark.parametrize(
    "value,triggered", [(0.249, False), (0.25, False), (0.251, True)]
)
def test_upstream_drawdown_rule_boundary(value: float, triggered: bool) -> None:
    result = MaverickDiagnostics().diagnose(
        DiagnosticContext(
            _scope(values={"max_drawdown": value}),
            None,
        )
    )
    assert bool(result.findings) is triggered
    assert result.evaluated_rules == ["drawdown"]
    assert result.unavailable_rules["win_rate"] == "metric_unavailable"
    if triggered:
        assert result.findings[0].evidence[0].value == value


def test_missing_metrics_are_not_zero_and_volume_is_not_portfolio_return() -> None:
    provider = MaverickDiagnostics()
    missing = provider.diagnose(
        DiagnosticContext(_scope(values={"win_rate": None}), None)
    )
    assert missing.status == "unavailable" and not missing.findings
    result = provider.diagnose(
        DiagnosticContext(
            _scope(mode="volume", values={"max_drawdown": 2000, "total_trades": 0}),
            None,
        )
    )
    assert [f.code for f in result.findings] == ["few_trades"]
    assert result.unavailable_rules["drawdown"] == "requires_portfolio_return_mode"


def test_report_persists_bound_analysis_and_diagnostics(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path)
    store = SQLiteStore(workspace)
    reports = ReportService(workspace=workspace, store=store)
    _save((workspace, store, reports))
    diagnostics = DiagnosticService(
        TradeHistoryService(workspace=workspace, store=store)
    )
    assert diagnostics.get("missing").success is False
    assert diagnostics.get("run", "wrong").success is False
    assert diagnostics.get("run").success
    result = reports.generate_report(["run"], sections=[], analysis=analysis())
    assert result["success"], result
    path = Path(result["path"])
    payload = _payload(path)
    manifest = json.loads(path.with_name("manifest.json").read_text())
    assert manifest["analysis"] == payload["analysis"] == analysis()
    assert (
        manifest["diagnostics"]["run"]["full"]
        == payload["runs"][0]["scopes"][0]["diagnostics"]
    )
    html = path.read_text()
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "下一步研究建議" in html
    assert reports.generate_report(["run"], sections=[], analysis=analysis()) == result
    for invalid in (
        analysis("other"),
        analysis(scope="missing"),
        {**analysis(), "references": []},
    ):
        rejected = reports.generate_report(["run"], sections=[], analysis=invalid)
        assert rejected["code"] == "invalid_analysis_reference"
    absent = reports.generate_report(["run"], sections=[])
    assert "尚未提供 LLM 研究解讀" in Path(absent["path"]).read_text()
