"""Independent generated-strategy semantics check, executed as a subprocess."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from copy import deepcopy
from datetime import UTC
from pathlib import Path
from typing import Any

import pandas as pd

from tests.e2e.strategy_scenarios import SCENARIOS, expected_signals, market_frame
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.artifact_service import ArtifactService
from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.run_service import RunService
from tradingdev.app.strategy_service import StrategyService
from tradingdev.domain.backtest.schemas import BacktestConfig
from tradingdev.domain.randomness import execution_randomness
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.shared.utils.config import load_config

_REPORT_FILES = {
    "research_report_html": ("html", "report.html"),
    "research_report_manifest": ("manifest", "manifest.json"),
}


def _report_path(workspace: WorkspacePaths, artifact: dict[str, Any]) -> str:
    """Require the two fixed content-addressed report paths."""
    report_id = artifact["metadata"].get("report_id")
    assert isinstance(report_id, str) and re.fullmatch(r"[0-9a-f]{64}", report_id), (
        "Report ID must be a SHA-256 digest"
    )
    suffix, filename = _REPORT_FILES[artifact["artifact_type"]]
    assert artifact["artifact_id"] == f"report:{report_id}:{suffix}", (
        "Report artifact ID differs"
    )
    expected = workspace.root
    for part in ("reports", report_id, filename):
        expected /= part
        assert not expected.is_symlink(), "Report path cannot contain symlinks"
    assert Path(artifact["path"]) == expected, "Report path differs from fixed location"
    assert expected.resolve().is_relative_to(workspace.root), (
        "Report path leaves workspace"
    )
    return report_id


def _verify_reports(
    workspace: WorkspacePaths, run: dict[str, Any], artifacts: list[dict[str, Any]]
) -> None:
    """Independently verify report scope/run linkage and saved hashes."""
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for artifact in artifacts:
        if artifact["artifact_type"] not in _REPORT_FILES:
            continue
        report_id = _report_path(workspace, artifact)
        grouped.setdefault(report_id, {})[artifact["artifact_type"]] = artifact
    run_id = run["run_id"]
    for report_id, pair in grouped.items():
        assert pair.keys() == _REPORT_FILES.keys(), (
            "Report must retain HTML and manifest"
        )
        html_artifact = pair["research_report_html"]
        manifest = json.loads(
            Path(pair["research_report_manifest"]["path"]).read_bytes()
        )
        assert manifest["schema_version"] == 2, "Report manifest schema differs"
        assert manifest["report_id"] == manifest["content_sha256"] == report_id, (
            "Report digest differs"
        )
        assert manifest["run_ids"] == [run_id], "Report run IDs differ"
        assert manifest["available_scopes"] == {run_id: run["available_scopes"]}, (
            "Report scopes differ"
        )
        content = Path(html_artifact["path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == manifest["html_sha256"], (
            "Report HTML digest differs"
        )


def verify(root: Path) -> None:
    """Verify runnable state, configurable periods and independent expected signals."""
    workspace = WorkspacePaths(root / "workspace")
    service = StrategyService(workspace)
    spec = service.resolve_executable("codex_sma_integration")
    assert spec.status.value == "runnable"
    assert Path(spec.source_path).is_relative_to(workspace.generated_strategies)
    assert Path(spec.config_path).parent == Path(spec.source_path).parent
    assert spec.revision_id
    raw_config = load_config(Path(spec.config_path))
    parameters = raw_config["strategy"]["parameters"]
    assert parameters["fast_period"] == 5
    assert parameters["slow_period"] == 20

    close = pd.Series([*range(1, 41), *range(40, 0, -1), *([1.0] * 30)], dtype=float)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(close), freq="h"),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 100.0,
        }
    )
    before = frame.copy(deep=True)
    for fast_period, slow_period in ((5, 20), (3, 8)):
        config = deepcopy(raw_config)
        config["strategy"]["parameters"].update(
            fast_period=fast_period, slow_period=slow_period
        )
        with execution_randomness(config.get("random_seed")):
            strategy = StrategyLoader(workspace_root=workspace.root).create_from_config(
                config, engine=None
            )
            result = strategy.generate_signals(frame)
            fast = close.rolling(fast_period).mean()
            slow = close.rolling(slow_period).mean()
            expected = pd.Series(0, index=frame.index, name="signal")
            expected.loc[fast > slow] = 1
            expected.loc[fast < slow] = -1
            pd.testing.assert_series_equal(
                result["signal"], expected, check_dtype=False
            )
            pd.testing.assert_frame_equal(frame, before)


def verify_execution_snapshot(
    snapshot: dict[str, Any],
    verified_config: dict[str, Any],
    *,
    parameters: dict[str, int],
    revision_id: str | None,
) -> None:
    """Bind actual execution settings to the independently checked requirements.

    Normalize schema defaults and equivalent UTC date representations. Runtime
    data paths and strategy metadata may be resolved by the backend; parameter
    experiments must use their explicitly expected overrides. Neither permits a
    change to any backtest assumption or the top-level random seed.
    """
    expected = BacktestConfig.model_validate(verified_config["backtest"])
    actual = BacktestConfig.model_validate(snapshot["backtest"])
    expected_fields = expected.model_dump(mode="python")
    actual_fields = actual.model_dump(mode="python")
    for fields in (expected_fields, actual_fields):
        for name in ("start_date", "end_date"):
            value = fields[name]
            fields[name] = (
                value.replace(tzinfo=UTC)
                if value.tzinfo is None
                else value.astimezone(UTC)
            )
    assert actual_fields == expected_fields, (
        "Execution backtest snapshot differs from verified configuration: "
        f"expected={expected_fields!r}, actual={actual_fields!r}"
    )
    assert snapshot.get("random_seed") == verified_config.get("random_seed"), (
        "Execution random_seed differs from verified configuration"
    )
    assert snapshot["strategy"]["parameters"] == parameters, (
        "Execution parameters differ from the requested scenario"
    )
    assert snapshot["strategy"]["revision_id"] == revision_id, (
        "Execution revision differs from the verified revision"
    )


def verify_workflow(root: Path, scenario_name: str) -> None:
    """Compare generated signals and stored backtest metrics with independent inputs."""
    scenario = SCENARIOS[scenario_name]
    workspace = WorkspacePaths(root / "workspace")
    spec = StrategyService(workspace).resolve_executable(scenario.strategy_id)
    assert spec.status.value in {"runnable", "promoted"}
    config = load_config(Path(spec.config_path))
    assert config["strategy"]["parameters"] == scenario.parameters
    bt = BacktestConfig(**config["backtest"])
    assert (bt.symbol, bt.timeframe, bt.mode) == ("BTC/USDT", "1h", "signal")
    assert str(bt.start_date.date()) == "2024-01-01"
    assert str(bt.end_date.date()) == "2024-01-08"
    assert bt.init_cash == 10000 and bt.fees == 0 and bt.slippage == 0
    assert config.get("random_seed") == 42, (
        "Expected top-level random_seed=42; "
        f"saved value={config.get('random_seed')!r}, "
        f"saved backtest={config['backtest']!r}"
    )
    assert bt.periods_per_year == 365
    frame = market_frame()
    original = frame.copy(deep=True)
    for parameters in (scenario.parameters, scenario.overrides):
        overridden = deepcopy(config)
        overridden["strategy"]["parameters"] = parameters
        with execution_randomness(config.get("random_seed")):
            strategy = StrategyLoader(workspace_root=workspace.root).create_from_config(
                overridden, engine=None
            )
            actual = strategy.generate_signals(frame)
            expected = expected_signals(frame, scenario, parameters)
            pd.testing.assert_series_equal(
                actual["signal"], expected, check_dtype=False
            )
            pd.testing.assert_frame_equal(frame, original)
            # Prefixes include warmup, a reversal, and a flat interval. A strategy
            # cannot use future rows to rewrite the earlier signal history.
            for length in (4, 24, 55, 92):
                prefix = frame.iloc[:length].copy()
                result = strategy.generate_signals(prefix)
                pd.testing.assert_series_equal(
                    result["signal"], expected.iloc[:length], check_dtype=False
                )
                pd.testing.assert_frame_equal(prefix, original.iloc[:length])

    run_service = RunService(workspace=workspace)
    runs = run_service.list_runs()
    assert len(runs) == 1, runs
    run = runs[0]
    assert run["strategy_id"] == scenario.strategy_id
    assert run["revision_id"] == spec.revision_id
    # Ordinary backtests treat end_date as a timestamp (midnight here).
    selected = frame.loc[
        frame["timestamp"] <= pd.Timestamp("2024-01-08", tz="UTC")
    ].copy()
    run_parameters = scenario.overrides if scenario.experiment else scenario.parameters
    selected["signal"] = expected_signals(selected, scenario, run_parameters)
    reference = BacktestService().create_engine(bt).run(selected).metrics
    assert reference["total_trades"] > 0
    for metric in ("total_return", "total_trades", "max_drawdown"):
        assert math.isclose(
            run["metrics"][metric], reference[metric], rel_tol=1e-9, abs_tol=1e-9
        ), (metric, run["metrics"][metric], reference[metric])
    detailed = run_service.get_run_metrics(run["run_id"])
    assert detailed["success"] and detailed["scope"] == "full", detailed
    stored = SQLiteStore(workspace).get_run(run["run_id"])
    assert stored is not None
    for metric, value in detailed["metrics"].items():
        assert stored["metrics"][metric] == value, (metric, "SQLite/artifact mismatch")
    for metric in ("daily_pnl_mean", "total_volume", "n_days"):
        assert metric not in run["metrics"]
        assert math.isclose(
            detailed["metrics"][metric], reference[metric], rel_tol=1e-9, abs_tol=1e-9
        ), (metric, detailed["metrics"][metric], reference[metric])
    artifacts = ArtifactService(workspace=workspace).list_artifacts(run["run_id"])
    by_type = {artifact["artifact_type"]: artifact for artifact in artifacts}
    assert {
        "strategy_source",
        "config_snapshot",
        "result_json",
        "dataset_fingerprint",
        "performance_json",
        "observations_json",
    } <= by_type.keys()
    for artifact in artifacts:
        path = Path(artifact["path"])
        if artifact["artifact_type"] in _REPORT_FILES:
            _report_path(workspace, artifact)
        else:
            assert path.is_relative_to(workspace.runs)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"], (
            f"Artifact SHA differs: {artifact['artifact_id']}"
        )
    _verify_reports(workspace, run, artifacts)
    assert (
        Path(by_type["strategy_source"]["path"]).read_bytes()
        == Path(spec.source_path).read_bytes()
    )
    snapshot = load_config(Path(by_type["config_snapshot"]["path"]))
    verify_execution_snapshot(
        snapshot, config, parameters=run_parameters, revision_id=spec.revision_id
    )


if __name__ == "__main__":
    root = Path(sys.argv[1])
    (root / "verifier.ready").write_text("ready", encoding="utf-8")
    if len(sys.argv) > 2:
        verify_workflow(root, sys.argv[2])
    else:
        verify(root)
