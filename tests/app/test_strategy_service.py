"""Strategy lifecycle service tests."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from tests.research_fixtures import research_for_config

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.contracts.strategy import (
    StrategyDryRunFailure,
    StrategyValidationFailure,
)
from tradingdev.app.strategy_service import StrategyNotExecutableError, StrategyService
from tradingdev.domain.strategies.schemas import StrategyMetadata

_STRATEGY_CODE = """\
from __future__ import annotations

from typing import Any

import pandas as pd

from tradingdev.domain.strategies.base import BaseStrategy


class FixtureStrategy(BaseStrategy):
    def __init__(
        self,
        backtest_engine: object | None = None,
        threshold: float = 0.0,
    ) -> None:
        self._engine = backtest_engine
        self._threshold = threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["signal"] = 0
        moves = result["close"].pct_change().fillna(0)
        result.loc[moves > self._threshold, "signal"] = 1
        return result

    def get_parameters(self) -> dict[str, Any]:
        return {"threshold": self._threshold}
"""

_YAML = """\
strategy:
  id: "fixture_strategy"
  version: "0.1.0"
  class_name: "FixtureStrategy"
  source_path: "workspace/generated_strategies/fixture_strategy.py"
  parameters:
    threshold: 0.0
backtest:
  symbol: "BTC/USDT"
  timeframe: "1h"
  start_date: "2024-01-01"
  end_date: "2024-01-31"
  init_cash: 10000.0
data:
  source: "binance_api"
  requirements:
    market:
      source: "binance_api"
      symbol: "BTC/USDT"
      timeframe: "1h"
    features: []
"""


def test_strategy_service_lifecycle(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = StrategyService(workspace)
    service._quality_gate_diagnostics = lambda _path: []  # type: ignore[assignment,method-assign]

    saved = service.save_draft(
        _STRATEGY_CODE,
        _YAML,
        research_for_config(_YAML),
        strategy_id="fixture_strategy",
    )

    assert saved.success is True
    assert saved.status == "draft"
    validated = service.validate("fixture_strategy")
    assert validated["success"] is True
    assert validated["status"] == "validated"
    assert validated["signal_analysis"]["rows"] == 80
    assert validated["effective_config"]["backtest"]["fees"] == 0.0006
    assert validated["effective_config"]["random_seed"] is None
    dry_run = service.dry_run("fixture_strategy")
    assert dry_run["success"] is True
    assert dry_run["status"] == "runnable"
    assert dry_run["signal_analysis"]["rows"] == 240
    assert dry_run["signal_analysis"]["transition_count"] >= 1
    assert dry_run["effective_config"] == validated["effective_config"]
    promoted = service.promote("fixture_strategy")
    assert promoted == {
        "success": True,
        "strategy_id": "fixture_strategy",
        "revision_id": saved.revision_id,
        "status": "promoted",
    }


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("field", ["random_seed", "random_sead"])
def test_config_errors_are_reported_before_constructing_strategy(
    tmp_path: Path, dry_run: bool, field: str
) -> None:
    service = StrategyService(WorkspacePaths(tmp_path / "workspace"))
    service._quality_gate_diagnostics = lambda _path: []  # type: ignore[assignment,method-assign]
    config = yaml.safe_load(_YAML)
    config["backtest"][field] = 42
    code = _STRATEGY_CODE.replace(
        "self._engine = backtest_engine",
        'raise RuntimeError("Constructor must not run for an invalid config")',
    )
    saved = service._revisions.create("invalid_config", code, config)
    if dry_run:
        # Simulate older validation evidence; dry-run must independently check config.
        recorded = service.record_validation_status(
            "invalid_config",
            {
                "revision_id": saved.revision_id,
                "checked_at": "2024-01-01T00:00:00+00:00",
                "success": True,
                "diagnostics": [],
            },
        )
        assert recorded["success"]
    checked = (
        service.dry_run("invalid_config")
        if dry_run
        else service.validate("invalid_config")
    )
    assert checked["success"] is False
    assert checked["effective_config"] is None
    assert checked["diagnostics"][0]["code"] == "invalid_execution_config"
    assert checked["diagnostics"][0]["phase"] == "config"
    assert f"backtest.{field}" in checked["diagnostics"][0]["message"]
    assert "YAML root" in checked["diagnostics"][0]["fix"]


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize(
    "invalid_case",
    [
        "nan_fees",
        "nan_string_fees",
        "overflow_string_fees",
        "nan_parameter",
        "infinite_parameter",
        "integer_key",
        "cycle",
    ],
)
def test_lossy_yaml_values_cannot_execute_or_advance_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dry_run: bool,
    invalid_case: str,
) -> None:
    service = StrategyService(WorkspacePaths(tmp_path / "workspace"))
    service._quality_gate_diagnostics = lambda _path: []  # type: ignore[assignment,method-assign]
    config = yaml.safe_load(_YAML)
    if invalid_case == "nan_fees":
        config["backtest"]["fees"] = float("nan")
    elif invalid_case == "nan_string_fees":
        config["backtest"]["fees"] = "NaN"
    elif invalid_case == "overflow_string_fees":
        config["backtest"]["fees"] = "1e999"
    elif invalid_case == "nan_parameter":
        config["strategy"]["parameters"]["threshold"] = float("nan")
    elif invalid_case == "infinite_parameter":
        config["strategy"]["parameters"]["threshold"] = {"nested": [float("inf")]}
    elif invalid_case == "integer_key":
        config["strategy"]["parameters"]["threshold"] = {1: "number", "1": "text"}
    else:
        circular: list[Any] = []
        circular.append(circular)
        config["strategy"]["parameters"]["threshold"] = circular
    saved = service._revisions.create("invalid_json", _STRATEGY_CODE, config)
    if dry_run:
        assert service.record_validation_status(
            "invalid_json",
            {
                "revision_id": saved.revision_id,
                "checked_at": "2024-01-01T00:00:00+00:00",
                "success": True,
                "diagnostics": [],
            },
        )["success"]
    called = False

    def unexpected_constructor(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("Invalid raw configuration reached the constructor")

    monkeypatch.setattr(service._loader, "create_from_config", unexpected_constructor)
    checked = (
        service.dry_run("invalid_json") if dry_run else service.validate("invalid_json")
    )
    assert not called
    assert checked["success"] is False
    assert checked["effective_config"] is None
    assert checked["diagnostics"][0]["code"] == "invalid_execution_config"
    assert checked["status"] == ("validated" if dry_run else "draft")
    response_model = StrategyDryRunFailure if dry_run else StrategyValidationFailure
    response_model.model_validate(checked).model_dump_json()
    metadata = service.get_strategy("invalid_json")["metadata"]
    assert metadata["status"] == checked["status"]
    evidence = metadata["dry_run" if dry_run else "validation"]
    assert not evidence["success"]
    assert evidence["effective_config"] is None


def test_validation_and_dry_run_seed_constructor_and_signals(tmp_path: Path) -> None:
    service = StrategyService(WorkspacePaths(tmp_path / "workspace"))
    service._quality_gate_diagnostics = lambda _path: []  # type: ignore[assignment,method-assign]
    code = (
        _STRATEGY_CODE.replace(
            "from tradingdev.domain.strategies.base import BaseStrategy",
            "from tradingdev.domain.randomness import (\n"
            "    get_numpy_rng, get_random, get_seed,\n)\n"
            "from tradingdev.domain.strategies.base import BaseStrategy",
        )
        .replace(
            "self._engine = backtest_engine",
            "self._engine = backtest_engine\n"
            "        assert get_seed() == 42\n"
            "        assert get_random().random() == 0.6394267984578837\n"
            "        assert get_numpy_rng().integers(100) == 8",
        )
        .replace(
            'result.loc[moves > self._threshold, "signal"] = 1',
            'result["signal"] = get_numpy_rng().integers(-1, 2, len(result))',
        )
    )
    saved = service.save_draft(
        code,
        "random_seed: 42\n" + _YAML,
        research_for_config("random_seed: 42\n" + _YAML),
        strategy_id="random_fixture",
    )
    assert saved.success
    first = service.validate("random_fixture")
    second = service.validate("random_fixture")
    assert first["success"] and second["success"]
    assert first["signal_analysis"] == second["signal_analysis"]
    assert first["effective_config"]["random_seed"] == 42
    assert "random_seed" not in first["effective_config"]["backtest"]
    checked = service.dry_run("random_fixture")
    assert checked["success"]
    assert checked["effective_config"] == first["effective_config"]
    retrieved = service.get_strategy("random_fixture")
    assert retrieved["metadata"]["validation"]["effective_config"]["random_seed"] == 42
    assert retrieved["metadata"]["dry_run"]["effective_config"]["random_seed"] == 42


def test_strategy_service_loads_generated_and_bundled_specs(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = StrategyService(workspace)

    assert service.save_draft(
        _STRATEGY_CODE,
        _YAML,
        research_for_config(_YAML),
        strategy_id="fixture_strategy",
    ).success

    generated = service.load("fixture_strategy")
    bundled = service.load("kd_crossover")

    assert generated is not None
    assert generated.strategy_id == "fixture_strategy"
    assert generated.kind == "generated"
    assert bundled is not None
    assert bundled.strategy_id == "kd_crossover"
    assert bundled.kind == "bundled"
    assert bundled.status == "promoted"


def test_record_validation_status_updates_draft_metadata(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = StrategyService(workspace)

    saved = service.save_draft(
        _STRATEGY_CODE,
        _YAML,
        research_for_config(_YAML),
        strategy_id="fixture_strategy",
    )
    assert saved.success

    response = service.record_validation_status(
        "fixture_strategy",
        {
            "revision_id": saved.revision_id,
            "checked_at": "2024-01-01T00:00:00+00:00",
            "success": True,
            "diagnostics": [],
            "signal_analysis": {"rows": 80},
        },
    )

    assert response["success"] is True
    assert response["status"] == "validated"
    spec = service.load("fixture_strategy")
    assert spec is not None
    assert spec.status == "validated"


def test_list_strategies_includes_requirements_and_recent_runs(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    service = StrategyService(workspace, store=store)
    assert service.save_draft(
        _STRATEGY_CODE,
        _YAML,
        research_for_config(_YAML),
        strategy_id="fixture_strategy",
    ).success
    store.create_run(
        run_id="run_a",
        job_id="job_a",
        strategy_id="fixture_strategy",
        artifact_dir=workspace.runs / "run_a",
        metrics={"total_return": 0.1},
        dataset_id="dataset-a",
    )

    item = next(
        item
        for item in service.list_strategies()
        if item["strategy_id"] == "fixture_strategy"
    )

    assert item["data_requirements"]["market"]["symbol"] == "BTC/USDT"
    assert item["recent_runs"][0]["run_id"] == "run_a"
    assert item["recent_runs"][0]["metrics"]["total_return"] == 0.1


def test_strategy_service_rejects_banned_import(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = StrategyService(workspace)
    service._quality_gate_diagnostics = lambda _path: []  # type: ignore[assignment,method-assign]

    code = "import os\n" + _STRATEGY_CODE
    assert service.save_draft(
        code,
        _YAML.replace("fixture_strategy", "bad_strategy"),
        research_for_config(_YAML.replace("fixture_strategy", "bad_strategy")),
        strategy_id="bad_strategy",
    ).success

    validated = service.validate("bad_strategy")

    assert validated["success"] is False
    messages = [item["message"] for item in validated["diagnostics"]]
    codes = [item["code"] for item in validated["diagnostics"]]
    assert "banned import: os" in messages
    assert "banned_import" in codes


def test_validate_fails_when_quality_gate_command_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = StrategyService(workspace)

    def raise_missing(
        *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("uv")

    monkeypatch.setattr(subprocess, "run", raise_missing)

    assert service.save_draft(
        _STRATEGY_CODE,
        _YAML,
        research_for_config(_YAML),
        strategy_id="fixture_strategy",
    ).success

    validated = service.validate("fixture_strategy")

    assert validated["success"] is False
    assert validated["status"] == "draft"
    diagnostics = validated["diagnostics"]
    assert {item["code"] for item in diagnostics} >= {
        "ruff_unavailable",
        "mypy_unavailable",
    }
    assert all(
        item["level"] == "error"
        for item in diagnostics
        if item["code"] in {"ruff_unavailable", "mypy_unavailable"}
    )


def test_validate_rejects_runnable_strategy(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = StrategyService(workspace)
    service._quality_gate_diagnostics = lambda _path: []  # type: ignore[assignment,method-assign]

    assert service.save_draft(
        _STRATEGY_CODE,
        _YAML,
        research_for_config(_YAML),
        strategy_id="fixture_strategy",
    ).success
    assert service.validate("fixture_strategy")["success"] is True
    assert service.dry_run("fixture_strategy")["status"] == "runnable"

    validated = service.validate("fixture_strategy")

    assert validated["success"] is False
    assert validated["status"] == "runnable"
    assert "only accepts draft or validated" in validated["error"]


def test_dry_run_only_accepts_validated_strategy(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = StrategyService(workspace)
    service._quality_gate_diagnostics = lambda _path: []  # type: ignore[assignment,method-assign]

    assert service.save_draft(
        _STRATEGY_CODE,
        _YAML,
        research_for_config(_YAML),
        strategy_id="fixture_strategy",
    ).success

    draft_dry_run = service.dry_run("fixture_strategy")
    assert draft_dry_run["success"] is False
    assert "requires validated" in draft_dry_run["error"]

    assert service.validate("fixture_strategy")["status"] == "validated"
    assert service.dry_run("fixture_strategy")["status"] == "runnable"

    runnable_dry_run = service.dry_run("fixture_strategy")
    assert runnable_dry_run["success"] is False
    assert "requires validated" in runnable_dry_run["error"]

    assert service.promote("fixture_strategy")["status"] == "promoted"
    promoted_dry_run = service.dry_run("fixture_strategy")
    assert promoted_dry_run["success"] is False
    assert "requires validated" in promoted_dry_run["error"]


def test_strategy_service_rejects_modified_snapshot_before_validation(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = StrategyService(workspace)
    saved = service.save_draft(
        _STRATEGY_CODE, _YAML, research_for_config(_YAML), strategy_id="syntax_strategy"
    )
    assert saved.success is True
    source_path = Path(saved.source_path)
    source_path.write_text("def broken(:\n", encoding="utf-8")

    validated = service.validate("syntax_strategy")

    assert validated["success"] is False
    assert validated["code"] == "strategy_revision_invalid"


@pytest.mark.parametrize(
    "import_source",
    ["import pandas_ta as ta", "from pandas_ta import sma"],
)
def test_strategy_service_rejects_removed_pandas_ta_import(
    tmp_path: Path, import_source: str
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = StrategyService(workspace)
    service._quality_gate_diagnostics = lambda _path: []  # type: ignore[assignment,method-assign]

    code = _STRATEGY_CODE.replace(
        "import pandas as pd",
        f"{import_source}\n\nimport pandas as pd",
    )
    assert service.save_draft(
        code,
        _YAML.replace("fixture_strategy", "bad_import_strategy"),
        research_for_config(_YAML.replace("fixture_strategy", "bad_import_strategy")),
        strategy_id="bad_import_strategy",
    ).success

    validated = service.validate("bad_import_strategy")

    assert validated["success"] is False
    codes = [item["code"] for item in validated["diagnostics"]]
    assert "import_not_allowed" in codes
    rejected = next(
        item
        for item in validated["diagnostics"]
        if item["code"] == "import_not_allowed"
    )
    assert rejected["message"] == "import not allowed: pandas_ta"
    assert "talib" in rejected["fix"]
    assert "tradingdev.domain.indicators" in rejected["fix"]


def test_strategy_service_rejects_invalid_signal_values(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = StrategyService(workspace)
    service._quality_gate_diagnostics = lambda _path: []  # type: ignore[assignment,method-assign]

    code = _STRATEGY_CODE.replace(
        'result.loc[moves > self._threshold, "signal"] = 1',
        'result.loc[moves > self._threshold, "signal"] = 2',
    )
    assert service.save_draft(
        code,
        _YAML.replace("fixture_strategy", "bad_signal_strategy"),
        research_for_config(_YAML.replace("fixture_strategy", "bad_signal_strategy")),
        strategy_id="bad_signal_strategy",
    ).success

    validated = service.validate("bad_signal_strategy")

    assert validated["success"] is False
    codes = [item["code"] for item in validated["diagnostics"]]
    assert "invalid_signal_values" in codes
    assert validated["signal_analysis"]["rows"] == 80


def test_strategy_service_rejects_input_mutation(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = StrategyService(workspace)
    service._quality_gate_diagnostics = lambda _path: []  # type: ignore[assignment,method-assign]

    code = _STRATEGY_CODE.replace("result = df.copy()", "result = df")
    assert service.save_draft(
        code,
        _YAML.replace("fixture_strategy", "mutating_strategy"),
        research_for_config(_YAML.replace("fixture_strategy", "mutating_strategy")),
        strategy_id="mutating_strategy",
    ).success

    validated = service.validate("mutating_strategy")

    assert validated["success"] is False
    codes = [item["code"] for item in validated["diagnostics"]]
    assert "input_mutated" in codes


_TALIB_CODE = """\
from __future__ import annotations

from typing import Any

import pandas as pd
import talib

from tradingdev.domain.strategies.base import BaseStrategy


class TalibStrategy(BaseStrategy):
    def __init__(
        self,
        backtest_engine: object | None = None,
        length: int = 3,
    ) -> None:
        self._engine = backtest_engine
        self._length = length

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        close = result["close"].to_numpy(dtype=float)
        _upper, middle, _lower = talib.BBANDS(close, timeperiod=self._length)
        sma = pd.Series(middle, index=result.index)
        result["signal"] = 0
        result.loc[result["close"] > sma, "signal"] = 1
        return result

    def get_parameters(self) -> dict[str, Any]:
        return {"length": self._length}
"""

_TALIB_YAML = (
    _YAML.replace("fixture_strategy", "talib_strategy")
    .replace("FixtureStrategy", "TalibStrategy")
    .replace("threshold: 0.0", "length: 3")
)


def test_strategy_service_validates_and_dry_runs_direct_talib_strategy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MYPY_CACHE_DIR", str(tmp_path / "mypy-cache"))
    monkeypatch.setenv("RUFF_CACHE_DIR", str(tmp_path / "ruff-cache"))
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = StrategyService(workspace)

    assert service.save_draft(
        _TALIB_CODE,
        _TALIB_YAML,
        research_for_config(_TALIB_YAML),
        strategy_id="talib_strategy",
    ).success

    validated = service.validate("talib_strategy")

    assert validated["success"] is True, validated["diagnostics"]
    assert validated["status"] == "validated"
    assert validated["signal_analysis"]["rows"] == 80
    assert validated["signal_analysis"]["signal_distribution"] == {"1": 78, "0": 2}

    dry_run = service.dry_run("talib_strategy")

    assert dry_run["success"] is True, dry_run["diagnostics"]
    assert dry_run["status"] == "runnable"
    assert dry_run["signal_analysis"]["rows"] == 240
    assert dry_run["signal_analysis"]["nan_count"] == 0
    assert dry_run["signal_analysis"]["transition_count"] > 0


def test_validation_finishes_on_captured_revision_after_new_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = StrategyService(WorkspacePaths(tmp_path / "workspace"))
    first = service.save_draft(
        _STRATEGY_CODE,
        _YAML,
        research_for_config(_YAML),
        strategy_id="fixture_strategy",
    )
    assert first.revision_id is not None
    saved_revisions: list[str | None] = []

    def save_during_check(_path: Path) -> list[Any]:
        second = service.save_draft(
            _STRATEGY_CODE,
            _YAML.replace("threshold: 0.0", "threshold: 0.5"),
            research_for_config(_YAML.replace("threshold: 0.0", "threshold: 0.5")),
            strategy_id="fixture_strategy",
            expected_revision_id=first.revision_id,
        )
        saved_revisions.append(second.revision_id)
        return []

    monkeypatch.setattr(service, "_quality_gate_diagnostics", save_during_check)
    checked = service.validate("fixture_strategy")

    assert checked["success"] is True
    assert checked["revision_id"] == first.revision_id
    current = service.load("fixture_strategy")
    assert current is not None
    assert current.revision_id == saved_revisions[0] != first.revision_id
    assert current.status == "draft"
    assert isinstance(current.metadata, StrategyMetadata)
    assert current.metadata.validation is None
    with pytest.raises(StrategyNotExecutableError, match="runnable or promoted"):
        service.resolve_executable("fixture_strategy")

    dry_run = service.dry_run("fixture_strategy", first.revision_id)
    assert dry_run["success"] is True
    assert dry_run["revision_id"] == first.revision_id
    assert service.promote("fixture_strategy", first.revision_id)["success"] is True
    assert (
        service.resolve_executable("fixture_strategy", first.revision_id).revision_id
        == first.revision_id
    )
    assert service.load("fixture_strategy") == current


def test_external_validation_result_cannot_validate_the_new_current_revision(
    tmp_path: Path,
) -> None:
    service = StrategyService(WorkspacePaths(tmp_path / "workspace"))
    first = service.save_draft(
        _STRATEGY_CODE,
        _YAML,
        research_for_config(_YAML),
        strategy_id="fixture_strategy",
    )
    second = service.save_draft(
        _STRATEGY_CODE,
        _YAML,
        research_for_config(_YAML),
        strategy_id="fixture_strategy",
        expected_revision_id=first.revision_id,
    )

    checked = service.record_validation_status(
        "fixture_strategy",
        {
            "revision_id": first.revision_id,
            "checked_at": "2024-01-01T00:00:00Z",
            "success": True,
        },
    )

    assert checked["revision_id"] == first.revision_id
    assert checked["status"] == "validated"
    current = service.load("fixture_strategy")
    assert current is not None
    assert current.revision_id == second.revision_id
    assert current.status == "draft"


def test_source_changed_during_check_does_not_gain_validation_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = StrategyService(WorkspacePaths(tmp_path / "workspace"))
    saved = service.save_draft(
        _STRATEGY_CODE,
        _YAML,
        research_for_config(_YAML),
        strategy_id="fixture_strategy",
    )

    def modify_source(path: Path) -> list[Any]:
        path.write_text(_STRATEGY_CODE + "\n# modified during validation\n")
        return []

    monkeypatch.setattr(service, "_quality_gate_diagnostics", modify_source)
    checked = service.validate("fixture_strategy", saved.revision_id)

    assert checked["success"] is False
    assert checked["code"] == "strategy_revision_invalid"
    # Restore only the test's bytes to inspect the persisted state.
    Path(saved.source_path).write_text(_STRATEGY_CODE)
    spec = service.load("fixture_strategy", saved.revision_id)
    assert spec is not None
    assert isinstance(spec.metadata, StrategyMetadata)
    assert spec.status == "draft"
    assert spec.metadata.validation is None


def test_generated_save_cannot_shadow_bundled_strategy(tmp_path: Path) -> None:
    service = StrategyService(WorkspacePaths(tmp_path / "workspace"))
    saved = service.save_draft(
        _STRATEGY_CODE, _YAML, research_for_config(_YAML), strategy_id="kd_crossover"
    )
    assert not saved.success
    assert saved.code == "reserved_strategy_id"


def test_legacy_id_collision_does_not_shadow_bundled_execution(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = StrategyService(workspace)
    bundled = service.resolve_executable("kd_crossover")
    legacy = workspace.generated_strategies / "kd_crossover.json"
    legacy.write_text('{"status": "runnable"}', encoding="utf-8")
    assert service.resolve_executable("kd_crossover") == bundled
    assert service.get_strategy("kd_crossover")["kind"] == "bundled"
    with pytest.raises(StrategyNotExecutableError, match="not found"):
        service.resolve_executable(
            "kd_crossover", revision_id="0123456789ab4def8123456789abcdef"
        )


def test_dry_run_cannot_replace_newer_validation_in_the_same_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = StrategyService(WorkspacePaths(tmp_path / "workspace"))
    monkeypatch.setattr(service, "_quality_gate_diagnostics", lambda _path: [])
    saved = service.save_draft(
        _STRATEGY_CODE,
        _YAML,
        research_for_config(_YAML),
        strategy_id="fixture_strategy",
    )
    assert service.validate("fixture_strategy")["success"]
    original_check = service._contract_checker.check

    def revalidate_during_dry_run(
        metadata: StrategyMetadata, *, fixture_rows: int
    ) -> dict[str, Any]:
        response = service.record_validation_status(
            metadata.strategy_id,
            {
                "revision_id": metadata.revision_id,
                "checked_at": "2030-01-01T00:00:00Z",
                "success": True,
                "signal_analysis": {"newer_evidence": True},
            },
        )
        assert response["success"]
        return original_check(metadata, fixture_rows=fixture_rows)

    monkeypatch.setattr(service._contract_checker, "check", revalidate_during_dry_run)
    checked = service.dry_run("fixture_strategy", saved.revision_id)

    assert checked["success"] is False
    assert checked["code"] == "strategy_revision_invalid"
    current = service.load("fixture_strategy")
    assert current is not None
    assert current.status == "validated"
    assert isinstance(current.metadata, StrategyMetadata)
    assert current.metadata.dry_run is None
    assert current.metadata.validation is not None
    assert current.metadata.validation.signal_analysis == {"newer_evidence": True}


def test_anonymous_creation_saves_one_research_outline_and_a_readable_name(
    tmp_path: Path,
) -> None:
    service = StrategyService(WorkspacePaths(tmp_path / "workspace"))
    research = research_for_config(_YAML, strategy_outline="均線向上時買入，向下時賣出")

    saved = service.save_draft(_STRATEGY_CODE, _YAML, research)

    assert saved.success, saved.error
    assert saved.strategy_id.startswith("strategy_")
    config = yaml.safe_load(Path(saved.config_path).read_text())
    assert config["research"]["strategy_outline"] == research.strategy_outline
    assert config["strategy"]["display_name"] == research.strategy_outline
    metadata = service.get_strategy(saved.strategy_id)["metadata"]
    assert metadata["request_summary"] == research.strategy_outline
    assert "display_name" not in metadata


def test_display_name_is_distinct_from_the_automatically_generated_identity(
    tmp_path: Path,
) -> None:
    service = StrategyService(WorkspacePaths(tmp_path / "workspace"))
    saved = service.save_draft(
        _STRATEGY_CODE,
        _YAML,
        research_for_config(_YAML),
        display_name="  我的均線研究  ",
    )
    assert saved.success, saved.error
    assert saved.strategy_id.startswith("strategy_")
    config = yaml.safe_load(Path(saved.config_path).read_text())
    assert config["strategy"]["display_name"] == "我的均線研究"


def test_anonymous_creation_retries_reserved_legacy_and_occupied_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = StrategyService(workspace)
    research = research_for_config(_YAML)
    original = service.save_draft(
        _STRATEGY_CODE, _YAML, research, strategy_id="existing"
    )
    assert original.success
    legacy = workspace.generated_strategies / "legacy.py"
    legacy.write_text("legacy source")
    residual = workspace.generated_strategies / "residual" / "revisions"
    residual.mkdir(parents=True)
    candidates = iter(["kd_crossover", "existing", "legacy", "residual", "unused"])
    monkeypatch.setattr(service, "_new_strategy_id", lambda: next(candidates))

    saved = service.save_draft(_STRATEGY_CODE, _YAML, research)

    assert saved.success and saved.strategy_id == "unused"
    current = service.load("existing")
    assert current is not None and current.revision_id == original.revision_id
    assert legacy.read_text() == "legacy source"
    assert not (residual.parent / "current.json").exists()
    assert not (workspace.generated_strategies / "kd_crossover").exists()


def test_exhausted_anonymous_names_fail_without_revising_the_existing_strategy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = StrategyService(WorkspacePaths(tmp_path / "workspace"))
    research = research_for_config(_YAML)
    original = service.save_draft(
        _STRATEGY_CODE, _YAML, research, strategy_id="occupied"
    )
    monkeypatch.setattr(service, "_new_strategy_id", lambda: "occupied")

    saved = service.save_draft(_STRATEGY_CODE, _YAML, research)

    assert not saved.success and saved.code == "strategy_id_generation_failed"
    assert service._revisions.revision_ids("occupied") == [original.revision_id]


def test_revisions_require_an_explicit_matching_current_revision(
    tmp_path: Path,
) -> None:
    service = StrategyService(WorkspacePaths(tmp_path / "workspace"))
    research = research_for_config(_YAML)
    first = service.save_draft(_STRATEGY_CODE, _YAML, research, strategy_id="fixture")
    duplicate = service.save_draft(
        _STRATEGY_CODE, _YAML, research, strategy_id="fixture"
    )
    assert not duplicate.success and duplicate.code == "strategy_id_exists"
    second = service.save_draft(
        _STRATEGY_CODE,
        _YAML,
        research,
        strategy_id="fixture",
        expected_revision_id=first.revision_id,
    )
    assert second.success and second.revision_id != first.revision_id
    stale = service.save_draft(
        _STRATEGY_CODE,
        _YAML,
        research,
        strategy_id="fixture",
        expected_revision_id=first.revision_id,
    )
    assert not stale.success and stale.code == "strategy_revision_conflict"
    current = service.load("fixture")
    assert current is not None and current.revision_id == second.revision_id
    assert len(service._revisions.revision_ids("fixture")) == 2


def test_revision_cannot_supply_only_an_expected_revision(tmp_path: Path) -> None:
    service = StrategyService(WorkspacePaths(tmp_path / "workspace"))
    saved = service.save_draft(
        _STRATEGY_CODE,
        _YAML,
        research_for_config(_YAML),
        expected_revision_id="0123456789ab4def8123456789abcdef",
    )
    assert not saved.success and saved.code == "invalid_revision_request"
    assert service._revisions.list_current() == []


def test_research_yaml_conflict_is_repairable_without_publishing(
    tmp_path: Path,
) -> None:
    service = StrategyService(WorkspacePaths(tmp_path / "workspace"))
    research = research_for_config(_YAML).model_copy(update={"symbol": "ETH/USDT"})
    saved = service.save_draft(
        _STRATEGY_CODE, _YAML, research, strategy_id="conflicting"
    )
    assert not saved.success and saved.code == "invalid_research_spec"
    assert service.load("conflicting") is None
    assert service._revisions.revision_ids("conflicting") == []


def test_inheritance_requires_and_uses_the_explicit_revision(tmp_path: Path) -> None:
    from tradingdev.domain.research import InheritResearchSpec

    service = StrategyService(WorkspacePaths(tmp_path / "workspace"))
    first = service.save_draft(
        _STRATEGY_CODE, _YAML, research_for_config(_YAML), strategy_id="fixture"
    )
    inherit = InheritResearchSpec(kind="inherit_selected_revision")
    rejected = service.save_draft(_STRATEGY_CODE, _YAML, inherit)
    assert not rejected.success and rejected.code == "invalid_research_spec"
    revised = service.save_draft(
        _STRATEGY_CODE,
        _YAML,
        inherit,
        strategy_id="fixture",
        expected_revision_id=first.revision_id,
    )
    assert revised.success, revised.error
    old = yaml.safe_load(Path(first.config_path).read_text())
    new = yaml.safe_load(Path(revised.config_path).read_text())
    assert new["research"]["strategy_outline"] == old["research"]["strategy_outline"]
    assert new["backtest"] == old["backtest"]
