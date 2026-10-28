"""Every fixed candidate receives offline evidence inside one supervised request."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError
from tests.app.test_preflight_service import _forbid_historical_execution
from tests.integration.execution_fixtures import research_input
from tests.preflight_fixtures import make_preflight_fixture

from tradingdev.adapters.execution.preflight_worker import execute_preflight
from tradingdev.adapters.execution.process_runner import (
    BoundedWorkerError,
    ProcessRunner,
)
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.contracts.execution_requests import ComparisonInput
from tradingdev.app.preflight_service import PreflightError, PreflightService
from tradingdev.app.prepared_execution import PreparedComparison
from tradingdev.app.research import resolve_research_config
from tradingdev.domain.comparison import ComparisonManifest
from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.preflight import ComparisonPreflightReceipt, PreflightRequest
from tradingdev.domain.strategies.capabilities import StrategyExecutionModeError
from tradingdev.domain.strategies.loader import StrategyLoader

if TYPE_CHECKING:
    import pandas as pd

    from tradingdev.domain.strategies.base import BaseStrategy


def _comparison(request: PreflightRequest, count: int = 2) -> PreflightRequest:
    arguments = deepcopy(request.arguments)
    arguments.pop("parameters", None)
    arguments["candidates"] = [
        {
            "label": f"Threshold {index}",
            "rationale": f"Evaluate fixed sensitivity proposal {index}.",
            "parameters": {"threshold": index / 1000},
        }
        for index in range(count)
    ]
    return PreflightRequest(kind="comparison", arguments=arguments)


@pytest.mark.parametrize(
    "invalid",
    [
        {"parameters": {}},
        {"validation_config": {"n_splits": 2}},
        {"param_ranges": {"threshold": [0, 1]}},
        {"candidates": []},
        {"candidates": [{"label": "x", "rationale": "x", "parameters": {}}] * 9},
        {"candidates": [{"label": " ", "rationale": "x", "parameters": {}}]},
        {"candidates": [{"label": "x", "rationale": " ", "parameters": {}}]},
        {"candidates": [{"label": "x", "rationale": "x"}]},
        {
            "candidates": [
                {"label": "x", "rationale": "x", "parameters": {}, "candidate_id": "x"}
            ]
        },
        {"research": {"kind": "inherit_selected_revision"}},
    ],
)
def test_comparison_input_rejects_search_global_parameters_and_incomplete_proposals(
    invalid: dict[str, Any],
) -> None:
    arguments = {
        "strategy_id": "example",
        "research": research_input(),
        "candidates": [
            {"label": "baseline", "rationale": "Explicit base", "parameters": {}}
        ],
        **invalid,
    }
    with pytest.raises(ValidationError):
        ComparisonInput.model_validate(arguments)


def test_all_eight_candidates_check_actual_parameters_without_history_or_base_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, single = make_preflight_fixture(tmp_path, monkeypatch)
    request = _comparison(single, count=8)
    request.arguments["research"] = {"kind": "inherit_selected_revision", "fees": 0}
    _forbid_historical_execution(monkeypatch)
    research_calls: list[dict[str, Any]] = []
    resolutions: list[dict[str, Any]] = []
    observed: list[tuple[int, float]] = []
    create = StrategyLoader.create_from_execution
    resolve = StrategyLoader.resolve_execution

    def capture_research(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = resolve_research_config(*args, **kwargs)
        research_calls.append(deepcopy(result))
        return result

    def capture_defaults(loader: StrategyLoader, config: dict[str, Any]) -> Any:
        resolutions.append(deepcopy(config))
        return resolve(loader, config)

    def capture(loader: StrategyLoader, *args: Any, **kwargs: Any) -> BaseStrategy:
        assert args[2] is None
        strategy = create(loader, *args, **kwargs)
        generate = strategy.generate_signals

        def no_training(*_args: Any, **_kwargs: Any) -> None:
            pytest.fail("Comparison preparation must not fit a strategy")

        def checked(frame: pd.DataFrame) -> pd.DataFrame:
            observed.append((len(frame), strategy.get_parameters()["threshold"]))
            return generate(frame)

        monkeypatch.setattr(strategy, "fit", no_training)
        monkeypatch.setattr(strategy, "generate_signals", checked)
        return strategy

    monkeypatch.setattr(
        "tradingdev.app.research_preparation.resolve_research_config", capture_research
    )
    monkeypatch.setattr(StrategyLoader, "resolve_execution", capture_defaults)
    monkeypatch.setattr(StrategyLoader, "create_from_execution", capture)
    before = {
        path: path.read_bytes() for path in workspace.root.rglob("*") if path.is_file()
    }
    result = execute_preflight(
        request, source_workspace=workspace, directory=tmp_path / "preflight"
    )
    assert isinstance(result.manifest, ComparisonManifest)
    assert isinstance(result.receipt, ComparisonPreflightReceipt)
    assert len(research_calls) == len(resolutions) == 1
    assert observed == [
        (rows, index / 1000) for index in range(8) for rows in (80, 240)
    ]
    assert {path: path.read_bytes() for path in before} == before
    assert not list(workspace.processed_data.iterdir())
    assert not list(workspace.runs.iterdir())
    result.receipt.verify_manifest(result.manifest)
    for index, (candidate, checked) in enumerate(
        zip(result.manifest.candidates, result.receipt.candidates, strict=True)
    ):
        assert (
            candidate.candidate_id == checked.candidate_id == f"candidate_{index + 1}"
        )
        assert checked.receipt.synthetic_fixture_rows == [80, 240]
        assert checked.receipt.manifest_hash == candidate.manifest.manifest_hash
        assert checked.receipt.parameter_combinations_checked is None
        config = candidate.manifest.config_copy()
        assert config["research"] == research_calls[0]["research"]
        assert config["backtest"]["fees"] == 0
        assert candidate.manifest.strategy_execution.constructor_kwargs == {
            "threshold": index / 1000
        }


def test_bundled_candidates_receive_both_synthetic_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_historical_execution(monkeypatch)
    request = PreflightRequest(
        kind="comparison",
        arguments={
            "strategy_id": "kd_crossover",
            "research": research_input(),
            "candidates": [
                {
                    "label": "Short",
                    "rationale": "Faster response",
                    "parameters": {"k_period": 9},
                },
                {"label": "Base", "rationale": "Reference setting", "parameters": {}},
            ],
        },
    )
    result = execute_preflight(
        request,
        source_workspace=WorkspacePaths(tmp_path / "source"),
        directory=tmp_path / "preflight",
    )
    assert isinstance(result.manifest, ComparisonManifest)
    assert isinstance(result.receipt, ComparisonPreflightReceipt)
    result.receipt.verify_manifest(result.manifest)
    periods = []
    for candidate in result.manifest.candidates:
        config = candidate.manifest.strategy_execution.constructor_kwargs["config"]
        assert isinstance(config, dict)
        periods.append(config["k_period"])
    assert periods == [9, 14]
    assert all(
        child.receipt.synthetic_fixture_rows == [80, 240]
        and child.receipt.signal_check == "synthetic_contract_passed"
        and child.receipt.training_requirement == "optional"
        for child in result.receipt.candidates
    )


@pytest.mark.parametrize("duplicate", ["effective_parameters", "label"])
def test_duplicate_candidates_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, duplicate: str
) -> None:
    workspace, single = make_preflight_fixture(tmp_path, monkeypatch)
    arguments = _comparison(single).model_dump(mode="json")["arguments"]
    if duplicate == "effective_parameters":
        # Empty overrides and an explicit default resolve to the same constructor.
        arguments["candidates"][1]["parameters"] = {}
    else:
        arguments["candidates"][1]["label"] = arguments["candidates"][0]["label"]
    with pytest.raises(PreflightError, match="duplicate|distinct"):
        execute_preflight(
            PreflightRequest(kind="comparison", arguments=arguments),
            source_workspace=workspace,
            directory=tmp_path / "preflight",
        )
    assert not list(workspace.runs.iterdir())


def test_later_candidate_contract_failure_produces_no_partial_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, single = make_preflight_fixture(
        tmp_path, monkeypatch, failure="exception"
    )
    arguments = _comparison(single).model_dump(mode="json")["arguments"]
    arguments["candidates"][1]["parameters"] = {"threshold": 9999.0}
    _forbid_historical_execution(monkeypatch)
    with pytest.raises(PreflightError, match="candidate_2.*contract failed"):
        execute_preflight(
            PreflightRequest(kind="comparison", arguments=arguments),
            source_workspace=workspace,
            directory=tmp_path / "preflight",
        )
    assert not list(workspace.runs.iterdir())


@pytest.mark.parametrize("serialize", ["child", "bundle"])
def test_comparison_requires_each_child_and_bundle_serialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, serialize: str
) -> None:
    workspace, single = make_preflight_fixture(tmp_path, monkeypatch)

    def fail(*_args: Any, **_kwargs: Any) -> str:
        raise ValueError("serialization failed")

    monkeypatch.setattr(
        ExecutionManifest if serialize == "child" else ComparisonManifest,
        "model_dump_json",
        fail,
    )
    with pytest.raises(ValueError, match="serialization failed"):
        execute_preflight(
            _comparison(single),
            source_workspace=workspace,
            directory=tmp_path / "preflight",
        )


@pytest.mark.parametrize("strategy_id", ["quantile_volume", "xgboost_direction"])
def test_training_required_comparison_returns_existing_guidance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strategy_id: str
) -> None:
    _forbid_historical_execution(monkeypatch)
    request = PreflightRequest(
        kind="comparison",
        arguments={
            "strategy_id": strategy_id,
            "research": research_input(),
            "candidates": [
                {"label": "Base", "rationale": "Reference", "parameters": {}}
            ],
        },
    )
    with pytest.raises(StrategyExecutionModeError) as caught:
        execute_preflight(
            request,
            source_workspace=WorkspacePaths(tmp_path / "source"),
            directory=tmp_path / "preflight",
        )
    assert caught.value.guidance.strategy_id == strategy_id
    assert caught.value.guidance.recommended_mode == "walk_forward"


@pytest.mark.parametrize("outcome", ["success", "timeout", "wrong_receipt"])
def test_parent_uses_one_whole_comparison_budget_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    workspace, single = make_preflight_fixture(tmp_path, monkeypatch)
    request = _comparison(single, count=8)
    directory = tmp_path / "parent"
    directory.mkdir()
    monkeypatch.setattr(
        "tradingdev.app.preflight_service.mkdtemp", lambda **_kw: str(directory)
    )
    budgets: list[float] = []

    def run(
        _runner: ProcessRunner, _module: str, *args: str, timeout_seconds: float
    ) -> None:
        budgets.append(timeout_seconds)
        assert json.loads(Path(args[0]).read_text())["request"]["kind"] == "comparison"
        if outcome == "timeout":
            raise BoundedWorkerError(
                "preflight_timeout", "timeout", cleanup_verified=True
            )
        payload = execute_preflight(
            request, source_workspace=workspace, directory=directory
        )
        result = payload.model_dump(mode="json")
        if outcome == "wrong_receipt":
            result["receipt"]["candidates"][1]["receipt"]["manifest_hash"] = "0" * 64
        Path(args[1]).write_text(json.dumps({"success": True, "result": result}))

    monkeypatch.setattr(ProcessRunner, "run_module", run)
    if outcome == "success":
        result = PreflightService(workspace).prepare(request)
        assert isinstance(result.prepared, PreparedComparison)
        assert isinstance(result.receipt, ComparisonPreflightReceipt)
        assert result.prepared.candidate_fixture_rows == {
            f"candidate_{index}": (80, 240) for index in range(1, 9)
        }
    else:
        with pytest.raises(PreflightError) as caught:
            PreflightService(workspace).prepare(request)
        assert caught.value.code == (
            "preflight_timeout"
            if outcome == "timeout"
            else "invalid_preflight_response"
        )
    assert budgets == [60]
    assert not directory.exists()
