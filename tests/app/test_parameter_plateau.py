"""Saved collections preserve missing coordinates and analysis identity."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest
from tests.adapters.test_ml_datasets import _frame

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.ml_datasets import MLDatasetStore
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.ml_surface import dataset_context
from tradingdev.app.plateau_service import PlateauService
from tradingdev.domain.data.cache import DataIdentity
from tradingdev.domain.ml.leaderboard import EvaluationContext, LeaderboardEntry
from tradingdev.domain.ml.market_evidence import raw_market_series
from tradingdev.domain.ml.registry.leaderboard_store import LeaderboardStore

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from sqlite3 import Connection

    from tradingdev.app.trade_history_service import LoadedHistoryRun

type Collection = tuple[WorkspacePaths, SQLiteStore, PlateauService]


def test_middle_undefined_keeps_its_axis(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path)
    context, dataset_id = _dataset(workspace)
    store = LeaderboardStore(workspace.ml_leaderboard)
    base = LeaderboardEntry(
        "a",
        "rule",
        context,
        {"ic_mean": 0.5},
        4,
        parameters={"threshold": 10},
        method_family="rule",
        method_name="threshold",
        dataset_id=dataset_id,
    )
    for index, value in enumerate((0.5, None, 0.6)):
        store.record(
            replace(
                base,
                entry_id=str(index),
                metrics={"ic_mean": value},
                parameters={"threshold": (index + 1) * 10},
            )
        )
    service = PlateauService(workspace=workspace, store=SQLiteStore(workspace))
    result = service.analyze(
        metric="signal.ic_mean",
        axes={"/threshold": [10, 20, 30]},
        entry_ids=["0", "1", "2"],
    )
    assert result["success"], result
    assert result["axes"] == {"/threshold": [10.0, 20.0, 30.0]}
    assert result["summary"]["best_neighborhood_size"] == 0
    assert [row["status"] for row in result["cells"]["items"]] == [
        "available",
        "undefined",
        "available",
    ]
    assert result["cells"]["items"][1]["value"] is None


@pytest.fixture
def collection(tmp_path: Path) -> tuple[WorkspacePaths, SQLiteStore, PlateauService]:
    workspace = WorkspacePaths(tmp_path)
    sql = SQLiteStore(workspace)
    return workspace, sql, PlateauService(workspace=workspace, store=sql)


def _record(
    workspace: WorkspacePaths,
    values: list[float | None],
    *,
    key: str = "ic_mean",
    parameter: str = "threshold",
) -> list[str]:
    context, dataset_id = _dataset(workspace)
    store = LeaderboardStore(workspace.ml_leaderboard)
    for index, value in enumerate(values):
        store.record(
            LeaderboardEntry(
                str(index),
                "rule",
                context,
                {key: value},
                4,
                parameters={parameter: index + 1},
                method_family="rule",
                method_name="threshold",
                dataset_id=dataset_id,
            )
        )
    return [str(index) for index in range(len(values))]


def test_minimize_ties_and_sample_threshold(collection: Collection) -> None:
    workspace, _, service = collection
    ids = _record(workspace, [2.0, 1.0, 1.0], key="ic_std")
    result = service.analyze(
        metric="signal.ic_std",
        axes={"/threshold": [1, 2, 3]},
        entry_ids=ids,
        min_neighborhood=2,
    )
    assert result["success"], result
    assert result["direction"] == "minimize"
    assert result["definition"]["unit"] == "fraction"
    assert result["summary"]["best"]["identity"]["entry_id"] == "1"
    assert result["summary"]["best"]["value"] == 1.0
    assert result["summary"]["best_score"] == -1.0
    assert result["summary"]["plateau_best"]["identity"]["entry_id"] == "1"
    assert result["summary"]["plateau_best_score"] == -1.5
    assert result["summary"]["plateau_best_value"] == 1.0
    assert result["cells"]["items"][0]["value"] == 2.0
    assert result["cells"]["items"][0]["score"] == -2.0


def test_full_undefined_layer_and_absent_cells(collection: Collection) -> None:
    workspace, _, service = collection
    ids = _record(workspace, [0.5, None, 0.6, 0.5, None, 0.6])
    store = LeaderboardStore(workspace.ml_leaderboard)
    for i, entry in enumerate(store.entries()):
        store.record(replace(entry, parameters={"x": i % 3 + 1, "y": i // 3 + 1}))
    result = service.analyze(
        metric="signal.ic_mean", axes={"/x": [1, 2, 3], "/y": [1, 2, 3]}, entry_ids=ids
    )
    assert result["success"], result
    assert (
        result["n_candidates"],
        result["n_available"],
        result["n_undefined"],
        result["n_absent"],
    ) == (6, 4, 2, 3)
    assert all(
        row["value"] is None
        for row in result["cells"]["items"]
        if row["status"] != "available"
    )
    assert all(
        row["parameters"]["/x"] == 2
        for row in result["cells"]["items"]
        if row["status"] == "undefined"
    )
    assert result["summary"]["best_neighborhood_size"] == 1
    assert result["summary"]["best_neighborhood_mean_score"] == 0.6


@pytest.mark.parametrize("values", [[], [None, None]])
def test_empty_or_all_undefined(
    collection: Collection, values: list[float | None]
) -> None:
    workspace, _, service = collection
    result = service.analyze(
        metric="signal.ic_mean",
        axes={"/threshold": [1, 2]},
        entry_ids=_record(workspace, values),
    )
    assert result["success"], result
    assert result["n_available"] == 0
    assert result["summary"]["best"] is None
    assert result["summary"]["plateau_best"] is None
    assert result["summary"]["best_score"] is None


@pytest.mark.parametrize(
    "axes, radius, minimum, message",
    [
        ({"/threshold": ["small", "large"]}, 1, None, "numeric"),
        ({"/threshold": [True]}, 1, None, "numeric"),
        ({"/threshold": [float("nan")]}, 1, None, "numeric"),
        ({"/threshold": [1, 1]}, 1, None, "unique"),
        ({}, 1, None, "numeric axes"),
        ({"/threshold": []}, 1, None, "nonempty"),
        ({"/threshold": list(range(4097))}, 1, None, "4096"),
        ({"/threshold": [1]}, 0, None, "radius"),
        ({"/threshold": [1]}, 1, 0, "min_neighborhood"),
    ],
)
def test_bad_request(
    collection: Collection,
    axes: dict[str, Any],
    radius: int,
    minimum: int | None,
    message: str,
) -> None:
    _, _, service = collection
    result = service.analyze(
        metric="signal.ic_mean",
        axes=axes,
        entry_ids=[],
        radius=radius,
        min_neighborhood=minimum,
    )
    assert result["success"] is False
    assert message in result["error"]
    assert result["next_step"]


def test_duplicate_coordinates_and_incompatible_parameters(
    collection: Collection,
) -> None:
    workspace, _, service = collection
    ids = _record(workspace, [0.5, 0.6])
    store = LeaderboardStore(workspace.ml_leaderboard)
    first, second = store.entries()
    store.record(replace(second, parameters=first.parameters))
    result = service.analyze(
        metric="signal.ic_mean", axes={"/threshold": [1, 2]}, entry_ids=ids
    )
    assert not result["success"]
    assert "Duplicate coordinate" in result["error"]
    assert "'0'" in result["error"] and "'1'" in result["error"]
    store.record(replace(second, parameters={"threshold": 2, "hidden": "different"}))
    result = service.analyze(
        metric="signal.ic_mean", axes={"/threshold": [1, 2]}, entry_ids=ids
    )
    assert not result["success"]
    assert "undeclared varying" in result["error"]


def test_page_snapshot_survives_re_evaluation_and_restart(
    collection: Collection,
) -> None:
    import json
    from pathlib import Path

    workspace, sql, service = collection
    ids = _record(workspace, [0.5, None, 0.6])
    first = service.analyze(
        metric="signal.ic_mean", axes={"/threshold": [1, 2, 3]}, entry_ids=ids, limit=1
    )
    identity = first["analysis_id"]
    blob = json.loads(Path(first["artifact"]["path"]).read_text())
    _record(workspace, [9.0, 9.0, 9.0])
    reopened = PlateauService(workspace=workspace, store=sql)
    pages = [reopened.read(identity, offset=i, limit=1) for i in range(3)]
    assert [page["cells"]["items"][0] for page in pages] == blob["cells"]
    assert all(page["summary"] == first["summary"] for page in pages)
    assert all(page["source_hash"] == first["source_hash"] for page in pages)
    assert pages[-1]["cells"]["next_offset"] is None
    assert "evidence" not in pages[0]
    assert blob["evidence"]["entries"][1]["metrics"]["ic_mean"] is None
    changed = reopened.analyze(
        metric="signal.ic_mean", axes={"/threshold": [1, 2, 3]}, entry_ids=ids
    )
    assert changed["analysis_id"] != identity
    assert changed["source_hash"] != first["source_hash"]


def test_artifact_tampering_and_republication(collection: Collection) -> None:
    from pathlib import Path

    workspace, _, service = collection
    ids = _record(workspace, [0.5])
    request: dict[str, Any] = {
        "metric": "signal.ic_mean",
        "axes": {"/threshold": [1]},
        "entry_ids": ids,
    }
    first = service.analyze(**request)
    second = service.analyze(**request)
    assert first == second
    path = Path(first["artifact"]["path"])
    import json

    altered = json.loads(path.read_text())
    altered["summary"]["best_score"] = 99.0
    changed_bytes = json.dumps(altered)
    path.write_text(changed_bytes)
    rejected = service.read(first["analysis_id"])
    assert not rejected["success"]
    assert "hash mismatch" in rejected["error"]
    result = service.analyze(**request)
    assert not result["success"]
    assert "changed" in result["error"]
    assert path.read_text() == changed_bytes
    assert not list(path.parent.glob(".publish-*"))


def test_saved_optimization_uses_every_training_scope(
    collection: Collection, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.app.test_selection_evidence import _publish

    workspace, sql, service = collection
    _publish(workspace, sql, "grid", 3)
    calls = []
    load = service.history.load_run

    def once(run_id: str) -> LoadedHistoryRun:
        calls.append(run_id)
        return load(run_id)

    monkeypatch.setattr(service.history, "load_run", once)
    result = service.analyze(
        metric="run.sharpe_ratio", axes={"/window": [1, 2, 3]}, run_id="grid"
    )
    assert result["success"], result
    assert calls == ["grid"]
    assert result["n_candidates"] == 3
    assert result["n_undefined"] == 1
    assert result["source"]["completeness"] == "verified_saved_grid"
    assert result["summary"]["best"]["identity"]["scope"] == "trial/2/train"
    assert {row["identity"]["scope"] for row in result["cells"]["items"]} == {
        f"trial/{i}/train" for i in range(3)
    }
    rejected = service.analyze(
        metric="run.sharpe_ratio", axes={"/window": [1, 3]}, run_id="grid"
    )
    assert not rejected["success"]
    assert "outside declared" in rejected["error"]
    extra = service.analyze(
        metric="run.sharpe_ratio", axes={"/window": [1, 2, 3, 4]}, run_id="grid"
    )
    assert not extra["success"]
    assert "exactly match" in extra["error"]


def test_partial_or_historical_optimization_is_refused(
    collection: Collection, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.app.test_selection_evidence import _publish

    workspace, sql, service = collection
    _publish(workspace, sql, "grid", 3)
    original = service.history.load_run("grid")
    for loaded in (
        replace(original, manifest=None),
        replace(
            original,
            performance=original.performance.model_copy(
                update={
                    "scopes": {
                        key: value
                        for key, value in original.performance.scopes.items()
                        if key != "trial/1/train"
                    }
                }
            ),
        ),
    ):
        monkeypatch.setattr(
            service.history, "load_run", lambda _, snapshot=loaded: snapshot
        )
        result = service.analyze(
            metric="run.sharpe_ratio", axes={"/window": [1, 2, 3]}, run_id="grid"
        )
        assert not result["success"]
        assert "manifest" in result["error"] or "complete grid" in result["error"]


def test_signal_model_numeric_axis_and_missing_metric(collection: Collection) -> None:
    workspace, _, service = collection
    ids = _record(workspace, [0.5, 0.6], parameter="n_estimators")
    store = LeaderboardStore(workspace.ml_leaderboard)
    for entry in store.entries():
        store.record(
            replace(entry, method_family="ml_model", method_name="same_algorithm")
        )
    result = service.analyze(
        metric="signal.ic_mean", axes={"/n_estimators": [1, 2]}, entry_ids=ids
    )
    assert result["success"], result
    absent_metric = service.analyze(
        metric="signal.ic_std", axes={"/n_estimators": [1, 2]}, entry_ids=ids
    )
    assert absent_metric["success"]
    assert all(
        row["reason"] == "metric_not_recorded"
        for row in absent_metric["cells"]["items"]
    )


def test_unknown_horizon_evidence_and_data_mismatch(collection: Collection) -> None:
    workspace, _, service = collection
    ids = _record(workspace, [0.5, 0.6])
    store = LeaderboardStore(workspace.ml_leaderboard)
    first, second = store.entries()
    store.record(
        replace(
            second,
            context=replace(second.context, horizon=2),
            parameters=first.parameters,
        )
    )
    result = service.analyze(
        metric="signal.ic_mean", axes={"horizon": [1, 2]}, entry_ids=ids
    )
    assert not result["success"]
    assert "context does not match" in result["error"]
    store.record(
        replace(second, context=replace(second.context, data_fingerprint="other"))
    )
    result = service.analyze(
        metric="signal.ic_mean", axes={"/threshold": [1, 2]}, entry_ids=ids
    )
    assert not result["success"]
    assert "context does not match" in result["error"]


def _dataset(
    workspace: WorkspacePaths,
    *,
    horizon: int = 1,
    shift: bool = False,
    raw_change: bool = False,
) -> tuple[EvaluationContext, str]:
    frame = _frame()
    if shift:
        frame["timestamp"] += pd.Timedelta(minutes=30)
    frame["label_start"] = frame["timestamp"]
    frame["label_end"] = frame["timestamp"] + pd.Timedelta(hours=horizon)
    frame["raw_row"] = range(len(frame))
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC"),
            **{
                key: [10.0 + int(raw_change)] * 10
                for key in ("open", "high", "low", "close", "volume")
            },
        }
    )
    identity = DataIdentity(
        source="binance_vision", market_type="spot", symbol="BTC/USDT", timeframe="1h"
    )
    _, manifest = MLDatasetStore(workspace.ml_datasets).save(
        frame,
        {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "horizon": horizon,
            "feature_set": "fixture",
            "code_fingerprint": "frozen-feature-code",
            "feature_columns": ["feature"],
            "kind": "single",
            "source": "binance_vision",
            "market_type": "spot",
            "raw_market": {"version": 1, "series": [raw_market_series(raw, identity)]},
        },
    )
    return dataset_context(manifest, window=60, n_quantiles=5), manifest["dataset_id"]


@pytest.mark.parametrize(
    "shift, raw_change, valid",
    [(False, False, True), (True, False, False), (False, True, False)],
)
def test_cross_horizon_verifies_both_raw_and_evaluated_rows(
    collection: Collection, shift: bool, raw_change: bool, valid: bool
) -> None:
    workspace, _, service = collection
    store = LeaderboardStore(workspace.ml_leaderboard)
    for index, horizon in enumerate((1, 2)):
        context, dataset_id = _dataset(
            workspace,
            horizon=horizon,
            shift=shift and index == 1,
            raw_change=raw_change and index == 1,
        )
        store.record(
            LeaderboardEntry(
                str(index),
                "rule",
                context,
                {"ic_mean": 0.5},
                4,
                dataset_id=dataset_id,
                parameters={"threshold": 1},
                method_family="rule",
                method_name="threshold",
            )
        )
    result = service.analyze(
        metric="signal.ic_mean", axes={"horizon": [1, 2]}, entry_ids=["0", "1"]
    )
    assert result["success"] is valid, result
    if valid:
        assert (
            result["source"]["market_comparison"]["raw_baseline"]["status"]
            == "verified"
        )
        assert (
            result["source"]["market_comparison"]["evaluation_rows"]["status"]
            == "verified"
        )
    else:
        assert (
            "baseline is mismatch" in result["error"]
            or "verified evaluation rows" in result["error"]
        )


@pytest.mark.parametrize("failure", [OSError("disk full"), TimeoutError("deadline")])
def test_failed_publication_cleans_staging(
    collection: Collection, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    import os

    workspace, sql, service = collection
    ids = _record(workspace, [0.5])

    def refuse(*args: object) -> None:
        raise failure

    monkeypatch.setattr(os, "link", refuse)
    result = service.analyze(
        metric="signal.ic_mean", axes={"/threshold": [1]}, entry_ids=ids
    )
    assert not result["success"]
    assert str(failure) in result["error"]
    assert list(service.artifacts.root.iterdir()) == []
    assert sql.list_artifacts() == []


def test_read_and_selection_failures_are_actionable(collection: Collection) -> None:
    workspace, _, service = collection
    ids = _record(workspace, [0.5])
    for request in (
        {"entry_ids": ["unknown"]},
        {"entry_ids": ["0", "0"]},
        {"entry_ids": ids, "run_id": "grid"},
        {},
        {"entry_ids": ids, "metric": "run.sharpe_ratio"},
        {"entry_ids": ids, "metric": "signal.missing"},
        {"entry_ids": ids, "axes": {"/missing": [1]}},
        {"entry_ids": ids, "limit": 0},
        {"run_id": "missing", "metric": "run.sharpe_ratio"},
    ):
        arguments: dict[str, Any] = {
            "metric": "signal.ic_mean",
            "axes": {"/threshold": [1]},
            **request,
        }
        result = service.analyze(**arguments)
        assert not result["success"], request
        assert result["error"] and result["next_step"]
    result = service.analyze(
        metric="signal.ic_mean", axes={"/threshold": [1]}, entry_ids=ids
    )
    identity = result["analysis_id"]
    for args in ({"offset": -1}, {"limit": 101}, {"offset": 2}):
        assert not service.read(identity, **args)["success"]
    assert service.read(identity, offset=1)["cells"]["items"] == []
    for identity in ("../unsafe", "a" * 64):
        assert not service.read(identity)["success"]


def test_real_rule_threshold_sweep(collection: Collection) -> None:
    from tradingdev.app.ml_service import MLService

    workspace, _, service = collection
    _, dataset_id = _dataset(workspace)
    ml = MLService(workspace=workspace)
    ids = [
        ml.evaluate_rule(
            dataset_id=dataset_id,
            column="feature",
            threshold=value,
            window=2,
            n_quantiles=2,
            record=True,
        )["recorded"]["entry_id"]
        for value in (0.0, 0.2)
    ]
    result = service.analyze(
        metric="signal.ic_mean", axes={"/threshold": [0.0, 0.2]}, entry_ids=ids
    )
    assert result["success"], result
    assert result["n_candidates"] == 2
    assert result["n_absent"] == 0
    # Changing direction remains a different method even when threshold is an axis.
    third = ml.evaluate_rule(
        dataset_id=dataset_id,
        column="feature",
        threshold=0.2,
        direction="below",
        window=2,
        n_quantiles=2,
        record=True,
    )
    mismatch = service.analyze(
        metric="signal.ic_mean",
        axes={"/threshold": [0.0, 0.2]},
        entry_ids=[ids[0], third["recorded"]["entry_id"]],
    )
    assert not mismatch["success"]
    assert "Incompatible" in mismatch["error"]


def _publish_custom_optimization(
    workspace: WorkspacePaths,
    sql: SQLiteStore,
    *,
    nested: bool = False,
    wrong_dates: bool = False,
) -> None:
    from tests.app.test_run_ranking import _manifest
    from tests.app.test_selection_evidence import _publish

    from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
    from tradingdev.adapters.storage.performance import PerformanceStore
    from tradingdev.domain.execution import ExecutionManifest, OptimizationSpec
    from tradingdev.domain.performance.artifacts import build_artifacts
    from tradingdev.domain.strategies.execution import StrategyExecution

    _publish(workspace, sql, "base-grid", 3)
    original = PlateauService(workspace=workspace, store=sql).history.load_run(
        "base-grid"
    )
    manifest = original.manifest
    assert manifest is not None
    assert manifest.optimization is not None
    scopes = dict(original.performance.scopes)
    if nested:
        base = _manifest()
        manifest = ExecutionManifest.create(
            kind="optimization",
            config=manifest.config_copy(),
            strategy_execution=StrategyExecution(
                kind="generated",
                program=base.strategy_execution.program,
                constructor_kwargs={"model": {"x": 1, "y": 2}},
            ),
            optimization=OptimizationSpec.model_validate(
                {
                    **manifest.optimization.model_dump(mode="json"),
                    "param_ranges": {"model": [{}, {"x": 2}, {"x": 3}]},
                }
            ),
        )
        for scope_id, scope in scopes.items():
            params = {
                "model": {}
                if scope.trial_index == 0
                else {"x": 2 if scope.trial_index == 1 else 3}
            }
            scopes[scope_id] = scope.model_copy(update={"parameters": params})
    observed = dict(original.observations.scopes)
    if wrong_dates:
        observed["trial/2/train"] = observed["trial/2/train"].model_copy(
            update={"timestamps": ["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"]}
        )
    ExecutionManifestStore(workspace).publish("custom-grid", manifest)
    sql.create_run(
        run_id="custom-grid",
        job_id="custom-grid",
        strategy_id="fixture",
        revision_id="rev-1",
        artifact_dir=workspace.runs / "custom-grid",
        metrics={},
        manifest_hash=manifest.manifest_hash,
    )
    PerformanceStore(workspace, sql).publish(
        build_artifacts(
            "custom-grid",
            manifest.manifest_hash,
            "test",
            scopes,
            observed,
            selected_train_scope="trial/2/train",
            optimization_selection=original.performance.optimization_selection,
        )
    )


def test_nested_search_uses_effective_coordinates(collection: Collection) -> None:
    workspace, sql, service = collection
    _publish_custom_optimization(workspace, sql, nested=True)
    result = service.analyze(
        metric="run.sharpe_ratio", axes={"/model/x": [1, 2, 3]}, run_id="custom-grid"
    )
    assert result["success"], result
    assert result["n_candidates"] == 3
    assert result["n_undefined"] == 1
    assert [cell["parameters"]["/model/x"] for cell in result["cells"]["items"]] == [
        1,
        2,
        3,
    ]


def test_optimization_checks_observation_range_without_research_record(
    collection: Collection,
) -> None:
    workspace, sql, service = collection
    _publish_custom_optimization(workspace, sql, wrong_dates=True)
    result = service.analyze(
        metric="run.sharpe_ratio", axes={"/window": [1, 2, 3]}, run_id="custom-grid"
    )
    assert not result["success"]
    assert "outside the approved scope range" in result["error"]


def test_sqlite_registration_failure_and_retry(
    collection: Collection, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import contextmanager

    workspace, sql, service = collection
    ids = _record(workspace, [0.5])
    request: dict[str, Any] = {
        "metric": "signal.ic_mean",
        "axes": {"/threshold": [1]},
        "entry_ids": ids,
    }
    connect = sql.connect

    @contextmanager
    def readonly() -> Iterator[Connection]:
        with connect() as connection:
            connection.execute("PRAGMA query_only=on")
            yield connection

    with monkeypatch.context() as patch:
        patch.setattr(sql, "connect", readonly)
        result = service.analyze(**request)
    assert not result["success"]
    assert "readonly" in result["error"]
    assert result["next_step"]
    assert len(list(service.artifacts.root.glob("*.json"))) == 1
    assert not list(service.artifacts.root.glob(".publish-*"))
    result = service.analyze(**request)
    assert result["success"], result
    with sql.connect() as connection:
        connection.execute("ALTER TABLE artifacts RENAME TO hidden_artifacts")
    failed = service.read(result["analysis_id"])
    assert not failed["success"]
    assert "no such table" in failed["error"]


def test_real_model_hyperparameter_sweep(collection: Collection) -> None:
    from tests.mcp.ml_fixtures import model_pair, seed_bars

    from tradingdev.app.ml_service import MLService

    workspace, _, service = collection
    seed_bars(workspace)
    ml = MLService(workspace=workspace)
    first = model_pair(ml)
    second = ml.train_model(
        dataset_id=first["dataset_id"],
        model_kind="lightgbm_direction",
        parameters={"n_estimators": 18, "max_depth": 3, "n_jobs": 1},
    )["model"]
    ids = [
        ml.evaluate_prediction(
            dataset_id=first["dataset_id"],
            model_id=identity,
            window=20,
            n_quantiles=5,
            record=True,
        )["recorded"]["entry_id"]
        for identity in (first["model_id"], second["model_id"])
    ]
    result = service.analyze(
        metric="signal.ic_mean", axes={"/n_estimators": [12, 18]}, entry_ids=ids
    )
    assert result["success"], result
    assert result["n_candidates"] == 2
    assert result["n_absent"] == 0
    assert {
        cell["parameters"]["/n_estimators"] for cell in result["cells"]["items"]
    } == {12, 18}
