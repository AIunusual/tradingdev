"""Application contract for paired comparison of saved series."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Any

import pytest
from tests.strategy_fixtures import GENERATED_PROGRAM

from tradingdev.adapters.storage.evaluation_series import EvaluationSeriesStore
from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths, sha256_text
from tradingdev.adapters.storage.performance import PerformanceStore
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.artifact_service import ArtifactService
from tradingdev.app.contracts.paired_comparison import (
    PairedComparisonResponse,
    SeriesSelection,
)
from tradingdev.app.paired_comparison_service import (
    ALGORITHM_VERSION,
    RAW_PRICE_ARTIFACT_TYPE,
    PairedComparisonError,
    PairedComparisonService,
)
from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.ml.evaluation.series import build_bundle, build_signal_series
from tradingdev.domain.performance.artifacts import (
    BenchmarkCoverage,
    BenchmarkDefinitionSnapshot,
    BenchmarkScope,
    PerformanceScope,
    ScopeObservations,
    build_artifacts,
)
from tradingdev.domain.strategies.execution import StrategyExecution

if TYPE_CHECKING:
    from collections.abc import Sequence

FIRST = "2024-01-01T00:00:00+00:00"


def _timestamps(count: int, *, start: int = 1, skip: int | None = None) -> list[str]:
    import pandas as pd

    first = pd.Timestamp("2024-01-01", tz="UTC")
    return [
        (first + pd.Timedelta(days=day)).isoformat()
        for day in range(start - 1, start - 1 + count)
        if day + 1 != skip
    ]


def _settings(**changes: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "frequency": "1d",
        "periods_per_year": 365.0,
        "risk_free_rate": 0.0,
        "return_std_ddof": 1,
        "terminal_close_policy": "saved_as_executed",
        "trade_scope": "closed",
        "calendar_timezone": "UTC",
    }
    values.update(changes)
    return values


def _publish_run(
    workspace: WorkspacePaths,
    store: SQLiteStore,
    run_id: str,
    *,
    returns: Sequence[float | None] | None = None,
    timestamps: list[str] | None = None,
    mode: str = "signal",
    settings: dict[str, Any] | None = None,
    scopes: list[str] | None = None,
    benchmark: BenchmarkScope | None = None,
) -> None:
    timestamps = _timestamps(30) if timestamps is None else timestamps
    returns = [0.01] * len(timestamps) if returns is None else list(returns)
    config: dict[str, Any] = {
        "strategy": {"id": "fixture", "parameters": {"fast": 12}},
        "backtest": {
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "mode": mode,
            "init_cash": None if mode == "volume" else 1000.0,
            "periods_per_year": 365.0,
            "risk_free_rate": 0.0,
        },
    }
    manifest = ExecutionManifest.create(
        kind="backtest",
        config=config,
        strategy_execution=StrategyExecution(
            kind="generated", program=GENERATED_PROGRAM, constructor_kwargs={"fast": 12}
        ),
    )
    store.create_run(
        run_id=run_id,
        job_id=run_id,
        strategy_id="fixture",
        artifact_dir=workspace.runs / run_id,
        metrics={"total_pnl": 1.0},
        manifest_hash=manifest.manifest_hash,
        dataset_id="dataset-a",
    )
    ExecutionManifestStore(workspace).publish(run_id, manifest)
    scope_names = ["full"] if scopes is None else scopes
    performance: dict[str, PerformanceScope] = {}
    observations: dict[str, ScopeObservations] = {}
    for name in scope_names:
        performance[name] = PerformanceScope.model_validate(
            {
                "mode": mode,
                "values": {"total_pnl": 1.0},
                "parameters": {"fast": 12},
                "metadata": {"settings": settings or _settings()},
            }
        )
        observations[name] = ScopeObservations(
            init_cash=None if mode == "volume" else 1000.0,
            equity_curve=[1000.0 + index for index in range(len(returns))],
            returns=returns,
            timestamps=timestamps,
            trades=[],
        )
    artifacts = build_artifacts(
        run_id,
        manifest.manifest_hash,
        scope_names[0],
        performance,
        observations,
        benchmarks={"full": benchmark} if benchmark is not None else None,
    )
    PerformanceStore(workspace, store).publish(artifacts)


def _publish_series(
    workspace: WorkspacePaths,
    artifact_id_name: str,
    *,
    ic: list[float | None],
    settings: dict[str, Any] | None = None,
    holdout: bool = False,
) -> str:
    length = len(ic)
    frame = build_signal_series(
        [float(index % 5) for index in range(length)],
        [0.01 * (index % 3 + 1) for index in range(length)],
        _timestamps(length),
        window=3,
        # The builder accepts null IC values at runtime; its annotation is too narrow.
        pearson_ic=ic,  # type: ignore[arg-type]
        spearman_ic=ic,  # type: ignore[arg-type]
    )
    bundle = build_bundle(
        frame,
        kind="signal_evaluation",
        evaluation_id=f"eval-{artifact_id_name}",
        dataset_id="dataset-a",
        dataset_hash="hash-a",
        settings=settings or {"horizon": 1, "window": 3, "n_quantiles": 5},
        method_family="rule",
        producer_source="saved_evaluation",
        signal_fingerprint=f"fingerprint-{artifact_id_name}",
        entry_id=f"entry-{artifact_id_name}",
        training_dataset_id="dataset-train" if holdout else "dataset-a",
        training_status="fitted" if holdout else "no_fitting",
        created_at="2024-02-01T00:00:00+00:00",
    )
    return EvaluationSeriesStore(workspace.ml_evaluation_series).save(bundle)


@pytest.fixture
def service(
    tmp_path: Path,
) -> tuple[WorkspacePaths, SQLiteStore, PairedComparisonService]:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    return (
        workspace,
        store,
        PairedComparisonService(workspace=workspace, store=store),
    )


def test_two_runs_use_one_shared_index_and_reported_direction(
    service: tuple[WorkspacePaths, SQLiteStore, PairedComparisonService],
) -> None:
    workspace, store, comparison = service
    a_returns = [0.01] * 30
    b_returns = [0.02 + (0.001 if index % 2 == 0 else -0.001) for index in range(30)]
    _publish_run(workspace, store, "run-a", returns=a_returns)
    _publish_run(
        workspace,
        store,
        "run-b",
        returns=b_returns,
        timestamps=_timestamps(30, start=2),
    )
    payload = comparison.compare(
        SeriesSelection(kind="run", run_id="run-a", scope="full"),
        SeriesSelection(kind="run", run_id="run-b", scope="full"),
    )
    response = PairedComparisonResponse.model_validate(payload)
    assert response.status == "available"
    assert response.result is not None
    # Hand-computed on the 29 shared dates: b keeps 15 positive and 14
    # negative offsets, so mean(b) = 0.02 + 0.001/29 and a - b is its negative.
    assert response.result.estimate == pytest.approx(-0.01 - 0.001 / 29, abs=1e-12)
    assert response.result.lower <= response.result.estimate <= response.result.upper
    assert response.statistic == "mean"
    assert response.difference_direction == "a_minus_b"
    assert response.result.seed == 20261127
    assert response.alignment.overlap == 29
    assert response.alignment.valid_observations == 29
    assert response.alignment.excluded_counts["not_in_a:a"] == 1
    assert response.alignment.excluded_counts["not_in_b:b"] == 1
    assert response.sources[0].source_hash != response.sources[1].source_hash


def test_two_days_are_insufficient_without_padding(
    service: tuple[WorkspacePaths, SQLiteStore, PairedComparisonService],
) -> None:
    workspace, store, comparison = service
    _publish_run(
        workspace,
        store,
        "run-a",
        returns=[0.01, 0.02, 0.03],
        timestamps=_timestamps(3),
    )
    _publish_run(
        workspace,
        store,
        "run-b",
        returns=[0.04, 0.05, 0.06],
        timestamps=_timestamps(3, start=2),
    )
    payload = comparison.compare(
        {"kind": "run", "run_id": "run-a"},
        {"kind": "run", "run_id": "run-b"},
    )
    assert payload["status"] == "unavailable"
    assert payload["unavailable"]["reason"] == "insufficient_observations"
    assert payload["alignment"]["original_lengths"] == {"a": 3, "b": 3}
    assert payload["alignment"]["overlap"] == 2
    assert payload["alignment"]["valid_observations"] == 2
    assert payload["result"] is None


def test_no_common_dates_has_its_own_reason(
    service: tuple[WorkspacePaths, SQLiteStore, PairedComparisonService],
) -> None:
    workspace, store, comparison = service
    _publish_run(
        workspace, store, "run-a", returns=[0.01] * 30, timestamps=_timestamps(30)
    )
    _publish_run(
        workspace,
        store,
        "run-b",
        returns=[0.02] * 30,
        timestamps=_timestamps(30, start=40),
    )
    payload = comparison.compare(
        {"kind": "run", "run_id": "run-a"},
        {"kind": "run", "run_id": "run-b"},
    )
    assert payload["unavailable"]["reason"] == "no_common_timestamps"


def test_two_scopes_of_one_run_are_distinct_sources(
    service: tuple[WorkspacePaths, SQLiteStore, PairedComparisonService],
) -> None:
    workspace, store, comparison = service
    _publish_run(
        workspace,
        store,
        "run-a",
        returns=[0.01] * 30,
        scopes=["train", "test"],
        settings=_settings(),
    )
    payload = comparison.compare(
        {"kind": "run", "run_id": "run-a", "scope": "train"},
        {"kind": "run", "run_id": "run-a", "scope": "test"},
    )
    assert payload["status"] == "available"
    assert [source["scope"] for source in payload["sources"]] == ["train", "test"]
    assert payload["sources"][0]["source_hash"] != payload["sources"][1]["source_hash"]


def test_incompatible_settings_and_frequency_are_refused(
    service: tuple[WorkspacePaths, SQLiteStore, PairedComparisonService],
) -> None:
    workspace, store, comparison = service
    _publish_run(workspace, store, "run-a", returns=[0.01] * 30)
    _publish_run(
        workspace,
        store,
        "run-b",
        returns=[0.02] * 30,
        settings=_settings(frequency="1h"),
    )
    _publish_run(
        workspace,
        store,
        "run-c",
        returns=[0.02] * 30,
        settings=_settings(risk_free_rate=0.01),
    )
    assert (
        comparison.compare(
            {"kind": "run", "run_id": "run-a"},
            {"kind": "run", "run_id": "run-b"},
        )["unavailable"]["reason"]
        == "frequency_mismatch"
    )
    assert (
        comparison.compare(
            {"kind": "run", "run_id": "run-a"},
            {"kind": "run", "run_id": "run-c"},
        )["unavailable"]["reason"]
        == "incompatible_definitions"
    )


def test_corrupted_snapshot_is_unavailable_not_recomputed(
    service: tuple[WorkspacePaths, SQLiteStore, PairedComparisonService],
) -> None:
    workspace, store, comparison = service
    _publish_run(workspace, store, "run-a", returns=[0.01] * 30)
    _publish_run(workspace, store, "run-b", returns=[0.02] * 30)
    target = next((workspace.runs / "run-b").glob("performance*.json"))
    target.write_text("{", encoding="utf-8")
    payload = comparison.compare(
        {"kind": "run", "run_id": "run-a"},
        {"kind": "run", "run_id": "run-b"},
    )
    assert payload["status"] == "unavailable"
    assert len(payload["sources"]) == 2
    assert all(source["source_hash"] == "unavailable" for source in payload["sources"])
    PairedComparisonResponse.model_validate(payload)


def test_volume_mode_and_zero_cost_benchmark_are_explicitly_unavailable(
    service: tuple[WorkspacePaths, SQLiteStore, PairedComparisonService],
) -> None:
    workspace, store, comparison = service
    _publish_run(workspace, store, "run-a", returns=[0.01] * 30)
    _publish_run(workspace, store, "volume", returns=None, mode="volume")
    volume = comparison.compare(
        {"kind": "run", "run_id": "volume"},
        {"kind": "run", "run_id": "run-a"},
    )
    assert volume["unavailable"]["reason"] == "not_applicable"
    zero = comparison.compare(
        {"kind": "run", "run_id": "run-a"},
        {
            "kind": "benchmark",
            "run_id": "run-a",
            "benchmark_mode": "zero_cost",
            "price_artifact_id": "missing",
        },
    )
    assert zero["unavailable"]["reason"] == "zero_cost_prices_unavailable"


def test_saved_product_benchmark_keeps_its_cost_identity(
    service: tuple[WorkspacePaths, SQLiteStore, PairedComparisonService],
) -> None:
    workspace, store, comparison = service
    benchmark = BenchmarkScope(
        status="available",
        definition=BenchmarkDefinitionSnapshot(
            id="saved-product",
            asset_class="crypto",
            symbol="BTC/USDT",
            label="product benchmark",
            weighting="equal",
            rationale="saved by producer",
        ),
        capital_basis="run_init_cash",
        init_cash=1000.0,
        fees=0.0006,
        slippage=0.0005,
        coverage=BenchmarkCoverage(
            scope_bars=30, aligned_bars=30, missing_bars=0, complete=True
        ),
        timestamps=_timestamps(30),
        equity_curve=[1000.0 + index for index in range(30)],
        returns=[0.02 + (0.001 if index % 2 else -0.001) for index in range(30)],
    )
    _publish_run(workspace, store, "run-a", returns=[0.01] * 30, benchmark=benchmark)
    payload = comparison.compare(
        {"kind": "run", "run_id": "run-a"},
        {
            "kind": "benchmark",
            "run_id": "run-a",
            "scope": "full",
            "benchmark_mode": "product",
        },
    )
    assert payload["status"] == "available"
    assert payload["sources"][1]["benchmark_definition"] == "saved-product"
    assert "0.0006" in payload["sources"][1]["benchmark_costs"]
    assert payload["result"]["estimate"] == pytest.approx(-0.01, abs=1e-12)


def test_signal_warm_up_keeps_coordinates_and_records_provenance(
    service: tuple[WorkspacePaths, SQLiteStore, PairedComparisonService],
) -> None:
    workspace, _, comparison = service
    ic_a = [None, None] + [0.01 + 0.001 * (index % 3) for index in range(28)]
    ic_b = [None, None] + [0.02 + 0.001 * (index % 2) for index in range(28)]
    ic_b[15] = None
    a_id = _publish_series(workspace, "a", ic=ic_a)
    b_id = _publish_series(workspace, "b", ic=ic_b, holdout=True)
    payload = comparison.compare(
        {"kind": "signal", "artifact_id": a_id, "column": "pearson_ic"},
        {"kind": "signal", "artifact_id": b_id, "column": "pearson_ic"},
    )
    response = PairedComparisonResponse.model_validate(payload)
    assert response.status == "available", payload
    assert response.alignment.overlap == 30
    assert response.alignment.valid_observations == 27
    assert response.alignment.excluded_counts["warmup:a"] == 2
    assert response.alignment.excluded_counts["warmup:b"] == 2
    assert response.sources[0].holdout_relationship == "not_applicable"
    assert response.sources[1].holdout_relationship == "different_dataset"
    assert response.claim_scope == "in_sample_or_unknown"


def test_realized_return_settings_must_match(
    service: tuple[WorkspacePaths, SQLiteStore, PairedComparisonService],
) -> None:
    workspace, _, comparison = service
    returns: list[float | None] = [0.01] * 30
    a_id = _publish_series(
        workspace,
        "a",
        ic=returns,
        settings={"horizon": 1, "window": 3, "n_quantiles": 5},
    )
    b_id = _publish_series(
        workspace,
        "b",
        ic=returns,
        settings={"horizon": 2, "window": 3, "n_quantiles": 5},
    )
    payload = comparison.compare(
        {"kind": "signal", "artifact_id": a_id, "column": "realized_return"},
        {"kind": "signal", "artifact_id": b_id, "column": "realized_return"},
    )
    assert payload["unavailable"]["reason"] == "incompatible_definitions"


def test_ic_and_return_are_never_mixed(
    service: tuple[WorkspacePaths, SQLiteStore, PairedComparisonService],
) -> None:
    workspace, _, comparison = service
    ic = [None, None] + [0.01] * 28
    a_id = _publish_series(workspace, "a", ic=ic)
    b_id = _publish_series(workspace, "b", ic=ic)
    payload = comparison.compare(
        {"kind": "signal", "artifact_id": a_id, "column": "pearson_ic"},
        {"kind": "signal", "artifact_id": b_id, "column": "realized_return"},
    )
    assert payload["unavailable"]["reason"] == "incompatible_definitions"


def test_inference_requires_complete_calendar_but_description_does_not(
    service: tuple[WorkspacePaths, SQLiteStore, PairedComparisonService],
) -> None:
    workspace, store, comparison = service
    a_times = _timestamps(30)
    b_times = _timestamps(31, skip=10)
    _publish_run(workspace, store, "run-a", returns=[0.01] * 30, timestamps=a_times)
    _publish_run(workspace, store, "run-b", returns=[0.02] * 30, timestamps=b_times)
    selection = (
        {"kind": "run", "run_id": "run-a"},
        {"kind": "run", "run_id": "run-b"},
    )
    descriptive = comparison.compare(*selection, mode="descriptive")
    inferential = comparison.compare(*selection, mode="inferential")
    assert descriptive["status"] == "available"
    assert descriptive["result"]["p_value_interpretation"] == "descriptive_only"
    assert inferential["status"] == "unavailable"
    assert inferential["unavailable"]["reason"] == "cadence_gap_unsupported"


def test_artifact_is_reusable_and_does_not_change_old_content(
    service: tuple[WorkspacePaths, SQLiteStore, PairedComparisonService],
) -> None:
    workspace, store, comparison = service
    _publish_run(workspace, store, "run-a", returns=[0.01] * 30)
    _publish_run(workspace, store, "run-b", returns=[0.02] * 30)
    arguments = (
        {"kind": "run", "run_id": "run-a"},
        {"kind": "run", "run_id": "run-b"},
    )
    first = comparison.compare(*arguments, save_artifact=True)
    PairedComparisonResponse.model_validate(first)
    second = comparison.compare(*arguments, save_artifact=True)
    assert first["artifact"] == second["artifact"]
    artifact_id = first["artifact"]["artifact_id"]
    read_back = ArtifactService(workspace=workspace, store=store).get_artifact(
        artifact_id,
        include_content=True,
        expected_sha256=first["artifact"]["sha256"],
    )
    assert read_back["success"] is True
    assert "paired comparison" in read_back["content"]
    assert ALGORITHM_VERSION in read_back["content"]
    assert '"valid_mask"' in read_back["content"]
    assert '"excluded"' in read_back["content"]

    assert first["html_artifact"] == second["html_artifact"]
    html = first["html_artifact"]
    assert html["artifact_id"].startswith("paired-comparison-html:")
    assert html["media_type"] == "text/html"
    html_record = store.get_artifact(html["artifact_id"])
    assert html_record is not None
    document = Path(html_record["path"]).read_text(encoding="utf-8")
    assert document.lstrip().lower().startswith("<!doctype html>")
    assert "<style>" in document and "<script" not in document.lower()
    assert "src=" not in document.lower()
    assert "估計差额" in document or "估計差額" in document
    assert ALGORITHM_VERSION in document
    assert "演算法版本" in document
    assert "完整分析 JSON" in document
    assert "完整對齊證據" in document
    assert '"valid_mask"' in document

    different = comparison.compare(*arguments, block_size=5, save_artifact=True)
    assert different["artifact"]["artifact_id"] != artifact_id
    assert different["html_artifact"]["artifact_id"] != html["artifact_id"]
    for registered_id, record in (
        (artifact_id, first["artifact"]),
        (html["artifact_id"], html),
    ):
        registered = store.get_artifact(registered_id)
        assert registered is not None
        assert registered["sha256"] == record["sha256"]


def test_verified_raw_price_artifact_enables_zero_cost_benchmark(
    service: tuple[WorkspacePaths, SQLiteStore, PairedComparisonService],
) -> None:
    workspace, store, comparison = service
    _publish_run(workspace, store, "run-a", returns=[0.01] * 30)
    document = json.dumps(
        {
            "timestamps": _timestamps(30),
            "closes": [100.0 + index for index in range(30)],
            "source": "verified-fixture",
        },
        sort_keys=True,
    )
    path = workspace.root / "raw_prices" / "prices.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    store.create_artifact(
        artifact_id="price-artifact",
        run_id=None,
        artifact_type=RAW_PRICE_ARTIFACT_TYPE,
        path=path,
        sha256=sha256_text(document),
        metadata={},
    )
    payload = comparison.compare(
        {"kind": "run", "run_id": "run-a"},
        {
            "kind": "benchmark",
            "run_id": "run-a",
            "benchmark_mode": "zero_cost",
            "price_artifact_id": "price-artifact",
        },
    )
    assert payload["status"] == "available"
    assert payload["sources"][1]["benchmark_definition"] == "zero_cost_buy_and_hold"
    assert payload["sources"][1]["benchmark_costs"] == "fees=0,slippage=0"
    assert payload["alignment"]["valid_observations"] >= 20


def test_missing_product_benchmark_is_unavailable(
    service: tuple[WorkspacePaths, SQLiteStore, PairedComparisonService],
) -> None:
    workspace, store, comparison = service
    _publish_run(workspace, store, "run-a", returns=[0.01] * 30)
    payload = comparison.compare(
        {"kind": "run", "run_id": "run-a"},
        {
            "kind": "benchmark",
            "run_id": "run-a",
            "benchmark_mode": "product",
        },
    )
    assert payload["unavailable"]["reason"] == "benchmark_unavailable"


def test_invalid_request_arguments_are_operation_errors(
    service: tuple[WorkspacePaths, SQLiteStore, PairedComparisonService],
) -> None:
    _, _, comparison = service
    selection = {"kind": "run", "run_id": "run-a"}
    with pytest.raises(PairedComparisonError, match="unknown comparison mode"):
        comparison.compare(selection, selection, mode="bad")  # type: ignore[arg-type]
    with pytest.raises(PairedComparisonError, match="reps must be"):
        comparison.compare(selection, selection, reps=1)
    with pytest.raises(PairedComparisonError, match="seed must be"):
        comparison.compare(selection, selection, seed=-1)
    with pytest.raises(PairedComparisonError, match="block_size must be"):
        comparison.compare(selection, selection, block_size=0)
    with pytest.raises(PairedComparisonError, match="confidence must be"):
        comparison.compare(selection, selection, confidence=1.0)
