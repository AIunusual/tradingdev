"""Saved search counts, identity and bounded reads through ranking and MCP."""

from __future__ import annotations

import json
from contextlib import closing
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal

import pytest
from mcp.server import MCPServer
from tests.app.test_run_ranking import _manifest, _observations
from tests.mcp.test_ml_leaderboard_runs import _dispatch

from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.performance import (
    PerformanceArtifactError,
    PerformanceStore,
)
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.ml_service import MLService
from tradingdev.app.ranking_service import RankingService
from tradingdev.app.run_service import RunService
from tradingdev.domain.execution import ExecutionManifest, OptimizationSpec
from tradingdev.domain.performance.artifacts import PerformanceScope, build_artifacts
from tradingdev.domain.research_scope import (
    optimization_backtest,
    verify_research_scopes,
)
from tradingdev.mcp.tools import ml, runs

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def storage() -> Iterator[tuple[WorkspacePaths, SQLiteStore]]:
    with TemporaryDirectory(prefix="tradingdev-selection-") as directory:
        workspace = WorkspacePaths(Path(directory))
        yield workspace, SQLiteStore(workspace)


def _publish(
    workspace: WorkspacePaths,
    store: SQLiteStore,
    run_id: str,
    count: int,
    *,
    historical: bool = False,
    wrong_test_params: bool = False,
) -> None:
    base = _manifest()
    config = base.config_copy()
    config["backtest"]["periods_per_year"] = 365
    manifest = ExecutionManifest.create(
        kind="optimization",
        config=config,
        strategy_execution=base.strategy_execution,
        optimization=OptimizationSpec.model_validate(
            {
                "param_ranges": {"window": list(range(1, count + 1))},
                "optimization_metric": "sharpe_ratio",
                "train_start": "2024-01-01",
                "train_end": "2024-01-31",
                "test_start": "2024-02-01",
                "test_end": "2024-03-01",
            }
        ),
    )

    def scope(
        split: Literal["train", "test"],
        index: int | None,
        window: int,
        sharpe: float | None,
    ) -> PerformanceScope:
        return PerformanceScope(
            mode="signal",
            split=split,
            trial_index=index,
            parameters={"window": window},
            values={"sharpe_ratio": sharpe},
            metadata={
                "execution_context": {
                    **optimization_backtest(manifest, split).model_dump(mode="json"),
                    "random_seed": config.get("random_seed"),
                },
                "settings": {"periods_per_year": 365},
                "unavailable": {"sharpe_ratio": "insufficient_data"}
                if sharpe is None
                else {},
            },
        )

    scopes = {
        f"trial/{index}/train": scope(
            "train", index, index + 1, None if index == 1 else float(index)
        )
        for index in range(count)
    }
    zone = "" if historical else "Z"
    train = _observations().model_copy(
        update={"timestamps": [f"2024-01-0{day}T00:00:00{zone}" for day in (1, 2)]}
    )
    test = _observations().model_copy(
        update={"timestamps": [f"2024-02-0{day}T00:00:00{zone}" for day in (1, 2)]}
    )
    observations = dict.fromkeys(scopes, train) | {"test": test}
    scopes["test"] = scope(
        "test", None, 1 if wrong_test_params else count, -1.0 / count
    )
    artifacts = build_artifacts(
        run_id,
        manifest.manifest_hash,
        "test",
        scopes,
        observations,
        selected_train_scope=f"trial/{count - 1}/train",
    )
    if not historical:
        selection = verify_research_scopes(
            manifest, artifacts.performance, artifacts.observations
        )
        assert selection is not None
        artifacts = build_artifacts(
            run_id,
            manifest.manifest_hash,
            "test",
            scopes,
            observations,
            selected_train_scope=selection.selected_train_scope,
            optimization_selection=selection,
        )
    ExecutionManifestStore(workspace).publish(run_id, manifest)
    store.create_run(
        run_id=run_id,
        job_id=run_id,
        strategy_id="fixture",
        revision_id="rev-1",
        artifact_dir=workspace.runs / run_id,
        metrics={"sharpe_ratio": -1.0 / count},
        manifest_hash=manifest.manifest_hash,
    )
    PerformanceStore(workspace, store).publish(artifacts)


def test_ranking_and_detail_pages_keep_each_search_size(
    storage: tuple[WorkspacePaths, SQLiteStore],
) -> None:
    workspace, store = storage
    _publish(workspace, store, "narrow", 5, historical=True)
    _publish(workspace, store, "wide", 500)
    server = MCPServer("selection-evidence")
    ranking = RankingService(workspace=workspace, store=store)
    ml.register(server, MLService(workspace=workspace))
    runs.register(server, RunService(workspace=workspace, store=store))
    args = {
        "metric": "run.sharpe_ratio",
        "runs": [{"run_id": "narrow"}, {"run_id": "wide"}],
    }
    for offset, run_id, count, source in (
        (0, "wide", 500, "trial_ledger"),
        (1, "narrow", 5, "verified_historical_bundle"),
    ):
        board = _dispatch(server, "leaderboard", {**args, "offset": offset, "limit": 1})
        (row,) = board["ranked"]
        assert row["run_id"] == run_id and row["scope"] == "test"
        evidence = row["selection_evidence"]
        assert evidence["scope_role"] == "selected_test"
        summary = evidence["summary"]
        assert (
            summary["planned"]
            == summary["raw_attempted"]
            == summary["completed"]
            == count
        )
        assert (
            summary["objective_available"] == count - 1
            and summary["objective_unavailable"] == 1
        )
        assert summary["selected_train_scope"] == f"trial/{count - 1}/train"
        assert "trials" not in evidence
        ref = evidence["reference"]
        assert ref["source"] == source
        read = {key: ref[key] for key in ("run_id", "performance_sha256")}
        for start, expected in ((1, None), (count - 1, float(count - 1))):
            page = _dispatch(
                server, "get_selection_evidence", {**read, "offset": start, "limit": 1}
            )
            assert page["summary"] == summary and page["reference"] == ref
            (trial,) = page["trials"]["items"]
            assert page["trials"]["total"] == count
            assert page["trials"]["next_offset"] == (
                start + 1 if start + 1 < count else None
            )
            assert trial["trial_index"] == start and trial["split"] == "train"
            assert trial["parameters"] == {"window": start + 1}
            assert trial["metrics"]["sharpe_ratio"] == {
                "value": expected,
                "reason": "insufficient_data" if expected is None else None,
            }
            assert trial["calculation_settings"]["periods_per_year"] == 365
            assert page["definitions"]["sharpe_ratio"]["requires_annualization"]
        assert (
            _dispatch(
                server,
                "get_selection_evidence",
                read | {"performance_sha256": "0" * 64},
            )["code"]
            == "selection_evidence_mismatch"
        )
    top = ranking.rank_run_scopes(
        "run.sharpe_ratio", [("wide", None), ("narrow", None)], top_k=1
    )
    assert top["ranked"][0]["selection_evidence"]["summary"]["raw_attempted"] == 500
    explicit = ranking.rank_run_scopes(
        "run.sharpe_ratio", [("wide", "trial/0/train"), ("wide", "trial/499/train")]
    )
    assert [row["selection_evidence"]["scope_role"] for row in explicit["ranked"]] == [
        "selected_train",
        "other_train",
    ]


@pytest.mark.parametrize(
    "failure", ["missing_observations", "wrong_test_params", "corrupt_performance"]
)
def test_unverifiable_evidence_never_produces_a_count(
    storage: tuple[WorkspacePaths, SQLiteStore],
    failure: str,
) -> None:
    workspace, store = storage
    _publish(
        workspace,
        store,
        "legacy",
        5,
        historical=True,
        wrong_test_params=failure == "wrong_test_params",
    )
    if failure == "missing_observations":
        with closing(store.connect()) as conn, conn:
            conn.execute(
                "delete from artifacts where artifact_id = 'legacy:observations_json'"
            )
    elif failure == "corrupt_performance":
        (workspace.runs / "legacy" / "performance.json").write_text("{}")
    result = RankingService(workspace=workspace, store=store).rank_run_scopes(
        "run.sharpe_ratio", [("legacy", None)]
    )
    if failure == "corrupt_performance":
        assert result["ranked"] == []
        assert "SHA-256" in result["skipped"][0]["reason"]
    else:
        evidence = result["ranked"][0]["selection_evidence"]
        assert (
            evidence["status"] == "unavailable"
            and "cannot be verified" in evidence["reason"]
        )
        assert "summary" not in evidence


@pytest.mark.parametrize(
    ("defect", "reason"),
    [("scope", "complete grid"), ("annualization", "metric settings differ")],
)
def test_registered_summary_must_match_the_manifest_at_read_time(
    storage: tuple[WorkspacePaths, SQLiteStore],
    defect: str,
    reason: str,
) -> None:
    workspace, store = storage
    _publish(workspace, store, "imported", 5)
    path = workspace.runs / "imported" / "performance.json"
    bundle = json.loads(path.read_bytes())
    scopes = bundle["scopes"]
    if defect == "scope":
        scopes["wrong-key"] = scopes.pop("trial/0/train")
    else:
        scopes["trial/0/train"]["metadata"]["settings"]["periods_per_year"] = 252
    content = json.dumps(bundle).encode()
    path.write_bytes(content)
    digest = sha256(content).hexdigest()
    with closing(store.connect()) as conn, conn:
        conn.execute(
            "update artifacts set sha256 = ? where artifact_id = ?",
            (digest, "imported:performance_json"),
        )
    result = RankingService(workspace=workspace, store=store).rank_run_scopes(
        "run.sharpe_ratio", [("imported", None)]
    )
    evidence = result["ranked"][0]["selection_evidence"]
    assert evidence["status"] == "unavailable" and reason in evidence["reason"]
    with pytest.raises(PerformanceArtifactError, match=reason) as error:
        RunService(workspace=workspace, store=store).get_selection_evidence(
            "imported", digest
        )
    assert error.value.code == "selection_evidence_invalid"
