"""Edge branches of the paired-comparison wiring, kept independently testable."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from tradingdev.app.contracts.paired_comparison import (
    PairedComparisonResponse,
    SeriesSelection,
)
from tradingdev.app.paired_comparison_service import (
    ARTIFACT_TYPE,
    HTML_ARTIFACT_TYPE,
    RAW_PRICE_ARTIFACT_TYPE,
    PairedComparisonError,
    PairedComparisonService,
    _fold_range,
    _LoadedSource,
    _LoadFailure,
    _render_html,
    _returns_from_equity,
    _run_claim_scope,
    _run_returns,
    _run_settings,
)
from tradingdev.domain.statistics.alignment import _utc_index, align_masked_pair
from tradingdev.domain.statistics.sharpe import IntervalUnavailable


def _series(values: list[float | None] | None = None) -> pd.Series:
    values = values if values is not None else [0.01] * 30
    return pd.Series(
        values,
        index=pd.date_range("2024-01-01", periods=len(values), freq="D", tz="UTC"),
        dtype="float64",
    )


def _loaded(
    *,
    values: list[float | None] | None = None,
    definition: str = "cost_bearing_trade_return",
    settings: dict[str, Any] | None = None,
    producer: str | None = "saved_backtest_observations",
) -> _LoadedSource:
    return _LoadedSource(
        series=_series(values),
        label="fixture",
        identity={
            "kind": "run",
            "label": "fixture",
            "source_hash": "hash",
            "producer_source": producer,
        },
        settings=settings or {},
        cadence="1d",
        definition=definition,
        cost_basis="saved_execution_with_costs",
        claim_scope="in_sample_or_unknown",
    )


_SELECTION = {"kind": "run", "run_id": "r"}


_SELECTION = {"kind": "run", "run_id": "r"}


def _reason(value: Any) -> str:
    assert isinstance(value, _LoadFailure)
    return value.reason


@pytest.fixture
def service(tmp_path: Path) -> PairedComparisonService:
    from tradingdev.adapters.storage.filesystem import WorkspacePaths
    from tradingdev.adapters.storage.sqlite import SQLiteStore

    workspace = WorkspacePaths(tmp_path / "workspace")
    return PairedComparisonService(workspace=workspace, store=SQLiteStore(workspace))


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"reps": True}, "reps must be"),
        ({"seed": True}, "seed must be"),
        ({"block_size": True}, "block_size must be"),
        ({"confidence": 1}, "confidence must be"),
        ({"mode": "unknown"}, "unknown comparison mode"),
        ({"reps": 20_000}, "reps must be"),
    ],
)
def test_invalid_argument_types_are_rejected(
    service: PairedComparisonService, arguments: dict[str, Any], message: str
) -> None:
    selection = {"kind": "run", "run_id": "run-a"}
    with pytest.raises(PairedComparisonError, match=message):
        service.compare(selection, selection, **arguments)


def test_alignment_failure_is_an_explicit_unavailable(
    service: PairedComparisonService, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_: Any, **__: Any) -> None:
        raise ValueError("duplicate timestamps")

    monkeypatch.setattr(
        "tradingdev.app.paired_comparison_service.align_masked_pair", broken
    )
    monkeypatch.setattr(service, "_load", lambda *_: _loaded())
    payload = service.compare(_SELECTION, _SELECTION)
    assert payload["unavailable"]["reason"] == "comparison_error"
    assert "duplicate timestamps" in payload["unavailable"]["detail"]


def test_no_jointly_valid_observations(
    service: PairedComparisonService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service, "_load", lambda *_: _loaded(values=[np.nan] * 30))
    payload = service.compare(_SELECTION, _SELECTION)
    assert payload["unavailable"]["reason"] == "no_valid_observations"


def test_inferential_requires_a_recorded_producer(
    service: PairedComparisonService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service, "_load", lambda *_: _loaded(producer=None))
    payload = service.compare(_SELECTION, _SELECTION, mode="inferential")
    assert payload["unavailable"]["reason"] == "source_qualification_unknown"


def test_inferential_rejects_interior_invalid_observations(
    service: PairedComparisonService, monkeypatch: pytest.MonkeyPatch
) -> None:
    values: list[float | None] = [0.01] * 30
    values[15] = float("nan")
    monkeypatch.setattr(service, "_load", lambda *_: _loaded(values=values))
    descriptive = service.compare(_SELECTION, _SELECTION, mode="descriptive")
    inferential = service.compare(_SELECTION, _SELECTION, mode="inferential")
    assert descriptive["status"] == "available"
    assert descriptive["alignment"]["excluded_counts"]["interior_gap:a"] == 1
    assert inferential["unavailable"]["reason"] == "cadence_gap_unsupported"


def test_unknown_producer_is_not_inferential(
    service: PairedComparisonService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service, "_load", lambda *_: _loaded(producer="unknown"))
    payload = service.compare(_SELECTION, _SELECTION, mode="inferential")
    assert payload["unavailable"]["reason"] == "source_qualification_unknown"


def test_cadence_is_inferred_from_saved_timestamps() -> None:
    from tradingdev.app.paired_comparison_service import _infer_cadence

    daily = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    hourly = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    irregular = pd.DatetimeIndex(
        [
            "2024-01-01T00:00:00+00:00",
            "2024-01-02T00:00:00+00:00",
            "2024-01-04T00:00:00+00:00",
        ]
    )
    assert _infer_cadence(daily) == "1d"
    assert _infer_cadence(hourly) == "1h"
    assert _infer_cadence(irregular) == "unsupported"
    assert _infer_cadence(daily[:1]) == "unsupported"


def test_bootstrap_errors_and_unavailable_intervals_are_separate(
    service: PairedComparisonService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service, "_load", lambda *_: _loaded())

    def broken(*_: Any, **__: Any) -> None:
        raise ValueError("bad bootstrap")

    monkeypatch.setattr(
        "tradingdev.app.paired_comparison_service.difference_interval_outcome", broken
    )
    assert (
        service.compare(_SELECTION, _SELECTION)["unavailable"]["reason"]
        == "comparison_error"
    )

    monkeypatch.setattr(
        "tradingdev.app.paired_comparison_service.difference_interval_outcome",
        lambda *_, **__: IntervalUnavailable("no_valid_resample", None, 30, 1000, 1000),
    )
    payload = service.compare(_SELECTION, _SELECTION)
    assert payload["unavailable"]["reason"] == "no_valid_resample"


def test_artifact_publication_failure_is_explicit(
    service: PairedComparisonService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service, "_load", lambda *_: _loaded())

    def broken(*_: Any, **__: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(service, "_publish_text", broken)
    with pytest.raises(PairedComparisonError, match="artifact failed: disk full"):
        service.compare(_SELECTION, _SELECTION, save_artifact=True)


def test_signal_settings_must_match(
    service: PairedComparisonService, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _loaded(
        definition="signal_pearson_ic", settings={"window": 3, "horizon": 1}
    )
    second = _loaded(
        definition="signal_pearson_ic", settings={"window": 5, "horizon": 1}
    )
    monkeypatch.setattr(
        service,
        "_load",
        lambda selection, _: first if selection.artifact_id == "a" else second,
    )
    payload = service.compare(
        {"kind": "signal", "artifact_id": "a", "column": "pearson_ic"},
        {"kind": "signal", "artifact_id": "b", "column": "pearson_ic"},
    )
    assert payload["unavailable"]["reason"] == "incompatible_definitions"


def test_zero_cost_signal_settings_must_match(
    service: PairedComparisonService, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _loaded(
        definition="zero_cost_signal_simulation", settings={"horizon": 1, "window": 3}
    )
    second = _loaded(
        definition="zero_cost_signal_simulation", settings={"horizon": 2, "window": 3}
    )
    monkeypatch.setattr(
        service,
        "_load",
        lambda selection, _: first if selection.artifact_id == "a" else second,
    )
    payload = service.compare(
        {"kind": "signal", "artifact_id": "a", "column": "realized_return"},
        {"kind": "signal", "artifact_id": "b", "column": "realized_return"},
    )
    assert payload["unavailable"]["reason"] == "incompatible_definitions"


def test_html_without_a_result_still_renders(service: PairedComparisonService) -> None:
    payload = service._unavailable_payload(
        _LoadFailure("x", "y"), "descriptive", [], []
    )
    document = _render_html(payload)
    assert "<!doctype html>" in document
    assert "x" in document
    assert "完整分析 JSON" in document


def test_html_interval_label_follows_confidence(
    service: PairedComparisonService,
) -> None:
    payload = service._unavailable_payload(
        _LoadFailure("x", "y"), "descriptive", [], []
    )
    payload["result"] = {
        "estimate": 0.0,
        "estimate_a": 0.0,
        "estimate_b": 0.0,
        "lower": -1.0,
        "upper": 1.0,
        "p_value": 1.0,
        "p_value_interpretation": "descriptive_only",
        "confidence": 0.9,
        "reps": 100,
        "valid_reps": 100,
        "seed": 1,
        "block_size": 2,
        "correlation": None,
    }
    document = _render_html(payload)
    assert "90% 區間" in document
    assert "95% 區間" not in document


def test_zero_cost_artifact_refusal_branches(
    service: PairedComparisonService, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tradingdev.app.contracts.paired_comparison import SeriesSelection

    empty = SeriesSelection.model_construct(
        kind="benchmark", run_id="r", benchmark_mode="zero_cost"
    )
    assert _reason(service._zero_cost_source(empty)) == "zero_cost_prices_unavailable"

    record = {"artifact_type": "something_else", "path": str(tmp_path / "p")}
    monkeypatch.setattr(service._store, "get_artifact", lambda _: record)
    selected = SeriesSelection(
        kind="benchmark",
        run_id="r",
        benchmark_mode="zero_cost",
        price_artifact_id="p",
    )
    assert (
        _reason(service._zero_cost_source(selected)) == "zero_cost_prices_unavailable"
    )

    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    record = {
        "artifact_type": RAW_PRICE_ARTIFACT_TYPE,
        "path": str(outside),
        "sha256": "0" * 64,
    }
    monkeypatch.setattr(service._store, "get_artifact", lambda _: record)
    assert (
        _reason(service._zero_cost_source(selected)) == "zero_cost_prices_unavailable"
    )

    inside = service._workspace.root / "prices.json"
    inside.parent.mkdir(parents=True, exist_ok=True)
    inside.write_text("{}", encoding="utf-8")
    record = {
        "artifact_type": RAW_PRICE_ARTIFACT_TYPE,
        "path": str(inside),
        "sha256": "0" * 64,
    }
    monkeypatch.setattr(service._store, "get_artifact", lambda _: record)
    assert (
        _reason(service._zero_cost_source(selected)) == "zero_cost_prices_unavailable"
    )

    import hashlib

    record["sha256"] = hashlib.sha256(inside.read_bytes()).hexdigest()
    assert (
        _reason(service._zero_cost_source(selected)) == "zero_cost_prices_unavailable"
    )

    inside.write_text(
        '{"closes": [1], "timestamps": ["2024-01-01"], "source": ""}',
        encoding="utf-8",
    )
    record["sha256"] = hashlib.sha256(inside.read_bytes()).hexdigest()
    assert (
        _reason(service._zero_cost_source(selected)) == "zero_cost_prices_unavailable"
    )


def test_zero_cost_hashes_and_parses_one_buffer(
    service: PairedComparisonService, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tradingdev.adapters.storage.filesystem import sha256_text
    from tradingdev.app.contracts.paired_comparison import SeriesSelection

    document = json.dumps(
        {
            "timestamps": [
                "2024-01-01T00:00:00+00:00",
                "2024-01-02T00:00:00+00:00",
            ],
            "closes": [100.0, 101.0],
            "source": "verified-fixture",
        }
    )
    path = service._workspace.root / "prices.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    monkeypatch.setattr(
        service._store,
        "get_artifact",
        lambda _: {
            "artifact_type": RAW_PRICE_ARTIFACT_TYPE,
            "path": str(path),
            "sha256": sha256_text(document),
        },
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *_args, **_kwargs: pytest.fail("price file was reopened after hashing"),
    )
    selected = SeriesSelection(
        kind="benchmark",
        run_id="r",
        benchmark_mode="zero_cost",
        price_artifact_id="p",
    )
    loaded = service._zero_cost_source(selected)
    assert not isinstance(loaded, _LoadFailure)
    assert loaded.definition == "zero_cost_price_return"


def test_signal_source_refusal_branches(
    service: PairedComparisonService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradingdev.app.contracts.paired_comparison import SeriesSelection

    class BrokenStore:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def load(self, *_: Any, **__: Any) -> Any:
            raise ValueError("series missing")

    monkeypatch.setattr(
        "tradingdev.adapters.storage.evaluation_series.EvaluationSeriesStore",
        BrokenStore,
    )
    selected = SeriesSelection(kind="signal", artifact_id="a" * 64, column="pearson_ic")
    assert _reason(service._signal_source(selected)) == "series_unavailable"

    class MissingColumnStore:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def load(self, *_: Any, **__: Any) -> Any:
            return type(
                "Bundle",
                (),
                {
                    "manifest": type("Manifest", (), {"columns": ()})(),
                    "frame": pd.DataFrame({"timestamp": []}),
                },
            )()

    monkeypatch.setattr(
        "tradingdev.adapters.storage.evaluation_series.EvaluationSeriesStore",
        MissingColumnStore,
    )
    assert _reason(service._signal_source(selected)) == "series_unavailable"
    other = SeriesSelection(
        kind="signal", artifact_id="a" * 64, column="realized_return"
    )
    assert _reason(service._signal_source(other)) == "series_unavailable"


def test_numeric_helpers_cover_degenerate_inputs() -> None:
    class Observations:
        init_cash: Any = 1000.0
        equity_curve: Any = [1000.0, 1010.0]
        returns: Any = None

    class Scope:
        observations = Observations()
        performance = type("Performance", (), {"metadata": {}})()

    assert _run_returns(cast("Any", Scope())) == pytest.approx([0.0, 0.01])
    Observations.returns = [0.01, 0.02]
    assert _run_returns(cast("Any", Scope())) == pytest.approx([0.01, 0.02])
    Observations.returns = [0.01]
    assert _run_returns(cast("Any", Scope())) is None
    Observations.returns = None
    Observations.init_cash = None
    assert _run_returns(cast("Any", Scope())) is None

    bridged = _returns_from_equity([1000.0, None, 1020.0], 1000.0)
    assert np.isnan(bridged[1])
    assert np.isnan(bridged[2])
    assert np.isnan(_returns_from_equity([10.0], 0.0)[0])
    assert _run_settings(cast("Any", Scope())) == {"init_cash": None}


def test_claim_scope_and_fold_range_helpers() -> None:
    test_scope = type(
        "Scope",
        (),
        {
            "performance": type("P", (), {"split": "test"})(),
            "research_evidence": object(),
        },
    )()
    train_scope = type(
        "Scope",
        (),
        {
            "performance": type("P", (), {"split": "train"})(),
            "research_evidence": None,
        },
    )()
    assert _run_claim_scope(test_scope) == "verified_out_of_sample"
    assert _run_claim_scope(train_scope) == "in_sample_or_unknown"

    manifest = type("Manifest", (), {"kind": "signal_evaluation"})()
    assert _fold_range(manifest, pd.DataFrame({"x": [1]})) is None
    manifest = type("Manifest", (), {"kind": "fold_validation"})()
    assert _fold_range(manifest, pd.DataFrame()) is None
    assert _fold_range(manifest, pd.DataFrame({"x": [1]})) is None
    frame = pd.DataFrame(
        {
            "train_start": ["a"],
            "train_end": ["b"],
            "eval_start": [None],
            "eval_end": ["d"],
        }
    )
    assert _fold_range(manifest, frame) == {
        "train_start": "a",
        "train_end": "b",
        "eval_start": None,
        "eval_end": "d",
    }


def test_alignment_validation_branches() -> None:
    with pytest.raises(ValueError, match="requires a DatetimeIndex"):
        _utc_index(cast("Any", [1.0]), "a")
    with pytest.raises(ValueError, match="requires datetime timestamps"):
        align_masked_pair(
            pd.Series([1.0, 2.0], index=pd.Index([1, 2])), pd.Series([1.0])
        )
    with pytest.raises(ValueError, match="real numbers"):
        align_masked_pair(
            pd.Series(
                [True, False],
                index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
            ),
            pd.Series(
                [1.0, 2.0], index=pd.date_range("2024-01-01", periods=2, tz="UTC")
            ),
        )
    with pytest.raises(ValueError, match="real numbers"):
        align_masked_pair(
            pd.Series(
                ["a", "b"], index=pd.date_range("2024-01-01", periods=2, tz="UTC")
            ),
            pd.Series(
                [1.0, 2.0], index=pd.date_range("2024-01-01", periods=2, tz="UTC")
            ),
        )
    empty = pd.Series([], dtype="float64", index=pd.DatetimeIndex([], tz="UTC"))
    assert align_masked_pair(empty, empty).valid_count == 0
    with pytest.raises(ValueError, match="both inputs must be pandas Series"):
        align_masked_pair(cast("Any", [1.0]), pd.Series([1.0]))


def test_contract_validators() -> None:
    with pytest.raises(ValidationError):
        SeriesSelection(kind="run")
    with pytest.raises(ValidationError):
        SeriesSelection(kind="benchmark", run_id="r")
    with pytest.raises(ValidationError):
        SeriesSelection(kind="benchmark", run_id="r", benchmark_mode="zero_cost")
    with pytest.raises(ValidationError):
        SeriesSelection(kind="signal", artifact_id="a")
    base = {
        "success": True,
        "schema_version": 1,
        "algorithm_version": "v1",
        "mode": "descriptive",
        "statistic": "mean",
        "difference_direction": "a_minus_b",
        "claim_scope": "in_sample_or_unknown",
        "qualification": {},
        "status": "available",
        "alignment": {
            "calendar_policy": "exact_utc_intersection_no_fill",
            "cadence": "1d",
            "cadence_complete": True,
            "original_lengths": {"a": 1, "b": 1},
            "overlap": 1,
            "valid_observations": 1,
            "common_period": {"start": None, "end": None},
            "excluded_counts": {},
            "excluded_sample": [],
            "missing_timestamps_count": 0,
            "missing_timestamps_sample": [],
            "valid_mask_count": 1,
        },
        "sources": [],
        "limitations": [],
    }
    with pytest.raises(ValidationError):
        PairedComparisonResponse.model_validate(base)
    with pytest.raises(ValidationError):
        PairedComparisonResponse.model_validate(
            {**base, "status": "available", "result": {}, "sources": [{}]}
        )
    with pytest.raises(ValidationError):
        PairedComparisonResponse.model_validate(
            {**base, "status": "unavailable", "sources": [{}, {}]}
        )


def test_registry_failure_is_a_publication_error(
    service: PairedComparisonService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service, "_load", lambda *_: _loaded())

    def registry_error(*_: Any, **__: Any) -> None:
        raise RuntimeError("registry offline")

    monkeypatch.setattr(service._store, "create_artifact", registry_error)
    with pytest.raises(PairedComparisonError, match="registry offline"):
        service.compare(_SELECTION, _SELECTION, save_artifact=True)


def test_artifact_conflict_paths(
    service: PairedComparisonService, tmp_path: Path
) -> None:
    from tradingdev.adapters.storage.filesystem import sha256_text

    document = "hello"
    kwargs: dict[str, Any] = {
        "document": document,
        "prefix": "paired-comparison-test",
        "suffix": ".md",
        "artifact_type": ARTIFACT_TYPE,
        "media_type": "text/markdown",
        "payload": {"mode": "descriptive", "claim_scope": "x", "status": "available"},
    }
    first = service._publish_text({}, **kwargs)
    assert first["artifact_id"].startswith("paired-comparison-test:")
    again = service._publish_text({}, **kwargs)
    assert again == first

    from tradingdev.adapters.storage.filesystem import sha256_text as text_digest

    conflict_path = (
        service._workspace.root
        / "paired_comparisons"
        / f"{text_digest(document)[:16]}.md"
    )
    conflict_path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="existing comparison artifact differs"):
        service._publish_text({}, **kwargs)
    conflict_path.write_text(document, encoding="utf-8")

    digest = sha256_text(document)
    service._store.create_artifact(
        artifact_id=first["artifact_id"],
        run_id=None,
        artifact_type=HTML_ARTIFACT_TYPE,
        path=service._workspace.root / "paired_comparisons" / f"{digest[:16]}.md",
        sha256="0" * 64,
        metadata={},
    )
    with pytest.raises(ValueError, match="registered"):
        service._publish_text({}, **kwargs)


def test_contract_status_and_source_count_branches() -> None:
    from tradingdev.app.contracts.paired_comparison import AlignmentRecord

    alignment = AlignmentRecord.model_validate(
        {
            "calendar_policy": "exact_utc_intersection_no_fill",
            "cadence": "1d",
            "cadence_complete": True,
            "original_lengths": {"a": 1, "b": 1},
            "overlap": 1,
            "valid_observations": 1,
            "common_period": {"start": None, "end": None},
            "excluded_counts": {},
            "excluded_sample": [],
            "missing_timestamps_count": 0,
            "missing_timestamps_sample": [],
            "valid_mask_count": 1,
        }
    )
    source = {"kind": "run", "label": "a", "source_hash": "h"}
    base = {
        "success": True,
        "schema_version": 1,
        "algorithm_version": "v1",
        "mode": "descriptive",
        "statistic": "mean",
        "difference_direction": "a_minus_b",
        "claim_scope": "in_sample_or_unknown",
        "qualification": {},
        "alignment": alignment.model_dump(),
        "sources": [source, source],
        "limitations": [],
    }
    with pytest.raises(ValidationError, match="requires a reason"):
        PairedComparisonResponse.model_validate({**base, "status": "unavailable"})
    result = {
        "estimate": 0.0,
        "estimate_a": 0.0,
        "estimate_b": 0.0,
        "lower": 0.0,
        "upper": 0.0,
        "p_value": 1.0,
        "p_value_interpretation": "descriptive_only",
        "confidence": 0.95,
        "reps": 100,
        "valid_reps": 100,
        "seed": 1,
        "block_size": 2,
        "correlation": None,
    }
    with pytest.raises(ValidationError, match="exactly two source identities"):
        PairedComparisonResponse.model_validate(
            {**base, "status": "available", "result": result, "sources": [source]}
        )


def test_finite_series_helper() -> None:
    from tradingdev.domain.statistics.alignment import finite_series

    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    assert finite_series(pd.Series([1.0, 2.0, 3.0], index=index))
    assert not finite_series(pd.Series([1.0, np.nan, 3.0], index=index))


def test_load_failures_are_explicit(
    service: PairedComparisonService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradingdev.app.trade_history_service import HistoryReadError

    def history_error(*_: Any, **__: Any) -> None:
        raise HistoryReadError("run_not_found", "missing run")

    monkeypatch.setattr(service, "_run_snapshot", history_error)
    assert (
        service.compare(_SELECTION, _SELECTION)["unavailable"]["reason"]
        == "run_not_found"
    )

    def value_error(*_: Any, **__: Any) -> None:
        raise ValueError("corrupt")

    monkeypatch.setattr(service, "_run_snapshot", value_error)
    payload = service.compare(_SELECTION, _SELECTION)
    assert payload["unavailable"]["reason"] == "comparison_error"
    assert "corrupt" in payload["unavailable"]["detail"]


def test_run_source_degenerate_snapshots(
    service: PairedComparisonService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradingdev.app.trade_history_service import HistoryReadError

    observations = type(
        "Observations",
        (),
        {
            "init_cash": 1000.0,
            "equity_curve": [1000.0],
            "returns": None,
            "timestamps": ["2024-01-01T00:00:00+00:00"],
        },
    )()
    loaded = type(
        "Loaded",
        (),
        {
            "performance": type(
                "Performance", (), {"default_scope": "full", "manifest_hash": None}
            )(),
            "observations": type(
                "ObservationsBundle", (), {"scopes": {"full": observations}}
            )(),
        },
    )()

    def fold_summary(*_: Any, **__: Any) -> Any:
        return type(
            "Scope",
            (),
            {
                "performance": type(
                    "P", (), {"kind": "fold_summary", "mode": "signal"}
                )(),
                "observations": observations,
            },
        )()

    monkeypatch.setattr(service, "_run_snapshot", lambda *_: loaded)
    monkeypatch.setattr(service._history, "select_scope", fold_summary)
    with pytest.raises(HistoryReadError, match="fold summaries"):
        service._run_source(SeriesSelection(kind="run", run_id="r"), {})

    def broken_returns(*_: Any, **__: Any) -> None:
        return None

    from tradingdev.app.paired_comparison_service import _run_returns as real_returns

    monkeypatch.setattr(
        "tradingdev.app.paired_comparison_service._run_returns", broken_returns
    )
    monkeypatch.setattr(
        service._history,
        "select_scope",
        lambda *_: type(
            "Scope",
            (),
            {
                "performance": type("P", (), {"kind": "backtest", "mode": "signal"})(),
                "observations": observations,
            },
        )(),
    )
    assert (
        _reason(service._run_source(SeriesSelection(kind="run", run_id="r"), {}))
        == "not_applicable"
    )

    monkeypatch.setattr(
        "tradingdev.app.paired_comparison_service._run_returns", real_returns
    )
    missing_time = type(
        "Observations",
        (),
        {
            "init_cash": 1000.0,
            "equity_curve": [1000.0],
            "returns": [0.01],
            "timestamps": None,
        },
    )()
    monkeypatch.setattr(
        service._history,
        "select_scope",
        lambda *_: type(
            "Scope",
            (),
            {
                "performance": type("P", (), {"kind": "backtest", "mode": "signal"})(),
                "observations": missing_time,
            },
        )(),
    )
    assert (
        _reason(service._run_source(SeriesSelection(kind="run", run_id="r"), {}))
        == "timestamps_unavailable"
    )


def test_benchmark_source_degenerate_snapshots(
    service: PairedComparisonService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from tradingdev.domain.performance.artifacts import BenchmarkScope

    scope = SimpleNamespace(
        performance=SimpleNamespace(metadata={}),
        observations=SimpleNamespace(init_cash=1000.0),
    )
    monkeypatch.setattr(service._history, "select_scope", lambda *_: scope)
    monkeypatch.setattr(
        service,
        "_run_snapshot",
        lambda *_: SimpleNamespace(performance=SimpleNamespace(default_scope="full")),
    )
    monkeypatch.setattr(
        "tradingdev.app.paired_comparison_service._run_identity",
        lambda *_: {"kind": "benchmark", "label": "b", "source_hash": "h"},
    )
    selection = SeriesSelection(kind="benchmark", run_id="r", benchmark_mode="product")

    def with_benchmark(benchmark: BenchmarkScope) -> Any:
        loaded = SimpleNamespace(
            performance=SimpleNamespace(
                default_scope="full", benchmarks={"full": benchmark}
            )
        )
        monkeypatch.setattr(service, "_run_snapshot", lambda *_: loaded)
        return service._benchmark_source(selection, {})

    absent = BenchmarkScope(status="unavailable", reason="no_baseline")
    assert _reason(with_benchmark(absent)) == "no_baseline"

    no_series = BenchmarkScope.model_construct(
        status="available",
        timestamps=None,
        equity_curve=None,
        returns=None,
        definition=None,
        capital_basis="run_init_cash",
        init_cash=1000.0,
        fees=None,
        slippage=None,
        reason=None,
        detail=None,
    )
    assert _reason(with_benchmark(no_series)) == "benchmark_unavailable"

    no_capital = BenchmarkScope.model_construct(
        status="available",
        timestamps=["2024-01-01T00:00:00+00:00"],
        equity_curve=[1000.0],
        returns=None,
        definition=None,
        capital_basis="run_init_cash",
        init_cash=None,
        fees=None,
        slippage=None,
        reason=None,
        detail=None,
    )
    assert _reason(with_benchmark(no_capital)) == "benchmark_unavailable"

    bad_length = BenchmarkScope.model_construct(
        status="available",
        timestamps=["2024-01-01T00:00:00+00:00", "2024-01-02T00:00:00+00:00"],
        equity_curve=[1000.0, 1001.0],
        returns=[0.01],
        definition=None,
        capital_basis="run_init_cash",
        init_cash=1000.0,
        fees=None,
        slippage=None,
        reason=None,
        detail=None,
    )
    assert _reason(with_benchmark(bad_length)) == "benchmark_unavailable"


def test_signal_numeric_column_refusal(
    service: PairedComparisonService, monkeypatch: pytest.MonkeyPatch
) -> None:
    class TextColumnStore:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def load(self, *_: Any, **__: Any) -> Any:
            return SimpleNamespace(
                manifest=SimpleNamespace(columns=("pearson_ic",)),
                frame=pd.DataFrame(
                    {
                        "pearson_ic": ["x"],
                        "timestamp": ["2024-01-01T00:00:00+00:00"],
                    }
                ),
            )

    from types import SimpleNamespace

    monkeypatch.setattr(
        "tradingdev.adapters.storage.evaluation_series.EvaluationSeriesStore",
        TextColumnStore,
    )
    selected = SeriesSelection(kind="signal", artifact_id="a" * 64, column="pearson_ic")
    assert _reason(service._signal_source(selected)) == "series_unavailable"


def test_artifact_symlink_and_link_race(
    service: PairedComparisonService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradingdev.adapters.storage.filesystem import sha256_text

    document = "race"
    digest = sha256_text(document)
    target = service._workspace.root / "outside.md"
    target.write_text("elsewhere", encoding="utf-8")
    link = service._workspace.root / "paired_comparisons" / f"{digest[:16]}.md"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    with pytest.raises(ValueError, match="leaves the workspace"):
        service._publish_text(
            {},
            document=document,
            prefix="paired-comparison-race",
            suffix=".md",
            artifact_type=ARTIFACT_TYPE,
            media_type="text/markdown",
            payload={"mode": "descriptive", "claim_scope": "x", "status": "available"},
        )
    link.unlink()

    real_link = service._workspace.root / "paired_comparisons" / f"{digest[:16]}.md"
    import os as os_module

    original_link = os_module.link

    def race(source: Any, destination: Any) -> None:
        real_link.write_text(document, encoding="utf-8")
        original_link(source, destination)

    monkeypatch.setattr("tradingdev.app.paired_comparison_service.os.link", race)
    result = service._publish_text(
        {},
        document=document,
        prefix="paired-comparison-race",
        suffix=".md",
        artifact_type=ARTIFACT_TYPE,
        media_type="text/markdown",
        payload={"mode": "descriptive", "claim_scope": "x", "status": "available"},
    )
    assert result["artifact_id"].startswith("paired-comparison-race:")
