"""The report projection: definitions, coordinates and what is shown by default.

Expected values here are declared, not copied from the code under test: the
visible set and the annualised set are written out by hand from the metric
catalogue's meaning, so a change in either direction has to be argued for.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from tradingdev.domain.ml.evaluation.artifacts import (
    SignalEvaluationBundle,
    build_signal_evaluation,
)
from tradingdev.domain.ml.evaluation.catalog import (
    SIGNAL_METRIC_CATALOG,
    definition_snapshot,
    report_visibility,
)
from tradingdev.domain.ml.leaderboard import (
    EvaluationContext,
    LeaderboardEntry,
)
from tradingdev.domain.ml.report import (
    ML_REPORT_SCHEMA_VERSION,
    ML_SIGNAL_KIND,
    ML_SIGNAL_MODE,
    ml_report_block,
    ml_signal_scope,
)

_CONTEXT = EvaluationContext(
    symbol="BTC/USDT",
    timeframe="1d",
    start_date="2024-01-01",
    end_date="2024-12-31",
    horizon=1,
)

# Declared by hand: a summary metric that says which direction is better is
# shown; everything else is stored but left unticked.
_VISIBLE = {
    "ic_mean",
    "icir",
    "rank_ic_mean",
    "rank_icir",
    "hit_rate",
    "strata_monotonicity",
    "long_short_spread",
    "annual_return",
    "max_drawdown",
    "sharpe",
    "alpha",
}

# Only these three divide or compound by a year; a drawdown is not annualised.
_ANNUALISED = {"annual_return", "sharpe", "alpha"}

_SNAPSHOT_FIELDS = {
    "id",
    "unit",
    "category",
    "provider",
    "description",
    "summary",
    "optimization_direction",
    "modes",
    "requires_annualization",
}


def _entry(
    entry_id: str,
    *,
    window: int = 20,
    horizon: int = 1,
    parameters: dict[str, Any] | None = None,
    **metrics: float | None,
) -> LeaderboardEntry:
    return LeaderboardEntry(
        entry_id=entry_id,
        label=f"model-{entry_id}",
        context=replace(_CONTEXT, window=window, horizon=horizon),
        metrics=metrics,
        n_observations=200,
        model_id=entry_id,
        dataset_id=f"ds_h{horizon}",
        parameters=parameters or {},
        signal_fingerprint="0123456789abcdef",
    )


class TestDefinitions:
    @pytest.mark.parametrize("metric_id", ["annual_return", "sharpe", "alpha", "beta"])
    def test_legacy_annualization_is_unknown_not_false(self, metric_id: str) -> None:
        bundle = build_signal_evaluation(
            "legacy",
            values={metric_id: 0.1},
            strata=[],
            horizon=1,
            window=20,
            n_bins=5,
            method="spearman",
        )
        payload = bundle.model_dump(mode="json")
        definition = payload["definitions"][metric_id]
        del definition["modes"]
        del definition["requires_annualization"]
        restored = SignalEvaluationBundle.model_validate_json(json.dumps(payload))
        assert restored.definitions[metric_id].requires_annualization is None
        assert restored.definitions[metric_id].modes == ("signal",)

    @pytest.mark.parametrize("metric_id", ["annual_return", "sharpe", "alpha", "beta"])
    def test_new_bundle_preserves_explicit_annualization(self, metric_id: str) -> None:
        bundle = build_signal_evaluation(
            "current",
            values={metric_id: 0.1},
            strata=[],
            horizon=1,
            window=20,
            n_bins=5,
            method="spearman",
        )
        restored = SignalEvaluationBundle.model_validate_json(bundle.model_dump_json())
        assert restored.definitions[metric_id].requires_annualization is (
            metric_id in _ANNUALISED
        )

    def test_every_signal_metric_declares_the_signal_mode(self) -> None:
        assert {
            mode for item in SIGNAL_METRIC_CATALOG.values() for mode in item.modes
        } == {"signal"}

    def test_only_the_compounding_metrics_require_annualisation(self) -> None:
        annualised = {
            key
            for key, item in SIGNAL_METRIC_CATALOG.items()
            if item.requires_annualization
        }
        assert annualised == _ANNUALISED

    def test_a_snapshot_uses_exactly_the_report_field_set(self) -> None:
        assert set(definition_snapshot("ic_mean")) == _SNAPSHOT_FIELDS

    def test_a_snapshot_keeps_the_optimisation_direction(self) -> None:
        assert definition_snapshot("ic_mean")["optimization_direction"] == "maximize"
        assert (
            definition_snapshot("max_drawdown")["optimization_direction"] == "minimize"
        )
        assert definition_snapshot("beta")["optimization_direction"] is None

    def test_a_snapshot_lists_modes_as_a_json_array(self) -> None:
        # The report stores modes as a JSON array, not a Python tuple.
        assert definition_snapshot("sharpe")["modes"] == ["signal"]

    def test_an_unknown_metric_has_no_snapshot(self) -> None:
        with pytest.raises(KeyError, match="unknown signal metric"):
            definition_snapshot("not_a_metric")

    def test_an_unknown_metric_has_no_visibility(self) -> None:
        with pytest.raises(KeyError, match="unknown signal metric"):
            report_visibility("not_a_metric")


class TestVisibility:
    def test_visibility_shows_exactly_the_rankable_summaries(self) -> None:
        shown = {key for key in SIGNAL_METRIC_CATALOG if report_visibility(key)}
        assert shown == _VISIBLE

    def test_beta_is_stored_but_not_shown(self) -> None:
        # Summary metric with no direction: exposure is described, not
        # preferred, so it is not put in the default view.
        assert SIGNAL_METRIC_CATALOG["beta"].summary is True
        assert report_visibility("beta") is False
        assert "beta" in definition_snapshot("beta")["id"]

    def test_every_shown_metric_can_also_rank(self) -> None:
        # A metric in the default view must have a direction, or a reader would
        # read an exposure or a count as quality.
        for key in SIGNAL_METRIC_CATALOG:
            if report_visibility(key):
                assert SIGNAL_METRIC_CATALOG[key].optimization_direction is not None

    def test_the_period_counts_stay_hidden(self) -> None:
        for key in ("periods_per_year", "n_return_periods", "n_periods"):
            assert report_visibility(key) is False


class TestScope:
    def test_a_scope_carries_the_report_scope_fields(self) -> None:
        scope = ml_signal_scope(_entry("aaa", ic_mean=0.05))
        assert set(scope) == {
            "scope_id",
            "kind",
            "mode",
            "split",
            "fold_index",
            "trial_index",
            "values",
            "parameters",
            "metadata",
            "observations",
        }
        assert scope["kind"] == ML_SIGNAL_KIND
        assert scope["mode"] == ML_SIGNAL_MODE
        assert scope["split"] == "full"

    def test_a_scope_says_it_has_no_observations(self) -> None:
        # A signal evaluation records no orders; the key is present and null.
        scope = ml_signal_scope(_entry("aaa", ic_mean=0.05))
        assert "observations" in scope
        assert scope["observations"] is None

    def test_a_failed_score_stays_at_its_coordinate(self) -> None:
        scope = ml_signal_scope(_entry("aaa", ic_mean=None, sharpe=1.25))
        assert scope["values"]["ic_mean"] is None
        assert scope["values"]["sharpe"] == 1.25

    def test_the_axis_coordinates_never_overwrite_a_fitted_parameter(self) -> None:
        # A fitted dictionary may legitimately hold a key called "window".
        scope = ml_signal_scope(
            _entry("aaa", window=30, parameters={"window": 99, "alpha": 0.1})
        )
        assert scope["parameters"] == {"window": 99, "alpha": 0.1}
        assert scope["metadata"]["window"] == 30

    def test_a_scope_reports_its_identity_and_fingerprint(self) -> None:
        scope = ml_signal_scope(_entry("aaa", ic_mean=0.05))
        assert scope["metadata"]["model_id"] == "aaa"
        assert scope["metadata"]["signal_fingerprint"] == "0123456789abcdef"
        assert scope["metadata"]["n_observations"] == 200
        assert scope["metadata"]["dataset_id"] == "ds_h1"


class TestReportBlock:
    def _block(self, *entries: LeaderboardEntry) -> dict[str, Any]:
        return ml_report_block(entries, "ic_mean", allow_parameter_axes=True)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
    def test_nonfinite_metrics_remain_json_null(self, value: float) -> None:
        block = self._block(
            _entry("failed", window=20, ic_mean=value, sharpe=value),
            _entry("valid", window=30, ic_mean=0.0),
        )
        encoded = json.loads(json.dumps(block, allow_nan=False))
        assert encoded["scopes"][0]["values"] == {"ic_mean": None, "sharpe": None}
        assert encoded["scopes"][1]["values"] == {"ic_mean": 0.0}
        assert encoded["parameter_axes"]["window"] == [20, 30]
        assert encoded["ranking"]["skipped"][0]["scope_id"] == "failed"

    def test_the_ranking_covers_every_candidate(self) -> None:
        entries = [_entry("a", ic_mean=0.1), _entry("b", ic_mean=0.2)]
        block = ml_report_block(entries, "ic_mean")
        assert block["ranking"]["ranked_scope_ids"] == ["b", "a"]

    @pytest.mark.parametrize("updated", [0.9, None])
    def test_mutated_scores_are_ranked_with_current_evidence(
        self, updated: float | None
    ) -> None:
        metrics: dict[str, float | None] = {"ic_mean": 0.1}
        a = replace(_entry("a"), metrics=metrics)
        b = _entry("b", ic_mean=0.2)
        first = ml_report_block([a, b], "ic_mean")
        metrics["ic_mean"] = updated
        block = ml_report_block([a, b], "ic_mean")
        assert first["ranking"]["ranked_scope_ids"] == ["b", "a"]
        assert first["scopes"][0]["values"]["ic_mean"] == 0.1
        assert block["ranking"]["ranked_scope_ids"] == (
            ["b"] if updated is None else ["a", "b"]
        )
        if updated is None:
            assert block["ranking"]["skipped"][0]["scope_id"] == "a"

    def test_mutated_significance_rebuilds_the_caveat(self) -> None:
        metrics: dict[str, float | None] = {"ic_mean": 0.9, "ic_t_stat": 0.1}
        a = replace(_entry("a"), metrics=metrics)
        b = _entry("b", ic_mean=0.2)
        first = ml_report_block([a, b], "ic_mean")
        metrics["ic_t_stat"] = 10.0
        block = ml_report_block([a, b], "ic_mean")
        assert any("is below" in text for text in first["caveats"])
        assert any("clears" in text for text in block["caveats"])

    def test_ranking_uses_the_requested_metric_and_alpha(self) -> None:
        entries = [
            _entry("a", ic_mean=0.9, ic_t_stat=2.4, max_drawdown=0.2),
            _entry("b", ic_mean=0.2, max_drawdown=0.1),
        ]
        block = ml_report_block(entries, "max_drawdown")
        assert block["ranking"]["direction"] == "minimize"
        assert block["ranking"]["ranked_scope_ids"] == ["b", "a"]
        assert any(
            "is below" in text
            for text in ml_report_block(entries, "ic_mean", alpha=0.01)["caveats"]
        )

    def test_parameter_surfaces_require_opt_in(self) -> None:
        entries = [_entry("a", window=10, ic_mean=0.1), _entry("b", ic_mean=0.2)]
        with pytest.raises(ValueError, match="same conditions"):
            ml_report_block(entries, "ic_mean")

    def test_parameter_surfaces_reject_incompatible_dataset_versions(self) -> None:
        entries = [
            _entry("a", window=10, ic_mean=0.1),
            replace(_entry("b", ic_mean=0.2), dataset_id="rebuilt"),
        ]
        with pytest.raises(ValueError, match="different dataset versions"):
            ml_report_block(entries, "ic_mean", allow_parameter_axes=True)

    def test_duplicate_candidate_ids_are_rejected(self) -> None:
        entries = [_entry("a", ic_mean=0.1), _entry("a", ic_mean=0.2)]
        with pytest.raises(ValueError, match="unique"):
            self._block(*entries)

    def test_round_tripped_entries_keep_the_same_evidence(self) -> None:
        entries = [_entry("a", ic_mean=float("nan")), _entry("b", ic_mean=0.1)]
        restored = [LeaderboardEntry.from_record(item.to_record()) for item in entries]
        block = ml_report_block(restored, "ic_mean")
        assert block == ml_report_block(entries, "ic_mean")
        assert block["ranking"]["ranked_scope_ids"] == ["b"]
        assert block["ranking"]["skipped"][0]["scope_id"] == "a"

    def test_an_empty_candidate_set_can_be_projected(self) -> None:
        block = self._block()
        assert block["scopes"] == []
        assert block["ranking"]["ranked_scope_ids"] == []
        assert block["ranking"]["skipped"] == []

    def test_the_block_keeps_every_candidate_including_failures(self) -> None:
        block = self._block(
            _entry("a", window=10, ic_mean=0.05),
            _entry("b", window=20, ic_mean=None),
            _entry("c", window=30, ic_mean=0.02),
        )
        assert len(block["scopes"]) == 3
        assert {scope["scope_id"] for scope in block["scopes"]} == {"a", "b", "c"}

    def test_a_failed_candidate_keeps_its_grid_position(self) -> None:
        block = self._block(
            _entry("a", window=10, ic_mean=0.05),
            _entry("b", window=20, ic_mean=None),
            _entry("c", window=30, ic_mean=0.02),
        )
        # The gap must not close: 20 is still an observed axis value.
        assert block["parameter_axes"]["window"] == [10, 20, 30]
        skipped = block["ranking"]["skipped"]
        assert len(skipped) == 1
        assert skipped[0]["scope_id"] == "b"
        assert skipped[0]["window"] == 20
        assert "not finite" in skipped[0]["reason"]

    def test_the_block_reports_the_axes_of_the_surface(self) -> None:
        block = self._block(
            _entry("a", window=10, horizon=1, ic_mean=0.05),
            _entry("b", window=30, horizon=3, ic_mean=0.02),
        )
        assert block["parameter_axes"] == {
            "horizon": [1, 3],
            "window": [10, 30],
            "n_quantiles": [5],
        }

    def test_definitions_are_stored_once_for_the_whole_block(self) -> None:
        block = self._block(_entry("a", ic_mean=0.05, sharpe=1.0))
        assert set(block["definitions"]) == {"ic_mean", "sharpe"}
        for scope in block["scopes"]:
            assert "definitions" not in scope

    def test_every_stored_value_is_defined(self) -> None:
        # The report rejects a value without a definition, so the projection
        # must never produce one.
        block = self._block(_entry("a", ic_mean=0.05, sharpe=1.0, beta=0.3))
        defined = set(block["definitions"])
        for scope in block["scopes"]:
            assert set(scope["values"]) <= defined

    def test_the_block_separates_shown_from_hidden_metrics(self) -> None:
        block = self._block(_entry("a", ic_mean=0.05, beta=0.3, n_periods=180))
        assert set(block["visible_metrics"]) == {"ic_mean"}
        assert set(block["hidden_metrics"]) == {"beta", "n_periods"}

    def test_the_block_keeps_the_ranking_and_its_caveats(self) -> None:
        block = self._block(
            _entry("a", window=10, ic_mean=0.05),
            _entry("b", window=30, ic_mean=0.02),
        )
        assert block["schema_version"] == ML_REPORT_SCHEMA_VERSION
        assert block["ranking"]["metric"] == "ic_mean"
        assert block["ranking"]["direction"] == "maximize"
        assert block["ranking"]["ranked_scope_ids"] == ["a", "b"]
        assert isinstance(block["caveats"], list)

    def test_a_surface_ranking_carries_the_surface_caveat(self) -> None:
        block = self._block(
            _entry("a", window=10, ic_mean=0.05),
            _entry("b", window=30, ic_mean=0.02),
        )
        assert any(
            "more than one evaluation setting" in text for text in block["caveats"]
        )

    def test_one_setting_needs_no_surface_caveat(self) -> None:
        block = self._block(
            _entry("a", window=20, ic_mean=0.05),
            _entry("b", window=20, ic_mean=0.02),
        )
        assert not any(
            "more than one evaluation setting" in text for text in block["caveats"]
        )

    def test_a_horizon_sweep_carries_the_provenance_caveat(self) -> None:
        # A surface may span horizons, but nothing verified that the source
        # bars match, so the block must say so rather than imply control.
        block = self._block(
            _entry("a", horizon=1, ic_mean=0.05),
            _entry("b", horizon=3, ic_mean=0.02),
        )
        assert any("cross-horizon source" in text for text in block["caveats"])

    def test_one_horizon_needs_no_provenance_caveat(self) -> None:
        block = self._block(
            _entry("a", horizon=1, ic_mean=0.05),
            _entry("b", horizon=1, ic_mean=0.02),
        )
        assert not any("cross-horizon source" in text for text in block["caveats"])

    def test_the_block_encodes_with_the_report_encoder(self) -> None:
        # The report hashes its payload with allow_nan=False; a None must
        # become null rather than an exception or a NaN token.
        block = self._block(
            _entry("a", ic_mean=None, sharpe=1.0),
            _entry("b", ic_mean=0.02),
        )
        encoded = json.dumps(
            block,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        assert '"ic_mean":null' in encoded
        assert json.loads(encoded)["scopes"][0]["values"]["ic_mean"] is None

    def test_a_metric_without_a_definition_cannot_be_projected(self) -> None:
        entries = [_entry("a", ic_mean=0.05, invented_metric=1.0)]
        with pytest.raises(KeyError, match="unknown signal metric"):
            ml_report_block(entries, "ic_mean")
