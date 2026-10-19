"""Ranking behaviour: direction, comparability, ties and selection pressure."""

from __future__ import annotations

from typing import Any

import pytest

from tradingdev.domain.ml.leaderboard import (
    EvaluationContext,
    LeaderboardEntry,
    build_entry_id,
    comparability_error,
    metric_direction,
    rank_entries,
    rankable_metrics,
    required_t_stat,
    signal_fingerprint,
)

_CONTEXT = EvaluationContext(
    symbol="BTC/USDT",
    timeframe="1d",
    start_date="2024-01-01",
    end_date="2024-12-31",
    horizon=1,
)


def _entry(
    entry_id: str,
    **metrics: float,
) -> LeaderboardEntry:
    return LeaderboardEntry(
        entry_id=entry_id,
        label=f"model-{entry_id}",
        context=_CONTEXT,
        metrics=metrics,
        n_observations=200,
        model_id=entry_id,
        dataset_id="ds_" + entry_id,
    )


def _ranked_ids(result: Any) -> list[str]:
    return [item.entry.entry_id for item in result.ranked]


class TestMetricDirection:
    def test_maximize_metrics_are_declared_higher_is_better(self) -> None:
        assert metric_direction("rank_icir") == "maximize"
        assert metric_direction("ic_mean") == "maximize"
        assert metric_direction("hit_rate") == "maximize"

    def test_minimize_metrics_are_declared_lower_is_better(self) -> None:
        assert metric_direction("signal_turnover") == "minimize"
        assert metric_direction("ic_std") == "minimize"

    def test_unknown_metric_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown signal metric"):
            metric_direction("sharpe_ratio")

    def test_metric_without_a_direction_cannot_rank(self) -> None:
        """A count is not better when larger, so it must not order candidates."""
        with pytest.raises(ValueError, match="no optimisation direction"):
            metric_direction("n_periods")

    def test_rankable_metrics_excludes_directionless_ones(self) -> None:
        rankable = rankable_metrics()
        assert "rank_icir" in rankable
        assert "signal_turnover" in rankable
        assert "n_periods" not in rankable
        assert "n_effective_periods" not in rankable


class TestRanking:
    def test_maximize_metric_puts_the_largest_first(self) -> None:
        result = rank_entries(
            [
                _entry("a", rank_icir=0.10),
                _entry("b", rank_icir=0.30),
                _entry("c", rank_icir=0.20),
            ],
            "rank_icir",
        )
        assert _ranked_ids(result) == ["b", "c", "a"]
        assert result.direction == "maximize"
        assert [item.rank for item in result.ranked] == [1, 2, 3]

    def test_minimize_metric_puts_the_smallest_first(self) -> None:
        result = rank_entries(
            [
                _entry("a", signal_turnover=0.9),
                _entry("b", signal_turnover=0.1),
                _entry("c", signal_turnover=0.5),
            ],
            "signal_turnover",
        )
        assert _ranked_ids(result) == ["b", "c", "a"]
        assert result.direction == "minimize"

    def test_equal_values_share_a_rank_and_the_next_rank_skips(self) -> None:
        result = rank_entries(
            [_entry("a", icir=0.5), _entry("b", icir=0.9), _entry("c", icir=0.5)],
            "icir",
        )
        ranks = {item.entry.entry_id: item.rank for item in result.ranked}
        assert ranks == {"b": 1, "a": 2, "c": 2}
        assert result.ranked[1].tied == 2

    def test_ties_are_broken_deterministically(self) -> None:
        entries = [_entry("c", icir=0.5), _entry("a", icir=0.5), _entry("b", icir=0.5)]
        first = _ranked_ids(rank_entries(entries, "icir"))
        second = _ranked_ids(rank_entries(list(reversed(entries)), "icir"))
        assert first == second == ["a", "b", "c"]

    def test_missing_metric_is_skipped_and_reported(self) -> None:
        result = rank_entries(
            [_entry("a", icir=0.4), _entry("b", hit_rate=0.6)], "icir"
        )
        assert _ranked_ids(result) == ["a"]
        assert [item.entry.entry_id for item in result.skipped] == ["b"]
        assert "no 'icir' recorded" in result.skipped[0].reason

    def test_non_finite_metric_is_skipped_rather_than_ranked(self) -> None:
        """NaN sorts above everything descending; it must never win."""
        result = rank_entries(
            [_entry("a", icir=0.4), _entry("b", icir=float("nan"))],
            "icir",
        )
        assert _ranked_ids(result) == ["a"]
        assert "not finite" in result.skipped[0].reason

    def test_infinite_metric_is_skipped(self) -> None:
        result = rank_entries(
            [_entry("a", icir=0.4), _entry("b", icir=float("inf"))],
            "icir",
        )
        assert _ranked_ids(result) == ["a"]

    def test_unrankable_entries_do_not_inflate_the_comparison(self) -> None:
        one = rank_entries([_entry("a", icir=0.4)], "icir")
        with_skips = rank_entries(
            [_entry("a", icir=0.4), _entry("b"), _entry("c")],
            "icir",
        )
        assert len(one.ranked) == len(with_skips.ranked) == 1
        assert len(with_skips.skipped) == 2
        assert "2 of 3" in with_skips.caveats[0]
        assert any("only one candidate" in caveat for caveat in with_skips.caveats)

    def test_top_k_truncates_the_ranking(self) -> None:
        entries = [
            _entry(name, icir=value)
            for name, value in (("a", 0.1), ("b", 0.5), ("c", 0.3))
        ]
        assert len(rank_entries(entries, "icir", top_k=2).ranked) == 2

    def test_top_k_below_one_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="top_k must be at least 1"):
            rank_entries([_entry("a", icir=0.1)], "icir", top_k=0)

    def test_nothing_rankable_leaves_an_empty_leaderboard(self) -> None:
        result = rank_entries([_entry("a")], "icir")
        assert result.ranked == ()
        assert result.leader is None

    def test_leader_is_the_top_entry(self) -> None:
        result = rank_entries([_entry("a", icir=0.1), _entry("b", icir=0.9)], "icir")
        assert result.leader is not None
        assert result.leader.entry.entry_id == "b"


class TestComparability:
    def test_same_context_is_comparable(self) -> None:
        assert (
            comparability_error([_entry("a", icir=0.1), _entry("b", icir=0.2)]) is None
        )

    def test_different_instruments_are_not_comparable(self) -> None:
        other = EvaluationContext("ETH/USDT", "1d", "2024-01-01", "2024-12-31", 1)
        entries = [
            _entry("a", icir=0.1),
            LeaderboardEntry("b", "model-b", other, {"icir": 0.2}, 200),
        ]
        error = comparability_error(entries)
        assert error is not None
        assert "ETH/USDT" in error and "BTC/USDT" in error

    def test_evaluation_settings_are_part_of_the_context(self) -> None:
        base = _CONTEXT
        assert (
            base.key
            != EvaluationContext(
                "BTC/USDT", "1d", "2024-01-01", "2024-12-31", 1, window=30
            ).key
        )
        assert (
            base.key
            != EvaluationContext(
                "BTC/USDT", "1d", "2024-01-01", "2024-12-31", 1, n_quantiles=3
            ).key
        )
        assert (
            base.key
            != EvaluationContext("BTC/USDT", "1d", "2024-01-01", "2024-12-31", 5).key
        )

    def test_filters_match_individual_fields(self) -> None:
        assert _CONTEXT.matches(symbol="BTC/USDT")
        assert not _CONTEXT.matches(symbol="ETH/USDT")
        assert _CONTEXT.matches(timeframe="1d", horizon=1)
        assert not _CONTEXT.matches(horizon=5)


class TestSelectionPressure:
    def test_threshold_rises_with_the_number_of_candidates(self) -> None:
        assert required_t_stat(1) == pytest.approx(1.96, abs=0.01)
        assert required_t_stat(10) > required_t_stat(1)
        assert required_t_stat(100) > required_t_stat(10)

    def test_threshold_rejects_bad_arguments(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            required_t_stat(0)
        with pytest.raises(ValueError, match="alpha"):
            required_t_stat(5, alpha=0.0)

    def test_a_weak_winner_is_called_out(self) -> None:
        """The t-stat companion is checked against the corrected threshold."""
        entries = [
            _entry("a", rank_icir=0.9, rank_ic_t_stat=0.4),
            _entry("b", rank_icir=0.1, rank_ic_t_stat=0.1),
        ]
        caveat = " ".join(rank_entries(entries, "rank_icir").caveats)
        assert "below the" in caveat
        assert "selection alone" in caveat

    def test_a_strong_winner_is_still_reported_with_its_threshold(self) -> None:
        entries = [
            _entry("a", rank_icir=0.9, rank_ic_t_stat=6.0),
            _entry("b", rank_icir=0.1, rank_ic_t_stat=0.1),
        ]
        caveat = " ".join(rank_entries(entries, "rank_icir").caveats)
        assert "clears the" in caveat

    def test_single_candidate_carries_its_own_caveat(self) -> None:
        caveat = " ".join(rank_entries([_entry("a", icir=0.5)], "icir").caveats)
        assert "only one candidate" in caveat

    def test_a_level_metric_is_checked_against_its_own_significance(self) -> None:
        entries = [
            _entry("a", icir=0.9, ic_t_stat=6.0),
            _entry("b", icir=0.1, ic_t_stat=0.1),
        ]
        caveat = " ".join(rank_entries(entries, "icir").caveats)
        assert "clears the" in caveat

    def test_a_spread_metric_is_not_credited_with_a_mean_test(self) -> None:
        """``ic_t_stat`` tests the mean IC, not the size of the IC spread."""
        entries = [
            _entry("a", ic_std=0.05, ic_t_stat=8.0),
            _entry("b", ic_std=0.40, ic_t_stat=0.1),
        ]
        caveat = " ".join(rank_entries(entries, "ic_std").caveats)
        assert "clears the" not in caveat
        assert "would be needed to call it signal" in caveat

    def test_short_samples_say_the_threshold_is_optimistic(self) -> None:
        entries = [
            _entry("a", icir=0.9, ic_t_stat=6.0, n_effective_periods=4),
            _entry("b", icir=0.1, ic_t_stat=0.1, n_effective_periods=4),
        ]
        caveat = " ".join(rank_entries(entries, "icir").caveats)
        assert "4 effective periods" in caveat
        assert "understates the bar" in caveat

    def test_a_long_sample_adds_no_such_warning(self) -> None:
        entries = [
            _entry("a", icir=0.9, ic_t_stat=6.0, n_effective_periods=200),
            _entry("b", icir=0.1, ic_t_stat=0.1, n_effective_periods=200),
        ]
        caveat = " ".join(rank_entries(entries, "icir").caveats)
        assert "understates the bar" not in caveat


class TestEntryIdentity:
    def test_entry_id_is_deterministic(self) -> None:
        first = build_entry_id(model_id="m", dataset_id="d", context=_CONTEXT)
        second = build_entry_id(model_id="m", dataset_id="d", context=_CONTEXT)
        assert first == second
        assert len(first) == 16

    def test_an_unmodelled_signal_is_identified_by_its_values(self) -> None:
        first = build_entry_id(
            model_id="",
            dataset_id="d",
            context=_CONTEXT,
            signal_fingerprint=signal_fingerprint([1.0, 2.0]),
        )
        second = build_entry_id(
            model_id="",
            dataset_id="d",
            context=_CONTEXT,
            signal_fingerprint=signal_fingerprint([1.0, 3.0]),
        )
        assert first != second

    def test_entry_id_separates_model_dataset_and_context(self) -> None:
        base = build_entry_id(model_id="m", dataset_id="d", context=_CONTEXT)
        assert base != build_entry_id(model_id="n", dataset_id="d", context=_CONTEXT)
        assert base != build_entry_id(model_id="m", dataset_id="e", context=_CONTEXT)
        assert base != build_entry_id(
            model_id="m",
            dataset_id="d",
            context=EvaluationContext("ETH/USDT", "1d", "2024-01-01", "2024-12-31", 1),
        )

    def test_signal_fingerprint_preserves_value_boundaries(self) -> None:
        # Both used to concatenate to "1.023.04.0" without separators.
        assert signal_fingerprint([1.0, 23.0, 4.0]) != signal_fingerprint(
            [1.02, 3.0, 4.0]
        )

    def test_signal_fingerprint_preserves_length_and_order(self) -> None:
        assert signal_fingerprint([1, 2]) == signal_fingerprint([1.0, 2.0])
        assert signal_fingerprint([1, 2]) != signal_fingerprint([2, 1])
        assert signal_fingerprint([1, 2]) != signal_fingerprint([1, 2, 0])

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
    def test_signal_fingerprint_refuses_non_finite_values(self, value: float) -> None:
        with pytest.raises(ValueError, match="Out of range float"):
            signal_fingerprint([value])

    def test_record_round_trip_preserves_everything(self) -> None:
        entry = _entry("a", icir=0.5, hit_rate=0.6)
        restored = LeaderboardEntry.from_record(entry.to_record())
        assert restored == entry

    def test_record_without_context_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="context"):
            LeaderboardEntry.from_record({"entry_id": "a", "metrics": {}})

    def test_record_without_metrics_is_rejected(self) -> None:
        payload = _entry("a", icir=0.5).to_record()
        del payload["metrics"]
        with pytest.raises(ValueError, match="metrics"):
            LeaderboardEntry.from_record(payload)

    def test_metric_lookup_is_none_for_missing_and_non_finite(self) -> None:
        entry = _entry("a", icir=0.5, bad=float("nan"))
        assert entry.metric("icir") == pytest.approx(0.5)
        assert entry.metric("hit_rate") is None
        assert entry.metric("bad") is None


class TestFeatureIdentity:
    """Feature set and code changes must create distinct, incomparable evidence."""

    @staticmethod
    def _context(**overrides: object) -> EvaluationContext:
        values: dict[str, object] = {
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "horizon": 1,
        }
        values.update(overrides)
        return EvaluationContext(**values)  # type: ignore[arg-type]

    def test_an_unknown_feature_identity_stays_unknown(self) -> None:
        assert self._context().feature_set == ""
        assert self._context().code_fingerprint == ""

    def test_the_feature_set_is_part_of_the_identity(self) -> None:
        assert (
            self._context(feature_set="returns").key
            != self._context(feature_set="technical").key
        )

    def test_the_code_fingerprint_is_part_of_the_identity(self) -> None:
        assert (
            self._context(code_fingerprint="v1").key
            != self._context(code_fingerprint="v2").key
        )

    def test_dataset_content_is_part_of_comparability_and_record_identity(self) -> None:
        before = self._context(data_fingerprint="old-bars")
        after = self._context(data_fingerprint="corrected-bars")
        first = self._entry("m", before, icir=0.5)
        second = self._entry("m", after, icir=0.6)
        assert before.key != after.key
        assert first.entry_id != second.entry_id
        assert comparability_error([first, second]) is not None
        assert LeaderboardEntry.from_record(first.to_record()).context == before
        assert "old-bars" in before.describe()

    @staticmethod
    def _entry(
        entry_id: str, context: EvaluationContext, **metrics: float
    ) -> LeaderboardEntry:
        return LeaderboardEntry(
            entry_id=build_entry_id(
                model_id=entry_id, dataset_id="ds_1", context=context
            ),
            label=f"model-{entry_id}",
            context=context,
            metrics=metrics,
            n_observations=200,
            model_id=entry_id,
            dataset_id="ds_1",
        )

    def test_a_code_change_is_not_comparable(self) -> None:
        entries = [
            self._entry("a", self._context(code_fingerprint="v1"), icir=0.5),
            self._entry("b", self._context(code_fingerprint="v2"), icir=0.9),
        ]
        error = comparability_error(entries)
        assert error is not None
        assert "v1" in error and "v2" in error

    def test_the_same_code_is_still_comparable(self) -> None:
        entries = [
            self._entry("a", self._context(code_fingerprint="v1"), icir=0.5),
            self._entry("b", self._context(code_fingerprint="v1"), icir=0.9),
        ]
        assert comparability_error(entries) is None

    def test_a_code_change_gives_a_new_entry_id(self) -> None:
        before = build_entry_id(
            model_id="m",
            dataset_id="ds_1",
            context=self._context(code_fingerprint="v1"),
        )
        after = build_entry_id(
            model_id="m",
            dataset_id="ds_1",
            context=self._context(code_fingerprint="v2"),
        )
        assert before != after

    def test_describe_names_the_feature_set_and_code(self) -> None:
        described = self._context(
            feature_set="returns", code_fingerprint="abc123"
        ).describe()
        assert "features=returns" in described
        assert "code=abc123" in described

    def test_matches_filters_by_feature_set(self) -> None:
        context = self._context(feature_set="returns")
        assert context.matches(feature_set="returns") is True
        assert context.matches(feature_set="technical") is False
        assert context.matches() is True

    def test_a_record_carries_the_feature_identity(self) -> None:
        entry = self._entry(
            "a", self._context(feature_set="returns", code_fingerprint="v1"), icir=0.5
        )
        restored = LeaderboardEntry.from_record(entry.to_record())
        assert restored.context.feature_set == "returns"
        assert restored.context.code_fingerprint == "v1"
        assert restored == entry

    def test_an_older_record_reads_as_unknown(self) -> None:
        payload = _entry("a", icir=0.5).to_record()
        del payload["context"]["feature_set"]
        del payload["context"]["code_fingerprint"]
        restored = LeaderboardEntry.from_record(payload)
        assert restored.context.feature_set == ""
        assert restored.context.code_fingerprint == ""
