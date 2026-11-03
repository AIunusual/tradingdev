"""The shared ranking core: direction, ties, missing values and groups."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from tradingdev.domain import ranking
from tradingdev.domain.ranking import (
    Candidate,
    MetricDefinition,
    RankingResult,
    rank_candidates,
    signal_candidate_id,
)

_MAXIMIZE = MetricDefinition(name="score", direction="maximize", unit="ratio")
_MINIMIZE = MetricDefinition(name="score", direction="minimize")


def _candidate(
    candidate_id: str, *, group: str = "ctx", **values: float | None
) -> Candidate[str]:
    return Candidate(
        candidate_id=candidate_id, values=values, group_key=group, payload=candidate_id
    )


def _ids(result: RankingResult[str]) -> list[str]:
    return [item.candidate.candidate_id for item in result.ranked]


class TestDirection:
    def test_maximize_puts_the_largest_first(self) -> None:
        result = rank_candidates(
            [_candidate("a", score=0.1), _candidate("b", score=0.3)], _MAXIMIZE
        )
        assert _ids(result) == ["b", "a"]
        assert [item.value for item in result.ranked] == [0.3, 0.1]
        assert result.direction == "maximize"

    def test_minimize_puts_the_smallest_first(self) -> None:
        result = rank_candidates(
            [_candidate("a", score=0.1), _candidate("b", score=0.3)], _MINIMIZE
        )
        assert _ids(result) == ["a", "b"]
        assert result.direction == "minimize"


class TestTies:
    def test_equal_values_share_a_rank_and_count_themselves(self) -> None:
        result = rank_candidates(
            [
                _candidate("a", score=0.3),
                _candidate("b", score=0.1),
                _candidate("c", score=0.3),
            ],
            _MAXIMIZE,
        )
        assert [
            (item.rank, item.candidate.candidate_id, item.tied)
            for item in result.ranked
        ] == [
            (1, "a", 2),
            (1, "c", 2),
            (3, "b", 1),
        ]

    def test_near_equal_values_are_ties(self) -> None:
        """A difference below the isclose tolerance is not a real ordering."""
        value = 0.3
        neighbour = math.nextafter(value, 1.0)
        result = rank_candidates(
            [_candidate("a", score=value), _candidate("b", score=neighbour)],
            _MAXIMIZE,
        )
        assert [item.rank for item in result.ranked] == [1, 1]
        assert result.ranked[0].tied == 2

    def test_clearly_different_values_are_not_ties(self) -> None:
        result = rank_candidates(
            [_candidate("a", score=0.3), _candidate("b", score=0.300001)], _MAXIMIZE
        )
        assert [item.rank for item in result.ranked] == [1, 2]

    @pytest.mark.parametrize("definition", [_MAXIMIZE, _MINIMIZE])
    def test_tie_order_follows_the_id_not_the_direction(
        self, definition: MetricDefinition
    ) -> None:
        """Reversing the value sort must not also reverse the tie-break."""
        result = rank_candidates(
            [
                _candidate("c", score=0.2),
                _candidate("a", score=0.2),
                _candidate("b", score=0.2),
            ],
            definition,
        )
        assert _ids(result) == ["a", "b", "c"]


class TestMissingValues:
    def test_an_unrecorded_metric_says_so(self) -> None:
        result = rank_candidates(
            [_candidate("a", score=0.1), _candidate("b")], _MAXIMIZE
        )
        assert [item.candidate.candidate_id for item in result.skipped] == ["b"]
        assert result.skipped[0].reason == "no 'score' recorded"

    @pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf")])
    def test_an_unusable_value_is_not_an_unrecorded_one(
        self, value: float | None
    ) -> None:
        result = rank_candidates([_candidate("a", score=value)], _MAXIMIZE)
        assert result.ranked == ()
        assert result.skipped[0].reason == "'score' is unavailable or not finite"

    def test_a_missing_value_is_never_treated_as_zero(self) -> None:
        """Zero would rank; an unavailable measurement must not."""
        result = rank_candidates(
            [_candidate("a", score=0.0), _candidate("b", score=None)], _MINIMIZE
        )
        assert _ids(result) == ["a"]
        assert len(result.skipped) == 1


class TestLimitsAndCaveats:
    def test_top_k_limits_the_slice_but_not_the_reported_total(self) -> None:
        result = rank_candidates(
            [
                _candidate("a", score=0.3),
                _candidate("b", score=0.2),
                _candidate("c", score=0.1),
            ],
            _MAXIMIZE,
            top_k=2,
        )
        assert _ids(result) == ["a", "b"]
        assert result.ranked_total == 3

    @pytest.mark.parametrize("top_k", [0, -1])
    def test_a_non_positive_limit_is_rejected(self, top_k: int) -> None:
        with pytest.raises(ValueError, match="top_k must be at least 1"):
            rank_candidates([_candidate("a", score=0.1)], _MAXIMIZE, top_k=top_k)

    def test_one_candidate_proves_nothing_and_says_so(self) -> None:
        result = rank_candidates([_candidate("a", score=0.1)], _MAXIMIZE)
        assert any("only one candidate" in text for text in result.caveats)

    def test_an_empty_selection_is_not_an_error(self) -> None:
        result: RankingResult[str] = rank_candidates([], _MAXIMIZE)
        assert result.ranked == () and result.skipped == ()
        assert result.ranked_total == 0

    def test_skipped_candidates_are_counted_in_the_caveat(self) -> None:
        result = rank_candidates(
            [_candidate("a", score=0.1), _candidate("b", score=None)], _MAXIMIZE
        )
        assert result.caveats[0].startswith(
            "1 of 2 entries were not ranked for 'score'"
        )


class TestGroups:
    def test_mixing_groups_without_permission_is_refused(self) -> None:
        with pytest.raises(ValueError, match="incomparable groups"):
            rank_candidates(
                [
                    _candidate("a", group="ctx-1", score=0.1),
                    _candidate("b", group="ctx-2", score=0.2),
                ],
                _MAXIMIZE,
            )

    def test_permitted_mixing_ranks_and_reports_both_groups(self) -> None:
        result = rank_candidates(
            [
                _candidate("a", group="ctx-1", score=0.1),
                _candidate("b", group="ctx-2", score=0.2),
            ],
            _MAXIMIZE,
            allow_mixed=True,
        )
        assert _ids(result) == ["b", "a"]
        assert result.groups == ("ctx-1", "ctx-2")

    def test_one_group_is_never_mixed(self) -> None:
        result = rank_candidates(
            [_candidate("a", score=0.1), _candidate("b", score=0.2)], _MAXIMIZE
        )
        assert result.groups == ("ctx",)


class TestPayloadAndLayering:
    def test_the_payload_comes_back_untouched(self) -> None:
        sentinel = object()
        candidate = Candidate(
            candidate_id="a", values={"score": 0.1}, group_key="ctx", payload=sentinel
        )
        result = rank_candidates([candidate], _MAXIMIZE)
        assert result.ranked[0].candidate.payload is sentinel

    def test_the_core_does_not_depend_on_the_ml_layer(self) -> None:
        """A shared core that imports domain.ml cannot serve backtest candidates."""
        source = Path(ranking.__file__).read_text(encoding="utf-8")
        assert "tradingdev.domain.ml" not in source

    def test_signal_ids_have_one_grammar(self) -> None:
        assert signal_candidate_id("abc123") == "signal:abc123"
