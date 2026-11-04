"""Pin the ranking output that existed before the core was extracted.

The expected values below were captured from ``rank_entries`` on the tree before
``domain/ranking.py`` existed. They are deliberately literal: this file exists to
prove the extraction changed no ordering, no tie, no reason and no caveat. A later
change that intends to alter the output should update this file in the same
commit, which makes the change visible in review instead of hiding inside a
refactor.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tradingdev.domain.ml.leaderboard import (
    EvaluationContext,
    LeaderboardEntry,
    LeaderboardResult,
    rank_entries,
)

_CONTEXT = EvaluationContext(
    symbol="BTC/USDT",
    timeframe="1d",
    start_date="2024-01-01",
    end_date="2024-12-31",
    horizon=1,
)

_FAMILY_CAVEAT = (
    "candidates span 2 method families (ml_model=6, rule=1); family labels do not "
    "establish equal evidence about selection history or out-of-sample performance. "
    "Inspect provenance and measurement conditions; filter by family to rank one kind"
)


def _entry(entry_id: str, **metrics: float | None) -> LeaderboardEntry:
    return LeaderboardEntry(
        entry_id=entry_id,
        label=f"model-{entry_id}",
        context=_CONTEXT,
        metrics=metrics,
        n_observations=200,
        model_id=entry_id,
        dataset_id=f"ds_{entry_id}",
    )


def _entries() -> list[LeaderboardEntry]:
    nan = float("nan")
    return [
        _entry("a", ic_mean=0.30, ic_std=0.10, signal_turnover=0.20),
        _entry("b", ic_mean=0.10, ic_std=0.10, signal_turnover=0.40),
        _entry("c", ic_mean=0.30, ic_std=0.20, signal_turnover=0.10),
        _entry("d", ic_mean=0.20, ic_std=0.05, signal_turnover=0.30),
        # Key absent: the metric was never recorded for this candidate.
        _entry("e", ic_std=0.30, signal_turnover=0.50),
        # Key present but not finite: the measurement failed.
        _entry("f", ic_mean=nan, ic_std=nan, signal_turnover=0.60),
        replace(
            _entry("r", ic_mean=0.05, ic_std=0.15, signal_turnover=0.25),
            model_id="",
            method_family="rule",
            method_name="rsi_14 < 30",
            parameters={"threshold": 30.0},
        ),
    ]


def _observed(result: LeaderboardResult) -> dict[str, object]:
    return {
        "metric": result.metric,
        "direction": result.direction,
        "ranked": [
            (item.rank, item.entry.entry_id, item.value, item.tied)
            for item in result.ranked
        ],
        "skipped": [(item.entry.entry_id, item.reason) for item in result.skipped],
        "caveats": list(result.caveats),
    }


_EXPECTED: dict[str, dict[str, object]] = {
    "ic_mean": {
        "metric": "ic_mean",
        "direction": "maximize",
        "ranked": [
            (1, "a", 0.30, 2),
            (1, "c", 0.30, 2),
            (3, "d", 0.20, 1),
            (4, "b", 0.10, 1),
            (5, "r", 0.05, 1),
        ],
        "skipped": [
            ("e", "no 'ic_mean' recorded"),
            ("f", "'ic_mean' is unavailable or not finite"),
        ],
        "caveats": [
            "2 of 7 entries were not ranked for 'ic_mean'. They stay recorded, so a "
            "parameter grid keeps their coordinates even when the score is "
            "unavailable; the ranked candidates are not necessarily adjacent "
            "settings.",
            "best of 5 candidates: the winner is drawn from 5 comparisons, so expect "
            "it to be optimistic; a t-statistic above 2.576 would be needed to call "
            "it signal",
            _FAMILY_CAVEAT,
        ],
    },
    "ic_std": {
        "metric": "ic_std",
        "direction": "minimize",
        "ranked": [
            (1, "d", 0.05, 1),
            (2, "a", 0.10, 2),
            (2, "b", 0.10, 2),
            (4, "r", 0.15, 1),
            (5, "c", 0.20, 1),
            (6, "e", 0.30, 1),
        ],
        "skipped": [("f", "'ic_std' is unavailable or not finite")],
        "caveats": [
            "1 of 7 entries were not ranked for 'ic_std'. They stay recorded, so a "
            "parameter grid keeps their coordinates even when the score is "
            "unavailable; the ranked candidates are not necessarily adjacent "
            "settings.",
            "best of 6 candidates: the winner is drawn from 6 comparisons, so expect "
            "it to be optimistic; a t-statistic above 2.638 would be needed to call "
            "it signal",
            _FAMILY_CAVEAT,
        ],
    },
    "signal_turnover": {
        "metric": "signal_turnover",
        "direction": "minimize",
        "ranked": [
            (1, "c", 0.10, 1),
            (2, "a", 0.20, 1),
            (3, "r", 0.25, 1),
            (4, "d", 0.30, 1),
            (5, "b", 0.40, 1),
            (6, "e", 0.50, 1),
            (7, "f", 0.60, 1),
        ],
        "skipped": [],
        "caveats": [
            "best of 7 candidates: the winner is drawn from 7 comparisons, so expect "
            "it to be optimistic; a t-statistic above 2.690 would be needed to call "
            "it signal",
            _FAMILY_CAVEAT,
        ],
    },
    "ic_mean_top2": {
        "metric": "ic_mean",
        "direction": "maximize",
        "ranked": [(1, "a", 0.30, 2), (1, "c", 0.30, 2)],
        "skipped": [
            ("e", "no 'ic_mean' recorded"),
            ("f", "'ic_mean' is unavailable or not finite"),
        ],
        "caveats": [
            "2 of 7 entries were not ranked for 'ic_mean'. They stay recorded, so a "
            "parameter grid keeps their coordinates even when the score is "
            "unavailable; the ranked candidates are not necessarily adjacent "
            "settings.",
            "best of 5 candidates: the winner is drawn from 5 comparisons, so expect "
            "it to be optimistic; a t-statistic above 2.576 would be needed to call "
            "it signal",
            _FAMILY_CAVEAT,
        ],
    },
}


@pytest.mark.parametrize("scenario", sorted(_EXPECTED))
def test_ranking_output_is_unchanged(scenario: str) -> None:
    """Every ordering, tie, reason and caveat matches the pre-extraction output."""
    entries = _entries()
    if scenario == "ic_mean_top2":
        result = rank_entries(entries, "ic_mean", top_k=2)
    else:
        result = rank_entries(entries, scenario)
    assert _observed(result) == _EXPECTED[scenario]
