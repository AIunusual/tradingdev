"""An entry says which dataset trained its model, and never guesses one."""

from __future__ import annotations

from tradingdev.domain.ml.leaderboard import EvaluationContext, LeaderboardEntry

_CONTEXT = EvaluationContext(
    symbol="BTC/USDT",
    timeframe="1d",
    start_date="2024-01-01",
    end_date="2024-12-31",
    horizon=1,
)


def test_the_training_dataset_survives_a_record_round_trip() -> None:
    entry = LeaderboardEntry(
        entry_id="a",
        label="xgboost_direction (xgboost)",
        context=_CONTEXT,
        metrics={"ic_mean": 0.2},
        n_observations=240,
        model_id="a" * 16,
        dataset_id="ds_evaluation",
        training_dataset_id="ds_training",
    )
    restored = LeaderboardEntry.from_record(entry.to_record())
    assert restored.training_dataset_id == "ds_training"
    assert restored == entry


def test_a_record_written_before_the_binding_reads_as_unknown() -> None:
    entry = LeaderboardEntry(
        entry_id="a",
        label="xgboost_direction (xgboost)",
        context=_CONTEXT,
        metrics={"ic_mean": 0.2},
        n_observations=240,
        model_id="a" * 16,
        dataset_id="ds_evaluation",
    )
    payload = entry.to_record()
    del payload["training_dataset_id"]
    restored = LeaderboardEntry.from_record(payload)
    assert restored.training_dataset_id == ""
    # The evaluation dataset must not stand in for the training one.
    assert restored.dataset_id == "ds_evaluation"


def test_a_stored_null_training_dataset_reads_as_unknown() -> None:
    entry = LeaderboardEntry(
        entry_id="a",
        label="xgboost_direction (xgboost)",
        context=_CONTEXT,
        metrics={"ic_mean": 0.2},
        n_observations=240,
        model_id="a" * 16,
        dataset_id="ds_evaluation",
    )
    payload = entry.to_record()
    payload["training_dataset_id"] = None
    assert LeaderboardEntry.from_record(payload).training_dataset_id == ""
