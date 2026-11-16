"""An entry says which dataset trained its model, and never guesses one."""

from __future__ import annotations

import json

import pytest

from tradingdev.domain.ml.leaderboard import (
    EvaluationContext,
    LeaderboardEntry,
    ProducerSource,
    build_entry_id,
)

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


@pytest.mark.parametrize("source", list(ProducerSource))
def test_producer_source_survives_json_round_trip(source: ProducerSource) -> None:
    entry = LeaderboardEntry(
        entry_id="a",
        label="model",
        context=_CONTEXT,
        metrics={"ic_mean": 0.2},
        n_observations=240,
        model_id="a" * 16,
        producer_source=source,
    )
    restored = LeaderboardEntry.from_record(json.loads(json.dumps(entry.to_record())))
    assert restored == entry
    assert restored.producer_source is source


def test_legacy_source_stays_unknown_despite_model_and_fingerprint() -> None:
    entry = LeaderboardEntry(
        entry_id="legacy",
        label="model",
        context=_CONTEXT,
        metrics={"ic_mean": 0.2},
        n_observations=240,
        model_id="a" * 16,
        training_dataset_id="ds_training",
        signal_fingerprint="f" * 16,
        method_family="ml_model",
    )
    payload = entry.to_record()
    del payload["producer_source"]
    restored = LeaderboardEntry.from_record(payload)
    assert restored.entry_id == "legacy"
    assert restored.producer_source is ProducerSource.UNKNOWN
    assert LeaderboardEntry.from_record(restored.to_record()) == restored


@pytest.mark.parametrize("source", ["verified", "", None, 1, []])
def test_invalid_stored_producer_source_is_not_treated_as_unknown(
    source: object,
) -> None:
    payload = LeaderboardEntry("a", "model", _CONTEXT, {}, 240).to_record()
    payload["producer_source"] = source
    with pytest.raises(ValueError):
        LeaderboardEntry.from_record(payload)


def test_known_producers_have_distinct_stable_candidate_ids() -> None:
    identities = {
        source: build_entry_id(
            model_id="m",
            dataset_id="ds",
            context=_CONTEXT,
            producer_source=source,
        )
        for source in ProducerSource
    }
    assert len(set(identities.values())) == len(ProducerSource)
    for source, identifier in identities.items():
        assert (
            build_entry_id(
                model_id="m",
                dataset_id="ds",
                context=_CONTEXT,
                producer_source=source,
            )
            == identifier
        )
