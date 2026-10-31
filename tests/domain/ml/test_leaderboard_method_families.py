"""Candidate identity beyond models: declared families, names and caveats."""

from __future__ import annotations

from typing import cast

import pytest

from tradingdev.domain.ml.leaderboard import (
    EvaluationContext,
    LeaderboardEntry,
    build_entry_id,
    family_caveats,
    family_counts,
    rank_entries,
    resolve_method,
)

_CONTEXT = EvaluationContext(
    symbol="BTC/USDT",
    timeframe="1h",
    start_date="2024-01-01",
    end_date="2024-12-31",
    horizon=1,
)
_MODEL = "a" * 16


def _entry(
    entry_id: str,
    *,
    model_id: str = "",
    family: str = "",
    name: str = "",
    **metrics: float | None,
) -> LeaderboardEntry:
    return LeaderboardEntry(
        entry_id=entry_id,
        label=name or f"candidate-{entry_id}",
        context=_CONTEXT,
        metrics=metrics,
        n_observations=200,
        model_id=model_id,
        dataset_id=f"ds_{entry_id}",
        method_family=family,
        method_name=name,
    )


class TestResolveMethod:
    def test_a_model_id_is_a_fitted_model(self) -> None:
        identity = resolve_method(
            model_id=_MODEL,
            method_family=None,
            method_name=None,
            method_parameters=None,
        )
        assert identity.family == "ml_model"
        assert identity.name == ""
        assert identity.parameters == {}

    def test_an_undeclared_signal_is_unnamed(self) -> None:
        identity = resolve_method(
            model_id=None,
            method_family=None,
            method_name=None,
            method_parameters=None,
        )
        assert identity.family == "other"

    def test_a_rule_keeps_its_name_and_numeric_settings(self) -> None:
        identity = resolve_method(
            model_id=None,
            method_family="rule",
            method_name="  rsi_14 < 30  ",
            method_parameters={"threshold": 30},
        )
        assert identity.family == "rule"
        assert identity.name == "rsi_14 < 30"
        assert identity.parameters == {"threshold": 30.0}

    def test_a_rule_without_a_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="method_name"):
            resolve_method(
                model_id=None,
                method_family="rule",
                method_name=None,
                method_parameters=None,
            )

    def test_a_model_cannot_declare_another_family(self) -> None:
        with pytest.raises(ValueError, match="ml_model"):
            resolve_method(
                model_id=_MODEL,
                method_family="rule",
                method_name="rsi_14 < 30",
                method_parameters=None,
            )

    def test_a_model_cannot_declare_its_own_parameters(self) -> None:
        """A model's settings come from its artifact, not from the caller."""
        with pytest.raises(ValueError, match="artifact"):
            resolve_method(
                model_id=_MODEL,
                method_family=None,
                method_name=None,
                method_parameters={"n_estimators": 60},
            )

    def test_a_model_cannot_declare_a_name(self) -> None:
        """A model is named by its artifact, so a second name would be unused."""
        with pytest.raises(ValueError, match="method_name"):
            resolve_method(
                model_id=_MODEL,
                method_family=None,
                method_name="rsi_14 < 30",
                method_parameters=None,
            )

    def test_an_unknown_family_is_refused(self) -> None:
        with pytest.raises(ValueError, match="method_family"):
            resolve_method(
                model_id=None,
                method_family="ensemble",
                method_name=None,
                method_parameters=None,
            )

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
    def test_a_non_finite_setting_is_refused(self, value: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            resolve_method(
                model_id=None,
                method_family=None,
                method_name="blend",
                method_parameters={"weight": value},
            )

    def test_a_non_numeric_setting_is_refused(self) -> None:
        with pytest.raises(ValueError, match="number"):
            resolve_method(
                model_id=None,
                method_family=None,
                method_name="blend",
                method_parameters=cast("dict[str, float]", {"weight": "half"}),
            )

    def test_an_overlong_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="160"):
            resolve_method(
                model_id=None,
                method_family="rule",
                method_name="r" * 161,
                method_parameters=None,
            )


class TestFamilyOnRecords:
    def test_a_stored_family_is_what_the_row_reports(self) -> None:
        assert _entry("a", family="rule", name="rsi_14 < 30").family == "rule"

    def test_a_row_with_a_model_reads_as_a_fitted_model(self) -> None:
        """Rows written before families existed still describe what they are."""
        assert _entry("a", model_id=_MODEL).family == "ml_model"

    def test_a_row_without_a_model_claims_nothing(self) -> None:
        assert _entry("a").family == "other"

    def test_the_round_trip_keeps_the_family_and_name(self) -> None:
        entry = _entry("a", family="rule", name="rsi_14 < 30", icir=0.4)
        restored = LeaderboardEntry.from_record(entry.to_record())
        assert restored == entry
        assert restored.family == "rule"
        assert restored.method_name == "rsi_14 < 30"

    def test_a_record_without_the_new_keys_reads_without_claiming_one(self) -> None:
        payload = _entry("a", model_id=_MODEL, icir=0.5).to_record()
        del payload["method_family"]
        del payload["method_name"]
        restored = LeaderboardEntry.from_record(payload)
        assert restored.method_family == ""
        assert restored.method_name == ""
        assert restored.family == "ml_model"

    def test_an_unknown_stored_family_fails_the_read(self) -> None:
        payload = _entry("a", icir=0.5).to_record()
        payload["method_family"] = "ensemble"
        with pytest.raises(ValueError, match="method family"):
            LeaderboardEntry.from_record(payload)


class TestEntryIdentity:
    def test_an_unnamed_signal_keeps_the_id_it_had_before_families(self) -> None:
        """Adding a family must not rename rows already on a leaderboard."""
        assert (
            build_entry_id(
                model_id="",
                dataset_id="d",
                context=_CONTEXT,
                signal_fingerprint="f",
            )
            == "3d79d39d247a8ba2"
        )

    def test_a_model_row_keeps_the_id_it_had_before_families(self) -> None:
        assert (
            build_entry_id(
                model_id=_MODEL,
                dataset_id="d",
                context=_CONTEXT,
                signal_fingerprint="f",
            )
            == "0ceaa8a60264ad59"
        )

    def test_two_named_rules_on_one_signal_stay_two_candidates(self) -> None:
        first = build_entry_id(
            model_id="",
            dataset_id="d",
            context=_CONTEXT,
            signal_fingerprint="f",
            method_name="rsi_14 < 30",
        )
        second = build_entry_id(
            model_id="",
            dataset_id="d",
            context=_CONTEXT,
            signal_fingerprint="f",
            method_name="rsi_14 < 35",
        )
        assert first != second

    def test_a_named_rule_is_not_the_unnamed_signal_it_produces(self) -> None:
        """Declaring what built a signal is a different candidate, not a relabel."""
        named = build_entry_id(
            model_id="",
            dataset_id="d",
            context=_CONTEXT,
            signal_fingerprint="f",
            method_name="rsi_14 < 30",
        )
        unnamed = build_entry_id(
            model_id="",
            dataset_id="d",
            context=_CONTEXT,
            signal_fingerprint="f",
        )
        assert named != unnamed


class TestFamilyReporting:
    def test_counts_follow_the_declared_family_order(self) -> None:
        entries = [
            _entry("a", family="rule", name="rsi_14 < 30"),
            _entry("b", model_id=_MODEL),
            _entry("c"),
            _entry("d", family="rule", name="rsi_14 < 35"),
        ]
        assert family_counts(entries) == {"ml_model": 1, "rule": 2, "other": 1}

    def test_one_family_needs_no_caveat(self) -> None:
        assert family_caveats({"ml_model": 2}) == ()

    def test_a_mixed_table_says_equal_scores_are_not_equal_evidence(self) -> None:
        caveats = family_caveats({"ml_model": 2, "rule": 1})
        assert len(caveats) == 1
        assert "ml_model=2" in caveats[0]
        assert "rule=1" in caveats[0]
        assert "equal evidence" in caveats[0]

    def test_unnamed_candidates_are_reported_on_their_own(self) -> None:
        caveats = family_caveats({"other": 2})
        assert len(caveats) == 1
        assert "no method family" in caveats[0]

    def test_a_mixed_table_with_unnamed_candidates_says_both(self) -> None:
        assert len(family_caveats({"rule": 1, "other": 1})) == 2

    def test_ranking_carries_the_family_mix(self) -> None:
        result = rank_entries(
            [
                _entry("a", family="rule", name="rsi_14 < 30", icir=0.4),
                _entry("b", model_id=_MODEL, icir=0.2),
            ],
            "icir",
        )
        assert [item.entry.entry_id for item in result.ranked] == ["a", "b"]
        assert any("method families" in caveat for caveat in result.caveats)

    def test_ranking_one_family_adds_no_mix_caveat(self) -> None:
        result = rank_entries(
            [
                _entry("a", model_id=_MODEL, icir=0.4),
                _entry("b", model_id="c" * 16, icir=0.2),
            ],
            "icir",
        )
        assert not any("method families" in caveat for caveat in result.caveats)
