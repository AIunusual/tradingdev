"""Tests for the fixed offline Sharpe inference calibration matrix."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
import tooling.sharpe_calibration as calibration
from tooling.sharpe_calibration import (
    CONFIGS,
    FULL_CONFIG,
    PROCESSES,
    SMOKE_CONFIG,
    _interpret,
    _markov_states,
    _summarize_proportion,
    _summarize_width,
    build_report,
    generate_returns,
    main,
    population_sharpe,
    run_dsr,
    run_matrix,
)

from tradingdev.domain.statistics.sharpe import (
    IntervalUnavailable,
    SharpeInterval,
    probabilistic_sharpe_ratio,
    sharpe_interval_outcome,
)


def _spec(name: str) -> Any:
    return next(spec for spec in PROCESSES if spec.name == name)


class TestPopulationTargets:
    def test_ar1_target_uses_stationary_variance(self) -> None:
        spec = _spec("ar1_positive")
        expected = 0.0005 / (0.01 / math.sqrt(1.0 - 0.5**2))

        assert population_sharpe(spec) == pytest.approx(expected)

    def test_regime_target_includes_between_regime_variance(self) -> None:
        spec = _spec("regime_switch")
        mean = (-0.001 + 0.002) / 2.0
        variance = (0.01**2 + 0.02**2) / 2.0 + (0.003**2) / 4.0

        assert population_sharpe(spec) == pytest.approx(mean / math.sqrt(variance))

    def test_generated_ar1_sample_matches_the_analytic_moments(self) -> None:
        spec = _spec("ar1_positive")
        values = generate_returns(spec, 200_000, seed=7)

        assert float(np.mean(values)) == pytest.approx(0.0005, abs=5e-5)
        assert float(np.std(values, ddof=1)) == pytest.approx(
            0.01 / math.sqrt(0.75), rel=0.02
        )


class TestFixedMatrix:
    def test_process_family_order_is_frozen(self) -> None:
        assert tuple(spec.name for spec in PROCESSES) == (
            "iid_normal",
            "ar1_positive",
            "ar1_negative",
            "volatility_persistence",
            "fat_tail_t5",
            "regime_switch",
        )

    def test_full_configuration_is_frozen(self) -> None:
        assert FULL_CONFIG.seeds == tuple(range(2000, 2100))
        assert FULL_CONFIG.bootstrap_reps == 200
        assert (FULL_CONFIG.short_n, FULL_CONFIG.long_n) == (60, 500)
        assert FULL_CONFIG.block_sizes == (None, 5, 20, 80)
        assert FULL_CONFIG.max_seconds == 900.0

    def test_interpretation_uses_nominal_monte_carlo_error(self) -> None:
        assert _interpret(0.94, 0.95, 1000) == "within_simulation_error"
        assert _interpret(0.90, 0.95, 1000) == "below_nominal"
        assert _interpret(0.99, 0.95, 1000) == "above_nominal"

    def test_boundary_rates_with_few_repetitions_stay_within_error(self) -> None:
        assert _interpret(0.0, 0.05, 10) == "within_simulation_error"
        assert _interpret(1.0, 0.95, 10) == "within_simulation_error"
        assert _interpret(0.0, 0.05, 1000) == "below_nominal"
        with pytest.raises(ValueError, match="repetitions"):
            _interpret(0.5, 0.5, 0)


class TestMatrixReport:
    @staticmethod
    def _tiny_config() -> Any:
        return replace(SMOKE_CONFIG, seeds=(2000, 2001, 2002), bootstrap_reps=20)

    def test_every_cell_reports_rates_errors_and_conventions(self) -> None:
        cells, limits = run_matrix(self._tiny_config())

        assert len(cells) == 12
        assert all(isinstance(item, str) for item in limits)
        for cell in cells:
            psr = cast("dict[str, float | str]", cell["psr_reject"])
            intervals = cast("dict[str, object]", cell["intervals"])
            point = cast("dict[str, float]", cell["point_estimate"])
            assert 0.0 <= float(psr["rate"]) <= 1.0
            assert float(psr["mcse"]) >= 0.0
            assert set(intervals) == {"auto", "20"}
            for summary in intervals.values():
                interval = cast("dict[str, object]", summary)
                coverage = cast("dict[str, float]", interval["coverage"])
                conditional = cast("dict[str, float]", interval["conditional_coverage"])
                assert 0.0 <= coverage["rate"] <= 1.0
                assert 0.0 <= conditional["rate"] <= 1.0
                unavailable = cast("float", interval["unavailable_rate"])
                assert 0.0 <= unavailable <= 1.0
                assert interval["coverage_label"] in {
                    "within_simulation_error",
                    "below_nominal",
                    "above_nominal",
                }
            assert point["mean_annualised_convention"] == pytest.approx(
                point["mean_per_period"] * math.sqrt(365.0), rel=1e-9
            )

    def test_report_states_the_no_change_decision(self) -> None:
        report = build_report(self._tiny_config())

        decision = cast("dict[str, object]", report["decision"])
        assert decision["algorithm_change"] is False
        assert decision["unverified"] != []
        assert isinstance(cast("list[object]", report["matrix"]), list)


class TestKnownRisks:
    def test_positive_autocorrelation_inflates_iid_psr_rejection(self) -> None:
        iid = _spec("iid_normal")
        ar = _spec("ar1_positive")
        iid_flags: list[bool] = []
        ar_flags: list[bool] = []
        for index in range(60):
            for spec, flags in ((iid, iid_flags), (ar, ar_flags)):
                values = generate_returns(spec, 500, seed=3000 + index)
                truth = population_sharpe(spec)
                psr = probabilistic_sharpe_ratio(values, benchmark_sr=truth)
                flags.append(psr is not None and psr > 0.95)

        iid_summary = _summarize(iid_flags)
        ar_summary = _summarize(ar_flags)
        # phi = 0.5 cuts the effective sample size by about (1-phi)/(1+phi)
        # = 1/3, so an iid PSR should reject far above 5%, near 17%.
        assert iid_summary == pytest.approx(0.05, abs=0.10)
        assert ar_summary > 0.10
        assert ar_summary > iid_summary + 0.08

    def test_extreme_block_size_has_an_explicit_outcome(self) -> None:
        values = np.random.default_rng(3).normal(0.001, 0.01, 25)

        outcome = sharpe_interval_outcome(
            values, periods_per_year=1.0, reps=10, seed=1, block_size=1000
        )

        assert isinstance(outcome, (SharpeInterval, IntervalUnavailable))
        if isinstance(outcome, SharpeInterval):
            assert outcome.block_size == 1000
        else:
            assert outcome.reason in {
                "insufficient_observations",
                "undefined_statistic",
                "no_valid_resample",
            }


class TestDsrBaseline:
    def test_independent_and_correlated_trials_are_reported_separately(self) -> None:
        config = replace(SMOKE_CONFIG, seeds=tuple(range(2000, 2010)))
        report = run_dsr(config)

        independent = cast("dict[str, object]", report["independent_trials"])
        correlated = cast("dict[str, object]", report["correlated_trials"])
        assert independent["trials"] == 10
        assert independent["cross_trial_correlation"] == 0.0
        assert correlated["cross_trial_correlation"] == pytest.approx(0.7)
        assert independent["within_series_time_correlation"] == 0.0
        assert correlated["within_series_time_correlation"] == 0.0
        independent_hurdle = cast("float", independent["mean_hurdle"])
        correlated_hurdle = cast("float", correlated["mean_hurdle"])
        assert independent_hurdle > 0.0
        assert correlated_hurdle != pytest.approx(independent_hurdle, abs=1e-12)


class TestDefensiveBranches:
    def test_invalid_process_settings_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="phi"):
            population_sharpe(replace(_spec("ar1_positive"), phi=1.0))
        with pytest.raises(ValueError, match="df"):
            population_sharpe(replace(_spec("fat_tail_t5"), df=2))
        with pytest.raises(ValueError, match="unknown process kind"):
            population_sharpe(replace(_spec("iid_normal"), kind=cast("Any", "other")))

    def test_invalid_generators_and_markov_settings_are_rejected(self) -> None:
        spec = _spec("iid_normal")
        with pytest.raises(ValueError, match="at least 2"):
            generate_returns(spec, 1, seed=0)
        with pytest.raises(ValueError, match="phi"):
            generate_returns(replace(spec, kind="ar1", phi=1.0), 10, seed=0)
        with pytest.raises(ValueError, match="df"):
            generate_returns(replace(spec, kind="fat_tail", df=2), 10, seed=0)
        with pytest.raises(ValueError, match="unknown process kind"):
            generate_returns(replace(spec, kind=cast("Any", "other")), 10, seed=0)
        with pytest.raises(ValueError, match="Markov"):
            _markov_states(0, 0.1, seed=np.random.default_rng(0))

    def test_empty_and_bounded_summaries_are_explicit(self) -> None:
        assert _summarize_proportion([]) == {
            "rate": 0.0,
            "mcse": 0.0,
            "low": 0.0,
            "high": 0.0,
        }
        assert _summarize_width([]) == {"mean": 0.0, "mcse": 0.0}
        assert _summarize_width([1.5])["mcse"] == 0.0
        assert _summarize_proportion([False, False])["low"] == 0.0
        assert _summarize_proportion([True, True])["high"] == 1.0

    def test_timeout_is_reported(self) -> None:
        config = replace(
            SMOKE_CONFIG, seeds=(2000,), bootstrap_reps=10, max_seconds=-1.0
        )

        with pytest.raises(TimeoutError, match="exceeded"):
            run_matrix(config)
        with pytest.raises(TimeoutError, match="exceeded"):
            run_dsr(config)

    def test_unavailable_intervals_are_counted(self, monkeypatch: Any) -> None:
        def unavailable(*_args: object, **_kwargs: object) -> IntervalUnavailable:
            return IntervalUnavailable("no_valid_resample", 0.1, 60, 20, 5)

        monkeypatch.setattr(calibration, "sharpe_interval_outcome", unavailable)
        config = replace(SMOKE_CONFIG, seeds=(2000,), bootstrap_reps=10)
        cells, _ = run_matrix(config)

        intervals = cast("dict[str, object]", cells[0]["intervals"])
        summary = cast("dict[str, object]", intervals["auto"])
        assert cast("float", summary["unavailable_rate"]) == 1.0

    def test_unexpected_interval_type_raises(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            calibration, "sharpe_interval_outcome", lambda *_a, **_k: object()
        )
        config = replace(SMOKE_CONFIG, seeds=(2000,), bootstrap_reps=10)

        with pytest.raises(TypeError, match="unexpected"):
            run_matrix(config)

    def test_undefined_point_estimate_raises(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(calibration, "sharpe_ratio", lambda *_a, **_k: None)
        config = replace(SMOKE_CONFIG, seeds=(2000,), bootstrap_reps=10)

        with pytest.raises(RuntimeError, match="undefined Sharpe"):
            run_matrix(config)

    def test_dsr_undefined_trial_and_result_raise(self, monkeypatch: Any) -> None:
        config = replace(SMOKE_CONFIG, seeds=(2000,))
        monkeypatch.setattr(calibration, "sharpe_ratio", lambda *_a, **_k: None)
        with pytest.raises(RuntimeError, match="undefined Sharpe"):
            run_dsr(config)
        monkeypatch.undo()
        monkeypatch.setattr(
            calibration, "deflated_sharpe_ratio", lambda *_a, **_k: None
        )
        with pytest.raises(RuntimeError, match="DSR is undefined"):
            run_dsr(config)


class TestCommandLine:
    def test_repository_root_is_independent_of_cwd(self, monkeypatch: Any) -> None:
        root = calibration._repository_root()
        assert (root / "tooling" / "sharpe_calibration.py").exists()

    def test_output_is_deterministic_and_outside_the_checkout(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        tiny = replace(SMOKE_CONFIG, seeds=(2000, 2001), bootstrap_reps=10)
        monkeypatch.setitem(CONFIGS, "smoke", tiny)
        checkout = tmp_path / "checkout"
        nested = checkout / "nested"
        nested.mkdir(parents=True)
        monkeypatch.setattr(calibration, "_repository_root", lambda: checkout)
        monkeypatch.chdir(nested)
        first = tmp_path / "first.json"
        second = tmp_path / "second.json"

        assert main(["--profile", "smoke", "--output", str(first)]) == 0
        assert main(["--profile", "smoke", "--output", str(second)]) == 0

        first_report = json.loads(first.read_text(encoding="utf-8"))
        second_report = json.loads(second.read_text(encoding="utf-8"))
        first_report.pop("elapsed_seconds")
        second_report.pop("elapsed_seconds")
        assert first_report == second_report

        with pytest.raises(SystemExit):
            main(["--profile", "smoke", "--output", str(checkout / "inside.json")])
        with pytest.raises(SystemExit):
            main(["--profile", "smoke", "--output", "relative.json"])


def _summarize(flags: list[bool]) -> float:
    return sum(flags) / len(flags)
