"""Fixed offline calibration matrix for Sharpe inference.

The matrix is deliberately small and pre-registered in this module: process
families, parameters, seeds, repetitions, bootstrap resamples, block sizes and
sample lengths do not change in response to results. Population Sharpes come
from the generator formulas, never from a sample estimate.

Usage::

    uv run --no-sync python -m tooling.sharpe_calibration \
        --profile full --output /tmp/tradingdev-sharpe-calibration.json
    uv run --no-sync python -m tooling.sharpe_calibration \
        --profile smoke --output /tmp/tradingdev-sharpe-smoke.json

The command is offline. It creates missing output parent directories and writes
only the requested JSON file; it creates no temporary workspace files.
``--output`` must be an absolute path outside the repository checkout.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import numpy.typing as npt

from tradingdev.domain.statistics.sharpe import (
    IntervalUnavailable,
    SharpeInterval,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    sharpe_interval_outcome,
    sharpe_ratio,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

ProcessKind = Literal["iid", "ar1", "volatility", "fat_tail", "regime"]


@dataclass(frozen=True)
class ProcessSpec:
    """One generator with an analytic per-period population Sharpe."""

    name: str
    kind: ProcessKind
    mean: float
    scale: float
    phi: float = 0.0
    switch_p: float = 0.0
    df: int = 0
    regime_means: tuple[float, float] = (0.0, 0.0)
    regime_scales: tuple[float, float] = (1.0, 1.0)


@dataclass(frozen=True)
class Config:
    """A fixed, reproducible experiment configuration."""

    name: str
    seeds: tuple[int, ...]
    bootstrap_reps: int
    short_n: int
    long_n: int
    block_sizes: tuple[int | None, ...]
    max_seconds: float
    confidence: float = 0.95
    periods_per_year: float = 1.0


@dataclass(frozen=True)
class IntervalRecord:
    """One bootstrap outcome for one generated sample."""

    ok: bool
    covered: bool
    width: float
    block_size: int | None


PROCESSES: tuple[ProcessSpec, ...] = (
    ProcessSpec("iid_normal", "iid", 0.0005, 0.01),
    ProcessSpec("ar1_positive", "ar1", 0.0005, 0.01, phi=0.5),
    ProcessSpec("ar1_negative", "ar1", 0.0005, 0.01, phi=-0.5),
    ProcessSpec(
        "volatility_persistence",
        "volatility",
        0.0005,
        1.0,
        switch_p=0.01,
        regime_scales=(0.001, 0.02),
    ),
    ProcessSpec("fat_tail_t5", "fat_tail", 0.0005, 0.006, df=5),
    ProcessSpec(
        "regime_switch",
        "regime",
        0.0,
        1.0,
        switch_p=0.005,
        regime_means=(-0.001, 0.002),
        regime_scales=(0.01, 0.02),
    ),
)

FULL_CONFIG = Config(
    name="full",
    seeds=tuple(range(2000, 2100)),
    bootstrap_reps=200,
    short_n=60,
    long_n=500,
    block_sizes=(None, 5, 20, 80),
    max_seconds=900.0,
)
SMOKE_CONFIG = Config(
    name="smoke",
    seeds=tuple(range(2000, 2010)),
    bootstrap_reps=50,
    short_n=60,
    long_n=500,
    block_sizes=(None, 20),
    max_seconds=120.0,
)
CONFIGS: dict[str, Config] = {"full": FULL_CONFIG, "smoke": SMOKE_CONFIG}
_DSR_TRIALS = 10
_BURN_IN = 1000
_CROSS_TRIAL_CORRELATION = 0.7


def population_sharpe(spec: ProcessSpec) -> float:
    """Per-period population Sharpe implied by ``spec``, independent of samples."""
    if spec.kind == "iid":
        sd = spec.scale
    elif spec.kind == "ar1":
        if not -1.0 < spec.phi < 1.0:
            raise ValueError("phi must be inside (-1, 1)")
        sd = spec.scale / math.sqrt(1.0 - spec.phi**2)
    elif spec.kind == "volatility":
        low, high = spec.regime_scales
        sd = math.sqrt((low**2 + high**2) / 2.0)
    elif spec.kind == "fat_tail":
        if spec.df <= 2:
            raise ValueError("fat_tail df must exceed 2")
        sd = spec.scale * math.sqrt(spec.df / (spec.df - 2))
    elif spec.kind == "regime":
        low_mean, high_mean = spec.regime_means
        low_sd, high_sd = spec.regime_scales
        mean = (low_mean + high_mean) / 2.0
        variance = (low_sd**2 + high_sd**2) / 2.0 + ((low_mean - high_mean) ** 2) / 4.0
        return mean / math.sqrt(variance)
    else:
        raise ValueError(f"unknown process kind: {spec.kind}")
    return spec.mean / sd


def generate_returns(
    spec: ProcessSpec, observations: int, seed: int
) -> npt.NDArray[np.float64]:
    """Generate one deterministic series from ``spec`` with no external data."""
    if observations < 2:
        raise ValueError("observations must be at least 2")
    rng = np.random.default_rng(seed)

    if spec.kind == "iid":
        return spec.mean + spec.scale * rng.normal(size=observations)

    if spec.kind == "ar1":
        if not -1.0 < spec.phi < 1.0:
            raise ValueError("phi must be inside (-1, 1)")
        shocks = spec.scale * rng.normal(size=observations + _BURN_IN)
        values = np.empty(observations + _BURN_IN, dtype=np.float64)
        values[0] = shocks[0] / math.sqrt(1.0 - spec.phi**2)
        for position in range(1, values.size):
            values[position] = spec.phi * values[position - 1] + shocks[position]
        return spec.mean + values[_BURN_IN:]

    if spec.kind in {"volatility", "regime"}:
        states = _markov_states(observations + _BURN_IN, spec.switch_p, seed=rng)
        innovations = rng.normal(size=states.size)
        if spec.kind == "volatility":
            low, high = spec.regime_scales
            scales = np.where(states == 0, low, high)
            return (spec.mean + scales * innovations)[_BURN_IN:]
        low_mean, high_mean = spec.regime_means
        low_sd, high_sd = spec.regime_scales
        means = np.where(states == 0, low_mean, high_mean)
        scales = np.where(states == 0, low_sd, high_sd)
        return (means + scales * innovations)[_BURN_IN:]

    if spec.kind == "fat_tail":
        if spec.df <= 2:
            raise ValueError("fat_tail df must exceed 2")
        return spec.mean + spec.scale * rng.standard_t(spec.df, size=observations)

    raise ValueError(f"unknown process kind: {spec.kind}")


def _markov_states(
    length: int, switch_p: float, *, seed: np.random.Generator
) -> npt.NDArray[np.int64]:
    """Symmetric two-state chain with a stationary probability of one half."""
    if length < 1 or not 0.0 < switch_p < 1.0:
        raise ValueError("invalid Markov chain settings")
    states = np.empty(length, dtype=np.int64)
    states[0] = int(seed.integers(2))
    switches = seed.random(length - 1) < switch_p
    for position in range(1, length):
        previous = int(states[position - 1])
        states[position] = 1 - previous if switches[position - 1] else previous
    return states


def _summarize_proportion(flags: Sequence[bool]) -> dict[str, float]:
    count = len(flags)
    if count == 0:
        raise ValueError("a proportion requires at least one observation")
    rate = sum(flags) / count
    mcse = math.sqrt(rate * (1.0 - rate) / count)
    return {
        "rate": rate,
        "mcse": mcse,
        "low": max(0.0, rate - 1.96 * mcse),
        "high": min(1.0, rate + 1.96 * mcse),
    }


def _summarize_width(widths: Sequence[float]) -> dict[str, float | None] | None:
    count = len(widths)
    if count == 0:
        return None
    mean = float(np.mean(widths))
    mcse = float(np.std(widths, ddof=1)) / math.sqrt(count) if count > 1 else None
    return {"mean": mean, "mcse": mcse}


def _classification_mcse(nominal: float, repetitions: int) -> float:
    """Monte Carlo standard error evaluated at the nominal rate."""
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    return math.sqrt(nominal * (1.0 - nominal) / repetitions)


def _classification_fields(nominal: float, repetitions: int) -> dict[str, float]:
    """Explicit tolerance fields so a reader can reproduce every label."""
    mcse = _classification_mcse(nominal, repetitions)
    return {
        "nominal": nominal,
        "repetitions": float(repetitions),
        "mcse": mcse,
        "tolerance": 2.0 * mcse,
    }


def _interpret(rate: float, nominal: float, repetitions: int) -> str:
    """Classify a deviation after allowing two Monte Carlo standard errors.

    The error is evaluated at the nominal rate, not the observed rate: the
    plug-in estimate collapses to zero at rates of 0 or 1 and would label
    ordinary boundary samples as real deviations.
    """
    tolerance = _classification_fields(nominal, repetitions)["tolerance"]
    if abs(rate - nominal) <= tolerance:
        return "within_simulation_error"
    return "below_nominal" if rate < nominal else "above_nominal"


def _block_label(block_size: int | None) -> str:
    return "auto" if block_size is None else str(block_size)


def _repository_root() -> Path:
    """Repository checkout containing this tool, independent of the caller's cwd."""
    return Path(__file__).resolve().parents[1]


def _check_deadline(deadline: float) -> float:
    """Return the current monotonic time if the calibration is within budget."""
    now = time.monotonic()
    if now > deadline:
        raise TimeoutError("calibration exceeded the time limit")
    return now


def run_matrix(
    config: Config, *, deadline: float | None = None
) -> tuple[list[dict[str, object]], list[str]]:
    """Run every process/sample/block cell and return cells plus measured limits."""
    if deadline is None:
        deadline = time.monotonic() + config.max_seconds
    cells: list[dict[str, object]] = []
    limits: list[str] = []
    annualisation = math.sqrt(config.periods_per_year)

    for process_index, spec in enumerate(PROCESSES):
        truth = population_sharpe(spec)
        interval_truth = truth * annualisation
        for sample_name, observations in (
            ("short", config.short_n),
            ("long", config.long_n),
        ):
            psr_flags: list[bool] = []
            records: dict[int | None, list[IntervalRecord]] = {
                block: [] for block in config.block_sizes
            }
            per_period: list[float] = []
            annualised: list[float] = []

            for seed in config.seeds:
                _check_deadline(deadline)
                values = generate_returns(spec, observations, seed)
                psr = probabilistic_sharpe_ratio(values, benchmark_sr=truth)
                psr_flags.append(psr is not None and psr > config.confidence)

                bootstrap_stream = np.random.SeedSequence(
                    seed, spawn_key=(process_index, observations)
                )
                bootstrap_seed = int(
                    bootstrap_stream.generate_state(1, dtype=np.uint64)[0]
                )
                for block_size in config.block_sizes:
                    outcome = sharpe_interval_outcome(
                        values,
                        periods_per_year=config.periods_per_year,
                        confidence=config.confidence,
                        reps=config.bootstrap_reps,
                        seed=bootstrap_seed,
                        block_size=block_size,
                    )
                    if isinstance(outcome, IntervalUnavailable):
                        records[block_size].append(
                            IntervalRecord(False, False, 0.0, None)
                        )
                    elif isinstance(outcome, SharpeInterval):
                        records[block_size].append(
                            IntervalRecord(
                                True,
                                outcome.lower <= interval_truth <= outcome.upper,
                                outcome.upper - outcome.lower,
                                outcome.block_size,
                            )
                        )
                    else:
                        raise TypeError("unexpected Sharpe interval outcome")

                point = sharpe_ratio(values, periods_per_year=1.0)
                annual = sharpe_ratio(values, periods_per_year=365.0)
                if point is None or annual is None:
                    raise RuntimeError("a generated sample has an undefined Sharpe")
                per_period.append(point)
                annualised.append(annual)

            psr_nominal = 1.0 - config.confidence
            psr_summary = _summarize_proportion(psr_flags)
            psr_label = _interpret(psr_summary["rate"], psr_nominal, len(psr_flags))
            if psr_label != "within_simulation_error":
                limits.append(f"{spec.name}/{sample_name}: PSR {psr_label}")

            interval_summaries: dict[str, object] = {}
            for block_size in config.block_sizes:
                block_records = records[block_size]
                unavailable = 1.0 - (
                    sum(record.ok for record in block_records) / len(block_records)
                )
                valid = [record for record in block_records if record.ok]
                # Unavailable intervals count as non-covering so the label is
                # not conditional on a selected subset of resamples.
                unconditional = _summarize_proportion(
                    [record.ok and record.covered for record in block_records]
                )
                conditional = (
                    _summarize_proportion([record.covered for record in valid])
                    if valid
                    else None
                )
                coverage_label = _interpret(
                    unconditional["rate"], config.confidence, len(block_records)
                )
                if coverage_label != "within_simulation_error":
                    limits.append(
                        f"{spec.name}/{sample_name}/block={_block_label(block_size)}: "
                        f"coverage {coverage_label}"
                    )
                chosen = [
                    record.block_size
                    for record in valid
                    if record.block_size is not None
                ]
                interval_summaries[_block_label(block_size)] = {
                    "coverage": unconditional,
                    "conditional_coverage": conditional,
                    "classification": _classification_fields(
                        config.confidence, len(block_records)
                    ),
                    "coverage_label": coverage_label,
                    "width": _summarize_width([record.width for record in valid]),
                    "unavailable_rate": unavailable,
                    "median_block_size": (float(np.median(chosen)) if chosen else None),
                    "repetitions": len(block_records),
                    "valid_repetitions": len(valid),
                }

            cells.append(
                {
                    "process": spec.name,
                    "sample": sample_name,
                    "observations": observations,
                    "population_sharpe_per_period": truth,
                    "interval_comparison_target": interval_truth,
                    "psr_reject": {
                        **psr_summary,
                        "classification": _classification_fields(
                            psr_nominal, len(psr_flags)
                        ),
                        "label": psr_label,
                    },
                    "intervals": interval_summaries,
                    "point_estimate": {
                        "mean_per_period": float(np.mean(per_period)),
                        "mean_annualised_convention": float(np.mean(annualised)),
                        "annualisation_factor": math.sqrt(365.0),
                    },
                }
            )

    _check_deadline(deadline)
    return cells, limits


def _correlated_trials(seed: int, observations: int) -> list[npt.NDArray[np.float64]]:
    """Return trials whose pairwise population correlation is exactly 0.7."""
    rng = np.random.default_rng(seed)
    common = rng.normal(size=observations)
    shared_sd = math.sqrt(_CROSS_TRIAL_CORRELATION)
    idiosyncratic_sd = math.sqrt(1.0 - _CROSS_TRIAL_CORRELATION)
    return [
        shared_sd * common + idiosyncratic_sd * rng.normal(size=observations)
        for _ in range(_DSR_TRIALS)
    ]


def run_dsr(config: Config, *, deadline: float | None = None) -> dict[str, object]:
    """Fixed independent-trial baseline plus a correlated-trial limitation."""
    if deadline is None:
        deadline = time.monotonic() + config.max_seconds
    results: dict[str, dict[str, list[float]]] = {
        "independent_trials": {"dsr": [], "hurdle": [], "reject": []},
        "correlated_trials": {"dsr": [], "hurdle": [], "reject": []},
    }

    for seed in config.seeds:
        _check_deadline(deadline)
        independent_rng = np.random.default_rng(seed)
        independent = [
            independent_rng.normal(size=config.long_n) for _ in range(_DSR_TRIALS)
        ]
        correlated = _correlated_trials(seed + 1_000_000, config.long_n)

        for label, trials in (
            ("independent_trials", independent),
            ("correlated_trials", correlated),
        ):
            sharpes = [
                sharpe_ratio(trial, periods_per_year=config.periods_per_year)
                for trial in trials
            ]
            if any(value is None for value in sharpes):
                raise RuntimeError("a DSR trial has an undefined Sharpe")
            values = [float(value) for value in sharpes if value is not None]
            best = trials[int(np.argmax(values))]
            result = deflated_sharpe_ratio(
                best,
                values,
                periods_per_year=config.periods_per_year,
                n_trials=_DSR_TRIALS,
            )
            if result is None:
                raise RuntimeError("DSR is undefined on a generated trial set")
            bucket = results[label]
            bucket["dsr"].append(result.dsr)
            bucket["hurdle"].append(result.hurdle_annualised)
            bucket["reject"].append(1.0 if result.dsr > config.confidence else 0.0)

    summaries: dict[str, object] = {}
    for label, bucket in results.items():
        reject = _summarize_proportion([bool(value) for value in bucket["reject"]])
        summaries[label] = {
            "trials": _DSR_TRIALS,
            "cross_trial_correlation": (
                0.0 if label == "independent_trials" else _CROSS_TRIAL_CORRELATION
            ),
            "within_series_time_correlation": 0.0,
            # Each candidate has N(0, 1) marginals; selecting on sample Sharpe
            # does not change the selected candidate's population moments.
            "selected_population_sharpe": 0.0,
            "mean_dsr": float(np.mean(bucket["dsr"])),
            "mean_hurdle": float(np.mean(bucket["hurdle"])),
            # The DSR hurdle is estimated from the same trials, so its null
            # rejection rate is not the PSR's 5% nominal. Report it only as a
            # descriptive baseline, not as a calibrated rejection rate.
            "dsr_reject": {**reject, "label": "descriptive_only"},
        }
    _check_deadline(deadline)
    return summaries


def build_report(config: Config) -> dict[str, object]:
    """Build the complete report for one fixed configuration.

    Coverage counts unavailable intervals as misses. Conditional coverage and
    width are null when no interval is available. A single valid interval has
    a mean width but no estimated standard error for that mean.
    """
    started = time.monotonic()
    deadline = started + config.max_seconds
    cells, limits = run_matrix(config, deadline=deadline)
    dsr = run_dsr(config, deadline=deadline)
    return {
        "schema_version": 1,
        "profile": config.name,
        "conventions": {
            "frequency": "one supplied period; no resampling of the input",
            "risk_free": "zero; callers supply excess returns",
            "inference_annualisation": (
                f"sqrt({config.periods_per_year}) = "
                f"{math.sqrt(config.periods_per_year)}; interval comparison "
                "uses the population target scaled by the same factor"
            ),
            "reporting_annualisation": "sqrt(365) applied only to the point estimate",
            "compounding": "simple per-period returns; no holding-period compounding",
            "bootstrap_seed": (
                "Data generation uses each configuration seed unchanged. "
                "The seed shared by all block sizes for that sample is "
                "int(SeedSequence(seed, spawn_key=(process_index, observations))"
                ".generate_state(1, dtype=uint64)[0]). The process_index is the "
                "zero-based position in configuration.processes."
            ),
            "interpretation_rule": (
                "abs(observed - nominal) <= 2 * Monte Carlo standard error "
                "means within_simulation_error"
            ),
            "mcse_basis": (
                "rate summaries report the observed plug-in mcse; labels use "
                "classification.mcse = sqrt(nominal * (1 - nominal) / repetitions) "
                "and classification.tolerance = 2 * classification.mcse"
            ),
        },
        "configuration": {
            "processes": [spec.name for spec in PROCESSES],
            "sample_sizes": {"short": config.short_n, "long": config.long_n},
            "repetitions": len(config.seeds),
            "seeds": list(config.seeds),
            "bootstrap_reps": config.bootstrap_reps,
            "confidence": config.confidence,
            "block_sizes": [_block_label(size) for size in config.block_sizes],
            "max_seconds": config.max_seconds,
            "dsr_trials": _DSR_TRIALS,
        },
        "matrix": cells,
        "dsr": dsr,
        "decision": {
            "algorithm_change": False,
            "rationale": (
                "No replacement algorithm was pre-registered. The fixed matrix "
                "reports limits and applicability; it is not evidence that a "
                "HAC or other correction is reliable in these cases."
            ),
            "measured_limits": limits,
            "unverified": [
                "non-stationary processes outside the six fixed families",
                "holding-period Sharpe under compounding",
                "effective trial count for correlated candidates",
                "any claim of nominal coverage for PSR or DSR",
            ],
        },
        "elapsed_seconds": _check_deadline(deadline) - started,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the offline calibration matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(CONFIGS), default="full")
    parser.add_argument(
        "--output",
        required=True,
        help="absolute JSON output path outside the repository checkout",
    )
    args = parser.parse_args(argv)

    output = Path(args.output).expanduser()
    if not output.is_absolute():
        parser.error("--output must be an absolute path")
    if output.resolve().is_relative_to(_repository_root()):
        parser.error("--output must be outside the repository checkout")

    report = build_report(CONFIGS[args.profile])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    cells = cast("list[object]", report["matrix"])
    decision = cast("dict[str, object]", report["decision"])
    limits = cast("list[str]", decision["measured_limits"])
    elapsed = cast("float", report["elapsed_seconds"])
    print(
        f"{args.profile}: {len(cells)} cells, "
        f"{len(limits)} measured limits, "
        f"{elapsed:.1f}s -> {output}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
