#!/usr/bin/env python3
"""Runnable evidence for the first seven testing commandments on issue #62."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NEW_TESTS = (
    "tests/domain/ml/test_evaluation_series.py",
    "tests/adapters/test_evaluation_series_store.py",
    "tests/app/test_ml_evaluation_series.py",
    "tests/app/test_ml_training_series.py",
    "tests/mcp/test_ml_evaluation_series_tools.py",
)

# A mutant must be killed by the named test before that test is trusted.
MUTANTS = (
    (
        "ic_constant",
        "src/tradingdev/domain/ml/evaluation/series.py",
        "return float(correlation(left_values, right_values))",
        "return 0.0",
        "tests/domain/ml/test_evaluation_series.py::"
        "test_rolling_ic_matches_independently_computed_correlations",
    ),
    (
        "content_hash_disabled",
        "src/tradingdev/domain/ml/evaluation/series.py",
        "if hashlib.sha256(frame_bytes(bundle.frame)).hexdigest() "
        "!= manifest.content_hash:",
        "if False:",
        "tests/domain/ml/test_evaluation_series.py::"
        "test_changed_values_are_rejected_against_the_saved_content_hash",
    ),
    (
        "settings_identity_ignored",
        "src/tradingdev/domain/ml/evaluation/series.py",
        "payload = {key: value for key, value in fields.items() "
        'if key != "artifact_id"}',
        'payload = {\n        key: value\n        for key, value in fields.items()\n'
        '        if key not in {"artifact_id", "settings"}\n    }',
        "tests/domain/ml/test_evaluation_series.py::"
        "test_changed_settings_are_rejected_against_the_manifest_identity",
    ),
    (
        "chronology_check_disabled",
        "src/tradingdev/domain/ml/evaluation/series.py",
        "if not stamps.is_unique or not stamps.is_monotonic_increasing:",
        "if False:",
        "tests/domain/ml/test_evaluation_series.py::"
        "test_reordered_timestamps_are_rejected_even_with_a_matching_hash",
    ),
    (
        "row_offset_ignored",
        "src/tradingdev/adapters/storage/evaluation_series.py",
        "end = min(offset + limit, total)",
        "end = min(limit, total)",
        "tests/adapters/test_evaluation_series_store.py::"
        "test_rows_pages_reassemble_to_the_full_export",
    ),
    (
        "protection_marker_ignored",
        "src/tradingdev/adapters/storage/evaluation_series.py",
        "if (self._protected / path.name).exists():",
        "if False:",
        "tests/adapters/test_evaluation_series_store.py::"
        "test_report_protection_marker_survives_unprotected_cleanup",
    ),
    (
        "staging_cleanup_disabled",
        "src/tradingdev/adapters/storage/evaluation_series.py",
        "            finally:\n"
        "                if staging.exists():\n"
        "                    shutil.rmtree(staging)",
        "            finally:\n"
        "                if False and staging.exists():\n"
        "                    shutil.rmtree(staging)",
        "tests/adapters/test_evaluation_series_store.py::"
        "test_failed_publication_leaves_no_partial_artifact",
    ),
    (
        "caller_source_forged",
        "src/tradingdev/app/ml_service.py",
        "producer_source=ProducerSource.CALLER_ARRAY,",
        "producer_source=ProducerSource.BACKEND_RULE,",
        "tests/app/test_ml_evaluation_series.py::"
        "test_saved_series_matches_the_dataset_rows_independently",
    ),
    (
        "fold_row_coordinates_forged",
        "src/tradingdev/app/ml_training.py",
        '"source_row": int(source_row),',
        '"source_row": -1,',
        "tests/app/test_ml_training_series.py::"
        "test_fold_rows_use_real_split_coordinates",
    ),
)

ORACLE_MARKERS = {
    "tests/domain/ml/test_evaluation_series.py": ("np.corrcoef",),
    "tests/app/test_ml_evaluation_series.py": ("pd.read_parquet",),
    "tests/app/test_ml_training_series.py": ("temporal_folds", "pd.read_parquet"),
}


@dataclass(frozen=True)
class Result:
    """One commandment check and its evidence."""

    name: str
    ok: bool
    detail: str


def _pytest_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/tradingdev-commandments-numba")
    return env


def _run_pytest(
    targets: tuple[str, ...] | list[str],
) -> subprocess.CompletedProcess[str]:
    args = [sys.executable, "-m", "pytest", "--rootdir", str(ROOT), *targets, "-q"]
    return subprocess.run(
        args,
        cwd=ROOT,
        env=_pytest_env(),
        capture_output=True,
        text=True,
        check=False,
    )


def _run_mutants() -> tuple[list[Result], float]:
    results: list[Result] = []
    killed = 0
    for name, relative, old, new, test in MUTANTS:
        path = ROOT / relative
        original = path.read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        text = original.decode("utf-8")
        if old not in text:
            results.append(Result(name, False, f"mutant anchor missing in {relative}"))
            continue
        try:
            path.write_text(text.replace(old, new, 1), encoding="utf-8")
            run = _run_pytest([test])
            output = run.stdout + run.stderr
            expected_failure = "failed" in output.lower() or "error" in output.lower()
            killed_by_this = run.returncode != 0 and expected_failure
            if not killed_by_this:
                results.append(Result(name, False, "mutant survived: " + test))
            else:
                killed += 1
                results.append(Result(name, True, "killed by " + test))
                print(f"[mutant] {name}: killed by {test}", flush=True)
        finally:
            path.write_bytes(original)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            results.append(Result(name, False, "source was not restored"))
    score = killed / len(MUTANTS)
    return results, score


def _check_imports() -> Result:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = set(data["project"]["dependencies"])
    declared.update(data["dependency-groups"]["dev"])
    required = ("pandas", "pyarrow", "filelock", "pydantic", "numpy", "pytest")
    missing = [
        name
        for name in required
        if not any(item.startswith(f"{name}>=") or item == name for item in declared)
    ]
    modules = (
        "tradingdev.domain.ml.evaluation.series",
        "tradingdev.adapters.storage.evaluation_series",
        "tradingdev.app.ml_service",
        "tradingdev.app.ml_training",
        "tradingdev.mcp.tools.ml",
    )
    try:
        for module in modules:
            importlib.import_module(module)
    except ImportError as error:
        return Result("imports", False, f"undeclared or unimportable module: {error}")
    if missing:
        return Result(
            "imports", False, "undeclared dependencies: " + ", ".join(missing)
        )
    detail = (
        f"{len(required)} dependencies and {len(modules)} modules verified"
    )
    return Result("imports", True, detail)


def _check_oracles() -> Result:
    missing: list[str] = []
    for path, markers in ORACLE_MARKERS.items():
        text = (ROOT / path).read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                missing.append(f"{path}:{marker}")
    if missing:
        return Result("independent_expected_values", False, ", ".join(missing))
    return Result("independent_expected_values", True, "independent oracles present")


def _rerun(repeat: int) -> Result:
    for index in range(1, repeat + 1):
        run = _run_pytest(list(NEW_TESTS))
        if run.returncode != 0:
            tail = run.stdout[-4000:]
            return Result("flaky_reruns", False, f"run {index} failed:\n{tail}")
        print(f"[rerun] {index}/{repeat}: passed", flush=True)
    return Result("flaky_reruns", True, f"{repeat} consecutive green runs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reruns", type=int, default=5)
    args = parser.parse_args()
    if args.reruns < 2:
        parser.error("--reruns must be at least 2")

    mutant_results, score = _run_mutants()
    checks = [
        Result("mutation_score", score == 1.0, f"{score:.1%} ({len(MUTANTS)} mutants)"),
        *_mutant_results,
        _check_imports(),
        _check_oracles(),
        _rerun(args.reruns),
    ]
    report = {
        "mutation_score": score,
        "mutants_total": len(MUTANTS),
        "reruns": args.reruns,
        "checks": [result.__dict__ for result in checks],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if any(not result.ok for result in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
