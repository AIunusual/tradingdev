"""Verify market evidence shared by signal rankings and fixed reports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from tradingdev.domain.ml.leaderboard import EvaluationContext

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tradingdev.adapters.storage.ml_datasets import MLDatasetStore
    from tradingdev.domain.ml.leaderboard import LeaderboardEntry
    from tradingdev.domain.ml.market_evidence import (
        EvidenceCheck,
        MarketComparison,
        RawMarketSeries,
    )

_RAW_FIELDS: dict[
    Literal[
        "source",
        "market_type",
        "timeframe",
        "first_utc",
        "last_utc",
        "rows",
        "fingerprint",
    ],
    str,
] = {
    "source": "provider differs",
    "market_type": "effective market type differs",
    "timeframe": "timeframe differs",
    "first_utc": "first UTC timestamp differs",
    "last_utc": "last UTC timestamp differs",
    "rows": "raw bar count differs",
    "fingerprint": "ordered OHLCV content or timestamps differ",
}


def dataset_context(
    manifest: dict[str, Any], *, window: int, n_quantiles: int
) -> EvaluationContext:
    """Bind saved evaluation settings to one immutable dataset."""
    return EvaluationContext(
        symbol=str(manifest.get("symbol") or ",".join(manifest.get("symbols") or ())),
        timeframe=str(manifest["timeframe"]),
        start_date=str(manifest["start_date"]),
        end_date=str(manifest["end_date"]),
        horizon=int(manifest["horizon"]),
        window=window,
        n_quantiles=n_quantiles,
        feature_set=str(manifest["feature_set"]),
        code_fingerprint=str(manifest["code_fingerprint"]),
        data_fingerprint=str(manifest["content_hash"]),
        raw_market_version=manifest.get("raw_market", {}).get("version"),
    )


def validate_dataset_context(entry: LeaderboardEntry, manifest: dict[str, Any]) -> None:
    expected = dataset_context(
        manifest, window=entry.context.window, n_quantiles=entry.context.n_quantiles
    )
    if entry.context != expected or entry.n_observations != manifest["rows"]:
        raise ValueError(
            "recorded evaluation context does not match dataset_id "
            f"{entry.dataset_id}: {entry.entry_id}"
        )


def _result(differences: list[str], missing: int, shared: str) -> EvidenceCheck:
    reasons = (
        [f"{missing} candidate(s) lack saved raw market evidence"] if missing else []
    )
    if differences:
        return {"status": "mismatch", "reasons": differences + reasons}
    if reasons:
        return {"status": "unknown", "reasons": reasons}
    return {"status": "verified", "reasons": [shared]}


def _raw_differences(
    series_by_dataset: list[dict[str, RawMarketSeries]],
) -> list[str]:
    if not series_by_dataset:
        return []
    first, *others = series_by_dataset
    reasons: dict[str, str] = {}
    for other in others:
        if first.keys() != other.keys():
            differing = sorted(first.keys() ^ other.keys())
            reasons.setdefault("members", f"symbol membership differs ({differing[0]})")
        for symbol in sorted(first.keys() & other.keys()):
            for field, message in _RAW_FIELDS.items():
                if first[symbol][field] != other[symbol][field]:
                    reasons.setdefault(field, f"{symbol}: {message}")
    return list(reasons.values())


def compare_market_evidence(
    entries: Sequence[LeaderboardEntry],
    store: MLDatasetStore,
    *,
    require_verified_baseline: bool,
    manifest: dict[str, Any] | None = None,
) -> MarketComparison | None:
    """Check every candidate before paging, opening each evidenced dataset once."""
    if (
        not require_verified_baseline
        and len({row.context.horizon for row in entries}) < 2
    ):
        return None
    selected: dict[str, dict[str, Any]] = {}
    supplied_id = manifest["dataset_id"] if manifest is not None else None
    missing = 0
    for row in entries:
        if row.context.raw_market_version is None:
            missing += 1
            continue
        if row.dataset_id not in selected:
            selected[row.dataset_id] = (
                manifest
                if manifest is not None and row.dataset_id == supplied_id
                else store.load(row.dataset_id)[1]
            )
        validate_dataset_context(row, selected[row.dataset_id])

    saved_datasets = [selected[key] for key in sorted(selected)]
    raw = _raw_differences(
        [
            {series["symbol"]: series for series in saved["raw_market"]["series"]}
            for saved in saved_datasets
        ]
    )
    rows = [saved["evaluation_rows"] for saved in saved_datasets]
    different_rows = (
        ["ordered evaluation timestamps or raw row origins differ"]
        if rows and any(mapping != rows[0] for mapping in rows[1:])
        else []
    )
    comparison: MarketComparison = {
        "raw_baseline": _result(raw, missing, "saved raw market inputs match"),
        "evaluation_rows": _result(
            different_rows, missing, "saved evaluation row coordinates match"
        ),
    }
    baseline = comparison["raw_baseline"]
    if require_verified_baseline and baseline["status"] != "verified":
        raise ValueError(
            f"raw market baseline is {baseline['status']}: "
            + "; ".join(baseline["reasons"])
        )
    return comparison
