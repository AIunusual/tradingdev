"""Readable metric values derived only from each run's saved unit definitions."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from tradingdev.domain.performance.artifacts import FoldStats

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tradingdev.domain.performance.artifacts import (
        MetricDefinitionSnapshot,
        MetricScalar,
    )

type DisplayValue = str | None | dict[str, str | int | None]


def display_metrics(
    values: Mapping[str, MetricScalar | FoldStats],
    definitions: Mapping[str, MetricDefinitionSnapshot],
) -> dict[str, DisplayValue]:
    """Format saved fractions as percentages without changing raw metric values."""
    result: dict[str, DisplayValue] = {}
    for name, value in values.items():
        unit = definitions[name].unit
        if isinstance(value, FoldStats):
            result[name] = {
                "mean": _scalar(value.mean, unit),
                "std": _scalar(value.std, unit),
                "min": _scalar(value.min, unit),
                "max": _scalar(value.max, unit),
                "valid_count": value.valid_count,
            }
        else:
            result[name] = _scalar(value, unit)
    return result


def _scalar(value: MetricScalar, unit: str) -> str | None:
    if value is None:
        return None
    if unit == "fraction":
        # Decimal formatting avoids float overflow when scaling finite values
        # above 1e306; z suppresses negative zero after display rounding.
        return format(Decimal(str(value)), "z.2%")
    return str(value) if isinstance(value, int) else format(value, "z.2f")
