"""Threshold rules over one saved dataset column, for candidates without a model.

A rule is a single comparison applied bar by bar to a feature that was built
from past bars only. It has no fitting step, and that is exactly what makes it
worth ranking beside a model: one table then answers whether fitting bought
anything over a hand-set level.

The rule reads a saved dataset column rather than raw prices, so it inherits
the dataset's own warm-up and row order and cannot silently disagree with the
rows it is scored on. Columns holding the label or the realised outcome are
refused: a rule on ``forward_return`` would be scored against the answer
instead of predicting it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence

RuleDirection = Literal["above", "below"]

RULE_DIRECTIONS: tuple[RuleDirection, ...] = ("above", "below")

# Row bookkeeping, the label, and the outcome the signal is scored against.
RESERVED_COLUMNS = frozenset(
    {"timestamp", "label_end", "forward_return", "target", "symbol"}
)


@dataclass(frozen=True)
class ThresholdRule:
    """Long while ``column`` is on one side of ``threshold``, short otherwise.

    Both legs are always taken, so the evaluation describes a rule that is
    always in the market rather than one whose flat periods would be scored as
    zero returns. ``parameters`` holds only the numeric setting, which is the
    axis a sweep varies; the column and side are part of :attr:`name`.
    """

    column: str
    threshold: float
    direction: RuleDirection = "above"

    def __post_init__(self) -> None:
        if not self.column or self.column in RESERVED_COLUMNS:
            raise ValueError(
                f"rule column {self.column!r} cannot be read; a rule may only "
                "use a feature column, never the label, the realised outcome or "
                "the row bookkeeping"
            )
        if isinstance(self.threshold, bool) or not isinstance(
            self.threshold, (int, float)
        ):
            raise ValueError(f"rule threshold must be a number, got {self.threshold!r}")
        if not math.isfinite(float(self.threshold)):
            raise ValueError("rule threshold must be finite")
        if self.direction not in RULE_DIRECTIONS:
            raise ValueError(
                f"rule direction must be one of {list(RULE_DIRECTIONS)}, "
                f"got {self.direction!r}"
            )

    @classmethod
    def create(
        cls, *, column: str, threshold: float, direction: str = "above"
    ) -> ThresholdRule:
        """Build a rule from untyped caller input, rejecting unknown sides."""
        if direction not in RULE_DIRECTIONS:
            raise ValueError(
                f"rule direction must be one of {list(RULE_DIRECTIONS)}, "
                f"got {direction!r}"
            )
        return cls(column=column, threshold=threshold, direction=direction)

    @property
    def name(self) -> str:
        """Readable identity, also used to tell two rules apart in a ranking."""
        operator = ">" if self.direction == "above" else "<"
        return f"{self.column} {operator} {self.threshold:g}"

    @property
    def parameters(self) -> dict[str, float]:
        """The numeric setting a sweep varies, for a parameter surface."""
        return {"threshold": float(self.threshold)}

    def signal(
        self, values: npt.NDArray[np.float64] | Sequence[float]
    ) -> npt.NDArray[np.float64]:
        """Return +1 where the condition holds and -1 where it does not.

        Refuses a column that never crosses the threshold: that rule is a
        constant position, whose simulated book has no positions at all and
        whose scores would describe nothing.
        """
        series = np.asarray(values, dtype=np.float64)
        if series.ndim != 1 or not len(series):
            raise ValueError("rule column must be a non-empty one-dimensional series")
        if not np.all(np.isfinite(series)):
            raise ValueError(
                "rule column contains non-finite values; rebuild the dataset so "
                "the rule is scored on complete rows"
            )
        if self.direction == "above":
            long = series > self.threshold
        else:
            long = series < self.threshold
        if long.all() or not long.any():
            raise ValueError(
                f"threshold {self.threshold:g} never changes {self.column!r} "
                f"({self.direction}); pick a level inside the observed range "
                f"[{float(series.min()):g}, {float(series.max()):g}] so the rule "
                "takes both legs"
            )
        return np.where(long, 1.0, -1.0)
