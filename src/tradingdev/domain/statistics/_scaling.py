"""Power-of-two rescaling for values at extreme magnitudes or spreads.

Statistics built from deviations (variances, higher moments, block length
estimates) overflow or underflow long before the values themselves do: a
fourth moment overflows once deviations pass about ``1e77``, and squared
deviations underflow when the spread is tiny even if the values are not.
So the decision looks at two numbers, the largest magnitude and the spread
(``max - min``). Values remain unchanged when the spread is at least
``2**-60`` and the largest magnitude is at most ``2**60``. Empty and all-zero
arrays also remain unchanged. Otherwise a power of two brings the spread
into ``[0.5, 1)``, or the magnitude when there is no finite, positive spread
to use.

Power-of-two scaling is exact for values that remain representable as normal
floats. It reduces overflow and underflow in deviation-based calculations,
but does not guarantee bit-for-bit equivalence or a finite statistic. Extreme
magnitude ranges can make smaller observations subnormal or underflow to zero,
losing information even for a mathematically scale-invariant statistic.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import numpy.typing as npt

_LOWER = 2.0**-60
_UPPER = 2.0**60


def scale_exponent(values: npt.NDArray[np.float64]) -> int:
    """Binary exponent ``e`` used to moderate magnitude or spread.

    Returns 0 when the values need no rescaling: empty, all zero, or with
    spread at least ``2**-60`` and largest magnitude at most ``2**60``.
    ``values`` must be finite. Rescaling can still lose small observations
    across extreme magnitude ranges, as described in the module docstring.
    """
    if values.size == 0:
        return 0
    peak = float(np.abs(values).max())
    if peak == 0.0:
        return 0
    # Python floats give inf for an overflowing spread without a warning.
    spread = float(values.max()) - float(values.min())
    if spread >= _LOWER and peak <= _UPPER:
        return 0
    if 0.0 < spread < math.inf:
        return math.frexp(spread)[1]
    # Constant values, or a spread that overflowed because the values reach
    # opposite ends of the float range: scale the magnitude below one, which
    # leaves the spread below two.
    return math.frexp(peak)[1]


def rescaled(
    values: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], int]:
    """``values * 2**-e`` and ``e``, with ``e`` from :func:`scale_exponent`."""
    exponent = scale_exponent(values)
    if exponent == 0:
        return values, 0
    return np.ldexp(values, -exponent), exponent


def rescaled_columns(columns: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Each column of a 2-D array rescaled on its own, as by :func:`rescaled`.

    For statistics computed per column that are invariant to each column's
    scale, such as a block length estimate.
    """
    exponents = [scale_exponent(columns[:, k]) for k in range(columns.shape[1])]
    if not any(exponents):
        return columns
    # int32: NumPy's ldexp takes a C int or long exponent, and long is 32
    # bits on Windows; binary exponents of doubles fit either way.
    return np.ldexp(columns, -np.array(exponents, dtype=np.int32))


def shift_for_moments(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Remove a shared offset when subtraction is exact (Sterbenz's lemma)."""
    low, high = float(values.min()), float(values.max())
    if high > 0.0 and low >= high * 0.5:
        return np.subtract(values, low)
    if low < 0.0 and high <= low * 0.5:
        return np.subtract(values, high)
    return values


def centered_dot(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    """Dot product of centered vectors without rounding each centered value.

    Call with nonempty, rescaled inputs whose shared offsets have been
    removed by :func:`shift_for_moments`. This preserves small observations
    that direct centering of a wide-range vector could round away.
    """
    products = math.sumprod(a, b)
    mean_a, total_b = math.fsum(a) / a.size, math.fsum(b)
    result = math.fma(-mean_a, total_b, products)
    if abs(result) < abs(products) * 2.0**-20:
        # The subtraction lost at least 20 bits to cancellation. Recompute
        # those moments exactly before rounding the small residual to float.
        exact_a = [Fraction(float(value)) for value in a]
        exact_b = [Fraction(float(value)) for value in b]
        exact_products = sum(x * y for x, y in zip(exact_a, exact_b, strict=True))
        return float(exact_products - sum(exact_a) * sum(exact_b) / a.size)
    return result


def rescaled_difference(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], int]:
    """Return an elementwise difference in units of a power of two."""
    with np.errstate(over="ignore"):
        difference = a - b
    if np.all(np.isfinite(difference)):
        return rescaled(difference)
    # A finite difference can exceed float64 before the subsequent statistic
    # is computed. Halving both sides keeps the subtraction in range.
    scaled, exponent = rescaled(a * 0.5 - b * 0.5)
    return scaled, exponent + 1
