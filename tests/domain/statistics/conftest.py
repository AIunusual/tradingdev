"""Force explicit bootstrap seeds in statistics tests.

Every resampling entry point under ``domain.statistics`` accepts a ``seed``.
A test that omits it draws from OS entropy, so the same test can pass today
and fail tomorrow. Reject unseeded generators for the duration of each test
instead of relying on repeated runs to notice the mistake.

Import-time randomness in third-party libraries (for example SciPy building
distribution docstrings) happens during collection, outside this fixture, and
is deliberately out of scope.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _require_seeded_generator(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    original = np.random.default_rng
    test_name = request.node.name

    def seeded_only(seed: Any = None, *args: Any, **kwargs: Any) -> Any:
        if seed is None:
            raise AssertionError(
                f"{test_name}: pass an explicit seed to the statistics bootstrap"
            )
        return original(seed, *args, **kwargs)

    monkeypatch.setattr(np.random, "default_rng", seeded_only)
