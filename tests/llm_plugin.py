"""Make model execution an explicit pytest choice, never an implicit expense."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_CODEX_MODEL = "gpt-6.1-sol"
DEFAULT_CODEX_REASONING_EFFORT = "high"
CODEX_REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")


def codex_settings(
    model: str | None = None, reasoning_effort: str | None = None
) -> tuple[str, str]:
    """Resolve explicit test settings without inheriting the review/host model."""
    if model is None:
        model = os.environ.get("TRADINGDEV_LLM_MODEL", DEFAULT_CODEX_MODEL)
    if not model.strip():
        raise ValueError("Codex test model must not be empty")
    if reasoning_effort is None:
        reasoning_effort = os.environ.get(
            "TRADINGDEV_LLM_REASONING_EFFORT", DEFAULT_CODEX_REASONING_EFFORT
        )
    if reasoning_effort not in CODEX_REASONING_EFFORTS:
        raise ValueError(
            "Codex --llm-reasoning-effort must be one of "
            + ", ".join(CODEX_REASONING_EFFORTS)
        )
    return model, reasoning_effort


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("TradingDev LLM tests")
    group.addoption("--llm-provider", choices=("codex", "local"), default=None)
    group.addoption("--llm-model", default=None)
    group.addoption("--llm-base-url", default="http://localhost:11434/v1")
    group.addoption("--llm-timeout", type=float, default=600.0)
    group.addoption(
        "--llm-temperature",
        type=float,
        default=None,
        help="Local model sampling override; omitted uses the server's API defaults",
    )
    group.addoption(
        "--llm-reasoning-effort",
        default=None,
        help=(
            "Reasoning effort; Codex uses the explicit test baseline, "
            "Local uses its server default"
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    provider = config.getoption("llm_provider")
    effort = config.getoption("llm_reasoning_effort")
    if effort is not None:
        if provider is None:
            raise pytest.UsageError(
                "--llm-reasoning-effort requires --llm-provider codex or local"
            )
        if not effort.strip():
            raise pytest.UsageError("--llm-reasoning-effort must not be empty")
    temperature = config.getoption("llm_temperature")
    if temperature is not None:
        if provider != "local":
            raise pytest.UsageError("--llm-temperature requires --llm-provider local")
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise pytest.UsageError(
                "--llm-temperature must be finite and between 0 and 2"
            )
    if provider is None:
        return
    timeout = config.getoption("llm_timeout")
    if not math.isfinite(timeout) or timeout <= 0:
        raise pytest.UsageError("--llm-timeout must be finite and positive")
    if provider == "codex":
        try:
            model, effort = codex_settings(config.getoption("llm_model"), effort)
        except ValueError as error:
            raise pytest.UsageError(str(error)) from error
        config.option.llm_model = model
        config.option.llm_reasoning_effort = effort
    if provider == "local":
        if not config.getoption("llm_model"):
            raise pytest.UsageError("Local LLM tests require --llm-model")
        diagnostic = "--llm-base-url must be a local loopback URL"
        try:
            url = urlsplit(config.getoption("llm_base_url"))
            # urllib validates the port's syntax and range only on access.
            _ = url.port
        except ValueError as error:
            raise pytest.UsageError(diagnostic) from error
        if (
            url.scheme not in {"http", "https"}
            or url.hostname not in {"localhost", "127.0.0.1", "::1"}
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise pytest.UsageError(diagnostic)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("llm_provider") is not None:
        return
    deselected: Sequence[pytest.Item] = [
        item for item in items if item.get_closest_marker("live_llm") is not None
    ]
    items[:] = [item for item in items if item not in deselected]
    config.hook.pytest_deselected(items=deselected)


def pytest_report_header(config: pytest.Config) -> str:
    provider = config.getoption("llm_provider")
    if provider == "codex":
        return (
            "LLM tests: codex (explicit opt-in); "
            f"model={config.getoption('llm_model')}, "
            f"reasoning_effort={config.getoption('llm_reasoning_effort')}"
        )
    return (
        f"LLM tests: {provider} (explicit opt-in)"
        if provider
        else "LLM tests excluded; use scripts/check-llm.sh codex|local"
    )
