"""Verify explicit model-test selection in isolated, entirely offline pytest runs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_SUITE = """\
from pathlib import Path

import pytest


def test_offline():
    Path("offline.ran").write_text("done", encoding="utf-8")


@pytest.mark.live_llm
def test_model(pytestconfig):
    provider = pytestconfig.getoption("llm_provider")
    assert provider in {"codex", "local"}
    Path("model.ran").write_text(provider, encoding="utf-8")
    Path("model_name.ran").write_text(
        repr(pytestconfig.getoption("llm_model")), encoding="utf-8")
    temperature = pytestconfig.getoption("llm_temperature")
    Path("temperature.ran").write_text(repr(temperature), encoding="utf-8")
    reasoning_effort = pytestconfig.getoption("llm_reasoning_effort")
    Path("reasoning_effort.ran").write_text(repr(reasoning_effort), encoding="utf-8")
"""


def _run_selection(
    root: Path, *arguments: str, overrides: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    (root / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    live_llm: explicit model test\n",
        encoding="utf-8",
    )
    (root / "test_selection.py").write_text(_SUITE, encoding="utf-8")
    environment = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_ADDOPTS": "",
        "PYTEST_PLUGINS": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for key in ("TRADINGDEV_LLM_MODEL", "TRADINGDEV_LLM_REASONING_EFFORT"):
        environment.pop(key, None)
    environment.update(overrides or {})
    return subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests.llm_plugin",
            "-p",
            "no:cacheprovider",
            "-q",
            *arguments,
            "test_selection.py",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def test_default_run_deselects_live_models_without_skipping(tmp_path: Path) -> None:
    result = _run_selection(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed, 1 deselected" in result.stdout
    assert "skipped" not in result.stdout
    assert (tmp_path / "offline.ran").exists()
    assert not (tmp_path / "model.ran").exists()


@pytest.mark.parametrize(
    ("provider", "options"),
    [("codex", []), ("local", ["--llm-model", "fixture-model"])],
)
def test_explicit_provider_includes_live_tests(
    tmp_path: Path, provider: str, options: list[str]
) -> None:
    result = _run_selection(tmp_path, "--llm-provider", provider, *options)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
    assert (tmp_path / "model.ran").read_text(encoding="utf-8") == provider
    assert (tmp_path / "offline.ran").exists()
    assert (tmp_path / "temperature.ran").read_text(encoding="utf-8") == "None"
    assert (tmp_path / "reasoning_effort.ran").read_text(encoding="utf-8") == (
        "'high'" if provider == "codex" else "None"
    )
    if provider == "codex":
        assert (tmp_path / "model_name.ran").read_text() == "'gpt-6.1-sol'"


@pytest.mark.parametrize(
    ("options", "diagnostic"),
    [
        ([], "Local LLM tests require --llm-model"),
        (
            ["--llm-model", "fixture", "--llm-base-url", "https://example.com/v1"],
            "local loopback",
        ),
        (
            [
                "--llm-model",
                "fixture",
                "--llm-base-url",
                "http://localhost.example.com/v1",
            ],
            "local loopback",
        ),
        (
            [
                "--llm-model",
                "fixture",
                "--llm-base-url",
                "http://user:secret@localhost/v1",
            ],
            "local loopback",
        ),
        (
            [
                "--llm-model",
                "fixture",
                "--llm-base-url",
                "http://localhost/v1?target=remote",
            ],
            "local loopback",
        ),
    ],
    ids=["missing-model", "remote-host", "deceptive-host", "credentials", "query"],
)
def test_invalid_local_configuration_fails_before_any_test_executes(
    tmp_path: Path, options: list[str], diagnostic: str
) -> None:
    result = _run_selection(tmp_path, "--llm-provider", "local", *options)
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert diagnostic in result.stderr
    assert not (tmp_path / "offline.ran").exists()
    assert not (tmp_path / "model.ran").exists()


@pytest.mark.parametrize(
    "base_url",
    [
        "http://[::1/v1",
        "http://[not-ipv6]/v1",
        "http://localhost:not-a-port/v1",
        "http://localhost:65536/v1",
        "http://127.0.0.1:-1/v1",
        "http://[::1]:65536/v1",
    ],
    ids=[
        "unclosed-ipv6",
        "invalid-ipv6",
        "nonnumeric-port",
        "oversized-port",
        "negative-port",
        "ipv6-oversized-port",
    ],
)
def test_malformed_local_url_is_usage_error_before_any_test_executes(
    tmp_path: Path, base_url: str
) -> None:
    result = _run_selection(
        tmp_path,
        "--llm-provider=local",
        "--llm-model=fixture",
        f"--llm-base-url={base_url}",
    )
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, result.stderr
    assert "--llm-base-url must be a local loopback URL" in result.stderr
    assert "INTERNALERROR" not in result.stderr
    assert not (tmp_path / "offline.ran").exists()
    assert not (tmp_path / "model.ran").exists()


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:11434/v1",
        "http://127.0.0.1:8080/v1",
        "http://[::1]:11434/v1",
        "https://localhost/v1",
        "http://[::1]/v1",
    ],
)
def test_valid_local_url_allows_selected_tests(tmp_path: Path, base_url: str) -> None:
    result = _run_selection(
        tmp_path,
        "--llm-provider=local",
        "--llm-model=fixture",
        f"--llm-base-url={base_url}",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
    assert (tmp_path / "offline.ran").exists()
    assert (tmp_path / "model.ran").read_text(encoding="utf-8") == "local"


@pytest.mark.parametrize("timeout", ["nan", "inf", "-inf", "0", "-1"])
def test_nonfinite_or_nonpositive_model_timeout_is_rejected(
    tmp_path: Path, timeout: str
) -> None:
    result = _run_selection(
        tmp_path, "--llm-provider", "codex", f"--llm-timeout={timeout}"
    )
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert "--llm-timeout must be finite and positive" in result.stderr
    assert not (tmp_path / "offline.ran").exists()
    assert not (tmp_path / "model.ran").exists()


@pytest.mark.parametrize("temperature", [0.0, 0.7, 2.0])
def test_explicit_local_temperature_reaches_selected_tests(
    tmp_path: Path, temperature: float
) -> None:
    result = _run_selection(
        tmp_path,
        "--llm-provider=local",
        "--llm-model=fixture",
        f"--llm-temperature={temperature}",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
    assert (tmp_path / "temperature.ran").read_text(encoding="utf-8") == repr(
        temperature
    )


@pytest.mark.parametrize("temperature", ["nan", "inf", "-inf", "-0.1", "2.1"])
def test_invalid_local_temperature_fails_before_any_test_executes(
    tmp_path: Path, temperature: str
) -> None:
    result = _run_selection(
        tmp_path,
        "--llm-provider=local",
        "--llm-model=fixture",
        f"--llm-temperature={temperature}",
    )
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert "--llm-temperature" in result.stderr
    assert not (tmp_path / "offline.ran").exists()
    assert not (tmp_path / "model.ran").exists()


@pytest.mark.parametrize("provider", [None, "codex"])
def test_temperature_requires_explicit_local_provider(
    tmp_path: Path, provider: str | None
) -> None:
    options = [] if provider is None else [f"--llm-provider={provider}"]
    result = _run_selection(tmp_path, *options, "--llm-temperature=0.7")
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert "--llm-temperature" in result.stderr
    assert not (tmp_path / "offline.ran").exists()
    assert not (tmp_path / "model.ran").exists()


def test_explicit_local_reasoning_effort_reaches_selected_tests(
    tmp_path: Path,
) -> None:
    result = _run_selection(
        tmp_path,
        "--llm-provider=local",
        "--llm-model=fixture",
        "--llm-reasoning-effort=low",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
    assert (tmp_path / "reasoning_effort.ran").read_text(encoding="utf-8") == "'low'"


def test_blank_local_reasoning_effort_fails_before_any_test_executes(
    tmp_path: Path,
) -> None:
    result = _run_selection(
        tmp_path,
        "--llm-provider=local",
        "--llm-model=fixture",
        "--llm-reasoning-effort= ",
    )
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert "--llm-reasoning-effort" in result.stderr
    assert not (tmp_path / "offline.ran").exists()
    assert not (tmp_path / "model.ran").exists()


def test_reasoning_effort_requires_explicit_provider(tmp_path: Path) -> None:
    result = _run_selection(tmp_path, "--llm-reasoning-effort=low")
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert "--llm-reasoning-effort" in result.stderr
    assert not (tmp_path / "offline.ran").exists()
    assert not (tmp_path / "model.ran").exists()


@pytest.mark.parametrize(
    ("options", "overrides", "expected_model", "expected_effort"),
    [
        (
            [],
            {
                "TRADINGDEV_CODEX_MODEL": "old-shared-model",
                "TRADINGDEV_DOC_REVIEW_MODEL": "review-model",
            },
            "gpt-6.1-sol",
            "high",
        ),
        (
            [],
            {
                "TRADINGDEV_LLM_MODEL": "configured-model",
                "TRADINGDEV_LLM_REASONING_EFFORT": "high",
            },
            "configured-model",
            "high",
        ),
        (
            ["--llm-model=explicit-model", "--llm-reasoning-effort=medium"],
            {
                "TRADINGDEV_LLM_MODEL": "configured-model",
                "TRADINGDEV_LLM_REASONING_EFFORT": "high",
            },
            "explicit-model",
            "medium",
        ),
    ],
    ids=["isolated-baseline", "test-environment", "explicit-overrides"],
)
def test_codex_settings_reach_selected_tests(
    tmp_path: Path,
    options: list[str],
    overrides: dict[str, str],
    expected_model: str,
    expected_effort: str,
) -> None:
    result = _run_selection(
        tmp_path, "--llm-provider=codex", *options, overrides=overrides
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "model_name.ran").read_text() == repr(expected_model)
    assert (tmp_path / "reasoning_effort.ran").read_text() == repr(expected_effort)


@pytest.mark.parametrize(
    ("options", "overrides"),
    [
        (["--llm-model= "], {}),
        (["--llm-reasoning-effort=ultra"], {}),
        (["--llm-reasoning-effort=typo"], {}),
        ([], {"TRADINGDEV_LLM_MODEL": ""}),
        ([], {"TRADINGDEV_LLM_REASONING_EFFORT": ""}),
    ],
)
def test_invalid_codex_settings_fail_before_tests(
    tmp_path: Path, options: list[str], overrides: dict[str, str]
) -> None:
    result = _run_selection(
        tmp_path, "--llm-provider=codex", *options, overrides=overrides
    )
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert not (tmp_path / "offline.ran").exists()
    assert not (tmp_path / "model.ran").exists()
