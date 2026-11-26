"""Launch selected live scenarios after their prerequisites have been checked."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from tooling import environment

if TYPE_CHECKING:
    from collections.abc import Generator

    import pytest

_SCENARIOS = (
    "workflows",
    "cleanup",
    "execution_plans",
    "ml",
    "evaluation_series",
    "automatic_walk_forward",
    "discovery",
    "novice_intake",
    "comparison",
    "plateau",
    "research_proposal",
    "strategy_execution_guidance",
    "research_delivery",
    "http_delivery",
)


class _Prerequisites:
    def pytest_collection_finish(self, session: pytest.Session) -> None:
        import pytest

        information_only = any(
            session.config.getoption(option, default=False)
            for option in (
                "collectonly",
                "showfixtures",
                "show_fixtures_per_test",
                "setupplan",
            )
        )
        if information_only or not any(
            item.get_closest_marker("live_llm") for item in session.items
        ):
            return
        from tooling.process_guard import ProcessCleanupError, report_cleanup_failure

        provider = session.config.getoption("llm_provider")
        try:
            image = environment.require_strategy_runtime()
            if provider == "codex":
                environment.require_codex()
            elif provider == "local":
                environment.require_local_service(
                    session.config.getoption("llm_base_url")
                )
            else:
                raise environment.EnvironmentCheckError("Select a live model provider.")
        except ProcessCleanupError as error:
            pytest.exit(
                "Live-test environment cleanup is unverified.",
                returncode=report_cleanup_failure(error),
            )
        except environment.EnvironmentCheckError as error:
            raise pytest.UsageError(
                f"Live-test environment check failed: {error}"
            ) from error
        print(
            f"Live-test prerequisites passed ({provider}; strategy image {image}).",
            flush=True,
        )


def main() -> int:
    arguments = sys.argv[1:]
    if not arguments or arguments[0] not in {"codex", "local"}:
        print(
            "Usage: scripts/check-llm.sh codex|local [pytest options]", file=sys.stderr
        )
        return 2
    try:
        environment.require_packages("pytest")
    except environment.EnvironmentCheckError as error:
        print(f"Environment check failed: {error}", file=sys.stderr)
        return 1
    import pytest

    class SerialExecution:
        @pytest.hookimpl(wrapper=True, tryfirst=True)
        def pytest_cmdline_main(
            self, config: pytest.Config
        ) -> Generator[None, int, int]:
            if any(
                config.getoption(option, default=False)
                for option in ("numprocesses", "tx", "looponfail")
            ):
                raise pytest.UsageError(
                    "scripts/check-llm.sh requires serial execution; remove parallel "
                    "and loop-on-failure options. "
                    "Live-test prerequisites run in this process."
                )
            return (yield)

    return int(
        pytest.main(
            [
                *(f"tests/e2e/test_llm_{name}.py" for name in _SCENARIOS),
                "--llm-provider",
                arguments[0],
                "-m",
                "live_llm",
                "-x",
                "-vv",
                "--capture=tee-sys",
                "--tb=short",
                "--durations=3",
                *arguments[1:],
            ],
            plugins=[_Prerequisites(), SerialExecution()],
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
