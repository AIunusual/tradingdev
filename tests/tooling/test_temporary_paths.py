"""Exercise the repository's tmp_path fixture in independent pytest sessions."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

_REPOSITORY = Path(__file__).resolve().parents[2]


def _run_pytest(root: Path, body: str) -> subprocess.CompletedProcess[str]:
    runtime = root / "runtime"
    runtime.mkdir()
    (root / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (root / "test_example.py").write_text(dedent(body), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests.conftest",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=root,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(_REPOSITORY), str(_REPOSITORY / "src")]),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "TMPDIR": str(runtime),
            "TMP": str(runtime),
            "TEMP": str(runtime),
        },
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize("outcome", ["passed", "assertion", "timeout", "interrupt"])
def test_tmp_path_is_removed_after_success_failure_timeout_and_interrupt(
    tmp_path: Path, outcome: str
) -> None:
    operation = {
        "passed": "pass",
        "assertion": 'raise AssertionError("intentional failure")',
        "timeout": (
            'subprocess.run([sys.executable, "-c", "import time; time.sleep(30)"], '
            "timeout=0.05, check=True)"
        ),
        "interrupt": "raise KeyboardInterrupt()",
    }[outcome]
    result = _run_pytest(
        tmp_path,
        f"""\
        import subprocess
        import sys
        from pathlib import Path
        import matplotlib as mpl
        from matplotlib import font_manager

        # Collection-time imports must already use a disposable plotting cache.
        Path('allocated_plot_cache').write_text(mpl.get_cachedir())
        assert list(Path(mpl.get_cachedir()).glob('fontlist-*.json'))

        def test_01_creates_strategy_and_data(tmp_path):
            Path('allocated_path').write_text(str(tmp_path))
            (tmp_path / 'strategy.py').write_text('temporary strategy')
            (tmp_path / 'market.parquet').write_bytes(b'temporary market data')
            {operation}

        def test_02_previous_artifacts_are_already_gone():
            assert not Path(Path('allocated_path').read_text()).exists()
        """,
    )
    expected_code = 0 if outcome == "passed" else 2 if outcome == "interrupt" else 1
    assert result.returncode == expected_code, result.stdout + result.stderr
    assert not Path((tmp_path / "allocated_path").read_text()).exists()
    assert not Path((tmp_path / "allocated_plot_cache").read_text()).exists()
    assert list((tmp_path / "runtime").iterdir()) == []
    if outcome != "interrupt":
        assert "1 passed" in result.stdout or "2 passed" in result.stdout
    if outcome == "timeout":
        assert "TimeoutExpired" in result.stdout


def test_tmp_path_cleanup_failure_is_a_test_error(tmp_path: Path) -> None:
    result = _run_pytest(
        tmp_path,
        """\
        from pathlib import Path
        from tempfile import TemporaryDirectory

        def test_cannot_remove_directory(tmp_path):
            Path('allocated_path').write_text(str(tmp_path))
            (tmp_path / 'strategy.py').write_text('temporary strategy')
            original = TemporaryDirectory._rmtree

            def refuse_directory(path, *args, **kwargs):
                if path == str(tmp_path):
                    raise PermissionError('intentional cleanup denial: ' + path)
                return original(path, *args, **kwargs)

            TemporaryDirectory._rmtree = staticmethod(refuse_directory)
        """,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "ERROR at teardown" in result.stdout
    assert "intentional cleanup denial" in result.stdout
    # The outer fixture owns the whole child session, including this deliberate
    # leftover. It removes it after we verify that cleanup was not reported as OK.
    leftover = Path((tmp_path / "allocated_path").read_text())
    assert leftover.is_relative_to(tmp_path / "runtime")
    assert (leftover / "strategy.py").exists()


def test_plot_cache_cleanup_failure_is_reported(tmp_path: Path) -> None:
    result = _run_pytest(
        tmp_path,
        """\
        import os
        from pathlib import Path
        from tempfile import TemporaryDirectory

        def test_cannot_remove_plot_cache():
            cache = os.environ['MPLCONFIGDIR']
            Path('allocated_plot_cache').write_text(cache)
            original = TemporaryDirectory._rmtree

            def refuse_cache(path, *args, **kwargs):
                if path == cache:
                    raise PermissionError('intentional plot cache cleanup denial')
                return original(path, *args, **kwargs)

            TemporaryDirectory._rmtree = staticmethod(refuse_cache)
        """,
    )
    assert result.returncode != 0
    assert "intentional plot cache cleanup denial" in result.stdout + result.stderr
    leftover = Path((tmp_path / "allocated_plot_cache").read_text())
    assert leftover.is_relative_to(tmp_path / "runtime") and leftover.is_dir()


@pytest.mark.parametrize(
    "arguments", ["monkeypatch, tmp_path", "tmp_path, monkeypatch"]
)
def test_tmp_path_cleanup_restores_nested_cwd_in_either_fixture_order(
    tmp_path: Path, arguments: str
) -> None:
    result = _run_pytest(
        tmp_path,
        f"""\
        from pathlib import Path

        def test_01_changes_directory({arguments}):
            original = Path.cwd()
            (original / 'allocated_path').write_text(str(tmp_path))
            nested = tmp_path / 'nested'
            nested.mkdir()
            marker = nested / 'strategy.py'
            marker.write_text('read-only test artifact')
            marker.chmod(0o400)
            monkeypatch.chdir(nested)

        def test_02_cwd_and_artifacts_are_restored():
            assert Path('allocated_path').exists()
            assert not Path(Path('allocated_path').read_text()).exists()
        """,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
    assert list((tmp_path / "runtime").iterdir()) == []
