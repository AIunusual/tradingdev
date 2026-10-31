"""Generated strategy failures return diagnostics without exiting the caller."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING

import pytest
import yaml
from tooling.process_guard import ProcessCleanupError, run_captured

from tests.integration.execution_fixtures import research_from_config

if TYPE_CHECKING:
    from tests.integration.mcp_harness import MCPWorkspace

pytestmark = pytest.mark.integration

_SERVICE_PROBE = """
import json
import sys
from pathlib import Path
import tradingdev
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.strategy_service import StrategyService
from tradingdev.domain.research import NewResearchSpec
from tradingdev.domain.strategies.templates import strategy_contract_payload

root, phase, fault = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
example = strategy_contract_payload(
    Path(tradingdev.__file__).parent, 'example'
)['content']
code = example['example_strategy_code']
if fault != 'success':
    statement = {
        'system_exit': 'raise SystemExit(7)',
        'exception': 'raise ValueError("intentional runtime failure")',
        'timeout': 'while True:\\n                pass',
    }[fault.removeprefix('import_').removeprefix('constructor_')]
    limit = 80 if phase == 'dry_run' else 0
    if fault.startswith('import_'):
        code += '\\n' + statement.replace('                pass', '    pass') + '\\n'
    elif fault.startswith('constructor_'):
        code = code.replace('        self._engine = backtest_engine',
            f'        if fast_period > 0:\\n            {statement}\\n'
            '        self._engine = backtest_engine')
    else:
        code = code.replace('        result = df.copy()\\n',
            f'        if len(df) > {limit}:\\n            {statement}\\n'
            '        result = df.copy()\\n')
options = {'contract_timeout_seconds': 3} if fault.endswith('timeout') else {}
service = StrategyService(WorkspacePaths(root / 'workspace'), **options)
saved = service.save_draft(code, example['example_yaml_config'], NewResearchSpec(
    strategy_outline='runtime fault isolation', symbol='BTC/USDT', timeframe='1h',
    start_date='2024-01-01', end_date='2024-12-31', init_cash=10000,
    fees=0.0006, slippage=0.0005, annualization=365,
))
assert saved.success, saved.error
if phase == 'dry_run':
    validated = service.validate(saved.strategy_id, saved.revision_id)
    assert validated['success'], validated
result = getattr(service, phase)(saved.strategy_id, saved.revision_id)
stored = service.get_strategy(saved.strategy_id, saved.revision_id)
print(json.dumps({'result': result, 'stored': stored}))
"""


@pytest.mark.parametrize(
    ("phase", "fault"),
    [
        (phase, fault)
        for phase in ("validate", "dry_run")
        for fault in ("success", "exception", "system_exit", "timeout")
    ]
    + [
        ("validate", fault)
        for fault in (
            "import_system_exit",
            "constructor_system_exit",
            "import_timeout",
            "constructor_timeout",
        )
    ],
)
def test_service_survives_runtime_failure(phase: str, fault: str) -> None:
    root = Path(mkdtemp(prefix="tradingdev-strategy-check-test-"))
    retained = False
    try:
        source = Path(__file__).resolve().parents[2] / "src"
        environment = {
            **os.environ,
            "PYTHONPATH": str(source),
            "MYPY_CACHE_DIR": str(root / "mypy"),
            "RUFF_CACHE_DIR": str(root / "ruff"),
            "NUMBA_CACHE_DIR": str(root / "numba"),
            "MPLCONFIGDIR": str(root / "matplotlib"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": str(root),
            "TMP": str(root),
            "TEMP": str(root),
        }
        child = run_captured(
            [sys.executable, "-c", _SERVICE_PROBE, str(root), phase, fault],
            cwd=root,
            env=environment,
            timeout=90,
        )
        assert child.returncode == 0, child.stderr
        reply = json.loads(child.stdout)
        result = reply["result"]
        assert result["success"] is (fault == "success")
        expected_status = (
            ("runnable" if phase == "dry_run" else "validated")
            if fault == "success"
            else ("validated" if phase == "dry_run" else "draft")
        )
        assert result["status"] == expected_status
        evidence_key = "dry_run" if phase == "dry_run" else "validation"
        evidence = reply["stored"]["metadata"][evidence_key]
        assert evidence["success"] is result["success"]
        assert evidence["revision_id"] == result["revision_id"]
        if fault != "success":
            expected_code = {
                "exception": "contract_execution_error",
                "system_exit": "strategy_check_failed",
                "timeout": "strategy_check_timeout",
            }[fault.removeprefix("import_").removeprefix("constructor_")]
            assert any(item["code"] == expected_code for item in result["diagnostics"])
        assert not list(root.glob("tradingdev-strategy-check-*/request.json"))
    except ProcessCleanupError as error:
        retained = True
        raise ProcessCleanupError(f"{error}; retained {root}") from error
    finally:
        if not retained:
            shutil.rmtree(root)


@pytest.mark.anyio
@pytest.mark.parametrize("phase", ["validate_strategy", "dry_run_strategy"])
async def test_real_mcp_survives_exit_and_accepts_repaired_revision(
    mcp_workspace: MCPWorkspace, phase: str
) -> None:
    async with mcp_workspace.connect() as client:
        example = (await client.call("get_strategy_contract", section="example"))[
            "content"
        ]
        good = example["example_strategy_code"]
        limit = 80 if phase == "dry_run_strategy" else 0
        broken = good.replace(
            "        result = df.copy()\n",
            f"        if len(df) > {limit}:\n            raise SystemExit(7)\n"
            "        result = df.copy()\n",
        )
        config = example["example_yaml_config"]
        saved = await client.call(
            "save_strategy",
            code=broken,
            yaml_config=config,
            research=research_from_config(yaml.safe_load(config)),
        )
        assert saved["success"], saved
        selected = {name: saved[name] for name in ("strategy_id", "revision_id")}
        if phase == "dry_run_strategy":
            assert (await client.call("validate_strategy", **selected))["success"]
        rejected = await client.call(phase, **selected)
        assert rejected["success"] is False
        assert any(
            item["code"] == "strategy_check_failed" for item in rejected["diagnostics"]
        )
        read = await client.call("get_strategy", **selected)
        assert read["success"] and read["revision_id"] == selected["revision_id"]
        repaired = await client.call(
            "save_strategy",
            strategy_id=selected["strategy_id"],
            expected_revision_id=selected["revision_id"],
            code=good,
            yaml_config=config,
            research={"kind": "inherit_selected_revision"},
        )
        assert (
            repaired["success"] and repaired["revision_id"] != selected["revision_id"]
        )
        selected["revision_id"] = repaired["revision_id"]
        assert (await client.call("validate_strategy", **selected))["success"]
        assert (await client.call("dry_run_strategy", **selected))[
            "status"
        ] == "runnable"
