"""Context measurements preserve scope, encoding, and cleanup boundaries."""

from __future__ import annotations

import importlib
import json
import os
import sys
from typing import TYPE_CHECKING

import pytest
from tooling.context_budget import (
    RequestSizes,
    audit_environment,
    json_size,
    text_size,
    tool_context_report,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_unicode_bytes_are_not_reported_as_characters_or_tokens() -> None:
    assert text_size("指標") == {"characters": 2, "utf8_bytes": 6}
    value = {"指標": "報酬"}
    serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    assert json_size(value) == text_size(serialized)


def test_tool_projection_excludes_output_schema_and_respects_selected_scope() -> None:
    tools = [
        {
            "name": name,
            "description": "Read evidence.",
            "inputSchema": {"type": "object"},
            "outputSchema": {"description": "output-only" * 100},
        }
        for name in ("beta", "alpha")
    ]
    report = tool_context_report("Core instructions", tools, names=["alpha"])
    assert report["tool_count"] == 1
    assert report["tool_names"] == ["alpha"]
    assert report["model_tools"] == json_size(
        [
            {
                "type": "function",
                "function": {
                    "name": "alpha",
                    "description": "Read evidence.",
                    "parameters": {"type": "object"},
                },
            }
        ]
    )
    assert report["mcp_wire_tools"]["characters"] > report["model_tools"]["characters"]
    with pytest.raises(ValueError, match="Unknown tool names: missing"):
        tool_context_report("Core", tools, names=["missing"])


def test_request_peak_counts_exact_utf8_bodies_without_retaining_content() -> None:
    observed = RequestSizes()
    assert "request_growth_bytes=0" in observed.observe(b"{}")
    body = '{"message":"資料"}'.encode()
    assert f"request_utf8_bytes={len(body)}" in observed.observe(body)
    assert "request_growth_bytes=-1" in observed.observe(b"0")
    assert observed.requests == 3
    assert observed.peak_bytes == len(body)


@pytest.mark.parametrize("failure", [None, RuntimeError, TimeoutError])
@pytest.mark.parametrize("original_bytecode_setting", [False, True])
@pytest.mark.parametrize("original_bytecode_env", [None, "0"])
def test_audit_restores_environment_and_removes_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: type[Exception] | None,
    original_bytecode_setting: bool,
    original_bytecode_env: str | None,
) -> None:
    original = str(tmp_path / "user-workspace")
    monkeypatch.setenv("TRADINGDEV_WORKSPACE", original)
    monkeypatch.delenv("TRADINGDEV_DATA_ROOT", raising=False)
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "user-matplotlib"))
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "user-cache"))
    if original_bytecode_env is None:
        monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    else:
        monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", original_bytecode_env)
    monkeypatch.setattr(sys, "dont_write_bytecode", original_bytecode_setting)
    previous_environment = dict(os.environ)
    module_dir = tmp_path / "external-module"
    module_dir.mkdir()
    module_name = "context_audit_bytecode_fixture"
    module_path = module_dir / f"{module_name}.py"
    module_path.write_text("value = 'imported outside audit root'\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(module_dir))
    assert module_name not in sys.modules
    root = None
    try:
        with audit_environment() as directory:
            root = directory
            (directory / "evidence").write_text("temporary")
            assert os.environ["TRADINGDEV_WORKSPACE"] != original
            assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
            assert sys.dont_write_bytecode is True
            assert not module_path.is_relative_to(directory)
            imported = importlib.import_module(module_name)
            assert imported.value == "imported outside audit root"
            assert not (module_dir / "__pycache__").exists()
            assert not list(module_dir.rglob("*.pyc"))
            if failure is not None:
                raise failure("fixture")
    except Exception as exc:
        assert failure is not None and type(exc) is failure
    finally:
        sys.modules.pop(module_name, None)
    assert root is not None and not root.exists()
    assert dict(os.environ) == previous_environment
    assert sys.dont_write_bytecode is original_bytecode_setting
    assert module_path.exists()
    assert not (module_dir / "__pycache__").exists()
    assert not list(module_dir.rglob("*.pyc"))
