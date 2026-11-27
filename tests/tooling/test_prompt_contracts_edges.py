"""Error branches of the prompt-contract auditor."""

from __future__ import annotations

import asyncio
import copy
from typing import Any

import pytest
from tooling.prompt_contracts import (
    ContractSnapshot,
    _section_values,
    check_snapshot,
    collect_snapshot,
    main,
)


def _schema(section: dict[str, Any]) -> dict[str, Any]:
    return {"properties": {"section": section}}


def test_section_values_rejects_nonlocal_and_cyclic_refs() -> None:
    with pytest.raises(ValueError, match="finite local reference"):
        _section_values(_schema({"$ref": "external"}))
    cyclic = {
        "properties": {"section": {"$ref": "#/definitions/a"}},
        "definitions": {"a": {"$ref": "#/definitions/a"}},
    }
    with pytest.raises(ValueError, match="finite local reference"):
        _section_values(cyclic)
    with pytest.raises(ValueError, match="nonempty string enum"):
        _section_values(_schema({"enum": []}))
    assert _section_values(_schema({"enum": ["core"]})) == ["core"]


@pytest.fixture(scope="module")
def registered() -> ContractSnapshot:
    return asyncio.run(collect_snapshot())


def test_check_snapshot_rejects_duplicate_tools(
    registered: ContractSnapshot,
) -> None:
    broken = copy.deepcopy(registered)
    broken.tools = broken.tools + [broken.tools[0]]
    with pytest.raises(ValueError, match="Duplicate registered tool names"):
        check_snapshot(broken)


def test_check_snapshot_rejects_wrong_section_payload(
    registered: ContractSnapshot,
) -> None:
    broken = copy.deepcopy(registered)
    broken.sections["core"]["section"] = "example"
    with pytest.raises(ValueError, match="returned the wrong section"):
        check_snapshot(broken)


def test_check_snapshot_rejects_indicator_drift(
    registered: ContractSnapshot,
) -> None:
    broken = copy.deepcopy(registered)
    broken.indicators["rsi"]["content"]["selected"]["name"] = "macd"
    with pytest.raises(ValueError, match="returned a different indicator"):
        check_snapshot(broken)


def test_check_snapshot_requires_every_discovery_response(
    registered: ContractSnapshot,
) -> None:
    broken = copy.deepcopy(registered)
    broken.discovery.pop("get_metric_catalog")
    with pytest.raises(ValueError, match="Missing static discovery response"):
        check_snapshot(broken)


def test_check_snapshot_rejects_failed_discovery(
    registered: ContractSnapshot,
) -> None:
    broken = copy.deepcopy(registered)
    broken.discovery["get_metric_catalog"]["success"] = False
    with pytest.raises(ValueError, match="discovery failed"):
        check_snapshot(broken)


def test_main_prints_a_report(
    registered: ContractSnapshot, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    async def snapshot() -> ContractSnapshot:
        return registered

    monkeypatch.setattr("tooling.prompt_contracts.collect_snapshot", snapshot)
    main()
    assert '"checked_utf8_bytes"' in capsys.readouterr().out
