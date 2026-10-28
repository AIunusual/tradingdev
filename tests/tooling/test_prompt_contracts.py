"""Mechanical contract checks reject drift without invoking research or models."""

from __future__ import annotations

import asyncio
import copy
import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
from jsonschema import ValidationError
from tooling import quality
from tooling.prompt_contracts import (
    BUDGETS,
    ContractSnapshot,
    check_snapshot,
    collect_snapshot,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tradingdev.adapters.storage.filesystem import WorkspacePaths


@pytest.fixture(scope="module")
def registered() -> ContractSnapshot:
    return asyncio.run(collect_snapshot())


@pytest.fixture
def candidate(registered: ContractSnapshot) -> ContractSnapshot:
    return copy.deepcopy(registered)


def test_registered_contracts_fit_explicit_serialization_budgets(
    candidate: ContractSnapshot,
) -> None:
    report = check_snapshot(candidate)
    sizes = report["checked_utf8_bytes"]
    assert sizes["initial_total"] == sizes["instructions"] + sizes["model_tools"]
    assert report["mcp_wire_tools"]["utf8_bytes"] > sizes["model_tools"]
    assert "not token counts" in report["measurement"]


@pytest.mark.parametrize("name", ["removed", "core2", "core-retired", "", "Core"])
def test_stale_section_reference_fails(candidate: ContractSnapshot, name: str) -> None:
    candidate.instructions += f'\nRead get_strategy_contract(section="{name}").'
    with pytest.raises(ValueError, match="Unknown strategy section reference:"):
        check_snapshot(candidate)


def test_section_hidden_from_every_menu_still_fails_input_schema_check(
    candidate: ContractSnapshot,
) -> None:
    del candidate.sections["example"]
    for payload in candidate.sections.values():
        payload["available_sections"] = [
            item for item in payload["available_sections"] if item["id"] != "example"
        ]
    with pytest.raises(ValueError, match="registered input schema"):
        check_snapshot(candidate)


def test_partial_menu_or_api_is_rejected(candidate: ContractSnapshot) -> None:
    candidate.sections["core"]["available_sections"].pop()
    with pytest.raises(ValueError, match="discovery menu"):
        check_snapshot(candidate)


def test_discovered_indicator_requires_readable_selected_reference(
    candidate: ContractSnapshot,
) -> None:
    del candidate.indicators["macd"]
    with pytest.raises(ValueError, match="Indicator catalog"):
        check_snapshot(candidate)


def test_selected_indicator_menu_and_documentation_cannot_drift(
    candidate: ContractSnapshot,
) -> None:
    payload = candidate.indicators["macd"]
    payload["available_sections"].pop()
    with pytest.raises(ValueError, match="discovery menu"):
        check_snapshot(candidate)
    payload["available_sections"] = candidate.sections["core"]["available_sections"]
    payload["content"]["selected"]["documentation"] = (
        'Read get_strategy_contract(section="removed").'
    )
    with pytest.raises(ValueError, match="Unknown strategy section reference: removed"):
        check_snapshot(candidate)


def test_output_contract_validation_is_not_replaced_by_size_check(
    candidate: ContractSnapshot,
) -> None:
    candidate.indicators["macd"]["content"]["selected"]["source"] = 123
    with pytest.raises(ValidationError):
        check_snapshot(candidate)


def test_growth_budget_counts_utf8_bytes_without_truncation(
    candidate: ContractSnapshot,
) -> None:
    candidate.instructions = "指" * (BUDGETS["instructions"] // 3 + 1)
    with pytest.raises(ValueError, match="UTF-8 bytes exceeds instructions budget"):
        check_snapshot(candidate)
    assert candidate.instructions.endswith("指")
    assert len(candidate.instructions) < BUDGETS["instructions"]


def test_selected_reference_growth_is_bounded(candidate: ContractSnapshot) -> None:
    candidate.indicators["macd"]["content"]["selected"]["source"] += (
        "x" * BUDGETS["indicator_reference"]
    )
    with pytest.raises(
        ValueError, match="indicator/macd: .* exceeds indicator_reference"
    ):
        check_snapshot(candidate)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("core", "base_strategy_source"),
        ("example", "example_strategy_code"),
        ("example", "example_yaml_config"),
    ],
)
def test_source_is_not_interpreted_as_workflow_prose(
    candidate: ContractSnapshot, section: str, field: str
) -> None:
    candidate.sections[section]["content"][field] += '\n# section="a_python_variable"\n'
    check_snapshot(candidate)


def test_fast_quality_runs_gate_and_propagates_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[str, ...]] = []

    def run(*arguments: str) -> None:
        commands.append(arguments)
        if arguments == ("tooling.prompt_contracts",):
            raise subprocess.CalledProcessError(1, arguments)

    monkeypatch.setattr(quality, "run", run)
    monkeypatch.setattr(sys, "argv", ["quality", "check"])
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    with pytest.raises(subprocess.CalledProcessError):
        quality.main()
    assert commands[-2:] == [("tooling.documentation",), ("tooling.prompt_contracts",)]


@pytest.mark.parametrize("failure", [RuntimeError, TimeoutError])
def test_collection_failure_cleans_workspace_and_restores_environment(
    monkeypatch: pytest.MonkeyPatch, failure: type[Exception]
) -> None:
    from tradingdev.mcp import server

    roots: list[Path] = []
    original = dict(os.environ)

    def fail(paths: WorkspacePaths) -> None:
        root = paths.root.parent
        roots.append(root)
        (root / "partial-reference").write_text("partial")
        raise failure("discovery unavailable")

    monkeypatch.setattr(server, "create_server", fail)
    with pytest.raises(failure, match="discovery unavailable"):
        asyncio.run(collect_snapshot())
    assert roots and all(not root.exists() for root in roots)
    assert dict(os.environ) == original
