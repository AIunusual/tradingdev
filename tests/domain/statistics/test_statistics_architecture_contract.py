"""Keep the statistics overview in step with the comparison entry points.

Expected values come from the module's own AST (declared return annotations and
public function names), never from the documentation under test. If an entry
point or its return contract changes, the overview must be updated as well.
"""

from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_ARCHITECTURE = _ROOT / "ARCHITECTURE.md"
_COMPARISON = _ROOT / "src/tradingdev/domain/statistics/comparison.py"

_OVERVIEW_START = "需要從原始觀察值獨立計算時"
_OVERVIEW_END = "摘要保留金額與比例回撤"
_INFERENCE_PREFIXES = ("difference_interval", "interval_table")


def _statistics_overview() -> str:
    """The prose overview of independent statistics, excluding the module table."""
    text = _ARCHITECTURE.read_text(encoding="utf-8")
    start = text.index(_OVERVIEW_START)
    end = text.index(_OVERVIEW_END, start)
    return text[start:end]


def _inference_entry_points() -> dict[str, str]:
    """Public paired-bootstrap APIs and their declared return annotations."""
    tree = ast.parse(_COMPARISON.read_text(encoding="utf-8"))
    return {
        node.name: ast.unparse(node.returns) if node.returns is not None else ""
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith(_INFERENCE_PREFIXES)
    }


def test_every_inference_entry_point_is_named_in_the_overview() -> None:
    entry_points = _inference_entry_points()
    assert {
        "difference_interval",
        "difference_interval_outcome",
        "interval_table",
        "interval_table_outcome",
    } <= set(entry_points)
    overview = _statistics_overview()
    missing = sorted(name for name in entry_points if name not in overview)
    assert not missing, f"statistics overview omits entry points: {missing}"


def test_overview_describes_the_unavailable_result_distinction() -> None:
    """The distinction in prose must match the declared return types."""
    entry_points = _inference_entry_points()
    for name in ("difference_interval", "interval_table"):
        assert "None" in entry_points[name], f"{name} no longer collapses to None"
    for name in ("difference_interval_outcome", "interval_table_outcome"):
        assert "IntervalUnavailable" in entry_points[name], (
            f"{name} no longer returns IntervalUnavailable"
        )
    overview = _statistics_overview()
    assert "IntervalUnavailable" in overview
    assert "None" in overview
    assert "不可用" in overview


def test_overview_points_to_the_full_contracts() -> None:
    assert "domain/statistics/comparison.py" in _statistics_overview()
