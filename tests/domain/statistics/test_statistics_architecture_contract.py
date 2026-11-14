"""Keep the architecture overview a single-source-of-truth boundary.

The architecture section records module responsibilities, public entry point
names and where the full contracts live. Method-level statistical contracts
and interpretation guidance belong to the module and function docstrings.

Expected values come from the modules' AST (public function names and their
declared return annotations, plus module docstrings), never from the
documentation under test. Both directions are checked so that neither
duplication nor loss of information passes silently.
"""

from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_ARCHITECTURE = _ROOT / "ARCHITECTURE.md"
_STATISTICS_DIR = _ROOT / "src/tradingdev/domain/statistics"
_COMPARISON = _STATISTICS_DIR / "comparison.py"
_BENCHMARK = _STATISTICS_DIR / "benchmark.py"

_SECTION_START = "## 績效計算與對話摘要"
_SECTION_END = "### 診斷擴充與替換"
_INFERENCE_PREFIXES = ("difference_interval", "interval_table")

# Criteria defined by the documentation review: these are method-level
# contracts or interpretation guidance, so they must not be restated in the
# architecture section. They are maintained in the module docstrings below.
_METHOD_SEMANTICS = (
    "IntervalUnavailable",
    "摺疊為 `None`",
    "保守篩選",
    "同一組重抽樣索引",
    "樣本外報酬",
    "主動報酬是",
    "beta 即迴歸斜率",
    "平均淨暴露",
)


def _statistics_section() -> str:
    text = _ARCHITECTURE.read_text(encoding="utf-8")
    start = text.index(_SECTION_START)
    end = text.index(_SECTION_END, start)
    return text[start:end]


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _docstring_text(path: Path) -> str:
    """Collapse docstring line wrapping so prose is matched by wording."""
    return " ".join((ast.get_docstring(_module_ast(path)) or "").split())


def _inference_entry_points() -> dict[str, str]:
    """Public paired-bootstrap APIs and their declared return annotations."""
    return {
        node.name: ast.unparse(node.returns) if node.returns is not None else ""
        for node in _module_ast(_COMPARISON).body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith(_INFERENCE_PREFIXES)
    }


def test_architecture_names_every_inference_entry_point() -> None:
    """The review requires the outcome entry points to be discoverable here."""
    entry_points = _inference_entry_points()
    assert {
        "difference_interval",
        "difference_interval_outcome",
        "interval_table",
        "interval_table_outcome",
    } <= set(entry_points)
    section = _statistics_section()
    missing = sorted(name for name in entry_points if name not in section)
    assert not missing, f"architecture section omits entry points: {missing}"


def test_architecture_does_not_restate_method_level_semantics() -> None:
    """Method contracts stay in the docstrings, not in two places at once."""
    section = _statistics_section()
    duplicated = [term for term in _METHOD_SEMANTICS if term in section]
    assert not duplicated, (
        f"architecture section duplicates method-level semantics: {duplicated}"
    )


def test_architecture_points_to_both_authoritative_contracts() -> None:
    section = _statistics_section()
    assert "domain/statistics/comparison.py" in section
    assert "domain/statistics/benchmark.py" in section
    assert "模組與函式 docstring" in section or "模組及函式 docstring" in section


def test_method_semantics_remain_in_the_module_docstrings() -> None:
    """Removing duplication must not remove the guidance itself."""
    comparison_doc = _docstring_text(_COMPARISON)
    benchmark_doc = _docstring_text(_BENCHMARK)

    # Unavailable-inference distinction, still declared by the return types.
    entry_points = _inference_entry_points()
    for name in ("difference_interval", "interval_table"):
        assert "None" in entry_points[name], f"{name} no longer collapses to None"
    for name in ("difference_interval_outcome", "interval_table_outcome"):
        assert "IntervalUnavailable" in entry_points[name], (
            f"{name} no longer returns IntervalUnavailable"
        )
    assert "IntervalUnavailable" in comparison_doc
    assert "out-of-sample" in comparison_doc
    assert "experimental design" in comparison_doc
    assert "common period" in comparison_doc

    # Benchmark interpretation guidance stays with the implementation.
    for term in ("active return", "excess return", "average net exposure", "costs"):
        assert term in benchmark_doc, f"benchmark docstring lost guidance: {term}"
