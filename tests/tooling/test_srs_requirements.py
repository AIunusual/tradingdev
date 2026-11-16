"""SRS maintenance catches broken records without claiming behavioral coverage."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest
from tooling import documentation
from tooling.requirements import (
    RequirementsError,
    check_requirements,
    contract_references,
    parse_requirement_records,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch


RECORD = """## FR-EXEC-001 固定執行

- **需求**：保存有效設定。
- **驗收**：預設值更新後，查詢仍回傳原設定。
- **契約**：[執行](../contract.md#設定)。
- **驗證**：[服務測試](../../tests/test_execution.py)，驗證歷史設定。
"""


def _srs(root: Path, record: str = RECORD) -> Path:
    (root / "docs/requirements").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "docs/requirements.md").write_text(
        "# SRS\n\n[執行](requirements/execution.md)\n", encoding="utf-8"
    )
    (root / "docs/contract.md").write_text("# 設定\n", encoding="utf-8")
    # Evidence must only be read as a reference, never imported or executed.
    (root / "tests/test_execution.py").write_text(
        "raise RuntimeError('never execute')\n"
    )
    chapter = root / "docs/requirements/execution.md"
    chapter.write_text("# 執行需求\n\n" + record, encoding="utf-8")
    return chapter


def test_documentation_entry_checks_srs_before_formatter(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _srs(tmp_path, RECORD.replace("- **驗收**：預設值更新後，查詢仍回傳原設定。\n", ""))
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("invalid SRS"))
    with pytest.raises(RequirementsError, match="missing fields 驗收"):
        documentation.check_documents(tmp_path)


def test_multiple_chapters_and_explicit_manual_verification(tmp_path: Path) -> None:
    _srs(tmp_path)
    (tmp_path / "docs/requirements/quality.md").write_text(
        "# 品質\n\n"
        + RECORD.replace("FR-EXEC-001", "NFR-UI-001").replace(
            "[服務測試](../../tests/test_execution.py)，驗證歷史設定。",
            "人工驗收：在離線瀏覽器開啟報告，確認可閱讀；瀏覽器差異尚待實測。",
        ),
        encoding="utf-8",
    )
    index = tmp_path / "docs/requirements.md"
    index.write_text(index.read_text() + "\n[品質](requirements/quality.md)\n")
    assert check_requirements(tmp_path) == 2


@pytest.mark.parametrize(
    "before, after, error",
    [
        ("FR-EXEC-001", "FR-EXEC-1", "malformed requirement"),
        ("## FR-EXEC-001", "### FR-EXEC-001", "malformed requirement"),
        ("保存有效設定。", "", "empty or duplicate 需求"),
        ("- **需求**：保存有效設定。", "", "missing fields 需求"),
        ("- **驗收**：", "- **驗収**：", "missing fields 驗收"),
        ("[執行](../contract.md#設定)", "詳見程式碼", "missing 契約 evidence"),
        (
            "[服務測試](../../tests/test_execution.py)",
            "[範例](https://example.invalid/test.py)",
            "missing 驗證 evidence",
        ),
        ("tests/test_execution.py", "tests/missing.py", "invalid local evidence"),
    ],
)
def test_invalid_requirement_records(
    tmp_path: Path, before: str, after: str, error: str
) -> None:
    _srs(tmp_path, RECORD.replace(before, after))
    with pytest.raises(RequirementsError, match=error):
        check_requirements(tmp_path)


def test_duplicate_identifiers_across_chapters(tmp_path: Path) -> None:
    _srs(tmp_path)
    (tmp_path / "docs/requirements/quality.md").write_text("# Other\n\n" + RECORD)
    index = tmp_path / "docs/requirements.md"
    index.write_text(index.read_text() + "\n[Other](requirements/quality.md)\n")
    with pytest.raises(RequirementsError, match="duplicate FR-EXEC-001"):
        check_requirements(tmp_path)


def test_duplicate_fields_and_code_examples_do_not_supply_evidence(
    tmp_path: Path,
) -> None:
    chapter = _srs(tmp_path, RECORD + "\n- **需求**：重複。\n")
    with pytest.raises(RequirementsError, match="duplicate 需求"):
        check_requirements(tmp_path)
    chapter.write_text("# Examples\n\n```markdown\n" + RECORD + "```\n")
    with pytest.raises(RequirementsError, match="no requirement records"):
        check_requirements(tmp_path)


@pytest.mark.parametrize("removed", ["entry", "navigation", "chapter", "test"])
def test_missing_navigation_and_deleted_evidence(tmp_path: Path, removed: str) -> None:
    chapter = _srs(tmp_path)
    if removed == "entry":
        (tmp_path / "docs/requirements.md").unlink()
    elif removed == "navigation":
        (tmp_path / "docs/requirements.md").write_text("# SRS\n")
    elif removed == "chapter":
        chapter.unlink()
    else:
        (tmp_path / "tests/test_execution.py").unlink()
    with pytest.raises(RequirementsError):
        check_requirements(tmp_path)


def test_repositories_before_srs_introduction_remain_valid(tmp_path: Path) -> None:
    assert check_requirements(tmp_path) == 0


def test_evidence_cannot_escape_repository(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _srs(root)
    outside = tmp_path / "outside.py"
    outside.write_text("raise RuntimeError('do not read')\n")
    target = root / "tests/test_execution.py"
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(RequirementsError, match="invalid local evidence"):
        check_requirements(root)


def test_new_chapter_requires_a_review_responsibility(tmp_path: Path) -> None:
    _srs(tmp_path)
    (tmp_path / "docs/requirements/new.md").write_text(RECORD)
    index = tmp_path / "docs/requirements.md"
    index.write_text(index.read_text() + "\n[New](requirements/new.md)\n")
    with pytest.raises(
        RequirementsError, match="unregistered SRS review responsibility"
    ):
        check_requirements(tmp_path)


def test_parsed_records_preserve_source_bounds_and_each_requirements_evidence() -> None:
    first = RECORD.replace("[執行](../contract.md#設定)", "[執行][contract]")
    first += (
        "\n### 補充\n\n原文說明。\n\n"
        "- **契約**：[補充](../extra.md)\n\n"
        "```markdown\n## FR-FAKE-001 範例\n```\n\n"
    )
    second = (
        RECORD.replace("FR-EXEC-001", "FR-EXEC-002")
        .replace("../contract.md#設定", "/docs/other.md#第二章")
        .replace("tests/test_execution.py", "tests/test_other.py")
    )
    second += "\n"
    text = (
        "# 執行需求\n\n" + first + second + "## 附錄\n\n"
        "[contract]: ../contract.md#%E8%A8%AD%E5%AE%9A\n"
    ).replace("\n", "\r\n")
    records = parse_requirement_records("docs/requirements/execution.md", text)
    assert [record.identifier for record in records] == ["FR-EXEC-001", "FR-EXEC-002"]
    start = 3
    for record, expected in zip(records, (first, second), strict=True):
        assert record.source == "docs/requirements/execution.md"
        assert record.start_line == start
        assert record.end_line == start + len(expected.splitlines()) - 1
        assert record.text == expected.replace("\n", "\r\n")
        assert record.text == "".join(
            text.splitlines(keepends=True)[record.start_line - 1 : record.end_line]
        )
        start = record.end_line + 1
    assert records[0].contract_references == (
        ("docs/contract.md", "設定"),
        ("docs/extra.md", ""),
    )
    assert records[0].verification_paths == ("tests/test_execution.py",)
    assert records[1].contract_references == (("docs/other.md", "第二章"),)
    assert records[1].verification_paths == ("tests/test_other.py",)
    assert contract_references("docs/requirements/execution.md", text) == {
        ("docs/contract.md", "設定"),
        ("docs/extra.md", ""),
        ("docs/other.md", "第二章"),
    }


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\v"])
@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_record_source_slices_treat_unicode_separators_as_inline_text(
    separator: str, newline: str
) -> None:
    first = RECORD.replace("保存有效設定。", f"第一段{separator}仍屬同一行。") + "\n"
    second = (
        RECORD.replace("FR-EXEC-001", "FR-EXEC-002")
        .replace("預設值更新後，查詢仍回傳原設定。", f"第二段{separator}不得混入前項。")
        .replace("../contract.md#設定", "../second.md#第二章")
        .replace("tests/test_execution.py", "tests/test_second.py")
        .rstrip("\n")
    )
    text = (f"# 執行需求{separator}導覽\n\n" + first + second).replace("\n", newline)

    records = parse_requirement_records("docs/requirements/execution.md", text)

    assert [record.identifier for record in records] == ["FR-EXEC-001", "FR-EXEC-002"]
    assert [(record.start_line, record.end_line) for record in records] == [
        (3, 9),
        (10, 15),
    ]
    assert [record.text for record in records] == [
        first.replace("\n", newline),
        second.replace("\n", newline),
    ]
    assert records[0].contract_references == (("docs/contract.md", "設定"),)
    assert records[0].verification_paths == ("tests/test_execution.py",)
    assert records[1].contract_references == (("docs/second.md", "第二章"),)
    assert records[1].verification_paths == ("tests/test_second.py",)


def test_parsing_manual_verification_needs_no_filesystem(
    monkeypatch: MonkeyPatch,
) -> None:
    from pathlib import Path

    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: pytest.fail("No I/O"))
    text = RECORD.replace(
        "[服務測試](../../tests/test_execution.py)，驗證歷史設定。",
        "人工驗收：在離線瀏覽器檢查互動。",
    ).rstrip("\n")
    (record,) = parse_requirement_records("docs/requirements/execution.md", text)
    assert record.text == text
    assert record.start_line == 1 and record.end_line == len(text.splitlines())
    assert record.verification_paths == ()
    assert record.contract_references == (("docs/contract.md", "設定"),)


def test_parsed_references_keep_inline_and_reference_links_but_not_examples() -> None:
    text = RECORD.replace(
        "[執行](../contract.md#設定)",
        "[第一](../contract.md#設定)、[重複][contract]、[另一個](../other.md)"
        "及[外部](https://example.invalid/guide.md)",
    ).replace(
        "[服務測試](../../tests/test_execution.py)",
        "[服務測試](../../tests/test_%65xecution.py?mode=read#case)、"
        "[重複](/tests/test_execution.py) 與[說明](../test_notes.md)",
    )
    text += (
        "\n[contract]: ../contract.md#設定\n\n"
        "```markdown\n- **契約**：[範例](../example.md)\n```\n\n"
        "> - **契約**：[引文](../quote.md)\n"
    )
    (record,) = parse_requirement_records("docs/requirements/execution.md", text)
    assert record.contract_references == (
        ("docs/contract.md", "設定"),
        ("docs/other.md", ""),
    )
    assert record.verification_paths == (
        "docs/test_notes.md",
        "tests/test_execution.py",
    )


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("# 無需求\n", "no requirement records"),
        ("```markdown\n" + RECORD + "```\n", "no requirement records"),
        (RECORD.replace("FR-EXEC-001", "FR-EXEC-1"), "malformed requirement"),
        (RECORD + "\n" + RECORD, "duplicate FR-EXEC-001"),
        (
            RECORD.replace("- **驗收**：預設值更新後，查詢仍回傳原設定。\n", ""),
            "missing fields 驗收",
        ),
        (RECORD + "\n- **需求**：重複。\n", "duplicate 需求"),
        (RECORD.replace("保存有效設定。", ""), "empty or duplicate 需求"),
        (
            RECORD.replace("../contract.md#設定", "../../../outside.md"),
            "invalid contract reference",
        ),
        (
            RECORD.replace("../contract.md#設定", "../../src/code.py"),
            "invalid contract reference",
        ),
        (RECORD.replace("../contract.md#設定", "ml.md"), "SRS cannot replace"),
        (
            RECORD.replace("../../tests/test_execution.py", "../../../test_outside.py"),
            "invalid local evidence reference",
        ),
    ],
)
def test_record_parser_rejects_incomplete_or_invalid_evidence(
    text: str, error: str
) -> None:
    with pytest.raises(RequirementsError, match=error):
        parse_requirement_records("docs/requirements/execution.md", text)


def test_contract_reference_extraction_still_accepts_fields_without_records() -> None:
    text = "- **契約**：[契約](../contract.md#設定)\n"
    assert contract_references("docs/requirements/execution.md", text) == {
        ("docs/contract.md", "設定")
    }
