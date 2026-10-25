"""Paging bounds and completeness are enforced independently of MCP clients."""

import pytest
from pydantic import ValidationError

from tradingdev.app.contracts.common import Page
from tradingdev.app.pagination import paginate


@pytest.mark.parametrize(
    ("offset", "limit", "expected", "next_offset"),
    [(0, 2, [0, 1], 2), (2, 2, [2], None), (3, 2, [], None), (9, 2, [], None)],
)
def test_page_projects_only_selected_items_and_reports_full_count(
    offset: int, limit: int, expected: list[int], next_offset: int | None
) -> None:
    projected = []

    def project(value: int) -> str:
        projected.append(value)
        return str(value)

    page = paginate([0, 1, 2], project=project, offset=offset, limit=limit)
    assert projected == expected
    assert page.items == [str(value) for value in expected]
    assert page.total == 3 and page.next_offset == next_offset
    assert page.offset == offset and page.limit == limit


@pytest.mark.parametrize(
    "arguments",
    [
        {"offset": -1},
        {"offset": True},
        {"offset": "0"},
        {"offset": 0.5},
        {"limit": 0},
        {"limit": 101},
        {"limit": False},
        {"limit": "20"},
        {"limit": 1.5},
    ],
)
def test_invalid_bounds_fail_before_projecting(arguments: dict[str, object]) -> None:
    def unexpected(_item: int) -> int:
        pytest.fail("Invalid page bounds must not project rows")

    with pytest.raises(ValidationError):
        paginate([1], project=unexpected, **arguments)  # type: ignore[arg-type]


def test_page_contract_rejects_silent_truncation_or_missing_continuation() -> None:
    with pytest.raises(ValidationError, match="items"):
        Page[int](items=[1], offset=0, limit=20, total=2, next_offset=None)
    with pytest.raises(ValidationError, match="next_offset"):
        Page[int](items=[1], offset=0, limit=1, total=2, next_offset=None)
    empty = paginate([], project=str)
    assert empty.model_dump() == {
        "items": [],
        "offset": 0,
        "limit": 20,
        "total": 0,
        "next_offset": None,
    }
