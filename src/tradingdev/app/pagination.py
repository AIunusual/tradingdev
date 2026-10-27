"""Bounded application projections without changing full internal list APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tradingdev.app.contracts.common import DEFAULT_PAGE_LIMIT, Page, PageRequest

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def paginate[T, U](
    items: Sequence[T],
    *,
    project: Callable[[T], U],
    offset: int = 0,
    limit: int = DEFAULT_PAGE_LIMIT,
) -> Page[U]:
    """Validate bounds, then project only the requested already-sorted items."""
    query = PageRequest(offset=offset, limit=limit)
    selected = items[query.offset : query.offset + query.limit]
    end = query.offset + len(selected)
    return Page(
        items=[project(item) for item in selected],
        offset=query.offset,
        limit=query.limit,
        total=len(items),
        next_offset=end if end < len(items) else None,
    )
