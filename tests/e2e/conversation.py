"""Provider-neutral evidence at completed model-turn boundaries."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.e2e.llm_client import ToolCall


@dataclass(frozen=True)
class ModelTurn:
    number: int
    text: str
    calls: list[ToolCall]


UserReply = Callable[[ModelTurn], Awaitable[str | None]]
