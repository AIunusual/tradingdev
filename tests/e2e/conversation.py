"""Provider-neutral evidence at completed model-turn boundaries."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt

if TYPE_CHECKING:
    from tests.e2e.llm_client import ToolCall


@dataclass(frozen=True)
class ModelTurn:
    number: int
    text: str
    calls: list[ToolCall]


UserReply = Callable[[ModelTurn], Awaitable[str | None]]


def links_to_file(text: str, path: Path) -> bool:
    """Require a rendered text link, not a path or Markdown example in code."""
    for block in MarkdownIt("commonmark").parse(text):
        children = block.children or []
        if any(child.type == "html_inline" for child in children):
            continue
        for index, token in enumerate(children):
            if token.type != "link_open":
                continue
            href = token.attrGet("href")
            if not isinstance(href, str):
                continue
            destination = urlsplit(href)
            if (
                destination.scheme
                or destination.netloc
                or destination.query
                or Path(unquote(destination.path)) != path
            ):
                continue
            for label in children[index + 1 :]:
                if label.type == "link_close":
                    break
                if label.type in {"text", "code_inline"} and label.content.strip():
                    return True
    return False


def visible_text(text: str) -> str:
    """Read Markdown wording without link targets, markup or raw HTML."""
    blocks: list[str] = []
    for block in MarkdownIt("commonmark").parse(text):
        if block.type in {"fence", "code_block"}:
            blocks.append(block.content)
        elif block.type == "inline":
            children = block.children or []
            # Raw HTML may hide its contents; it is not visible text evidence.
            if any(child.type == "html_inline" for child in children):
                continue
            blocks.append(
                "".join(
                    child.content
                    if child.type in {"text", "code_inline"}
                    else " "
                    if child.type in {"softbreak", "hardbreak"}
                    else ""
                    for child in children
                )
            )
    return " ".join(" ".join(blocks).split())
