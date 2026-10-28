"""Shared model-review selection and rendered-prompt budget options."""

from __future__ import annotations

import argparse

from tooling.review_scope import SCOPES


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _context_for(value: str) -> tuple[str, str]:
    scope, separator, path = value.partition(":")
    if not separator or scope not in {*SCOPES, "cross_scope"} or not path:
        raise argparse.ArgumentTypeError(
            "expected SCOPE:PATH with a known scope or cross_scope and nonempty path"
        )
    return scope, path


def add_review_arguments(parser: argparse.ArgumentParser) -> None:
    """Add model-only controls, without changing the entrypoint's check route."""
    parser.add_argument(
        "--review-scope",
        action="append",
        dest="review_scopes",
        choices=sorted(SCOPES),
        help="Review this ownership scope (repeat; must cover all changed owners)",
    )
    parser.add_argument(
        "--review-context",
        action="append",
        default=[],
        help="Include this exact repository file in model evidence (repeat)",
    )
    parser.add_argument(
        "--review-context-for",
        action="append",
        type=_context_for,
        default=[],
        metavar="SCOPE:PATH",
        help="Include a file only in this scope (repeat; cross_scope allowed)",
    )
    parser.add_argument(
        "--max-prompt-bytes",
        type=_positive_integer,
        help="Limit each complete rendered model prompt in UTF-8 bytes",
    )
    parser.add_argument(
        "--max-total-prompt-bytes",
        type=_positive_integer,
        help="Limit all rendered model prompts together in UTF-8 bytes",
    )


def validate_review_options(
    *,
    review_scopes: tuple[str, ...] | None = None,
    review_context: tuple[str, ...] = (),
    review_context_for: tuple[tuple[str, str], ...] = (),
    max_prompt_bytes: int | None = None,
    max_total_prompt_bytes: int | None = None,
) -> None:
    """Reject malformed options before Git, model, or check side effects."""
    if review_scopes is not None:
        if not isinstance(review_scopes, tuple) or not review_scopes:
            raise RuntimeError("Review scopes must be a nonempty tuple of names.")
        for scope in review_scopes:
            if not isinstance(scope, str) or scope not in SCOPES:
                raise RuntimeError(f"Unknown review scope: {scope!r}.")
    if not isinstance(review_context, tuple) or any(
        not isinstance(path, str) or not path for path in review_context
    ):
        raise RuntimeError("Review context must contain nonempty repository paths.")
    if not isinstance(review_context_for, tuple):
        raise RuntimeError("Scoped review context must be a tuple of scope/path pairs.")
    for item in review_context_for:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not all(isinstance(value, str) for value in item)
            or item[0] not in {*SCOPES, "cross_scope"}
            or not item[1]
        ):
            raise RuntimeError("Scoped review context requires a known scope and path.")
    for name, value in (
        ("Per-prompt", max_prompt_bytes),
        ("Total prompt", max_total_prompt_bytes),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise RuntimeError(f"{name} budget must be a positive integer byte count.")


def review_arguments(
    *,
    review_scopes: tuple[str, ...] | None = None,
    review_context: tuple[str, ...] = (),
    review_context_for: tuple[tuple[str, str], ...] = (),
    max_prompt_bytes: int | None = None,
    max_total_prompt_bytes: int | None = None,
) -> list[str]:
    """Forward only explicit options as separate argv entries, never shell text."""
    validate_review_options(
        review_scopes=review_scopes,
        review_context=review_context,
        review_context_for=review_context_for,
        max_prompt_bytes=max_prompt_bytes,
        max_total_prompt_bytes=max_total_prompt_bytes,
    )
    arguments: list[str] = []
    for scope in review_scopes or ():
        arguments.extend(["--review-scope", scope])
    for path in review_context:
        if path.startswith("-"):
            arguments.append(f"--review-context={path}")
        else:
            arguments.extend(["--review-context", path])
    for scope, path in review_context_for:
        arguments.append(f"--review-context-for={scope}:{path}")
    if max_prompt_bytes is not None:
        arguments.extend(["--max-prompt-bytes", str(max_prompt_bytes)])
    if max_total_prompt_bytes is not None:
        arguments.extend(["--max-total-prompt-bytes", str(max_total_prompt_bytes)])
    return arguments
