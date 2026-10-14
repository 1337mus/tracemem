"""Provider-neutral context packing under caller-supplied token counters."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .retrieval import SearchHit, _context_line


@dataclass(frozen=True, slots=True)
class TokenPackedContext:
    """A token-counted context and the exact hits that contributed to it."""

    text: str
    selected_hits: tuple[SearchHit, ...]
    prompt_tokens: int
    max_prompt_tokens: int
    counter_id: str


def _count_tokens(count_prompt_tokens: Callable[[str], int], context: str) -> int:
    count = count_prompt_tokens(context)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("token counter must return a non-negative integer")
    return count


def pack_context_by_tokens(
    hits: Sequence[SearchHit],
    max_prompt_tokens: int,
    count_prompt_tokens: Callable[[str], int],
    *,
    counter_id: str,
) -> TokenPackedContext:
    """Pack complete evidence lines using an exact caller-provided prompt counter."""
    if (
        isinstance(max_prompt_tokens, bool)
        or not isinstance(max_prompt_tokens, int)
        or max_prompt_tokens < 0
    ):
        raise ValueError("max_prompt_tokens must be a non-negative integer")
    if not callable(count_prompt_tokens):
        raise TypeError("count_prompt_tokens must be callable")
    if not isinstance(counter_id, str):
        raise TypeError("counter_id must be a nonempty string")
    if not counter_id.strip():
        raise ValueError("counter_id must be a nonempty string")

    base_tokens = _count_tokens(count_prompt_tokens, "")
    if base_tokens > max_prompt_tokens:
        raise ValueError(
            f"empty prompt uses {base_tokens} tokens, "
            f"exceeding the {max_prompt_tokens}-token budget"
        )
    lines: list[str] = []
    selected_hits: list[SearchHit] = []
    used_tokens = base_tokens
    for hit in hits:
        line = _context_line(hit)
        candidate_context = "\n".join((*lines, line))
        candidate_tokens = _count_tokens(count_prompt_tokens, candidate_context)
        if candidate_tokens <= max_prompt_tokens:
            lines.append(line)
            selected_hits.append(hit)
            used_tokens = candidate_tokens
        else:
            continue
    return TokenPackedContext(
        "\n".join(lines), tuple(selected_hits), used_tokens, max_prompt_tokens, counter_id
    )
