"""``rag_answer`` -- a grounded answer over an explicit set of movie ids (R-052, R-057).

This is layer 1 of ADR-0012's grounding guard, and it is the layer that does most of the
work. The tool takes **movie ids**, not free text, and builds its context from exactly
those records. Un-retrieved movies cannot enter the prompt, because there is no path by
which they could -- grounding holds by construction rather than by instruction.

That also makes it testable: the closed payload is a precise thing to verify an answer
against, which is what ADR-0012's post-hoc check does downstream.
"""

from __future__ import annotations

import pandas as pd

from movieagent.data import schema as S
from movieagent.logging import get_logger
from movieagent.tools.base import Outcome, ToolContext, ToolResult

log = get_logger("tools.rag_answer")

NAME = "rag_answer"
DESCRIPTION = """\
Write a grounded natural-language answer about specific movies.

Pass the question and the ids of the movies to ground it in -- normally the ids returned
by semantic_search, structured_search or fuzzy_movie_search in this same turn.

The answer is built only from those movies' stored records. Use this when the user wants
prose about films rather than a table."""

_SYSTEM = """\
You are a movie analyst answering strictly from supplied dataset records.

Rules, without exception:
- Use ONLY the movie records below. They are the entire world of facts available.
- Never add a movie that is not in the records, even if you know one that fits better.
- Never state a fact -- runtime, year, cast, budget, revenue, rating -- that is not in
  the records. If a field is marked unknown, say it is unknown; do not estimate it.
- If the records do not answer the question, say so plainly.
- Refer to movies by their exact titles as written in the records.
- Be concise and specific. No preamble, no invented detail, no filler.
"""


def _context_block(context: ToolContext, movie_ids: list[int]) -> tuple[str, list[int]]:
    """Render the closed context, and report which ids were not found."""
    repo = context.repository
    blocks: list[str] = []
    missing: list[int] = []

    for movie_id in movie_ids:
        row = repo.get(movie_id)
        if row is None:
            missing.append(movie_id)
            continue
        year = row[S.RELEASE_YEAR]
        header = f"{row[S.TITLE]}" + (f" ({int(year)})" if pd.notna(year) else "")
        lines = [f"### {header}", f"id: {int(row[S.ID])}"]

        def add(label: str, value: object, unknown: str = "unknown") -> None:
            if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)):
                lines.append(f"{label}: {unknown}")
            elif isinstance(value, list):
                lines.append(f"{label}: {', '.join(str(v) for v in value) or unknown}")
            else:
                lines.append(f"{label}: {value}")

        add("genres", list(row[S.GENRES] or []))
        add("director", list(row[S.DIRECTORS] or []))
        add("starring", list(row[S.TOP_CAST] or []))
        add("release_date", row[S.RELEASE_DATE])
        add("runtime_minutes", None if pd.isna(row[S.RUNTIME]) else int(row[S.RUNTIME]))
        add("rating", None if pd.isna(row[S.VOTE_AVERAGE]) else float(row[S.VOTE_AVERAGE]))
        add("vote_count", None if pd.isna(row[S.VOTE_COUNT]) else int(row[S.VOTE_COUNT]))
        # Explicitly "unknown", never 0 -- R-016 carried all the way to the prompt.
        add("budget_usd", None if not row[S.BUDGET_KNOWN] else int(row[S.BUDGET]))
        add("revenue_usd", None if not row[S.REVENUE_KNOWN] else int(row[S.REVENUE]))
        add("tagline", row[S.TAGLINE])
        add("keywords", list(row[S.KEYWORDS] or [])[:15])
        add("overview", row[S.OVERVIEW])
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks), missing


def run(context: ToolContext, question: str, movie_ids: list[int]) -> ToolResult:
    """Answer ``question`` using only the records for ``movie_ids``."""
    if not movie_ids:
        return ToolResult.invalid(
            ["movie_ids is empty -- retrieve movies first, then ground the answer in them"]
        )
    if context.answer_fn is None:
        return ToolResult.failure(
            "No language model is configured, so a grounded answer cannot be generated."
        )

    block, missing = _context_block(context, movie_ids)
    if not block:
        return ToolResult(
            status=Outcome.NOT_FOUND,
            message=f"None of the ids {movie_ids} are in this dataset.",
            payload={"missing_ids": missing},
        )

    refs = [r for r in (context.repository.ref(i) for i in movie_ids) if r is not None]
    prompt = f"Movie records:\n\n{block}\n\n---\n\nQuestion: {question}"

    try:
        answer = context.answer_fn(_SYSTEM, prompt)
    except Exception as exc:  # noqa: BLE001 - provider failures are expected (R-087)
        log.exception("rag_answer generation failed")
        return ToolResult.failure(f"The language model could not be reached: {exc}")

    return ToolResult.ok_result(
        answer.strip(),
        refs=refs,
        payload={"answer": answer.strip(), "grounded_in": [r.movie_id for r in refs]},
        meta={
            "context_movies": [r.to_dict() for r in refs],
            "missing_ids": missing,
            "context_chars": len(block),
        },
    )
