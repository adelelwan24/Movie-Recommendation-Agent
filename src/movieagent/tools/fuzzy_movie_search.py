"""``fuzzy_movie_search`` -- title string to movie identity (R-040…R-045).

The contract is narrow on purpose: it answers *"which movie is this string?"* and
nothing else. It does not fetch details and it does not find similar films; those are
``movie_details`` and ``semantic_search``.

It is also allowed to refuse. Below the accept band, or when the top two candidates are
within a few points, it returns ``AMBIGUOUS`` and the graph routes to a clarification
interrupt (ADR-0021) rather than guessing.
"""

from __future__ import annotations

from movieagent.logging import get_logger
from movieagent.retrieval.fuzzy import MatchOutcome
from movieagent.tools.base import Outcome, ToolContext, ToolResult

log = get_logger("tools.fuzzy_movie_search")

NAME = "fuzzy_movie_search"
DESCRIPTION = """\
Resolve an approximate, partial or misspelled movie title to a specific movie.

Use for: "Avatr", "Intersteler", "the dark knight rises", "lord of the rings" -- any
time the user names a film and you need its identity before doing anything else.

Do NOT use for: plot, theme or mood descriptions (use semantic_search), or for filtering
by genre/year/rating (use structured_search).

Returns status "ambiguous" with candidates when it cannot confidently pick one. When
that happens, ask the user which they meant -- do not choose for them."""


def run(context: ToolContext, title: str) -> ToolResult:
    """Resolve a title, or decline to guess."""
    match = context.matcher.match(title)
    candidates = [c.to_dict() for c in match.candidates]

    match match.outcome:
        case MatchOutcome.MATCH:
            assert match.best is not None
            log.info("resolved %r -> %s (%.0f)", title, match.best.ref.title, match.best.score)
            return ToolResult.ok_result(
                f"{title!r} resolves to {match.best.ref.label()} ({match.reason}).",
                refs=[match.best.ref],
                payload={"candidates": candidates, "record": match.best.ref.to_dict()},
                meta={"score": match.best.score, "normalized": match.normalized},
            )

        case MatchOutcome.AMBIGUOUS:
            options = ", ".join(f"{c.ref.label()} ({c.score:.0f})" for c in match.candidates)
            return ToolResult(
                status=Outcome.AMBIGUOUS,
                message=(
                    f"{title!r} could be several movies -- {match.reason}. "
                    f"Candidates: {options}. Ask the user which one they meant."
                ),
                refs=[c.ref for c in match.candidates],
                payload={"candidates": candidates},
                meta={"normalized": match.normalized, "reason": match.reason},
            )

        case _:
            return ToolResult(
                status=Outcome.NOT_FOUND,
                message=(
                    f"No movie in this dataset matches {title!r} ({match.reason}). "
                    "The dataset ends around 2016, so a recent film will not be here."
                ),
                payload={"candidates": candidates},
                meta={"normalized": match.normalized, "reason": match.reason},
            )
