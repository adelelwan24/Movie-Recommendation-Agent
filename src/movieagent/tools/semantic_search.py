"""``semantic_search`` -- meaning to movies, with optional hybrid pre-filtering.

Covers R-050…R-057 (conceptual retrieval) and R-070 (hybrid). Two entry shapes:

* a natural-language description ("someone trying to survive alone on another planet");
* ``similar_to`` a movie id, which ranks by that film's own document vector.

Constraints are applied **before** ranking (ADR-0011), so every result satisfies them and
k results come back whenever k exist. Post-filtering would quietly return three when you
asked for ten and look like scarcity rather than filtering.
"""

from __future__ import annotations

import numpy as np

from movieagent.data.query import SearchQuery
from movieagent.data.schema import RetrievedDoc
from movieagent.logging import get_logger
from movieagent.tools.base import Outcome, ToolContext, ToolResult

log = get_logger("tools.semantic_search")

NAME = "semantic_search"
DESCRIPTION = """\
Find movies by meaning: plot, theme, mood, or similarity to another movie.

Use for: "a movie about someone surviving alone on another planet", "dark and
psychological", "time travel and changing the past", "similar to Inception". Pass
`filters` as well to combine meaning with hard constraints -- "a funny science-fiction
movie from after 2010 under 2 hours" is semantic intent plus genre/year/runtime filters.

Do NOT use for: resolving a specific title the user typed (use fuzzy_movie_search), or
for pure counting and aggregation (use structured_search).

Filters are applied before ranking, so every result satisfies them. If the candidate
pool is small, the ranking within it means little -- the pool size is reported."""


def run(
    context: ToolContext,
    query: str | None = None,
    similar_to: int | None = None,
    filters: SearchQuery | None = None,
    k: int | None = None,
) -> ToolResult:
    """Retrieve semantically, optionally restricted by structured filters."""
    repo = context.repository
    settings = context.settings.retrieval
    k = k or settings.top_k

    if query is None and similar_to is None:
        return ToolResult.invalid(["provide either `query` text or a `similar_to` movie id"])

    mask: np.ndarray | None = None
    constraints: list[str] = []
    if filters is not None and not filters.is_empty():
        problems = repo.validate(filters)
        if problems:
            return ToolResult.invalid(problems)
        mask = repo.mask_for(filters)
        constraints = filters.describe()

    # R-056, revised by ADR-0026. Checked *before* embedding: if the query is not made
    # of words this corpus uses, no amount of vector geometry will tell us so afterwards.
    if query is not None and context.vocabulary is not None:
        cov = context.vocabulary.coverage(query)
        if cov.ratio < settings.min_lexical_coverage:
            return ToolResult(
                status=Outcome.LOW_CONFIDENCE,
                message=(
                    f"{query!r} does not read like a description of anything in this "
                    f"dataset -- {', '.join(repr(w) for w in cov.unknown[:5])} "
                    f"{'do' if len(cov.unknown) > 1 else 'does'} not appear anywhere in "
                    "the ~4,800 movie documents. Ask the user to rephrase rather than "
                    "presenting whatever ranked highest."
                ),
                payload={"lexical_coverage": round(cov.ratio, 2), "shown": 0},
                meta={
                    "unknown_words": list(cov.unknown),
                    "known_words": list(cov.known),
                    "min_lexical_coverage": settings.min_lexical_coverage,
                },
            )

    pool = context.index.pool_size(mask)
    if pool == 0:
        return ToolResult.empty(
            "No movies satisfy those constraints, so there is nothing to rank.",
            binding=constraints,
        )

    if similar_to is not None:
        position = repo.position(similar_to)
        if position is None:
            return ToolResult(
                status=Outcome.NOT_FOUND,
                message=f"Movie id {similar_to} is not in this dataset.",
            )
        hits = context.index.similar_to(position, k, mask)
        seed = repo.ref(similar_to)
        subject = f"movies similar to {seed.label()}" if seed else f"movies similar to {similar_to}"
    else:
        vector = context.embedder.embed_query(query or "")
        hits = context.index.search(vector, k, mask)
        subject = f"movies matching {query!r}"

    if not hits:
        return ToolResult.empty(f"No results for {subject}.", binding=constraints)

    retrieved = [
        RetrievedDoc(
            ref=repo.refs_at(np.array([hit.position]))[0],
            score=hit.score,
            document=context.documents[hit.position],
        )
        for hit in hits
    ]
    best = retrieved[0].score

    meta = {
        "pool_size": pool,
        "corpus_size": len(context.index),
        "filters": filters.to_display() if filters else {},
        "constraints": constraints,
        "documents": [d.to_dict() for d in retrieved],
        "scores": [round(d.score, 4) for d in retrieved],
        "similarity_floor": settings.similarity_floor,
        "lexical_coverage": (
            round(context.vocabulary.coverage(query).ratio, 2)
            if query is not None and context.vocabulary is not None
            else None
        ),
    }
    payload = {
        "pool_size": pool,
        "shown": len(retrieved),
        "top_score": round(best, 4),
    }

    # R-056. The floor is low because ADR-0008's labelled template raises baseline
    # similarity across the whole corpus -- every document shares its field labels, so
    # nothing is ever truly dissimilar. An intuitive 0.5 would reject good matches.
    if best < settings.similarity_floor:
        return ToolResult(
            status=Outcome.LOW_CONFIDENCE,
            message=(
                f"Nothing matches {subject} well -- the best similarity is "
                f"{best:.2f}, below the {settings.similarity_floor:.2f} floor. "
                "Show these only with an explicit caveat that they may not be what was meant."
            ),
            refs=[d.ref for d in retrieved],
            payload=payload,
            meta=meta,
        )

    message = f"Retrieved {len(retrieved)} {subject}."
    if mask is not None:
        message += f" Ranked within a filtered pool of {pool} movies."
        if pool < k * 2:
            # Honest signal: ranking 8 out of 12 is close to arbitrary, and saying so
            # is better than presenting a confident-looking order.
            message += " That pool is small, so the ordering is weakly meaningful."
    return ToolResult.ok_result(
        message, refs=[d.ref for d in retrieved], payload=payload, meta=meta
    )
