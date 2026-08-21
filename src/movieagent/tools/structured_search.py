"""``structured_search`` -- deterministic filtering, sorting and aggregation (R-030…R-039).

The LLM interprets the request into a ``SearchQuery``; every number and record here comes
from pandas. No model output reaches the arithmetic.
"""

from __future__ import annotations

from typing import Any

from movieagent.data.query import SearchQuery
from movieagent.logging import get_logger
from movieagent.tools.base import ToolContext, ToolResult

log = get_logger("tools.structured_search")

NAME = "structured_search"
DESCRIPTION = """\
Filter, sort, count and aggregate movies using exact dataset fields.

Use for: counts and aggregations ("how many movies per genre", "10 most common genres",
"which production companies have the most movies", "movies released per year"), numeric
comparisons (rating, votes, runtime, budget, revenue, popularity), year ranges, genre /
company / country / language / person filters, sorting and top-N.

Do NOT use for: finding a movie by an approximate or misspelled title (use
fuzzy_movie_search), or for plot/theme/mood queries (use semantic_search).

Unknown budget or revenue is stored as unknown, not zero, so it is excluded from
rankings and comparisons rather than sorted to the bottom."""


def run(context: ToolContext, query: SearchQuery) -> ToolResult:
    """Execute a structured query."""
    repo = context.repository

    # Values are validated against the dataset's real vocabularies, not just the schema
    # (R-038). The failure that actually bites here is a *plausible* term with the wrong
    # wording -- "Sci-Fi" rather than "Science Fiction" -- which would otherwise return
    # zero rows and a confident "no movies match".
    problems = repo.validate(query)
    if problems:
        log.info("rejected structured query: %s", problems)
        return ToolResult.invalid(problems)

    if query.aggregate is not None:
        return _aggregate(context, query)
    return _search(context, query)


def _aggregate(context: ToolContext, query: SearchQuery) -> ToolResult:
    frame = context.repository.aggregate(query)
    spec = query.aggregate
    assert spec is not None

    if frame.empty:
        return ToolResult.empty(
            f"No data to aggregate for {spec.describe()}.",
            binding=query.describe(),
        )

    records: list[dict[str, Any]] = frame.to_dict(orient="records")
    return ToolResult.ok_result(
        f"{spec.describe()} over {len(frame)} groups.",
        payload={
            "aggregate": records,
            "columns": list(frame.columns),
            "group_by": spec.group_by.value,
            "metric": spec.metric.value,
        },
        meta={"filters": query.to_display(), "rows": records},
    )


def _search(context: ToolContext, query: SearchQuery) -> ToolResult:
    result = context.repository.search(query)

    if result.total == 0:
        binding = result.binding or ["no filters"]
        return ToolResult.empty(
            "No movies match those filters.", binding=binding
        )

    # The true total always travels with the capped rows (OQ-007). A truncated view must
    # never be able to pass for the complete answer.
    shown = len(result.rows)
    message = f"{result.total} movies match; showing {shown}."
    if result.excluded_unknown:
        message += (
            f" {result.excluded_unknown} excluded from the ranking because "
            f"{query.sort_by} is unknown for them."
        )

    return ToolResult.ok_result(
        message,
        refs=result.refs,
        payload={
            "total": result.total,
            "shown": shown,
            "excluded_unknown": result.excluded_unknown,
        },
        meta={
            "filters": query.to_display(),
            "constraints": query.describe(),
            "rows": result.rows.to_dict(orient="records"),
            "columns": list(result.rows.columns),
        },
    )
