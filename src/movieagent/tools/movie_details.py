"""``movie_details`` -- the complete record for one movie (R-060, R-061).

Missing fields are reported as unknown rather than omitted or invented. That is R-061,
and it is also the second layer of ADR-0012's grounding: if the payload says
``"budget": null, "budget_known": false``, the model has nothing plausible to substitute.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from movieagent.data import schema as S
from movieagent.tools.base import Outcome, ToolContext, ToolResult

NAME = "movie_details"
DESCRIPTION = """\
Return the complete stored record for one movie, given its numeric id.

Use after fuzzy_movie_search has resolved a title, or on an id from a previous result
set. Returns cast, crew, genres, keywords, ratings, runtime, budget, revenue, release
date, companies, countries and the full overview.

Fields the dataset does not have are returned as null with a "known" flag -- report them
as unknown. Never fill one in from your own knowledge."""


def _scalar(value: Any) -> Any:
    """Convert pandas scalars to plain Python, preserving null as ``None``."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def _listed(value: Any) -> list[str]:
    if value is None:
        return []
    return [str(v) for v in value]


def run(context: ToolContext, movie_id: int) -> ToolResult:
    """Fetch one movie's full record."""
    row = context.repository.get(movie_id)
    if row is None:
        return ToolResult(
            status=Outcome.NOT_FOUND,
            message=(
                f"No movie with id {movie_id} in this dataset. "
                "Resolve the title with fuzzy_movie_search first."
            ),
        )

    ref = context.repository.ref(movie_id)
    assert ref is not None

    record: dict[str, Any] = {
        "id": int(row[S.ID]),
        "title": _scalar(row[S.TITLE]),
        "original_title": _scalar(row[S.ORIGINAL_TITLE]),
        "tagline": _scalar(row[S.TAGLINE]),
        "overview": _scalar(row[S.OVERVIEW]),
        "release_date": str(row[S.RELEASE_DATE]) if row[S.RELEASE_DATE] is not None else None,
        "release_year": _scalar(row[S.RELEASE_YEAR]),
        "runtime_minutes": _scalar(row[S.RUNTIME]),
        "genres": _listed(row[S.GENRES]),
        "keywords": _listed(row[S.KEYWORDS]),
        "director": _listed(row[S.DIRECTORS]),
        "top_cast": _listed(row[S.TOP_CAST]),
        "full_cast": _listed(row[S.FULL_CAST]),
        "production_companies": _listed(row[S.COMPANIES]),
        "production_countries": _listed(row[S.COUNTRIES]),
        "spoken_languages": _listed(row[S.LANGUAGES]),
        "original_language": _scalar(row[S.ORIGINAL_LANGUAGE]),
        "vote_average": _scalar(row[S.VOTE_AVERAGE]),
        "vote_count": _scalar(row[S.VOTE_COUNT]),
        "popularity": _scalar(row[S.POPULARITY]),
        "budget_usd": _scalar(row[S.BUDGET]),
        "budget_known": bool(_scalar(row[S.BUDGET_KNOWN])),
        "revenue_usd": _scalar(row[S.REVENUE]),
        "revenue_known": bool(_scalar(row[S.REVENUE_KNOWN])),
        "status": _scalar(row[S.STATUS]),
        "homepage": _scalar(row[S.HOMEPAGE]),
    }

    unknown = sorted(
        key
        for key, value in record.items()
        if value in (None, [], "") and not key.endswith("_known")
    )
    message = f"Full record for {ref.label()}."
    if unknown:
        message += f" Unknown in this dataset: {', '.join(unknown)}."

    return ToolResult.ok_result(
        message,
        refs=[ref],
        payload={"record": record},
        meta={"unknown_fields": unknown},
    )
