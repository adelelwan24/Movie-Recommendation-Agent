"""The processed movie schema and its typed records (ADR-0005).

Every derived column here maps to a requirement. R-019 says explicitly *not* to
normalize every possible field, so anything the system does not consume is absent by
design rather than by oversight.

Dtype choices are requirements, not preferences:

* nullable ``Int64`` for ``runtime``/``vote_count``/``budget``/``revenue`` so a missing
  value is ``NA`` and never ``0`` (R-014);
* real dates for ``release_date`` (R-015);
* ``budget``/``revenue`` of 0 stored as ``NA`` with companion ``*_known`` flags, so the
  UI can honestly say "unknown" and aggregations can report coverage (R-016).

Parquet is what preserves these across reloads; CSV would silently undo them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

# --------------------------------------------------------------------------- columns

ID = "id"
TITLE = "title"
ORIGINAL_TITLE = "original_title"
TITLE_NORM = "title_norm"
ORIGINAL_TITLE_NORM = "original_title_norm"

OVERVIEW = "overview"
TAGLINE = "tagline"

GENRES = "genre_names"
KEYWORDS = "keyword_names"
COMPANIES = "company_names"
COUNTRIES = "country_names"
LANGUAGES = "language_names"
TOP_CAST = "top_cast"
FULL_CAST = "full_cast"
DIRECTORS = "directors"

VOTE_AVERAGE = "vote_average"
VOTE_COUNT = "vote_count"
POPULARITY = "popularity"
RUNTIME = "runtime"
BUDGET = "budget"
REVENUE = "revenue"
BUDGET_KNOWN = "budget_known"
REVENUE_KNOWN = "revenue_known"

RELEASE_DATE = "release_date"
RELEASE_YEAR = "release_year"
ORIGINAL_LANGUAGE = "original_language"
STATUS = "status"
HOMEPAGE = "homepage"

#: List-valued columns, in the shape the aggregation and filter paths expect.
LIST_COLUMNS: tuple[str, ...] = (
    GENRES,
    KEYWORDS,
    COMPANIES,
    COUNTRIES,
    LANGUAGES,
    TOP_CAST,
    FULL_CAST,
    DIRECTORS,
)

#: Cast members kept in the embedded document (ADR-0008 / OQ-014). The full cast stays
#: in metadata so structured search can still answer "which films star X".
TOP_CAST_SIZE = 5


class NumericField(StrEnum):
    """Fields a structured condition may compare against (ADR-0004).

    A closed enum is the point: an unknown field is a validation error the agent can
    act on (R-038), not a silent empty result.
    """

    VOTE_AVERAGE = VOTE_AVERAGE
    VOTE_COUNT = VOTE_COUNT
    POPULARITY = POPULARITY
    RUNTIME = RUNTIME
    BUDGET = BUDGET
    REVENUE = REVENUE
    RELEASE_YEAR = RELEASE_YEAR


class ListField(StrEnum):
    """Fields that can be grouped over or filtered by membership."""

    GENRES = GENRES
    KEYWORDS = KEYWORDS
    COMPANIES = COMPANIES
    COUNTRIES = COUNTRIES
    LANGUAGES = LANGUAGES
    CAST = FULL_CAST
    DIRECTORS = DIRECTORS


#: Fields whose zero value means "unknown", never "zero" (R-016).
UNKNOWN_WHEN_ZERO: tuple[str, ...] = (BUDGET, REVENUE)


# --------------------------------------------------------------------------- records


@dataclass(frozen=True, slots=True)
class MovieRef:
    """A lightweight reference: the unit of conversational memory (ADR-0022).

    Deliberately small. Result *rows* are not kept in graph state -- only these -- so
    the checkpointer does not accumulate full payloads across a long session.
    """

    movie_id: int
    title: str
    year: int | None = None

    def label(self) -> str:
        return f"{self.title} ({self.year})" if self.year else self.title

    def to_dict(self) -> dict[str, Any]:
        return {"movie_id": self.movie_id, "title": self.title, "year": self.year}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MovieRef:
        return cls(
            movie_id=int(raw["movie_id"]),
            title=str(raw["title"]),
            year=int(raw["year"]) if raw.get("year") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class FuzzyCandidate:
    """One scored title match (ADR-0009). Scores are surfaced in the trace so the user
    can see *why* the system asked for clarification rather than guessing."""

    ref: MovieRef
    score: float
    matched_on: str  # "title" or "original_title"

    def to_dict(self) -> dict[str, Any]:
        return {**self.ref.to_dict(), "score": round(self.score, 1), "matched_on": self.matched_on}


@dataclass(frozen=True, slots=True)
class RetrievedDoc:
    """One semantic hit: the reference, its similarity, and the document text that was
    actually embedded -- which is also what the UI shows as "retrieved context" (R-103)
    and what grounds generation, so there is no second rendering path to drift."""

    ref: MovieRef
    score: float
    document: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.ref.to_dict(),
            "score": round(self.score, 4),
            "document": self.document,
        }
