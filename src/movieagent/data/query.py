"""The structured filter DSL (ADR-0004).

A closed, validated query object rather than model-generated SQL. Three requirements
pushed the same way:

* **R-038** -- an invalid filter must produce a clear, actionable error. Pydantic's
  ``ValidationError`` already is one, and it feeds straight back to the model.
* **R-103** -- the UI must show the filters applied. This object *is* that display.
* **R-148** -- follow-up turns carry the filter set forward and re-execute. A structured
  value merges; a SQL string would need rewriting.

The expressive ceiling is real and deliberate: conditions are ANDed, there is no OR
across fields and no nested boolean groups. None of the PDF's twelve example queries
need them. See ADR-0004's "when to revisit".

Lives under ``data/`` rather than ``tools/`` (where ADR-0004 sketched it) because the
repository consumes it to build masks -- putting it in ``tools/`` would invert the
layering that ADR-0019 enforces.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from movieagent.data.schema import ListField, NumericField


class ComparisonOp(StrEnum):
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    EQ = "eq"
    BETWEEN = "between"


class Condition(BaseModel):
    """One numeric comparison. ``field`` is a closed enum, so an unknown field is a
    validation error the agent can act on rather than a silent empty result."""

    # `populate_by_name` so the model may emit either "field" (the natural name, and
    # what the JSON schema advertises) or "field_name" without a validation failure.
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    field_name: NumericField = Field(alias="field", description="Numeric field to compare")
    op: ComparisonOp
    value: float | list[float] = Field(description="Scalar, or [low, high] when op=between")

    @model_validator(mode="after")
    def _check_value_shape(self) -> Self:
        if self.op is ComparisonOp.BETWEEN:
            if not isinstance(self.value, list) or len(self.value) != 2:
                raise ValueError("op=between requires value=[low, high]")
            if self.value[0] > self.value[1]:
                raise ValueError("op=between requires low <= high")
        elif isinstance(self.value, list):
            raise ValueError(f"op={self.op} requires a scalar value, not a list")
        return self

    def describe(self) -> str:
        symbols = {
            ComparisonOp.GT: ">",
            ComparisonOp.GTE: ">=",
            ComparisonOp.LT: "<",
            ComparisonOp.LTE: "<=",
            ComparisonOp.EQ: "=",
        }
        if self.op is ComparisonOp.BETWEEN:
            low, high = self.value  # type: ignore[misc]
            return f"{self.field_name} between {low} and {high}"
        return f"{self.field_name} {symbols[self.op]} {self.value}"


class AggregateMetric(StrEnum):
    COUNT = "count"
    AVG_RATING = "avg_rating"
    SUM_REVENUE = "sum_revenue"


class GroupBy(StrEnum):
    """What an aggregation can group by.

    List fields are exploded first; ``release_year`` is a scalar. These are the only
    groupings the PDF's aggregation questions need (movies per genre, most common
    genres, companies with most movies, movies per year).
    """

    GENRES = ListField.GENRES.value
    KEYWORDS = ListField.KEYWORDS.value
    COMPANIES = ListField.COMPANIES.value
    COUNTRIES = ListField.COUNTRIES.value
    LANGUAGES = ListField.LANGUAGES.value
    CAST = ListField.CAST.value
    DIRECTORS = ListField.DIRECTORS.value
    RELEASE_YEAR = "release_year"


class AggregateSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    group_by: GroupBy
    metric: AggregateMetric = AggregateMetric.COUNT
    limit: int = Field(default=25, ge=1, le=500)
    sort_desc: bool = True

    def describe(self) -> str:
        return f"{self.metric} by {self.group_by} (top {self.limit})"


class SearchQuery(BaseModel):
    """A validated, mergeable description of a structured query.

    Also doubles as the hybrid pre-filter (ADR-0011) and as the unit of conversational
    memory (ADR-0022) -- one object, three jobs, which is why its shape is load-bearing.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    conditions: list[Condition] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    companies: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    people: list[str] = Field(
        default_factory=list, description="Matches cast or director"
    )
    directors: list[str] = Field(default_factory=list)
    year_from: int | None = Field(default=None, ge=1870, le=2100)
    year_to: int | None = Field(default=None, ge=1870, le=2100)
    sort_by: NumericField | None = None
    sort_desc: bool = True
    limit: int = Field(default=25, ge=1, le=500)
    aggregate: AggregateSpec | None = None

    @model_validator(mode="after")
    def _check_years(self) -> Self:
        if self.year_from is not None and self.year_to is not None:
            if self.year_from > self.year_to:
                raise ValueError("year_from must not exceed year_to")
        return self

    # ---------------------------------------------------------------- introspection

    def is_empty(self) -> bool:
        """True when nothing constrains the result set. Used to decide whether a
        'filter' is really a filter or just a default."""
        return not any(
            (
                self.conditions,
                self.genres,
                self.keywords,
                self.companies,
                self.countries,
                self.languages,
                self.people,
                self.directors,
                self.year_from is not None,
                self.year_to is not None,
                self.aggregate is not None,
            )
        )

    def describe(self) -> list[str]:
        """Human-readable constraint list for the trace panel (R-103) and for naming
        the binding constraints when a result set comes back empty (R-039)."""
        parts: list[str] = []
        for condition in self.conditions:
            parts.append(condition.describe())
        if self.year_from is not None:
            parts.append(f"release_year >= {self.year_from}")
        if self.year_to is not None:
            parts.append(f"release_year <= {self.year_to}")
        for label, values in (
            ("genre", self.genres),
            ("keyword", self.keywords),
            ("company", self.companies),
            ("country", self.countries),
            ("language", self.languages),
            ("person", self.people),
            ("director", self.directors),
        ):
            if values:
                parts.append(f"{label} in {values}")
        if self.aggregate is not None:
            parts.append(self.aggregate.describe())
        if self.sort_by is not None:
            parts.append(f"sort by {self.sort_by} {'desc' if self.sort_desc else 'asc'}")
        return parts

    # ---------------------------------------------------------------------- merging

    def merged_with(self, other: SearchQuery) -> SearchQuery:
        """Carry this query forward and layer ``other`` on top (R-148, ADR-0022).

        Merge rules, stated once and implemented here because this is the *only* write
        path -- the reducer calls it, so it cannot be skipped:

        * a new condition on an **already-constrained field replaces** the old one
          ("above 7.5" then "above 8" gives ``> 8``, not both);
        * a new condition on a **new field is added** (AND);
        * a non-empty list filter **replaces** its counterpart rather than unioning --
          "actually, comedies" means comedies, not sci-fi *and* comedies;
        * scalars (sort, limit, aggregate) are overridden only when explicitly set.

        A *fresh topic* is not handled here: the planner signals it and the reducer
        drops the old query entirely, because otherwise filters accumulate forever and
        turn nine returns nothing.
        """
        replaced = {c.field_name for c in other.conditions}
        conditions = [c for c in self.conditions if c.field_name not in replaced]
        conditions.extend(other.conditions)

        def pick(new: list[str], old: list[str]) -> list[str]:
            return list(new) if new else list(old)

        return SearchQuery(
            conditions=conditions,
            genres=pick(other.genres, self.genres),
            keywords=pick(other.keywords, self.keywords),
            companies=pick(other.companies, self.companies),
            countries=pick(other.countries, self.countries),
            languages=pick(other.languages, self.languages),
            people=pick(other.people, self.people),
            directors=pick(other.directors, self.directors),
            year_from=other.year_from if other.year_from is not None else self.year_from,
            year_to=other.year_to if other.year_to is not None else self.year_to,
            sort_by=other.sort_by if other.sort_by is not None else self.sort_by,
            sort_desc=other.sort_desc,
            limit=other.limit,
            aggregate=other.aggregate if other.aggregate is not None else self.aggregate,
        )

    def to_display(self) -> dict[str, Any]:
        """Only the fields actually set -- defaults would be noise in the trace."""
        return self.model_dump(exclude_defaults=True, by_alias=True, mode="json")
