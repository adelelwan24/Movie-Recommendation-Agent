"""The twelve PDF example queries, as executable specifications (R-030 – R-039).

Every one of these runs with **no LLM anywhere**. That is the point of ADR-0004's filter
DSL: the queries are values, so determinism (R-030) is something a test can assert rather
than something we hope the model preserves.

Query labels Q1–Q12 match `docs/REQUIREMENTS.md` §3.
"""

from __future__ import annotations

import pytest

from movieagent.data import schema as S
from movieagent.data.query import (
    AggregateMetric,
    AggregateSpec,
    ComparisonOp,
    Condition,
    GroupBy,
    NumericField,
    SearchQuery,
)
from movieagent.tools.base import Outcome, ToolContext
from movieagent.tools.structured_search import run as structured_search


@pytest.fixture
def context(repo, matcher, settings) -> ToolContext:
    """Structured search needs no index or embedder, so they stay None."""
    return ToolContext(
        settings=settings,
        repository=repo,
        matcher=matcher,
        index=None,  # type: ignore[arg-type]
        embedder=None,  # type: ignore[arg-type]
        documents=[],
    )


def condition(field: NumericField, op: ComparisonOp, value: float) -> Condition:
    return Condition(field=field, op=op, value=value)


class TestAggregations:
    def test_q1_movies_per_genre(self, context) -> None:
        result = structured_search(
            context,
            SearchQuery(aggregate=AggregateSpec(group_by=GroupBy.GENRES, limit=100)),
        )
        assert result.status is Outcome.OK
        rows = {r["genre_names"]: r["movie_count"] for r in result.payload["aggregate"]}
        assert rows["Drama"] > rows["Western"]
        # A film in three genres counts once per genre, so the total exceeds the corpus.
        assert sum(rows.values()) > len(context.repository)

    def test_q2_ten_most_common_genres(self, context) -> None:
        result = structured_search(
            context,
            SearchQuery(aggregate=AggregateSpec(group_by=GroupBy.GENRES, limit=10)),
        )
        rows = result.payload["aggregate"]
        assert len(rows) == 10
        assert rows[0]["genre_names"] == "Drama"
        counts = [r["movie_count"] for r in rows]
        assert counts == sorted(counts, reverse=True)

    def test_q3_companies_with_most_movies(self, context) -> None:
        result = structured_search(
            context,
            SearchQuery(aggregate=AggregateSpec(group_by=GroupBy.COMPANIES, limit=10)),
        )
        names = [r["company_names"] for r in result.payload["aggregate"]]
        assert "Warner Bros." in names
        assert len(names) == 10

    def test_q4_movies_per_release_year(self, context) -> None:
        result = structured_search(
            context,
            SearchQuery(
                aggregate=AggregateSpec(group_by=GroupBy.RELEASE_YEAR, limit=200, sort_desc=False)
            ),
        )
        years = [int(r["release_year"]) for r in result.payload["aggregate"]]
        assert min(years) < 1930
        assert max(years) >= 2016


class TestFiltersAndSorting:
    def test_q5_top_ten_by_revenue_excludes_unknowns(self, context) -> None:
        """R-016 in action: unknown revenue must not be ranked as zero."""
        result = structured_search(
            context, SearchQuery(sort_by=NumericField.REVENUE, sort_desc=True, limit=10)
        )
        assert result.status is Outcome.OK
        assert len(result.refs) == 10
        assert result.refs[0].title == "Avatar"
        assert result.payload["excluded_unknown"] > 1_000

    def test_q6_rated_above_eight(self, context) -> None:
        query = SearchQuery(
            conditions=[condition(NumericField.VOTE_AVERAGE, ComparisonOp.GT, 8)]
        )
        result = structured_search(context, query)
        assert result.status is Outcome.OK
        # The cap must never masquerade as the whole answer (OQ-007).
        assert result.payload["total"] > result.payload["shown"]
        frame = context.repository.frame
        matched = frame[frame[S.VOTE_AVERAGE] > 8]
        assert result.payload["total"] == len(matched)

    def test_q7_longer_than_150_minutes(self, context) -> None:
        result = structured_search(
            context,
            SearchQuery(conditions=[condition(NumericField.RUNTIME, ComparisonOp.GT, 150)]),
        )
        frame = context.repository.frame
        expected = int((frame[S.RUNTIME] > 150).sum())
        assert result.payload["total"] == expected
        # R-014: the 37 films with unknown runtime satisfy no comparison.
        assert expected < len(frame) - int(frame[S.RUNTIME].isna().sum())

    def test_q8_after_2010_and_rated_above_7_5(self, context) -> None:
        """OQ-012: "after 2010" is strict, so 2011 onward."""
        query = SearchQuery(
            year_from=2011,
            conditions=[condition(NumericField.VOTE_AVERAGE, ComparisonOp.GT, 7.5)],
        )
        result = structured_search(context, query)
        assert result.status is Outcome.OK
        frame = context.repository.frame
        for ref in result.refs:
            row = frame[frame[S.ID] == ref.movie_id].iloc[0]
            assert row[S.RELEASE_YEAR] >= 2011
            assert row[S.VOTE_AVERAGE] > 7.5

    def test_q9_action_with_over_5000_votes(self, context) -> None:
        query = SearchQuery(
            genres=["Action"],
            conditions=[condition(NumericField.VOTE_COUNT, ComparisonOp.GT, 5000)],
        )
        result = structured_search(context, query)
        assert result.status is Outcome.OK
        frame = context.repository.frame
        for ref in result.refs:
            row = frame[frame[S.ID] == ref.movie_id].iloc[0]
            assert "Action" in row[S.GENRES]
            assert row[S.VOTE_COUNT] > 5000

    def test_q10_budget_above_100_million(self, context) -> None:
        result = structured_search(
            context,
            SearchQuery(
                conditions=[condition(NumericField.BUDGET, ComparisonOp.GT, 100_000_000)]
            ),
        )
        assert result.status is Outcome.OK
        assert result.payload["total"] > 50

    def test_q11_action_after_2010_rated_above_7_5(self, context) -> None:
        query = SearchQuery(
            genres=["Action"],
            year_from=2011,
            conditions=[condition(NumericField.VOTE_AVERAGE, ComparisonOp.GT, 7.5)],
        )
        result = structured_search(context, query)
        assert result.status is Outcome.OK
        titles = {ref.title for ref in result.refs}
        assert "The Dark Knight Rises" in titles

    def test_q12_top_scifi_with_1000_votes(self, context) -> None:
        query = SearchQuery(
            genres=["Science Fiction"],
            conditions=[condition(NumericField.VOTE_COUNT, ComparisonOp.GTE, 1000)],
            sort_by=NumericField.VOTE_AVERAGE,
            sort_desc=True,
            limit=10,
        )
        result = structured_search(context, query)
        assert len(result.refs) == 10
        assert "Inception" in {ref.title for ref in result.refs}

    def test_director_aggregate_for_nolan(self, context) -> None:
        """R-084's worked example: entity resolution into a structured aggregate."""
        result = structured_search(context, SearchQuery(directors=["Christopher Nolan"]))
        assert result.status is Outcome.OK
        assert result.payload["total"] == 8


class TestErrorPaths:
    def test_unknown_genre_is_rejected_with_a_suggestion(self, context) -> None:
        """R-038. The failure that actually bites is a *plausible* wrong term."""
        result = structured_search(context, SearchQuery(genres=["Sci-Fi"]))
        assert result.status is Outcome.INVALID_INPUT
        assert any("Science Fiction" in error for error in result.payload["errors"])

    def test_unknown_director_is_rejected(self, context) -> None:
        result = structured_search(context, SearchQuery(directors=["Christofer Nolan"]))
        assert result.status is Outcome.INVALID_INPUT
        assert "Christopher Nolan" in result.payload["errors"][0]

    def test_impossible_filter_is_empty_not_error(self, context) -> None:
        """R-039: empty and invalid are different states, and must stay different."""
        result = structured_search(
            context,
            SearchQuery(conditions=[condition(NumericField.VOTE_AVERAGE, ComparisonOp.GT, 10)]),
        )
        assert result.status is Outcome.EMPTY
        assert result.payload["binding_constraints"]

    def test_condition_schema_rejects_bad_shapes(self) -> None:
        """Structural validation happens before the data is ever touched."""
        with pytest.raises(ValueError):
            Condition(field=NumericField.RUNTIME, op=ComparisonOp.BETWEEN, value=90)
        with pytest.raises(ValueError):
            Condition(field=NumericField.RUNTIME, op=ComparisonOp.GT, value=[1, 2])
        with pytest.raises(ValueError):
            SearchQuery(year_from=2015, year_to=2010)


class TestFilterMerging:
    def test_same_field_replaces_rather_than_accumulates(self) -> None:
        """R-148 / ADR-0010: "above 7.5" then "above 8" means > 8, not both."""
        first = SearchQuery(
            conditions=[condition(NumericField.VOTE_AVERAGE, ComparisonOp.GT, 7.5)],
            genres=["Science Fiction"],
        )
        second = SearchQuery(
            conditions=[condition(NumericField.VOTE_AVERAGE, ComparisonOp.GT, 8)]
        )
        merged = first.merged_with(second)
        assert len(merged.conditions) == 1
        assert merged.conditions[0].value == 8
        # The genre must survive: the user only refined the rating.
        assert merged.genres == ["Science Fiction"]

    def test_new_field_is_anded(self) -> None:
        first = SearchQuery(year_from=2011)
        second = SearchQuery(
            conditions=[condition(NumericField.RUNTIME, ComparisonOp.LT, 120)]
        )
        merged = first.merged_with(second)
        assert merged.year_from == 2011
        assert len(merged.conditions) == 1

    def test_carried_filters_re_execute_over_the_whole_dataset(self, context) -> None:
        """The requester's OQ-006 decision, verified end to end.

        Turn 1 displays 25 of N. Turn 2 must re-query the *full* matching set, not the
        25 rows that happened to be shown -- otherwise a display cap silently truncates
        the answer.
        """
        turn1 = SearchQuery(genres=["Science Fiction"], year_from=2011, limit=25)
        first = structured_search(context, turn1)
        assert first.payload["total"] > first.payload["shown"]

        turn2 = turn1.merged_with(
            SearchQuery(conditions=[condition(NumericField.VOTE_AVERAGE, ComparisonOp.GT, 7.5)])
        )
        second = structured_search(context, turn2)
        assert second.status is Outcome.OK

        frame = context.repository.frame
        expected = frame[
            frame[S.GENRES].map(lambda g: "Science Fiction" in g)
            & (frame[S.RELEASE_YEAR] >= 2011)
            & (frame[S.VOTE_AVERAGE] > 7.5)
        ]
        assert second.payload["total"] == len(expected)
