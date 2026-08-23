"""Fuzzy title matching and its confidence bands (R-040 – R-045).

The PDF's five acceptance cases are the core of this file, but the more important tests
are the ones asserting the system **refuses to guess** -- that is the actual requirement
(p3 §2, p5 §10), and it is the part a naive "return the best match" implementation gets
wrong while still passing the five happy cases.
"""

from __future__ import annotations

import pytest

from movieagent.retrieval.fuzzy import MatchOutcome
from movieagent.tools.base import Outcome, ToolContext
from movieagent.tools.fuzzy_movie_search import run as fuzzy_search


@pytest.fixture
def context(repo, matcher, settings) -> ToolContext:
    return ToolContext(
        settings=settings,
        repository=repo,
        matcher=matcher,
        index=None,  # type: ignore[arg-type]
        embedder=None,  # type: ignore[arg-type]
        documents=[],
    )


class TestPdfAcceptanceCases:
    """R-041: the five cases the PDF names explicitly."""

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("Avatar", "Avatar"),
            ("Avatr", "Avatar"),
            ("Intersteler", "Interstellar"),
            ("the dark knight rises", "The Dark Knight Rises"),
            ("Titanik", "Titanic"),
            ("jurasic park", "Jurassic Park"),
        ],
    )
    def test_resolves_to_expected_title(self, matcher, query: str, expected: str) -> None:
        match = matcher.match(query)
        assert match.outcome is MatchOutcome.MATCH, f"{query!r}: {match.reason}"
        assert match.best is not None
        assert match.best.ref.title == expected

    def test_lord_of_the_rings_is_ambiguous_not_arbitrary(self, matcher) -> None:
        """The fifth PDF case, and the one that matters most.

        "lord of the rings" is a partial title shared by three films. Returning any one
        of them would satisfy a naive reading and violate the actual instruction.
        """
        match = matcher.match("lord of the rings")
        assert match.outcome is MatchOutcome.AMBIGUOUS
        assert len(match.candidates) >= 2
        titles = " ".join(c.ref.title for c in match.candidates)
        assert "Lord of the Rings" in titles


class TestRefusalToGuess:
    def test_near_tie_forces_ambiguity_regardless_of_score(self, matcher, settings) -> None:
        """The rule that does the real work (ADR-0009).

        A floor alone does not prevent arbitrariness: returning film A at 90 when film B
        scored 89 is exactly the failure the PDF describes. A clear winner must be clear.
        """
        match = matcher.match("lord of the rings")
        assert match.outcome is MatchOutcome.AMBIGUOUS
        assert match.best is None, "an ambiguous match must not nominate a winner"
        top_two = match.candidates[:2]
        assert (top_two[0].score - top_two[1].score) <= settings.fuzzy.tie_margin

    def test_garbage_is_not_found_not_a_wrong_answer(self, matcher) -> None:
        match = matcher.match("zzzqqq not a movie at all xyzzy")
        assert match.outcome is MatchOutcome.NOT_FOUND

    def test_empty_input_is_handled(self, matcher) -> None:
        assert matcher.match("").outcome is MatchOutcome.NOT_FOUND
        assert matcher.match("   ").outcome is MatchOutcome.NOT_FOUND

    @pytest.mark.parametrize("query", ["the", "a", "zzz"])
    def test_very_short_input_is_refused(self, matcher, query: str) -> None:
        """ADR-0025's length guard.

        ``ratio("the", "They")`` is 86 -- correct for edit distance, since at three
        characters one edit is most of the string. No threshold fixes that; refusing to
        match approximately below four characters does.
        """
        match = matcher.match(query)
        assert match.outcome is MatchOutcome.NOT_FOUND
        assert "too short" in match.reason

    def test_short_titles_still_resolve_exactly(self, matcher) -> None:
        """The length guard must not break real two-letter titles.

        The exact short-circuit runs first, so "Up" resolves while "the" does not.
        """
        match = matcher.match("Up")
        assert match.outcome is MatchOutcome.MATCH
        assert match.best is not None and match.best.ref.title == "Up"

    @pytest.mark.parametrize(
        "query",
        [
            "qqqzzz nonexistent film",
            "asdfghjkl",
            "a movie about stuff",
            "Låt den rätte komma in",
        ],
    )
    def test_unmatched_input_scores_far_below_the_bands(self, matcher, query: str) -> None:
        """The evidence behind ADR-0025's thresholds.

        The last case is a real film that is genuinely absent from TMDB 5000 -- exactly
        the situation where a confident wrong answer would be worst.
        """
        match = matcher.match(query)
        assert match.outcome is MatchOutcome.NOT_FOUND


class TestToolEnvelope:
    def test_match_returns_ok_with_one_ref(self, context) -> None:
        result = fuzzy_search(context, "Intersteler")
        assert result.status is Outcome.OK
        assert len(result.refs) == 1
        assert result.refs[0].title == "Interstellar"
        assert result.meta["score"] >= 75

    def test_ambiguous_returns_candidates_for_the_agent_to_ask_with(self, context) -> None:
        """R-043: ambiguity must reach the agent as *data it can act on*."""
        result = fuzzy_search(context, "lord of the rings")
        assert result.status is Outcome.AMBIGUOUS
        assert len(result.payload["candidates"]) >= 2
        assert "ask the user" in result.message.lower()
        # The scores must be visible so the trace can explain the refusal.
        assert all("score" in c for c in result.payload["candidates"])

    def test_not_found_is_distinct_from_ambiguous(self, context) -> None:
        """R-045: two different states, not one."""
        result = fuzzy_search(context, "qqqzzz nonexistent film")
        assert result.status is Outcome.NOT_FOUND
        assert result.refs == []

    def test_original_title_is_searchable(self, matcher) -> None:
        """Non-English films are often searched by their original name.

        `Amélie`'s stored title is the English one; `Le Fabuleux Destin d'Amélie Poulain`
        is its `original_title`, so this only passes because both columns are indexed.
        """
        match = matcher.match("Le Fabuleux Destin d'Amelie Poulain")
        assert match.outcome is MatchOutcome.MATCH
        assert match.best is not None
        assert match.best.matched_on == "original_title"


class TestPersonResolution:
    def test_misspelled_director_resolves(self, matcher) -> None:
        """R-084: entity resolution before a structured aggregate."""
        assert "Christopher Nolan" in matcher.resolve_person("Christofer Nolan")

    def test_partial_name_resolves(self, matcher) -> None:
        assert "Quentin Tarantino" in matcher.resolve_person("Tarantino")


class TestResolvedPayload:
    """What the tool returns is a reference, and it says so (R-060).

    It used to return the match under a ``record`` key. The UI draws a detail card for
    any ``record``, so "Tell me about Intersteler" rendered an empty Interstellar card
    -- every field "unknown", because a reference has no fields -- directly above the
    real one from ``movie_details``.
    """

    def test_a_match_is_reported_as_resolved_not_as_a_record(self, context) -> None:
        from movieagent.ui.components import is_record

        result = fuzzy_search(context, title="Intersteler")
        assert result.payload["resolved"]["title"] == "Interstellar"
        assert "record" not in result.payload
        assert not is_record(result.payload["resolved"])
