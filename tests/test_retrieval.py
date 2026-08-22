"""Semantic retrieval, hybrid pre-filtering and document construction.

Covers R-050 – R-057 and R-070. The golden-set assertions are marked ``model_coupled``:
they depend on the pinned embedding model, so a legitimate model upgrade will redden them
and that has to be triaged rather than trusted (ADR-0024).
"""

from __future__ import annotations

import numpy as np
import pytest

from movieagent.data import schema as S
from movieagent.data.query import ComparisonOp, Condition, NumericField, SearchQuery
from movieagent.retrieval.documents import build_document
from movieagent.retrieval.vector_index import VectorIndex
from movieagent.tools.base import Outcome
from movieagent.tools.semantic_search import run as semantic_search


@pytest.fixture
def context(runtime):
    return runtime.tool_context()


class TestDocumentConstruction:
    def test_document_is_labelled_not_a_raw_row(self, repo) -> None:
        """R-053: embedding raw CSV rows is explicitly not acceptable."""
        row = repo.frame[repo.frame[S.TITLE] == "Interstellar"].iloc[0]
        document = build_document(row)
        assert document.startswith("Title: Interstellar (2014)")
        for label in ("Genres:", "Keywords:", "Director:", "Starring:", "Overview:"):
            assert label in document
        assert "Christopher Nolan" in document

    def test_metadata_fields_are_excluded_from_the_document(self, repo) -> None:
        """ADR-0008's field allocation: embed what is thematic, filter what is exact.

        Production companies and countries are kept as metadata precisely so they can be
        filtered exactly rather than matched fuzzily -- and so they do not dilute the
        plot signal the R-051 queries depend on.
        """
        row = repo.frame[repo.frame[S.TITLE] == "Interstellar"].iloc[0]
        document = build_document(row)
        assert "Paramount" not in document
        assert "United States of America" not in document
        assert "vote_average" not in document

    def test_empty_fields_are_omitted_not_left_blank(self, repo) -> None:
        """A bare "Tagline:" with nothing after it is noise every document would share."""
        no_tagline = repo.frame[repo.frame[S.TAGLINE].isna()]
        if no_tagline.empty:
            pytest.skip("every movie has a tagline")
        document = build_document(no_tagline.iloc[0])
        assert "Tagline:" not in document

    def test_one_document_per_movie(self, runtime) -> None:
        """R-055: no artificial chunking of short overviews."""
        assert len(runtime.documents) == len(runtime.repository)
        assert len(runtime.index) == len(runtime.repository)


class TestVectorIndex:
    def test_rows_map_back_to_the_right_movie(self, runtime) -> None:
        """The id<->row mapping is load-bearing.

        An off-by-one here returns confidently wrong movies, which is exactly the failure
        R-004 exists to prevent -- and it would look completely plausible.
        """
        for movie_id in runtime.repository.frame[S.ID].head(50):
            position = runtime.repository.position(int(movie_id))
            assert position is not None
            assert runtime.repository.refs_at(np.array([position]))[0].movie_id == movie_id

    def test_a_movie_is_its_own_nearest_neighbour(self, runtime, vectors) -> None:
        """Sanity check on normalization: self-similarity must be ~1.0."""
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
        hits = runtime.index.search(vectors[10], k=1)
        assert hits[0].position == 10
        assert hits[0].score == pytest.approx(1.0, abs=1e-4)

    def test_similar_to_excludes_the_seed(self, runtime) -> None:
        """"Films like Inception" answering "Inception" is correct and useless."""
        position = runtime.repository.position(27205)  # Inception
        hits = runtime.index.similar_to(position, k=5)
        assert all(hit.position != position for hit in hits)
        assert len(hits) == 5

    def test_dimension_mismatch_is_refused(self) -> None:
        """ADR-0007's two backends produce incompatible spaces; a silent mismatch would
        return confident nonsense."""
        index = VectorIndex(np.eye(4, dtype=np.float32))
        with pytest.raises(ValueError, match="dimension"):
            index.search(np.ones(8, dtype=np.float32), k=1)


class TestHybridSearch:
    """R-070: semantic intent combined with hard constraints."""

    def test_filters_are_applied_before_ranking(self, context) -> None:
        """ADR-0011's guarantee: every result satisfies every constraint."""
        filters = SearchQuery(
            genres=["Science Fiction"],
            year_from=2011,
            conditions=[Condition(field=NumericField.RUNTIME, op=ComparisonOp.LT, value=120)],
        )
        result = semantic_search(context, query="funny science fiction", filters=filters, k=8)
        assert result.status is Outcome.OK

        frame = context.repository.frame
        for ref in result.refs:
            row = frame[frame[S.ID] == ref.movie_id].iloc[0]
            assert "Science Fiction" in row[S.GENRES]
            assert row[S.RELEASE_YEAR] >= 2011
            assert row[S.RUNTIME] < 120

    def test_returns_k_results_when_the_pool_allows(self, context) -> None:
        """The reason pre-filtering was chosen over post-filtering.

        Post-filtering would return 3 of a requested 8 and look like scarcity rather
        than filtering.
        """
        filters = SearchQuery(genres=["Drama"])
        result = semantic_search(context, query="family conflict", filters=filters, k=8)
        assert len(result.refs) == 8
        assert result.payload["pool_size"] > 8

    def test_pool_size_is_reported(self, context) -> None:
        """Honest signal: ranking 8 of 12 is close to arbitrary, and the user sees that."""
        filters = SearchQuery(genres=["Western"], year_from=2011)
        result = semantic_search(context, query="revenge", filters=filters, k=8)
        assert "pool_size" in result.payload
        assert result.meta["pool_size"] == result.payload["pool_size"]

    def test_impossible_constraints_are_empty_not_low_confidence(self, context) -> None:
        filters = SearchQuery(
            genres=["Western"],
            conditions=[
                Condition(field=NumericField.VOTE_AVERAGE, op=ComparisonOp.GT, value=9.9)
            ],
        )
        result = semantic_search(context, query="revenge", filters=filters)
        assert result.status is Outcome.EMPTY
        assert result.payload["binding_constraints"]


class TestLowConfidence:
    """R-056, as revised by ADR-0026.

    The absolute cosine floor these tests originally used does not work: measured over
    this corpus, genuine queries top out at 0.575-0.730 and pure gibberish at
    0.566-0.629. The populations overlap, so no threshold separates them. Lexical
    coverage does.
    """

    @pytest.mark.parametrize(
        "query",
        [
            "qwertyuiop zxcvbnm asdfgh flurble wizzlewop",
            "florb glorb snorb",
            "blorp zonk quix vex",
        ],
    )
    def test_gibberish_reports_low_confidence(self, context, query: str) -> None:
        result = semantic_search(context, query=query, k=5)
        assert result.status is Outcome.LOW_CONFIDENCE
        assert result.payload["lexical_coverage"] < 0.4
        assert result.meta["unknown_words"]

    def test_gibberish_would_have_passed_the_similarity_floor(self, context) -> None:
        """The evidence for ADR-0026, pinned as a regression guard.

        If someone reinstates an absolute floor as the primary signal, this test shows
        why it fails: this gibberish scores *higher* than several genuine queries do.
        """
        vector = context.embedder.embed_query("qwertyuiop zxcvbnm asdfgh flurble wizzlewop")
        top = context.index.search(vector, k=1)[0].score
        assert top > context.settings.retrieval.similarity_floor

    @pytest.mark.parametrize(
        "query",
        [
            "a heist inside someone's dreams",
            "time travel and changing the past",
            "movies about the Rwandan genocide",
        ],
    )
    def test_real_queries_pass_both_checks(self, context, query: str) -> None:
        """Including one (Rwandan genocide) with a rare proper noun, which is where a
        coverage check is most at risk of a false positive."""
        result = semantic_search(context, query=query, k=5)
        assert result.status is Outcome.OK

    def test_valid_english_about_an_absent_topic_is_not_detected(self, context) -> None:
        """The honest limit of R-056 in this system (ADR-0026, R-126).

        The dataset ends in 2016. This query is fluent, fully in-vocabulary, and has no
        answer -- and nothing in the retrieval layer can tell. The films returned will
        look confident. Only the agent's prompt (which knows the cutoff) mitigates it.
        """
        result = semantic_search(context, query="the invasion of Ukraine in 2022", k=5)
        assert result.status is Outcome.OK  # not flagged -- documenting, not asserting good


class TestInvalidInput:
    def test_requires_a_query_or_a_seed(self, context) -> None:
        result = semantic_search(context)
        assert result.status is Outcome.INVALID_INPUT

    def test_unknown_seed_id_is_not_found(self, context) -> None:
        result = semantic_search(context, similar_to=999_999_999)
        assert result.status is Outcome.NOT_FOUND

    def test_invalid_filter_vocabulary_is_rejected(self, context) -> None:
        result = semantic_search(context, query="space", filters=SearchQuery(genres=["Sci-Fi"]))
        assert result.status is Outcome.INVALID_INPUT
