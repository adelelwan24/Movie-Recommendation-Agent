"""Backend parity: the Qdrant collection must answer exactly like the numpy index.

Tier 2 in ADR-0024's scheme -- no network, no API key. The Qdrant tests skip when the
collection has not been built, the same way the golden retrieval tests skip without the
vector index.

The premise of these tests is that ``MovieRepository.mask_for`` is the *definition* of
what a filter means and ``qdrant_filters`` is a second implementation of it. Two
implementations of one definition drift silently unless something compares them, so the
comparison is the test: same pool, same ids, same order. A filter-translation bug does
not raise -- it quietly returns plausible movies that violate a constraint the user
asked for, which is the R-004 failure mode the whole system is shaped to avoid.
"""

from __future__ import annotations

import numpy as np
import pytest

from movieagent.config import Settings, VectorBackend
from movieagent.data import schema as S
from movieagent.data.query import ComparisonOp, Condition, SearchQuery
from movieagent.data.schema import NumericField
from movieagent.errors import ArtifactError
from movieagent.retrieval.backend import Restriction
from movieagent.retrieval.qdrant_filters import to_qdrant_filter
from movieagent.retrieval.vector_index import VectorIndex

pytest.importorskip("qdrant_client", reason="qdrant-client not installed")

from movieagent.retrieval.qdrant_index import QdrantIndex  # noqa: E402

#: Every translation rule in ``qdrant_filters`` gets at least one case here.
FILTER_CASES: tuple[tuple[str, SearchQuery], ...] = (
    ("single genre", SearchQuery(genres=["Drama"])),
    ("case-insensitive genre", SearchQuery(genres=["sCiEnCe FiCtIoN"])),
    ("multi-value membership", SearchQuery(genres=["Horror", "Thriller"])),
    ("year range", SearchQuery(year_from=2011, year_to=2014)),
    (
        "runtime + genre + year",
        SearchQuery(
            genres=["Science Fiction"],
            year_from=2011,
            conditions=[Condition(field=NumericField.RUNTIME, op=ComparisonOp.LT, value=120)],
        ),
    ),
    (
        "vote thresholds",
        SearchQuery(
            conditions=[
                Condition(field=NumericField.VOTE_AVERAGE, op=ComparisonOp.GTE, value=7.5),
                Condition(field=NumericField.VOTE_COUNT, op=ComparisonOp.GT, value=1000),
            ]
        ),
    ),
    (
        "budget between",
        SearchQuery(
            conditions=[
                Condition(
                    field=NumericField.BUDGET, op=ComparisonOp.BETWEEN, value=[50e6, 200e6]
                )
            ]
        ),
    ),
    ("person matches cast or director", SearchQuery(people=["Quentin Tarantino"])),
    ("director only", SearchQuery(directors=["Christopher Nolan"])),
    ("company", SearchQuery(companies=["Pixar Animation Studios"])),
    ("language", SearchQuery(languages=["English"])),
    ("narrow pool", SearchQuery(genres=["Western"], year_from=2011)),
    ("empty pool", SearchQuery(genres=["Western"], year_from=2030)),
)


@pytest.fixture(scope="module")
def backends(request, repo):
    """The two backends over the same corpus. Skips when either is unbuilt."""
    settings = Settings()
    paths = settings.paths
    store = settings.vector_store

    try:
        numpy_index = VectorIndex.load(paths.embeddings_npy)
    except ArtifactError as exc:
        pytest.skip(f"numpy index not built: {exc}")

    if store.backend is VectorBackend.QDRANT and not store.url:
        # Embedded Qdrant is single-process and the session ``runtime`` already holds
        # the lock when the app is configured this way, so opening a second client
        # would fail. Reuse the one the runtime built rather than skipping the parity
        # checks in exactly the configuration they matter most.
        qdrant_index = request.getfixturevalue("runtime").index
        owned = False
    else:
        try:
            qdrant_index = QdrantIndex.connect(
                path=None if store.url else paths.qdrant_dir,
                url=store.url,
                api_key=store.api_key,
                collection=store.collection,
            )
        except (ArtifactError, RuntimeError) as exc:
            pytest.skip(f"qdrant collection not built: {exc}")
        owned = True

    if len(numpy_index) != len(repo) or len(qdrant_index) != len(repo):
        if owned:
            qdrant_index.close()
        pytest.skip("backends and dataset are out of sync; rebuild the artifacts")

    yield numpy_index, qdrant_index
    if owned:
        qdrant_index.close()


class TestFilterTranslation:
    """``qdrant_filters`` must mean exactly what ``mask_for`` means."""

    @pytest.mark.parametrize("label,query", FILTER_CASES, ids=[c[0] for c in FILTER_CASES])
    def test_pools_match(self, backends, repo, label: str, query: SearchQuery) -> None:
        """Pool size first: it isolates the filter from the ranking.

        If the pools differ the result sets could never have matched, and a recall
        number would only obscure which half was wrong.
        """
        numpy_index, qdrant_index = backends
        restriction = Restriction.from_query(repo, query)
        assert qdrant_index.pool_size(restriction) == numpy_index.pool_size(restriction)

    def test_an_empty_filter_is_no_filter(self) -> None:
        assert to_qdrant_filter(SearchQuery()) is None
        assert to_qdrant_filter(None) is None

    def test_nulls_never_satisfy_a_comparison(self, backends, repo) -> None:
        """R-014/R-016 across the seam.

        37 movies have no runtime and ~1,000 have unknown budget. In pandas those are
        excluded by ``fillna(False)``; in Qdrant by the payload key being absent. Both
        must reach the same set, or "under 90 minutes" silently gains films whose
        runtime nobody knows.
        """
        _, qdrant_index = backends
        query = SearchQuery(
            conditions=[Condition(field=NumericField.RUNTIME, op=ComparisonOp.LT, value=90)]
        )
        restriction = Restriction.from_query(repo, query)
        frame = repo.frame
        expected = int((frame[S.RUNTIME] < 90).fillna(False).sum())
        assert qdrant_index.pool_size(restriction) == expected
        assert frame[S.RUNTIME].isna().sum() > 0  # the case is actually exercised


class TestSearchParity:
    def test_unfiltered_top_k_is_identical(self, backends) -> None:
        numpy_index, qdrant_index = backends
        for position in (0, 10, 500, 2500, 4802):
            vector = numpy_index.matrix[position]
            left = numpy_index.search(vector, 8)
            right = qdrant_index.search(vector, 8)
            assert [h.position for h in left] == [h.position for h in right]
            for a, b in zip(left, right):
                assert a.score == pytest.approx(b.score, abs=1e-5)

    @pytest.mark.parametrize("label,query", FILTER_CASES, ids=[c[0] for c in FILTER_CASES])
    def test_filtered_top_k_is_identical(self, backends, repo, label, query) -> None:
        numpy_index, qdrant_index = backends
        restriction = Restriction.from_query(repo, query)
        vector = numpy_index.matrix[100]
        left = numpy_index.search(vector, 8, restriction)
        right = qdrant_index.search(vector, 8, restriction)
        assert [h.position for h in left] == [h.position for h in right]

    def test_every_result_satisfies_the_constraints(self, backends, repo) -> None:
        """ADR-0011's guarantee, asserted against the store rather than the mask."""
        numpy_index, qdrant_index = backends
        query = SearchQuery(genres=["Science Fiction"], year_from=2011)
        restriction = Restriction.from_query(repo, query)
        hits = qdrant_index.search(numpy_index.matrix[100], 8, restriction)

        assert len(hits) == 8  # the pool is large enough; k results must come back
        frame = repo.frame
        for hit in hits:
            row = frame.iloc[hit.position]
            assert "Science Fiction" in row[S.GENRES]
            assert row[S.RELEASE_YEAR] >= 2011

    def test_an_empty_pool_returns_nothing(self, backends, repo) -> None:
        numpy_index, qdrant_index = backends
        restriction = Restriction.from_query(
            repo, SearchQuery(genres=["Western"], year_from=2030)
        )
        assert qdrant_index.pool_size(restriction) == 0
        assert qdrant_index.search(numpy_index.matrix[0], 8, restriction) == []

    def test_similar_to_excludes_the_seed(self, backends, repo) -> None:
        numpy_index, qdrant_index = backends
        position = repo.position(27205)  # Inception
        left = numpy_index.similar_to(position, 5)
        right = qdrant_index.similar_to(position, 5)
        assert all(hit.position != position for hit in right)
        assert [h.position for h in left] == [h.position for h in right]

    def test_dimension_mismatch_is_refused(self, backends) -> None:
        """A collection queried with another model's vectors returns confident nonsense."""
        _, qdrant_index = backends
        with pytest.raises(ValueError, match="dimension"):
            qdrant_index.search(np.ones(8, dtype=np.float32), k=1)

    def test_a_raw_mask_still_works(self, backends, repo) -> None:
        """Callers holding a bare mask get the id-list path, which must agree too."""
        numpy_index, qdrant_index = backends
        mask = repo.mask_for(SearchQuery(genres=["Western"]))
        vector = numpy_index.matrix[100]
        assert qdrant_index.pool_size(mask) == int(np.count_nonzero(mask))
        assert [h.position for h in qdrant_index.search(vector, 5, mask)] == [
            h.position for h in numpy_index.search(vector, 5, mask)
        ]
