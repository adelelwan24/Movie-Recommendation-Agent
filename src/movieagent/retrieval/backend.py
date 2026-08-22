"""The search-backend seam (ADR-0006, ADR-0011).

Two implementations sit behind this protocol: the in-process numpy index that has always
been here, and a Qdrant collection. They are interchangeable by configuration.

The load-bearing detail is ``Restriction``. ADR-0011's guarantee -- constraints applied
*before* ranking, so k results come back whenever k exist -- is expressed differently by
each backend: numpy wants a boolean row mask, Qdrant wants a payload filter evaluated
server-side. Rather than pick one and translate at every call site, a ``Restriction``
carries the *source* query plus the mask, and each backend takes the form it needs.

That dual representation is also the correctness check: the two must select the same
rows, and ``scripts/benchmark_vector_backends.py`` asserts exactly that. A filter
translation that silently drifts is the failure mode a vector database introduces here,
so it is measured rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from movieagent.data.query import SearchQuery


@dataclass(frozen=True, slots=True)
class Hit:
    """One result: a row position into the repository frame, and its cosine score.

    Positions rather than movie ids because the repository frame, the document list and
    the vector rows share one order -- and every consumer already speaks positions.
    """

    position: int
    score: float


@dataclass(frozen=True, slots=True)
class Restriction:
    """A pre-filter, carried in both the forms the backends consume.

    ``mask`` is authoritative: it comes from ``MovieRepository.mask_for``, which is the
    single definition of what a ``SearchQuery`` means. ``query`` is what a database-side
    filter is derived from. When a backend cannot express some predicate it can always
    fall back to the mask.
    """

    mask: np.ndarray
    query: SearchQuery | None = None

    @classmethod
    def from_query(cls, repository, query: SearchQuery) -> Restriction:
        """Build from a validated ``SearchQuery`` -- the normal path."""
        return cls(mask=repository.mask_for(query), query=query)

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> Restriction:
        """Build from a bare mask, for callers that computed one themselves."""
        return cls(mask=np.asarray(mask, dtype=bool))

    @property
    def positions(self) -> np.ndarray:
        """Row positions the restriction admits."""
        return np.flatnonzero(self.mask)

    def __len__(self) -> int:
        return int(np.count_nonzero(self.mask))


def coerce_restriction(value: Restriction | np.ndarray | None) -> Restriction | None:
    """Accept either a ``Restriction`` or a raw boolean mask.

    Raw masks stay supported because they read naturally in tests and in any caller that
    already holds one; there is no reason to make those construct a wrapper.
    """
    if value is None or isinstance(value, Restriction):
        return value
    return Restriction.from_mask(value)


@runtime_checkable
class SearchBackend(Protocol):
    """What ``semantic_search`` needs from a vector store."""

    #: Short identifier used in logs, traces and the benchmark report.
    name: str

    def __len__(self) -> int: ...

    @property
    def dimension(self) -> int: ...

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        restriction: Restriction | np.ndarray | None = None,
    ) -> list[Hit]: ...

    def similar_to(
        self,
        position: int,
        k: int,
        restriction: Restriction | np.ndarray | None = None,
    ) -> list[Hit]: ...

    def pool_size(self, restriction: Restriction | np.ndarray | None) -> int: ...
