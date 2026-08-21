"""In-process exact-cosine vector index (ADR-0006, ADR-0011).

~4,800 documents at 384 dimensions is roughly 7 MB. One matmul searches the whole
corpus in a couple of milliseconds, which is 0.1% of a turn dominated by LLM latency.
Approximate search would buy nothing and cost test stability.

The property that actually earned this design is **hybrid pre-filtering**: because the
matrix is a plain array in the same row order as the repository, restricting to
"science fiction, after 2010, under 120 minutes" is a boolean mask applied *before*
ranking. That guarantees k constraint-satisfying results whenever k exist. Post-filtering
(retrieve 50, then discard) returns 3 when you asked for 10 and looks like "only 3 exist"
-- a silent failure this shape cannot produce.

Its expiry date is stated in ADR-0006: O(N*d) does not bend, and past ~500k documents
this must be replaced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from movieagent.errors import ArtifactError


@dataclass(frozen=True, slots=True)
class Hit:
    """One result: a row position into the repository frame, and its cosine score."""

    position: int
    score: float


class VectorIndex:
    """Exact cosine search over L2-normalized row vectors.

    Immutable after construction -- it is shared across Streamlit sessions via
    ``@st.cache_resource`` (ADR-0014).
    """

    def __init__(self, matrix: np.ndarray) -> None:
        if matrix.ndim != 2:
            raise ValueError(f"expected a 2-D matrix, got shape {matrix.shape}")
        self._matrix = np.ascontiguousarray(matrix, dtype=np.float32)

    @classmethod
    def load(cls, path: Path) -> VectorIndex:
        if not path.exists():
            raise ArtifactError(
                f"No vector index at {path}.\n"
                "  Build it first:  python scripts/build_index.py"
            )
        return cls(np.load(path))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self._matrix)

    def __len__(self) -> int:
        return int(self._matrix.shape[0])

    @property
    def dimension(self) -> int:
        return int(self._matrix.shape[1])

    @property
    def matrix(self) -> np.ndarray:
        """The raw matrix. Treat as read-only."""
        return self._matrix

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        mask: np.ndarray | None = None,
    ) -> list[Hit]:
        """Top-k by cosine similarity, optionally restricted to ``mask``.

        ``mask`` is applied **before** scoring (ADR-0011). Returns fewer than ``k``
        results only when the candidate pool is genuinely smaller -- and the caller is
        expected to report the pool size rather than let a short list imply scarcity.
        """
        if k <= 0:
            return []

        vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dimension:
            raise ValueError(
                f"query dimension {vector.shape[0]} does not match index dimension "
                f"{self.dimension} -- the index was built with a different embedding "
                "model; rebuild it"
            )

        if mask is None:
            positions = np.arange(self._matrix.shape[0])
            candidates = self._matrix
        else:
            positions = np.flatnonzero(mask)
            if positions.size == 0:
                return []
            candidates = self._matrix[positions]

        scores = candidates @ vector
        k = min(k, scores.shape[0])
        # argpartition is O(n) where a full sort is O(n log n); only the top-k slice is
        # then sorted. Immaterial at 4,800 rows, but it is the right shape.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="mergesort")]
        return [Hit(position=int(positions[i]), score=float(scores[i])) for i in top]

    def similar_to(
        self,
        position: int,
        k: int,
        mask: np.ndarray | None = None,
    ) -> list[Hit]:
        """Movies similar to an existing movie, by its own document vector.

        The seed itself is excluded -- "find films similar to Inception" answering
        "Inception" is technically correct and practically useless.
        """
        hits = self.search(self._matrix[position], k + 1, mask)
        return [hit for hit in hits if hit.position != position][:k]

    def pool_size(self, mask: np.ndarray | None) -> int:
        """How many candidates a mask admits.

        Surfaced in the trace because it is the honest confidence signal for hybrid
        search: ranking 8 films out of a pool of 12 is close to arbitrary, and the user
        deserves to see that rather than a confident-looking ordering.
        """
        return len(self) if mask is None else int(np.count_nonzero(mask))
