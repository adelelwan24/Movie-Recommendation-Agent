"""Qdrant-backed vector search (ADR-0006's successor path).

Same contract as ``VectorIndex``, different engine. Two deployment shapes from one class:

* **embedded** -- ``QdrantClient(path=...)`` runs the engine in-process against a local
  directory. No Docker, no service to start, and a fresh clone still works offline,
  which is the precondition ADR-0016 refuses to give up.
* **server** -- set ``QDRANT_URL`` and the same code talks to a real cluster. That is
  the entire migration.

**Point ids are row positions**, not movie ids. Every consumer above this module -- the
repository frame, the document list, ``Hit.position`` -- already speaks positions, and
translating in one place at write time is cheaper than translating at every read.

Filtering is server-side (``qdrant_filters``), so ADR-0011's pre-filter guarantee is
preserved rather than approximated: Qdrant applies the payload filter during traversal,
so k constraint-satisfying results come back whenever k exist. When a query cannot be
translated the backend falls back to an explicit id list built from the mask, which is
exact by construction and slower only in the tail.
"""

from __future__ import annotations

import atexit
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from movieagent.data import schema as S
from movieagent.errors import ArtifactError
from movieagent.logging import get_logger
from movieagent.retrieval.backend import Hit, Restriction, coerce_restriction
from movieagent.retrieval.qdrant_filters import (
    KEYWORD_PAYLOAD_KEYS,
    NUMERIC_PAYLOAD_KEYS,
    UntranslatableFilter,
    to_qdrant_filter,
)

log = get_logger("qdrant")

DEFAULT_COLLECTION = "movies"

#: Scalar payload fields carried alongside the filterable ones, for debugging a point
#: without a round-trip to the frame.
_IDENTITY_KEYS: tuple[str, ...] = (S.ID, S.TITLE)


def _require_qdrant() -> Any:
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:  # pragma: no cover - install-dependent
        raise ArtifactError(
            "qdrant-client is not installed, but VECTOR_BACKEND=qdrant.\n"
            "  Install it:  uv pip install -e .\n"
            "  Or switch back:  VECTOR_BACKEND=numpy"
        ) from exc
    return QdrantClient


def _payload_for_row(row: pd.Series) -> dict[str, Any]:
    """One point's payload.

    Null numerics are **omitted rather than stored as null** -- that is what makes
    ``vote_average > 7`` skip unrated films, matching ``fillna(False)`` in the pandas
    path (R-014, R-016). String lists are case-folded to match the repository's
    term index, which keys on ``casefold()``.
    """
    payload: dict[str, Any] = {}

    for key in _IDENTITY_KEYS:
        value = row.get(key)
        if value is not None and not pd.isna(value):
            payload[key] = int(value) if key == S.ID else str(value)

    for key in KEYWORD_PAYLOAD_KEYS:
        values = row.get(key)
        if values is None or (isinstance(values, float) and pd.isna(values)):
            continue
        folded = [str(v).casefold() for v in values if v is not None]
        if folded:
            payload[key] = folded

    for key in NUMERIC_PAYLOAD_KEYS:
        value = row.get(key)
        if value is None or pd.isna(value):
            continue
        payload[key] = float(value)

    return payload


class QdrantIndex:
    """Cosine search over a Qdrant collection, filtered server-side."""

    name = "qdrant"

    def __init__(self, client: Any, collection: str, size: int, dimension: int) -> None:
        self._client = client
        self._collection = collection
        self._size = size
        self._dimension = dimension
        self._closed = False
        # The client releases its storage lock from ``__del__``, which in embedded mode
        # runs *after* interpreter teardown has unloaded the modules that the unlock
        # itself needs -- producing an alarming "Exception ignored in" traceback on every
        # clean exit. Closing at atexit happens while the interpreter is still whole.
        atexit.register(self.close)

    # ------------------------------------------------------------------ construction

    @classmethod
    def connect(
        cls,
        *,
        path: Path | None = None,
        url: str | None = None,
        api_key: str | None = None,
        collection: str = DEFAULT_COLLECTION,
    ) -> QdrantIndex:
        """Open an existing collection. Raises when it has not been built."""
        client = cls._client_for(path=path, url=url, api_key=api_key)
        try:
            info = client.get_collection(collection)
        except Exception as exc:  # noqa: BLE001 - surfaced as a typed, actionable error
            client.close()
            where = url or (path and str(path)) or "memory"
            raise ArtifactError(
                f"No Qdrant collection {collection!r} at {where}.\n"
                "  Build it first:  python scripts/build_index.py --with-qdrant"
            ) from exc

        vectors = info.config.params.vectors
        dimension = int(getattr(vectors, "size", 0))
        size = int(client.count(collection, exact=True).count)
        log.info("qdrant collection %s: %d points, dim %d", collection, size, dimension)
        return cls(client, collection, size, dimension)

    @classmethod
    def build(
        cls,
        matrix: np.ndarray,
        frame: pd.DataFrame,
        *,
        path: Path | None = None,
        url: str | None = None,
        api_key: str | None = None,
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = 512,
    ) -> QdrantIndex:
        """(Re)create the collection from a vector matrix and the processed frame.

        Recreates rather than upserts: the matrix and the frame are rebuilt together by
        ``build_index.py``, so a partial overwrite could leave points from two different
        embedding models in one collection -- silently, and answering wrongly.
        """
        from qdrant_client import models as qm

        if len(matrix) != len(frame):
            raise ValueError(
                f"matrix has {len(matrix)} rows but the frame has {len(frame)} -- "
                "these must be built together"
            )

        client = cls._client_for(path=path, url=url, api_key=api_key)
        dimension = int(matrix.shape[1])
        if client.collection_exists(collection):
            client.delete_collection(collection)
        client.create_collection(
            collection_name=collection,
            vectors_config=qm.VectorParams(size=dimension, distance=qm.Distance.COSINE),
        )

        frame = frame.reset_index(drop=True)
        vectors = np.ascontiguousarray(matrix, dtype=np.float32)
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            chunk = frame.iloc[start:stop]
            client.upsert(
                collection_name=collection,
                points=qm.Batch(
                    ids=list(range(start, stop)),
                    vectors=vectors[start:stop].tolist(),
                    payloads=[_payload_for_row(row) for _, row in chunk.iterrows()],
                ),
            )
            log.debug("upserted %d/%d points", stop, len(frame))

        # Payload indexes are what make filtered search fast on a server. The embedded
        # engine ignores them (and says so, loudly), so they are declared only for the
        # deployment where they do something.
        if url:
            for key in KEYWORD_PAYLOAD_KEYS:
                client.create_payload_index(
                    collection, key, field_schema=qm.PayloadSchemaType.KEYWORD
                )
            for key in NUMERIC_PAYLOAD_KEYS:
                client.create_payload_index(
                    collection, key, field_schema=qm.PayloadSchemaType.FLOAT
                )

        size = int(client.count(collection, exact=True).count)
        log.info("built qdrant collection %s: %d points, dim %d", collection, size, dimension)
        return cls(client, collection, size, dimension)

    @staticmethod
    def _client_for(*, path: Path | None, url: str | None, api_key: str | None) -> Any:
        QdrantClient = _require_qdrant()
        if url:
            return QdrantClient(url=url, api_key=api_key)
        if path is None:
            return QdrantClient(location=":memory:")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            return QdrantClient(path=str(path))
        except RuntimeError as exc:
            # Embedded storage takes an exclusive lock, so the app and a script cannot
            # both hold it. That is a deployment property worth stating plainly rather
            # than leaving as a stack trace about file locks.
            if "already accessed" not in str(exc):
                raise
            raise ArtifactError(
                f"The embedded Qdrant store at {path} is locked by another process.\n"
                "  Embedded mode is single-process: stop the app (or the other script) "
                "and retry.\n"
                "  For concurrent access, run a Qdrant server and set QDRANT_URL."
            ) from exc

    def close(self) -> None:
        """Release the storage lock. Idempotent; also runs at interpreter exit."""
        if self._closed:
            return
        self._closed = True
        self._client.close()

    # -------------------------------------------------------------------- properties

    def __len__(self) -> int:
        return self._size

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def client(self) -> Any:
        """The underlying client, for scripts that need collection-level operations."""
        return self._client

    @property
    def collection(self) -> str:
        return self._collection

    # ----------------------------------------------------------------------- queries

    def _filter_for(self, restriction: Restriction | None) -> Any:
        """Payload filter when the query translates; an explicit id list otherwise."""
        if restriction is None:
            return None
        if restriction.query is not None:
            try:
                return to_qdrant_filter(restriction.query)
            except UntranslatableFilter as exc:  # pragma: no cover - no such predicate yet
                log.warning("falling back to id filter: %s", exc)
        return self._id_filter(restriction)

    @staticmethod
    def _id_filter(restriction: Restriction) -> Any:
        from qdrant_client import models as qm

        return qm.Filter(
            must=[qm.HasIdCondition(has_id=[int(p) for p in restriction.positions])]
        )

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        restriction: Restriction | np.ndarray | None = None,
    ) -> list[Hit]:
        """Top-k by cosine similarity, filtered before ranking."""
        if k <= 0:
            return []
        restriction = coerce_restriction(restriction)

        vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dimension:
            raise ValueError(
                f"query dimension {vector.shape[0]} does not match index dimension "
                f"{self.dimension} -- the collection was built with a different "
                "embedding model; rebuild it"
            )

        response = self._client.query_points(
            collection_name=self._collection,
            query=vector.tolist(),
            query_filter=self._filter_for(restriction),
            limit=k,
            with_payload=False,
        )
        return [Hit(position=int(point.id), score=float(point.score)) for point in response.points]

    def similar_to(
        self,
        position: int,
        k: int,
        restriction: Restriction | np.ndarray | None = None,
    ) -> list[Hit]:
        """Movies similar to an existing movie, ranked by that movie's own vector.

        Queries by point id so the vector never leaves the store. The seed is excluded --
        "films similar to Inception" answering "Inception" is correct and useless.
        """
        if k <= 0:
            return []
        restriction = coerce_restriction(restriction)
        response = self._client.query_points(
            collection_name=self._collection,
            query=int(position),
            query_filter=self._filter_for(restriction),
            limit=k + 1,
            with_payload=False,
        )
        hits = [Hit(position=int(p.id), score=float(p.score)) for p in response.points]
        return [hit for hit in hits if hit.position != position][:k]

    def pool_size(self, restriction: Restriction | np.ndarray | None) -> int:
        """How many candidates the restriction admits, counted by the engine.

        Counted server-side rather than read off the mask so the number reported in the
        trace is the pool the *ranking actually ran over*. If the payload filter and the
        pandas mask ever disagree, this is where it becomes visible instead of hiding.
        """
        restriction = coerce_restriction(restriction)
        if restriction is None:
            return len(self)
        return int(
            self._client.count(
                self._collection, count_filter=self._filter_for(restriction), exact=True
            ).count
        )
