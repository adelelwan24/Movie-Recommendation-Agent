"""Embedding backends (ADR-0007, R-118).

Two implementations behind one protocol, selected by configuration:

* ``SentenceTransformerBackend`` -- local, default, no API key, no network after the
  first weight download. This is what makes the index build and the retrieval tests run
  free and offline on a fresh clone (ADR-0016's precondition).
* ``OpenAICompatibleBackend`` -- any endpoint serving ``/v1/embeddings``: OpenAI proper,
  vLLM, Ollama, LM Studio. **Not OpenRouter**, which proxies chat completions only --
  which is why embedding configuration is independent of chat configuration.

Documents and queries are embedded through *separate methods* on purpose. The default
model (``bge-small-en-v1.5``) requires an instruction prefix on **queries only**;
collapsing both into one ``embed()`` is the standard way to silently lose retrieval
quality with BGE models. The asymmetry lives inside the backend so no call site has to
remember it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from movieagent.config import EmbeddingProvider, EmbeddingSettings
from movieagent.errors import ConfigurationError, EmbeddingBackendError
from movieagent.logging import get_logger

log = get_logger("embeddings")

#: Models that want a retrieval instruction on the query side only.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_QUERY_PREFIXES: dict[str, str] = {
    "BAAI/bge-small-en-v1.5": _BGE_QUERY_PREFIX,
    "BAAI/bge-base-en-v1.5": _BGE_QUERY_PREFIX,
    "BAAI/bge-large-en-v1.5": _BGE_QUERY_PREFIX,
}


@runtime_checkable
class EmbeddingBackend(Protocol):
    """The contract both backends satisfy."""

    model_id: str

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalize rows so cosine similarity is a plain dot product (ADR-0006)."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class SentenceTransformerBackend:
    """Local embeddings via ``sentence-transformers``.

    Chosen as the default despite pulling ~2 GB of torch, because OQ-002 (is there an
    API budget? are live calls acceptable?) went unanswered -- and a default that might
    not run for the reader is not a default.
    """

    def __init__(self, model_id: str, batch_size: int = 64) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - install-dependent
            raise EmbeddingBackendError(
                "sentence-transformers is not installed. Either install it "
                "(uv pip install -e .) or set EMBEDDING_PROVIDER=openai_compatible."
            ) from exc

        self.model_id = model_id
        self._batch_size = batch_size
        log.info("loading local embedding model %s", model_id)
        self._model = SentenceTransformer(model_id)
        self._query_prefix = _QUERY_PREFIXES.get(model_id, "")

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=len(texts) > 512,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return _l2_normalize(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        vectors = self._model.encode(
            [self._query_prefix + text], convert_to_numpy=True, normalize_embeddings=False
        )
        return _l2_normalize(vectors)[0]


class OpenAICompatibleBackend:
    """Embeddings from any OpenAI-compatible ``/v1/embeddings`` endpoint.

    No query prefix: hosted embedding models are symmetric, and adding BGE's
    instruction to an OpenAI model would degrade rather than help.
    """

    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str | None,
        batch_size: int = 64,
    ) -> None:
        from openai import OpenAI

        if not api_key:
            raise ConfigurationError(
                "EMBEDDING_PROVIDER=openai_compatible requires EMBEDDING_API_KEY.\n"
                f"  endpoint: {base_url}\n"
                "  Or switch to the local backend: EMBEDDING_PROVIDER=sentence_transformers"
            )
        self.model_id = model_id
        self._batch_size = batch_size
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = int(self.embed_query("dimension probe").shape[0])
        return self._dimension

    def _embed(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            try:
                response = self._client.embeddings.create(model=self.model_id, input=batch)
            except Exception as exc:  # noqa: BLE001 - surfaced as a typed error
                raise EmbeddingBackendError(
                    f"embedding request failed against {self.model_id}: {exc}"
                ) from exc
            out.extend(item.embedding for item in response.data)
        return _l2_normalize(np.asarray(out, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]


def build_embedding_backend(settings: EmbeddingSettings) -> EmbeddingBackend:
    """Construct the configured backend."""
    match settings.provider:
        case EmbeddingProvider.SENTENCE_TRANSFORMERS:
            return SentenceTransformerBackend(settings.model, settings.batch_size)
        case EmbeddingProvider.OPENAI_COMPATIBLE:
            if not settings.base_url:  # pragma: no cover - validated in config
                raise ConfigurationError("EMBEDDING_BASE_URL is required for this provider")
            return OpenAICompatibleBackend(
                settings.model, settings.base_url, settings.api_key, settings.batch_size
            )
        case _:  # pragma: no cover
            raise ConfigurationError(f"unknown embedding provider {settings.provider!r}")
