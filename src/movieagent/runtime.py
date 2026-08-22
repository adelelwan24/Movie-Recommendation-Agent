"""Assembly: artifacts on disk -> a runnable agent.

One place that knows how the pieces fit together, so ``app.py``, the scripts and the
tests all construct the system the same way.

The objects built here are the ones ADR-0014 shares across Streamlit sessions via
``@st.cache_resource``, which is why every one of them is read-only after construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from movieagent.config import PREPROCESS_VERSION, Settings, VectorBackend, get_settings
from movieagent.data.manifest import Manifest
from movieagent.data.preprocess import file_sha256
from movieagent.data.repository import MovieRepository
from movieagent.errors import ArtifactError
from movieagent.llm.embeddings import EmbeddingBackend, build_embedding_backend
from movieagent.logging import get_logger
from movieagent.retrieval.coverage import CorpusVocabulary
from movieagent.retrieval.backend import SearchBackend
from movieagent.retrieval.fuzzy import FuzzyTitleMatcher
from movieagent.retrieval.vector_index import VectorIndex
from movieagent.tools.base import ToolContext

log = get_logger("runtime")


@dataclass(frozen=True, slots=True)
class Runtime:
    """Everything long-lived, built once per process."""

    settings: Settings
    repository: MovieRepository
    matcher: FuzzyTitleMatcher
    index: SearchBackend
    embedder: EmbeddingBackend
    documents: list[str]
    vocabulary: CorpusVocabulary
    manifest: Manifest

    def tool_context(self) -> ToolContext:
        return ToolContext(
            settings=self.settings,
            repository=self.repository,
            matcher=self.matcher,
            index=self.index,
            embedder=self.embedder,
            documents=self.documents,
            vocabulary=self.vocabulary,
        )


def build_search_backend(settings: Settings) -> SearchBackend:
    """Construct the configured vector-search backend.

    Both read the same artifacts produced by ``build_index.py``; they differ only in
    where the vectors live and who evaluates the filter. Imported lazily so a project
    running on the default backend never needs ``qdrant-client`` installed.
    """
    paths = settings.paths
    store = settings.vector_store
    if store.backend is VectorBackend.QDRANT:
        from movieagent.retrieval.qdrant_index import QdrantIndex

        return QdrantIndex.connect(
            path=None if store.url else paths.qdrant_dir,
            url=store.url,
            api_key=store.api_key,
            collection=store.collection,
        )
    return VectorIndex.load(paths.embeddings_npy)


def _optional_sha(path: Path) -> str | None:
    """Hash a source file if it is present.

    Absent is legitimate: a deployment may ship artifacts without the raw CSVs. The
    preprocessing version is always checked regardless.
    """
    return file_sha256(path) if path.exists() else None


def load_runtime(settings: Settings | None = None) -> Runtime:
    """Load artifacts and construct the read-only runtime.

    Fails loudly and with instructions when artifacts are missing or stale (ADR-0005) --
    a stale artifact would otherwise answer questions quietly and wrongly.
    """
    settings = settings or get_settings()
    paths = settings.paths

    if not paths.manifest_json.exists():
        raise ArtifactError(
            "The dataset has not been built yet.\n"
            f"  Expected artifacts in: {paths.artifacts_dir}\n"
            "  Run:  python scripts/build_index.py"
        )

    manifest = Manifest.read(paths.manifest_json)
    manifest.check_against_sources(
        preprocess_version=PREPROCESS_VERSION,
        movies_sha=_optional_sha(paths.movies_csv),
        credits_sha=_optional_sha(paths.credits_csv),
    )
    # Checked before the backend is constructed: loading a 130 MB local model only to
    # discover the index was built with a different one is a waste of everyone's time.
    manifest.check_embedding(model=settings.embedding.model)

    repository = MovieRepository.from_parquet(paths.movies_parquet, manifest)
    index = build_search_backend(settings)
    if len(index) != len(repository):
        raise ArtifactError(
            f"Index has {len(index)} vectors but the dataset has {len(repository)} movies. "
            "Rebuild:  python scripts/build_index.py"
        )

    documents = pd.read_parquet(paths.documents_parquet)["document"].tolist()
    embedder = build_embedding_backend(settings.embedding)
    manifest.check_embedding(model=settings.embedding.model, dimension=embedder.dimension)

    matcher = FuzzyTitleMatcher(repository, settings.fuzzy)
    vocabulary = CorpusVocabulary(documents)
    log.info(
        "runtime ready: %d movies, %d vectors (%s backend), %d vocabulary words",
        len(repository),
        len(index),
        index.name,
        len(vocabulary),
    )

    return Runtime(
        settings=settings,
        repository=repository,
        matcher=matcher,
        index=index,
        embedder=embedder,
        documents=documents,
        vocabulary=vocabulary,
        manifest=manifest,
    )


def build_default_agent(runtime: Runtime | None = None, settings: Settings | None = None):
    """Construct the agent against the configured chat model.

    Imported lazily so that everything above this line stays usable with no API key --
    ADR-0015's layered validation is only real if the deterministic half never touches
    the LLM import path.
    """
    from movieagent.agent.graph import build_agent
    from movieagent.llm.models import build_chat_model

    settings = settings or get_settings()
    runtime = runtime or load_runtime(settings)
    model = build_chat_model(settings.llm)
    return build_agent(settings, runtime.tool_context(), model)
