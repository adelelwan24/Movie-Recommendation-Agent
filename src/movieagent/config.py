"""Configuration (ADR-0015).

Grouped settings with **layered validation**: structure (types, ranges, enums) is
checked at construction; credentials are checked at the point of use. That is
deliberate -- preprocessing, structured search, fuzzy matching and the local embedding
backend all work with no API key, and refusing to boot without one would be a
self-inflicted limitation.

`LLM_*` and `EMBEDDING_*` are separate groups on purpose (ADR-0007): OpenRouter serves
chat completions but **not** embeddings, so the embedding backend needs its own base
URL and can point at OpenAI proper, a self-hosted vLLM, Ollama or LM Studio.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from movieagent.errors import ConfigurationError

# Bumped whenever preprocessing changes shape. Stamped into the artifact manifest so
# a stale artifact fails loudly instead of silently (ADR-0005).
PREPROCESS_VERSION = "1"

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class EmbeddingProvider(StrEnum):
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI_COMPATIBLE = "openai_compatible"


class VectorBackend(StrEnum):
    """Which engine answers vector search.

    `numpy` is the in-process exact-cosine index ADR-0006 chose and still the default:
    it needs no service, and at ~4,800 documents it is not the bottleneck. `qdrant` is
    the same search backed by a real vector database, embedded by default and pointed
    at a server by setting `QDRANT_URL`.
    """

    NUMPY = "numpy"
    QDRANT = "qdrant"


class LLMSettings(BaseModel):
    """Chat / tool-calling endpoint. Defaults to OpenRouter; a vLLM swap is a base-URL
    change with no code edit (R-117)."""

    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str | None = None
    model: str = "openai/gpt-4o-mini"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)

    def require_key(self) -> str:
        """Layered validation: called only on paths that actually need a credential."""
        if not self.api_key:
            raise ConfigurationError(
                "LLM_API_KEY is not set, so the agent cannot call the model.\n"
                f"  endpoint: {self.base_url}\n"
                "  Set it in .env (see .env.example) and restart.\n"
                "  Structured search, fuzzy matching and index building do not need it."
            )
        return self.api_key


class EmbeddingSettings(BaseModel):
    """Embedding backend (ADR-0007). Independent of `LLMSettings` -- do not point
    `base_url` at OpenRouter, which serves no `/v1/embeddings` endpoint."""

    provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    model: str = "BAAI/bge-small-en-v1.5"
    base_url: str | None = None
    api_key: str | None = None
    batch_size: int = Field(default=64, ge=1, le=2048)

    @model_validator(mode="after")
    def _check_api_backend(self) -> EmbeddingSettings:
        if self.provider is EmbeddingProvider.OPENAI_COMPATIBLE and not self.base_url:
            raise ValueError(
                "EMBEDDING_PROVIDER=openai_compatible requires EMBEDDING_BASE_URL. "
                "OpenRouter does not serve /v1/embeddings -- point this at OpenAI, "
                "vLLM, Ollama or LM Studio."
            )
        return self


class VectorStoreSettings(BaseModel):
    """Vector-store selection (ADR-0006, revisited).

    `url` is the only difference between embedded and served Qdrant: unset means the
    engine runs in-process against `artifacts/qdrant`, set means a cluster. Nothing else
    in the system changes shape, which is the point of keeping both behind one protocol.
    """

    backend: VectorBackend = VectorBackend.NUMPY
    collection: str = "movies"
    url: str | None = None
    api_key: str | None = None

    @model_validator(mode="after")
    def _check_server(self) -> VectorStoreSettings:
        if self.url and self.backend is not VectorBackend.QDRANT:
            raise ValueError("QDRANT_URL is set but VECTOR_BACKEND is not 'qdrant'")
        return self


class RetrievalSettings(BaseModel):
    """Retrieval parameters (ADR-0011, revised by ADR-0026).

    `similarity_floor` is a **backstop only**. Measurement showed an absolute cosine
    threshold cannot serve R-056 on this corpus: genuine queries top out at 0.575-0.730
    and gibberish at 0.566-0.629, so the populations overlap. 0.35 catches catastrophic
    mismatch and nothing subtler; `min_lexical_coverage` is the signal that actually
    separates them.
    """

    top_k: int = Field(default=8, ge=1, le=50)
    similarity_floor: float = Field(default=0.35, ge=0.0, le=1.0)
    min_lexical_coverage: float = Field(default=0.4, ge=0.0, le=1.0)
    result_display_limit: int = Field(default=25, ge=1, le=200)


class FuzzySettings(BaseModel):
    """Fuzzy confidence bands.

    Recalibrated in ADR-0025 against a measured set rather than ADR-0009's original
    guess of 90/75. With the corrected scorer the two populations separate cleanly:
    correct matches score 86-100 (lowest: "Titanik" -> "Titanic" at 86) and unmatched
    input scores 33-59. 82 and 65 sit inside that gap with margin on both sides.
    """

    accept: int = Field(default=82, ge=0, le=100)
    ambiguous: int = Field(default=65, ge=0, le=100)
    tie_margin: int = Field(default=3, ge=0, le=50)
    max_candidates: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def _check_bands(self) -> FuzzySettings:
        if self.ambiguous >= self.accept:
            raise ValueError("FUZZY_AMBIGUOUS must be below FUZZY_ACCEPT")
        return self


class AgentSettings(BaseModel):
    """ADR-0021 loop guards and ADR-0022 memory window."""

    recursion_limit: int = Field(default=25, ge=5, le=100)
    max_tool_iterations: int = Field(default=6, ge=1, le=20)
    turn_timeout_s: float = Field(default=90.0, gt=0)
    message_window: int = Field(default=8, ge=2, le=100)


class PathSettings(BaseModel):
    """Relative by default so the app stays deployment-ready (ADR-0015)."""

    data_dir: Path = _PROJECT_ROOT / "data"
    artifacts_dir: Path = _PROJECT_ROOT / "artifacts"

    @property
    def movies_csv(self) -> Path:
        return self.data_dir / "tmdb_5000_movies.csv"

    @property
    def credits_csv(self) -> Path:
        return self.data_dir / "tmdb_5000_credits.csv"

    @property
    def movies_parquet(self) -> Path:
        return self.artifacts_dir / "movies.parquet"

    @property
    def embeddings_npy(self) -> Path:
        return self.artifacts_dir / "embeddings.npy"

    @property
    def qdrant_dir(self) -> Path:
        """Embedded Qdrant storage. One directory, single-process -- it takes a lock."""
        return self.artifacts_dir / "qdrant"

    @property
    def documents_parquet(self) -> Path:
        return self.artifacts_dir / "documents.parquet"

    @property
    def manifest_json(self) -> Path:
        return self.artifacts_dir / "manifest.json"


class Settings(BaseSettings):
    """Root settings object. Immutable, cached, and shared via `@st.cache_resource`
    (ADR-0014's read-only rule)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # Flat aliases keep .env readable rather than using nested delimiters.
    llm_base_url: str = LLMSettings.model_fields["base_url"].default
    llm_api_key: str | None = None
    llm_model: str = LLMSettings.model_fields["model"].default
    llm_temperature: float = 0.0
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3

    embedding_provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    embedding_model: str = EmbeddingSettings.model_fields["model"].default
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_batch_size: int = 64

    vector_backend: VectorBackend = VectorBackend.NUMPY
    qdrant_collection: str = VectorStoreSettings.model_fields["collection"].default
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None

    top_k: int = 8
    similarity_floor: float = 0.35
    min_lexical_coverage: float = 0.4
    result_display_limit: int = 25

    fuzzy_accept: int = 82
    fuzzy_ambiguous: int = 65
    fuzzy_tie_margin: int = 3
    fuzzy_max_candidates: int = 5

    recursion_limit: int = 25
    max_tool_iterations: int = 6
    turn_timeout_s: float = 90.0
    message_window: int = 8

    data_dir: Path = PathSettings.model_fields["data_dir"].default
    artifacts_dir: Path = PathSettings.model_fields["artifacts_dir"].default

    log_level: str = "INFO"
    log_file: Path | None = None

    @field_validator(
        "llm_api_key",
        "embedding_base_url",
        "embedding_api_key",
        "qdrant_url",
        "qdrant_api_key",
        "log_file",
        mode="before",
    )
    @classmethod
    def _blank_means_unset(cls, value: object) -> object:
        """Treat an empty env value as absent.

        `.env` files habitually carry blank placeholders (`LOG_FILE=`), and dotenv
        reads those as `""`, not as missing. For `Path | None` that is actively
        dangerous: pydantic coerces `""` to `Path(".")`, which resolves to the project
        directory -- so `logging.FileHandler` was handed a *directory* and raised
        `PermissionError`. Optional means optional; a blank line means unset.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def llm(self) -> LLMSettings:
        return LLMSettings(
            base_url=self.llm_base_url,
            api_key=self.llm_api_key,
            model=self.llm_model,
            temperature=self.llm_temperature,
            timeout_s=self.llm_timeout_s,
            max_retries=self.llm_max_retries,
        )

    @property
    def embedding(self) -> EmbeddingSettings:
        return EmbeddingSettings(
            provider=self.embedding_provider,
            model=self.embedding_model,
            base_url=self.embedding_base_url,
            api_key=self.embedding_api_key,
            batch_size=self.embedding_batch_size,
        )

    @property
    def vector_store(self) -> VectorStoreSettings:
        return VectorStoreSettings(
            backend=self.vector_backend,
            collection=self.qdrant_collection,
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
        )

    @property
    def retrieval(self) -> RetrievalSettings:
        return RetrievalSettings(
            top_k=self.top_k,
            similarity_floor=self.similarity_floor,
            min_lexical_coverage=self.min_lexical_coverage,
            result_display_limit=self.result_display_limit,
        )

    @property
    def fuzzy(self) -> FuzzySettings:
        return FuzzySettings(
            accept=self.fuzzy_accept,
            ambiguous=self.fuzzy_ambiguous,
            tie_margin=self.fuzzy_tie_margin,
            max_candidates=self.fuzzy_max_candidates,
        )

    @property
    def agent(self) -> AgentSettings:
        return AgentSettings(
            recursion_limit=self.recursion_limit,
            max_tool_iterations=self.max_tool_iterations,
            turn_timeout_s=self.turn_timeout_s,
            message_window=self.message_window,
        )

    @property
    def paths(self) -> PathSettings:
        return PathSettings(data_dir=self.data_dir, artifacts_dir=self.artifacts_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
