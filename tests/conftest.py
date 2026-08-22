"""Shared fixtures (ADR-0016, ADR-0024).

Tiers 1, 2 and 5 run with **no network at all**. That is not frugality -- OQ-002 (is
there an API budget? are live calls acceptable?) went unanswered, so a suite that needs
a key is a suite that might not run for whoever reviews this.

Tier 3 (golden retrieval) needs the real embedding model and is skipped when the index
has not been built.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("LLM_API_KEY", "test-key-not-used")

from movieagent.config import Settings  # noqa: E402
from movieagent.data.preprocess import build_movies_frame  # noqa: E402
from movieagent.data.repository import MovieRepository  # noqa: E402
from movieagent.retrieval.fuzzy import FuzzyTitleMatcher  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="session")
def frame() -> pd.DataFrame:
    """The processed frame.

    Prefers the built Parquet -- that is what the app actually loads, so testing it
    verifies the dtype round-trip that R-014–R-016 depend on. Falls back to building
    from CSV so the suite works before the first build.
    """
    parquet = ARTIFACTS_DIR / "movies.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)

    movies = DATA_DIR / "tmdb_5000_movies.csv"
    credits = DATA_DIR / "tmdb_5000_credits.csv"
    if not (movies.exists() and credits.exists()):
        pytest.skip("dataset CSVs not present")
    built, _ = build_movies_frame(movies, credits)
    return built


@pytest.fixture(scope="session")
def repo(frame: pd.DataFrame) -> MovieRepository:
    return MovieRepository(frame)


@pytest.fixture(scope="session")
def matcher(repo: MovieRepository, settings: Settings) -> FuzzyTitleMatcher:
    return FuzzyTitleMatcher(repo, settings.fuzzy)


@pytest.fixture(scope="session")
def vectors():
    """The raw document vectors, independent of which backend serves search.

    Read from the artifact rather than off ``runtime.index``: a matrix is a numpy-index
    implementation detail, and tests about vector *geometry* should not stop working
    when the store behind the protocol changes.
    """
    import numpy as np

    path = ARTIFACTS_DIR / "embeddings.npy"
    if not path.exists():
        pytest.skip("vector index not built")
    return np.load(path)


@pytest.fixture(scope="session")
def runtime():
    """Full runtime including the vector index. Skips when unbuilt."""
    from movieagent.errors import ArtifactError
    from movieagent.runtime import load_runtime

    try:
        return load_runtime(Settings())
    except ArtifactError as exc:
        pytest.skip(f"artifacts not built: {exc}")
