"""The artifact manifest and its staleness guard (ADR-0005, ADR-0007).

An artifact that has drifted from its sources fails *silently and plausibly*: the app
starts, answers questions, and is quietly wrong. That is the worst available failure, so
this turns it into a loud one at load time.

Two independent things are guarded:

* the **data** artifact against the source CSVs and the preprocessing version;
* the **embedding** artifact against the model id and dimension that produced it --
  because ADR-0007 ships two backends, and querying a ``bge-small`` index with
  ``text-embedding-3-small`` vectors is meaningless. If the dimensions happened to
  match it would be *silently* meaningless, which is worse.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from movieagent.errors import ArtifactError, ArtifactStaleError


@dataclass(slots=True)
class Manifest:
    preprocess_version: str
    movies_csv_sha256: str
    credits_csv_sha256: str
    row_count: int
    embedding_model: str
    embedding_dimension: int
    document_template_version: str
    built_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    report: dict[str, Any] = field(default_factory=dict)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> Manifest:
        if not path.exists():
            raise ArtifactError(
                f"No artifact manifest at {path}.\n"
                "  Build the index first:  python scripts/build_index.py"
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def check_against_sources(
        self,
        *,
        preprocess_version: str,
        movies_sha: str | None,
        credits_sha: str | None,
    ) -> None:
        """Raise if the built artifact no longer matches its inputs.

        Source hashes are optional: a deployment may ship artifacts without the raw
        CSVs (the app is kept deployment-ready per ADR-0015), and in that case there is
        nothing to compare against. The preprocessing version is always checked.
        """
        problems: list[str] = []
        if self.preprocess_version != preprocess_version:
            problems.append(
                f"preprocessing version {self.preprocess_version!r} "
                f"but this build expects {preprocess_version!r}"
            )
        if movies_sha is not None and self.movies_csv_sha256 != movies_sha:
            problems.append("tmdb_5000_movies.csv has changed since the artifact was built")
        if credits_sha is not None and self.credits_csv_sha256 != credits_sha:
            problems.append("tmdb_5000_credits.csv has changed since the artifact was built")
        if problems:
            raise ArtifactStaleError(
                "Artifacts are stale:\n  - "
                + "\n  - ".join(problems)
                + "\n  Rebuild:  python scripts/build_index.py"
            )

    def check_embedding(self, *, model: str, dimension: int | None = None) -> None:
        """Raise if the index was built with a different embedding model.

        This guard is not optional -- without it ADR-0007's two-backend design is
        actively dangerous.
        """
        if self.embedding_model != model:
            raise ArtifactStaleError(
                f"The vector index was built with embedding model "
                f"{self.embedding_model!r}, but the current configuration selects "
                f"{model!r}. These are different vector spaces; querying one with the "
                f"other returns confident nonsense.\n"
                "  Either set EMBEDDING_MODEL back, or rebuild:  "
                "python scripts/build_index.py"
            )
        if dimension is not None and self.embedding_dimension != dimension:
            raise ArtifactStaleError(
                f"Embedding dimension mismatch: index has {self.embedding_dimension}, "
                f"backend produces {dimension}. Rebuild the index."
            )
