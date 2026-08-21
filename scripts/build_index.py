"""Build the dataset and vector-index artifacts (ADR-0005, ADR-0007).

Run once before starting the app::

    python scripts/build_index.py

Why this is a separate step rather than something the app does at startup: embedding
~4,800 documents cannot happen on every Streamlit rerun, so a build step exists no matter
what. Given that, putting preprocessing in the same lifecycle is strictly simpler than
having two. Parquet then preserves the nullable/date/unknown semantics that R-014–R-016
require and that a CSV round-trip would silently undo.

Everything written here is stamped into ``manifest.json``, which the loader checks. A
stale artifact fails loudly instead of answering questions quietly and wrongly.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:  # allow running without an editable install
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402

from movieagent.config import PREPROCESS_VERSION, get_settings  # noqa: E402
from movieagent.data.manifest import Manifest  # noqa: E402
from movieagent.data.preprocess import build_movies_frame, file_sha256  # noqa: E402
from movieagent.llm.embeddings import build_embedding_backend  # noqa: E402
from movieagent.logging import configure_logging, get_logger  # noqa: E402
from movieagent.retrieval.documents import (  # noqa: E402
    DOCUMENT_TEMPLATE_VERSION,
    build_documents,
)
from movieagent.retrieval.vector_index import VectorIndex  # noqa: E402

log = get_logger("build")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Rebuild the dataset only. Useful when iterating on preprocessing.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if up to date.")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    paths = settings.paths

    for csv in (paths.movies_csv, paths.credits_csv):
        if not csv.exists():
            log.error("missing source file: %s", csv)
            print(
                f"\nSource CSV not found: {csv}\n"
                "Download the TMDB 5000 Movie Dataset from Kaggle and place both CSVs "
                f"in {paths.data_dir}.\n",
                file=sys.stderr,
            )
            return 2

    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    log.info("reading %s and %s", paths.movies_csv.name, paths.credits_csv.name)
    frame, report = build_movies_frame(paths.movies_csv, paths.credits_csv)
    frame.to_parquet(paths.movies_parquet, index=False)
    log.info("wrote %s (%d movies)", paths.movies_parquet.name, len(frame))

    log.info("building semantic documents (template v%s)", DOCUMENT_TEMPLATE_VERSION)
    documents = build_documents(frame)
    pd.DataFrame({"id": frame["id"], "document": documents}).to_parquet(
        paths.documents_parquet, index=False
    )

    embedding_model = settings.embedding.model
    dimension = 0
    if args.skip_embeddings:
        log.warning("skipping embeddings; the app will not start until they are built")
    else:
        backend = build_embedding_backend(settings.embedding)
        embedding_model = backend.model_id
        log.info("embedding %d documents with %s", len(documents), embedding_model)
        matrix = backend.embed_documents(documents)
        dimension = int(matrix.shape[1])
        VectorIndex(matrix).save(paths.embeddings_npy)
        log.info("wrote %s %s", paths.embeddings_npy.name, matrix.shape)

    manifest = Manifest(
        preprocess_version=PREPROCESS_VERSION,
        movies_csv_sha256=file_sha256(paths.movies_csv),
        credits_csv_sha256=file_sha256(paths.credits_csv),
        row_count=len(frame),
        embedding_model=embedding_model,
        embedding_dimension=dimension,
        document_template_version=DOCUMENT_TEMPLATE_VERSION,
        report=report.to_dict(),
    )
    manifest.write(paths.manifest_json)

    elapsed = time.perf_counter() - started
    print(_summary(len(frame), report.to_dict(), dimension, elapsed, paths.artifacts_dir))
    return 0


def _summary(rows: int, report: dict, dimension: int, elapsed: float, out: Path) -> str:
    """Print the numbers the documentation has to quote (R-018), rather than estimates."""
    lines = [
        "",
        "Build complete.",
        f"  movies processed          {rows}",
        f"  joined from               {report['movies_in']} movies / {report['credits_in']} credits",
        f"  unmatched on join         {report['unmatched_movies']} movies, "
        f"{report['unmatched_credits']} credits",
        f"  missing release_date      {report['missing_release_date']}",
        f"  missing runtime           {report['missing_runtime']}",
        f"  empty overview            {report['missing_overview']}",
        f"  budget 0 -> unknown       {report['zero_budget_treated_as_unknown']}",
        f"  revenue 0 -> unknown      {report['zero_revenue_treated_as_unknown']}",
        f"  no director in crew       {report['no_director']}",
        f"  no keywords               {report['no_keywords']}",
    ]
    if report.get("malformed_json"):
        lines.append(f"  malformed JSON            {report['malformed_json']}")
    if dimension:
        lines.append(f"  embedding dimension       {dimension}")
    lines += [
        f"  artifacts                 {out}",
        f"  elapsed                   {elapsed:.1f}s",
        "",
        "Next:  streamlit run app.py",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
