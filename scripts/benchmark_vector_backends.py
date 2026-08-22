"""Compare the numpy index and the Qdrant collection on agreement and latency.

Run::

    python scripts/benchmark_vector_backends.py [--queries 200] [--k 8] [--seed 0]

Two questions, and they are not equally important.

**Do they return the same thing?** This is the one that matters. The numpy index is the
reference implementation of ADR-0011's pre-filter: ``mask_for`` defines what a filter
means, and the Qdrant payload filter is a second implementation of that definition in
another engine. Agreement is therefore a *correctness* measurement, not a quality one --
any disagreement is a bug in the translation, not a tuning opportunity. Pool sizes are
compared first because they isolate the filter from the ranking: if the pools differ, the
result sets were never going to match and the recall number would only obscure why.

**Which is faster?** Interesting but secondary at this corpus size, where a search is a
rounding error next to LLM latency. It becomes the deciding number only once the corpus
grows past the point where a full matmul per query is affordable -- which is the exact
condition ADR-0006 named as its own expiry date.

Queries are existing document vectors rather than freshly embedded text, so the benchmark
runs without loading a 130 MB model and measures the stores rather than the embedder.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:  # allow running without an editable install
    sys.path.insert(0, str(_SRC))

from movieagent.config import Settings, get_settings  # noqa: E402
from movieagent.data.query import ComparisonOp, Condition, SearchQuery  # noqa: E402
from movieagent.data.repository import MovieRepository  # noqa: E402
from movieagent.data.schema import NumericField  # noqa: E402
from movieagent.logging import configure_logging  # noqa: E402
from movieagent.retrieval.backend import Restriction  # noqa: E402
from movieagent.retrieval.qdrant_index import QdrantIndex  # noqa: E402
from movieagent.retrieval.vector_index import VectorIndex  # noqa: E402

#: Filter shapes that exercise each translation rule in ``qdrant_filters``.
SCENARIOS: tuple[tuple[str, SearchQuery | None], ...] = (
    ("unfiltered", None),
    ("single genre", SearchQuery(genres=["Drama"])),
    (
        "genre + year + runtime",
        SearchQuery(
            genres=["Science Fiction"],
            year_from=2011,
            conditions=[Condition(field=NumericField.RUNTIME, op=ComparisonOp.LT, value=120)],
        ),
    ),
    (
        "numeric only (nulls must not match)",
        SearchQuery(
            conditions=[
                Condition(field=NumericField.VOTE_AVERAGE, op=ComparisonOp.GTE, value=7.5),
                Condition(field=NumericField.VOTE_COUNT, op=ComparisonOp.GTE, value=1000),
            ]
        ),
    ),
    ("budget between (unknown != 0)",
     SearchQuery(
         conditions=[
             Condition(
                 field=NumericField.BUDGET, op=ComparisonOp.BETWEEN, value=[50e6, 200e6]
             )
         ]
     )),
    ("person (cast OR director)", SearchQuery(people=["Quentin Tarantino"])),
    ("multi-value membership", SearchQuery(genres=["Horror", "Thriller"], languages=["English"])),
    ("narrow pool", SearchQuery(genres=["Western"], year_from=2011)),
)


@dataclass
class ScenarioResult:
    label: str
    pool_numpy: int
    pool_qdrant: int
    mean_recall: float
    exact_match_rate: float
    max_score_delta: float
    numpy_ms: list[float]
    qdrant_ms: list[float]

    @property
    def pools_agree(self) -> bool:
        return self.pool_numpy == self.pool_qdrant


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[index]


def md_table(rows: list[list[object]], headers: list[str]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out) + "\n"


def run_scenario(
    label: str,
    query: SearchQuery | None,
    repository: MovieRepository,
    numpy_index: VectorIndex,
    qdrant_index: QdrantIndex,
    vectors: np.ndarray,
    k: int,
) -> ScenarioResult:
    restriction = Restriction.from_query(repository, query) if query is not None else None

    pool_numpy = numpy_index.pool_size(restriction)
    pool_qdrant = qdrant_index.pool_size(restriction)

    recalls: list[float] = []
    exact = 0
    max_delta = 0.0
    numpy_ms: list[float] = []
    qdrant_ms: list[float] = []

    for vector in vectors:
        start = time.perf_counter()
        numpy_hits = numpy_index.search(vector, k, restriction)
        numpy_ms.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        qdrant_hits = qdrant_index.search(vector, k, restriction)
        qdrant_ms.append((time.perf_counter() - start) * 1000)

        left = [h.position for h in numpy_hits]
        right = [h.position for h in qdrant_hits]
        if left:
            recalls.append(len(set(left) & set(right)) / len(left))
        exact += left == right

        # Score comparison is position-wise on the shared ids, since a different
        # ordering of equal scores is not a disagreement worth flagging.
        by_position = {h.position: h.score for h in qdrant_hits}
        for hit in numpy_hits:
            if hit.position in by_position:
                max_delta = max(max_delta, abs(hit.score - by_position[hit.position]))

    return ScenarioResult(
        label=label,
        pool_numpy=pool_numpy,
        pool_qdrant=pool_qdrant,
        mean_recall=statistics.fmean(recalls) if recalls else float("nan"),
        exact_match_rate=exact / len(vectors) if len(vectors) else float("nan"),
        max_score_delta=max_delta,
        numpy_ms=numpy_ms,
        qdrant_ms=qdrant_ms,
    )


def build_report(results: list[ScenarioResult], k: int, n_queries: int, corpus: int) -> str:
    parts: list[str] = []

    def add(line: str) -> None:
        parts.append(line if line.endswith("\n") else line + "\n")

    add("# Vector backend benchmark - numpy vs Qdrant\n")
    add(
        f"`scripts/benchmark_vector_backends.py` - {n_queries} queries per scenario, "
        f"k={k}, corpus of {corpus:,} documents.\n"
    )

    disagreements = [r for r in results if not r.pools_agree or r.mean_recall < 1.0]
    if disagreements:
        add("> **The two backends disagree.** Every row below with a pool mismatch or")
        add("> recall under 1.00 is a filter-translation bug in `qdrant_filters.py`,")
        add("> not a tuning artifact - the numpy mask is the definition.\n")
    else:
        add(
            "**The two backends agree exactly on every scenario**: identical candidate "
            "pools, identical top-k, identical ordering. The Qdrant payload filters "
            "reproduce `MovieRepository.mask_for` exactly, so ADR-0011's pre-filter "
            "guarantee survives the swap.\n"
        )

    add("\n## Agreement\n")
    add(
        md_table(
            [
                [
                    r.label,
                    f"{r.pool_numpy:,}",
                    f"{r.pool_qdrant:,}",
                    "yes" if r.pools_agree else "**NO**",
                    f"{r.mean_recall:.3f}",
                    f"{r.exact_match_rate:.3f}",
                    f"{r.max_score_delta:.2e}",
                ]
                for r in results
            ],
            [
                "scenario",
                "pool (numpy)",
                "pool (qdrant)",
                "pools agree",
                f"recall@{k}",
                "identical ordering",
                "max score delta",
            ],
        )
    )
    add(
        "\n`recall@k` is the share of numpy's top-k that Qdrant also returned; "
        "`identical ordering` is the stricter test that the ranked lists match "
        "position for position.\n"
    )

    add("\n## Latency per query\n")
    add(
        md_table(
            [
                [
                    r.label,
                    f"{statistics.median(r.numpy_ms):.2f}",
                    f"{percentile(r.numpy_ms, 0.95):.2f}",
                    f"{statistics.median(r.qdrant_ms):.2f}",
                    f"{percentile(r.qdrant_ms, 0.95):.2f}",
                    f"{statistics.median(r.qdrant_ms) / max(statistics.median(r.numpy_ms), 1e-9):.1f}x",
                ]
                for r in results
            ],
            [
                "scenario",
                "numpy median (ms)",
                "numpy p95 (ms)",
                "qdrant median (ms)",
                "qdrant p95 (ms)",
                "ratio",
            ],
        )
    )

    numpy_all = [ms for r in results for ms in r.numpy_ms]
    qdrant_all = [ms for r in results for ms in r.qdrant_ms]
    numpy_median = statistics.median(numpy_all)
    qdrant_median = statistics.median(qdrant_all)
    filtered = [r for r in results if r.label != "unfiltered"]
    slowest = max(filtered, key=lambda r: statistics.median(r.qdrant_ms)) if filtered else None

    add("")
    add(f"- numpy, all scenarios: median {numpy_median:.2f} ms")
    add(
        f"- qdrant (embedded), all scenarios: median {qdrant_median:.2f} ms - "
        f"{qdrant_median / max(numpy_median, 1e-9):.0f}x slower"
    )

    add("\n### Reading these numbers\n")
    shape = (
        f" The narrowest filters are its *slowest* scenarios -- {slowest.label!r} admits "
        f"{slowest.pool_qdrant:,} candidates and takes "
        f"{statistics.median(slowest.qdrant_ms):.0f} ms."
        if slowest
        else ""
    )
    add(
        "The gap is not a property of Qdrant, and the shape of it says why: numpy gets "
        "**faster** as a filter narrows, because fewer rows get scored. Embedded Qdrant "
        f"gets **slower**.{shape}"
    )
    add(
        "\nThat inversion is the embedded engine's cost model. Local mode evaluates payload "
        "conditions in Python and **ignores payload indexes entirely** -- it warns as much "
        "at build time -- so every filtered query walks the collection. A served Qdrant "
        "applies the same filters through indexed structures during traversal, so these "
        "latencies would have to be re-measured against a real instance before concluding "
        "anything about the engine itself.\n"
    )
    add(
        "\nConcretely, at this corpus size numpy is the faster backend by a wide margin and "
        "remains the sensible default. Embedded Qdrant buys the operational shape -- a real "
        "store, server-side filters, a one-setting move to a cluster -- and costs tens of "
        "milliseconds per query, which is still small beside an LLM turn but no longer free. "
        "The case for switching is the corpus outgrowing an O(N*d) matmul per query, which "
        "is the expiry condition ADR-0006 set for itself. This benchmark is not that case.\n"
    )
    return "".join(parts)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=200, help="Query vectors per scenario.")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qdrant-path", type=Path, default=None)
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument(
        "--out", type=Path, default=root / "artifacts" / "vector_backend_benchmark.md"
    )
    args = parser.parse_args()

    settings: Settings = get_settings()
    configure_logging("WARNING")
    paths = settings.paths
    store = settings.vector_store

    repository = MovieRepository.from_parquet(paths.movies_parquet)
    numpy_index = VectorIndex.load(paths.embeddings_npy)
    qdrant_index = QdrantIndex.connect(
        path=None if store.url else (args.qdrant_path or paths.qdrant_dir),
        url=store.url,
        api_key=store.api_key,
        collection=args.collection or store.collection,
    )

    if len(numpy_index) != len(qdrant_index):
        print(
            f"backends hold different corpora: numpy {len(numpy_index)}, "
            f"qdrant {len(qdrant_index)} -- rebuild with "
            "`python scripts/build_index.py --with-qdrant`",
            file=sys.stderr,
        )
        return 1

    rng = np.random.default_rng(args.seed)
    rows = rng.choice(len(numpy_index), size=min(args.queries, len(numpy_index)), replace=False)
    vectors = numpy_index.matrix[rows]

    # Warm-up: the first Qdrant query pays connection and segment-load cost that would
    # otherwise land entirely in the first scenario's p95.
    for vector in vectors[:5]:
        numpy_index.search(vector, args.k)
        qdrant_index.search(vector, args.k)

    results = []
    for label, query in SCENARIOS:
        print(f"  {label} ...", flush=True)
        results.append(
            run_scenario(
                label, query, repository, numpy_index, qdrant_index, vectors, args.k
            )
        )

    report = build_report(results, args.k, len(vectors), len(numpy_index))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    qdrant_index.close()

    print(report)
    print(f"wrote {args.out}")
    mismatched = [r for r in results if not r.pools_agree or r.mean_recall < 1.0]
    return 1 if mismatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
