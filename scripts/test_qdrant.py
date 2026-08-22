"""Exercise the Qdrant collection with real queries and report retrieval latency.

    python scripts/test_qdrant.py                 # all cases
    python scripts/test_qdrant.py --repeats 10    # steadier timings
    python scripts/test_qdrant.py --case sci-fi   # only cases matching a substring
    python scripts/test_qdrant.py --quiet         # summary tables only

Different from ``benchmark_vector_backends.py``, which replays stored document vectors to
compare two backends. This one drives Qdrant the way the app does -- real text, embedded
at query time, through the same filter translation -- so the results are inspectable and
the timings include the split that matters in practice: embedding the query is usually
the larger half, and it is not the database's fault.

Latency is reported as median over repeats rather than a single sample, because the first
query against an embedded collection pays segment-load cost that says nothing about the
steady state.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:  # allow running without an editable install
    sys.path.insert(0, str(_SRC))

from movieagent.config import get_settings  # noqa: E402
from movieagent.data import schema as S  # noqa: E402
from movieagent.data.query import ComparisonOp, Condition, SearchQuery  # noqa: E402
from movieagent.data.repository import MovieRepository  # noqa: E402
from movieagent.data.schema import NumericField  # noqa: E402
from movieagent.errors import ArtifactError  # noqa: E402
from movieagent.llm.embeddings import build_embedding_backend  # noqa: E402
from movieagent.logging import configure_logging  # noqa: E402
from movieagent.retrieval.backend import Restriction  # noqa: E402
from movieagent.retrieval.qdrant_index import QdrantIndex  # noqa: E402


@dataclass(frozen=True)
class Case:
    """One retrieval to run: text or a seed movie, optionally filtered."""

    label: str
    query: str | None = None
    similar_to: int | None = None
    filters: SearchQuery | None = None
    note: str = ""


CASES: tuple[Case, ...] = (
    Case("plain-survival", query="someone trying to survive alone on another planet"),
    Case("plain-psychological", query="a dark psychological thriller that messes with your head"),
    Case("plain-timetravel", query="time travel and changing the past"),
    Case("plain-heist", query="a heist crew pulling off an impossible robbery"),
    Case(
        "filter-scifi-recent-short",
        query="funny science fiction",
        filters=SearchQuery(
            genres=["Science Fiction"],
            year_from=2011,
            conditions=[Condition(field=NumericField.RUNTIME, op=ComparisonOp.LT, value=120)],
        ),
        note="three filter kinds at once: membership, year range, numeric",
    ),
    Case(
        "filter-acclaimed-drama",
        query="a family torn apart by a secret",
        filters=SearchQuery(
            genres=["Drama"],
            conditions=[
                Condition(field=NumericField.VOTE_AVERAGE, op=ComparisonOp.GTE, value=7.5),
                Condition(field=NumericField.VOTE_COUNT, op=ComparisonOp.GTE, value=1000),
            ],
        ),
    ),
    Case(
        "filter-person",
        query="war and survival",
        filters=SearchQuery(people=["Tom Hanks"]),
        note="the DSL's only disjunction: cast OR director",
    ),
    Case(
        "filter-keyword",
        query="a villain with a plan",
        filters=SearchQuery(keywords=["superhero"]),
    ),
    Case(
        "filter-budget-range",
        query="spectacular visual effects",
        filters=SearchQuery(
            conditions=[
                Condition(
                    field=NumericField.BUDGET, op=ComparisonOp.BETWEEN, value=[150e6, 300e6]
                )
            ]
        ),
        note="unknown budgets must be excluded, not treated as 0",
    ),
    Case(
        "filter-narrow-pool",
        query="revenge",
        filters=SearchQuery(genres=["Western"], year_from=2011),
        note="ranking within a tiny pool is weakly meaningful -- the pool is reported",
    ),
    Case(
        "filter-empty-pool",
        query="revenge",
        filters=SearchQuery(genres=["Western"], year_from=2030),
        note="no candidates: must return nothing rather than ignoring the filter",
    ),
    Case("similar-inception", similar_to=27205, note="ranked by the seed's own vector"),
    Case(
        "similar-filtered",
        similar_to=27205,
        filters=SearchQuery(year_from=2015),
        note="similarity plus a hard constraint",
    ),
)

#: k values for the latency-vs-k sweep.
K_SWEEP: tuple[int, ...] = (1, 8, 25, 50, 100)


@dataclass
class Result:
    case: Case
    pool: int
    embed_ms: float
    search_ms: list[float] = field(default_factory=list)
    rows: list[tuple[int, str, str, str, float]] = field(default_factory=list)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.search_ms) if self.search_ms else float("nan")

    @property
    def min_ms(self) -> float:
        return min(self.search_ms) if self.search_ms else float("nan")

    @property
    def max_ms(self) -> float:
        return max(self.search_ms) if self.search_ms else float("nan")


# ------------------------------------------------------------------------------ output


def rule(char: str = "-", width: int = 78) -> str:
    return char * width


def table(headers: list[str], rows: list[list[str]], indent: str = "  ") -> str:
    if not rows:
        return f"{indent}(no rows)\n"
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))
    ]
    lines = [indent + "  ".join(h.ljust(w) for h, w in zip(headers, widths))]
    lines.append(indent + "  ".join("-" * w for w in widths))
    for row in rows:
        lines.append(indent + "  ".join(cell.ljust(w) for cell, w in zip(row, widths)))
    return "\n".join(lines) + "\n"


def describe_filters(query: SearchQuery | None) -> str:
    if query is None or query.is_empty():
        return "none"
    return "; ".join(query.describe())


# ----------------------------------------------------------------------------- running


def run_case(
    case: Case,
    index: QdrantIndex,
    repository: MovieRepository,
    embedder,
    k: int,
    repeats: int,
) -> Result:
    restriction = (
        Restriction.from_query(repository, case.filters)
        if case.filters is not None and not case.filters.is_empty()
        else None
    )

    vector = None
    embed_ms = 0.0
    if case.query is not None:
        start = time.perf_counter()
        vector = embedder.embed_query(case.query)
        embed_ms = (time.perf_counter() - start) * 1000

    pool = index.pool_size(restriction)
    result = Result(case=case, pool=pool, embed_ms=embed_ms)

    hits = []
    for _ in range(repeats):
        start = time.perf_counter()
        if case.similar_to is not None:
            position = repository.position(case.similar_to)
            hits = index.similar_to(position, k, restriction)
        else:
            hits = index.search(vector, k, restriction)
        result.search_ms.append((time.perf_counter() - start) * 1000)

    frame = repository.frame
    for rank, hit in enumerate(hits, 1):
        row = frame.iloc[hit.position]
        year = row[S.RELEASE_YEAR]
        result.rows.append(
            (
                rank,
                str(row[S.TITLE])[:38],
                "-" if year is None or year != year else str(int(year)),
                ", ".join(list(row[S.GENRES])[:3])[:30],
                hit.score,
            )
        )
    return result


def print_case(result: Result, k: int, repeats: int) -> None:
    case = result.case
    subject = (
        f'"{case.query}"' if case.query else f"movies similar to id {case.similar_to}"
    )
    print(rule("="))
    print(f"{case.label}")
    print(rule("="))
    print(f"  query    {subject}")
    print(f"  filters  {describe_filters(case.filters)}")
    if case.note:
        print(f"  why      {case.note}")
    print(
        f"  pool     {result.pool:,} of the corpus"
        + ("  <- filtered before ranking" if case.filters else "")
    )
    if case.query is not None:
        print(f"  embed    {result.embed_ms:7.2f} ms  (once per query)")
    print(
        f"  search   {result.median_ms:7.2f} ms median"
        f"   [{result.min_ms:.2f} - {result.max_ms:.2f}]  over {repeats} runs, k={k}"
    )
    print()

    if not result.rows:
        print("  no results -- the filter admits nothing, which is the correct answer\n")
        return
    print(
        table(
            ["#", "title", "year", "genres", "score"],
            [
                [str(rank), title, year, genres, f"{score:.4f}"]
                for rank, title, year, genres, score in result.rows
            ],
        )
    )


def print_summary(results: list[Result], k: int) -> None:
    print(rule("="))
    print("SUMMARY")
    print(rule("="))
    print(
        table(
            ["case", "pool", "hits", "embed ms", "search ms", "min", "max"],
            [
                [
                    r.case.label,
                    f"{r.pool:,}",
                    str(len(r.rows)),
                    f"{r.embed_ms:.2f}" if r.case.query else "-",
                    f"{r.median_ms:.2f}",
                    f"{r.min_ms:.2f}",
                    f"{r.max_ms:.2f}",
                ]
                for r in results
            ],
        )
    )

    searches = [r.median_ms for r in results]
    embeds = [r.embed_ms for r in results if r.case.query]
    filtered = [r for r in results if r.case.filters is not None]
    unfiltered = [r for r in results if r.case.filters is None]

    print(f"  search, all cases      median {statistics.median(searches):.2f} ms")
    if unfiltered:
        print(
            "  unfiltered             median "
            f"{statistics.median([r.median_ms for r in unfiltered]):.2f} ms"
        )
    if filtered:
        print(
            "  filtered               median "
            f"{statistics.median([r.median_ms for r in filtered]):.2f} ms"
        )
    if embeds:
        print(f"  embedding the query    median {statistics.median(embeds):.2f} ms")
    print(f"  k                      {k}\n")

    if filtered and unfiltered:
        ratio = statistics.median([r.median_ms for r in filtered]) / max(
            statistics.median([r.median_ms for r in unfiltered]), 1e-9
        )
        print(
            f"  Filtered queries are {ratio:.1f}x the cost of unfiltered ones. Embedded\n"
            "  Qdrant evaluates payload conditions in Python and ignores payload indexes,\n"
            "  so a filter adds work instead of removing it. On a server those same\n"
            "  filters run through indexed structures during traversal.\n"
        )


def print_k_sweep(
    index: QdrantIndex, repository: MovieRepository, vector, repeats: int
) -> None:
    print(rule("="))
    print("LATENCY vs k")
    print(rule("="))
    filters = SearchQuery(genres=["Drama"])
    restriction = Restriction.from_query(repository, filters)

    rows = []
    for k in K_SWEEP:
        plain, filtered = [], []
        for _ in range(repeats):
            start = time.perf_counter()
            index.search(vector, k)
            plain.append((time.perf_counter() - start) * 1000)

            start = time.perf_counter()
            index.search(vector, k, restriction)
            filtered.append((time.perf_counter() - start) * 1000)
        rows.append(
            [str(k), f"{statistics.median(plain):.2f}", f"{statistics.median(filtered):.2f}"]
        )
    print(table(["k", "unfiltered ms", "genre-filtered ms"], rows))
    print("  Asking for more neighbours is close to free; the filter is what costs.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=8, help="Results per query.")
    parser.add_argument("--repeats", type=int, default=5, help="Timed runs per case.")
    parser.add_argument("--case", type=str, default=None, help="Only cases matching this.")
    parser.add_argument("--quiet", action="store_true", help="Summary tables only.")
    parser.add_argument("--no-sweep", action="store_true", help="Skip the latency-vs-k sweep.")
    parser.add_argument("--qdrant-path", type=Path, default=None)
    args = parser.parse_args()

    configure_logging("WARNING")
    settings = get_settings()
    paths = settings.paths
    store = settings.vector_store

    cases = [c for c in CASES if not args.case or args.case in c.label]
    if not cases:
        print(f"no case matches {args.case!r}", file=sys.stderr)
        return 1

    try:
        index = QdrantIndex.connect(
            path=None if store.url else (args.qdrant_path or paths.qdrant_dir),
            url=store.url,
            api_key=store.api_key,
            collection=store.collection,
        )
    except ArtifactError as exc:
        print(exc, file=sys.stderr)
        return 1

    repository = MovieRepository.from_parquet(paths.movies_parquet)
    if len(index) != len(repository):
        print(
            f"collection has {len(index)} points but the dataset has {len(repository)} "
            "movies -- rebuild with `python scripts/build_index.py --with-qdrant`",
            file=sys.stderr,
        )
        return 1

    print(rule("="))
    print("QDRANT COLLECTION")
    print(rule("="))
    location = store.url or (args.qdrant_path or paths.qdrant_dir)
    print(f"  location    {location}")
    print(f"  mode        {'server' if store.url else 'embedded (single-process)'}")
    print(f"  collection  {index.collection}")
    print(f"  points      {len(index):,}   dimension {index.dimension}")
    print(f"  embedder    {settings.embedding.model}")
    print(f"  cases       {len(cases)}   k={args.k}   repeats={args.repeats}\n")

    embedder = build_embedding_backend(settings.embedding)

    # Warm-up: the first query pays model and segment-load costs that would otherwise be
    # charged to whichever case happened to run first.
    warm = embedder.embed_query("warm up the model and the collection")
    index.search(warm, args.k)

    results = []
    for case in cases:
        result = run_case(case, index, repository, embedder, args.k, args.repeats)
        results.append(result)
        if not args.quiet:
            print_case(result, args.k, args.repeats)

    print_summary(results, args.k)
    if not args.no_sweep:
        print_k_sweep(index, repository, warm, args.repeats)

    index.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
