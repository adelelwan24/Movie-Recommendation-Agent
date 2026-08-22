"""Build a SQLite database from ``artifacts/movies.parquet`` and query it.

    python scripts/build_sqlite.py                       # build artifacts/movies.db
    python scripts/build_sqlite.py --query "SELECT ..."  # run one query
    python scripts/build_sqlite.py --samples             # run the example queries

Schema: one `movies` row per film for the scalar columns, plus a link table per
list-valued column (`genres`, `keywords`, `companies`, `countries`, `languages`,
`cast_members`, `directors`), each `(movie_id, name, ord)` where `ord` preserves the
source ordering -- billing order for cast, which is the one you want for "top billed".

Missing stays missing: runtime, budget and revenue are NULL rather than 0, because a
zero budget in the source meant "unknown" and summing it as zero would quietly understate
every average.

    sqlite3 artifacts/movies.db
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

#: parquet list column -> table name.
LIST_TABLES: dict[str, str] = {
    "genre_names": "genres",
    "keyword_names": "keywords",
    "company_names": "companies",
    "country_names": "countries",
    "language_names": "languages",
    "full_cast": "cast_members",
    "directors": "directors",
}

MOVIE_COLUMNS: tuple[str, ...] = (
    "id",
    "title",
    "original_title",
    "overview",
    "tagline",
    "release_date",
    "release_year",
    "original_language",
    "status",
    "runtime",
    "budget",
    "revenue",
    "vote_average",
    "vote_count",
    "popularity",
    "homepage",
)

SCHEMA = """
CREATE TABLE movies (
    id                INTEGER PRIMARY KEY,
    title             TEXT NOT NULL,
    original_title    TEXT,
    overview          TEXT,
    tagline           TEXT,
    release_date      TEXT,      -- ISO 8601, NULL when unknown
    release_year      INTEGER,
    original_language TEXT,
    status            TEXT,
    runtime           INTEGER,   -- NULL when unknown
    budget            INTEGER,   -- NULL when unknown (0 in the source meant unknown)
    revenue           INTEGER,   -- NULL when unknown
    vote_average      REAL,
    vote_count        INTEGER,
    popularity        REAL,
    homepage          TEXT
);
CREATE INDEX movies_year  ON movies(release_year);
CREATE INDEX movies_title ON movies(title);
"""

LINK_SCHEMA = """
CREATE TABLE {table} (
    movie_id INTEGER NOT NULL REFERENCES movies(id),
    name     TEXT    NOT NULL,
    ord      INTEGER NOT NULL   -- position in the source list
);
CREATE INDEX {table}_movie ON {table}(movie_id);
CREATE INDEX {table}_name  ON {table}(name);
"""

SAMPLE_QUERIES: tuple[tuple[str, str], ...] = (
    (
        "Top 10 rated films with at least 1000 votes",
        """
        SELECT title, release_year, vote_average, vote_count
        FROM movies
        WHERE vote_count >= 1000
        ORDER BY vote_average DESC, vote_count DESC
        LIMIT 10
        """,
    ),
    (
        "Films per genre",
        """
        SELECT g.name AS genre, COUNT(*) AS films, ROUND(AVG(m.vote_average), 2) AS avg_rating
        FROM genres g JOIN movies m ON m.id = g.movie_id
        GROUP BY g.name
        ORDER BY films DESC
        """,
    ),
    (
        "Directors with the most films (min 8)",
        """
        SELECT d.name AS director, COUNT(*) AS films,
               ROUND(AVG(m.vote_average), 2) AS avg_rating
        FROM directors d JOIN movies m ON m.id = d.movie_id
        GROUP BY d.name
        HAVING films >= 8
        ORDER BY films DESC, avg_rating DESC
        LIMIT 10
        """,
    ),
    (
        "Best return on budget (both known, budget over $10M)",
        """
        SELECT title, release_year, budget, revenue,
               ROUND(CAST(revenue AS REAL) / budget, 1) AS multiple
        FROM movies
        WHERE budget > 10000000 AND revenue IS NOT NULL
        ORDER BY multiple DESC
        LIMIT 10
        """,
    ),
    (
        "Runtime and output by decade",
        """
        SELECT (release_year / 10) * 10 AS decade, COUNT(*) AS films,
               ROUND(AVG(runtime), 1) AS avg_runtime
        FROM movies
        WHERE release_year IS NOT NULL
        GROUP BY decade
        ORDER BY decade DESC
        LIMIT 8
        """,
    ),
    (
        "Most frequent top-billed actors",
        """
        SELECT name AS actor, COUNT(*) AS lead_roles
        FROM cast_members
        WHERE ord = 0
        GROUP BY name
        ORDER BY lead_roles DESC
        LIMIT 10
        """,
    ),
    (
        "Leading actors by the films they headline (min 10 leads)",
        """
        SELECT c.name AS actor,
               COUNT(*) AS leads,
               ROUND(AVG(m.vote_average), 2) AS avg_rating,
               ROUND(SUM(m.revenue) / 1e9, 2) AS box_office_bn,
               MIN(m.release_year) || '-' || MAX(m.release_year) AS span
        FROM cast_members c
        JOIN movies m ON m.id = c.movie_id
        WHERE c.ord = 0 AND m.vote_count >= 100
        GROUP BY c.name
        HAVING leads >= 10
        ORDER BY avg_rating DESC
        LIMIT 10
        """,
    ),
    (
        "Sci-fi since 2010, under two hours, well rated",
        """
        SELECT m.title, m.release_year, m.runtime, m.vote_average
        FROM movies m JOIN genres g ON g.movie_id = m.id
        WHERE g.name = 'Science Fiction'
          AND m.release_year >= 2010
          AND m.runtime < 120
          AND m.vote_count >= 500
        ORDER BY m.vote_average DESC
        LIMIT 10
        """,
    ),
)


def build(parquet: Path, db_path: Path) -> sqlite3.Connection:
    frame = pd.read_parquet(parquet)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)

    connection = sqlite3.connect(db_path)
    connection.executescript(SCHEMA)
    for table in LIST_TABLES.values():
        connection.executescript(LINK_SCHEMA.format(table=table))

    movies = frame[list(MOVIE_COLUMNS)].astype(object).where(frame[list(MOVIE_COLUMNS)].notna(), None)
    movies["release_date"] = [
        None if d is None else str(d) for d in movies["release_date"]
    ]
    connection.executemany(
        f"INSERT INTO movies VALUES ({','.join('?' * len(MOVIE_COLUMNS))})",
        movies.itertuples(index=False, name=None),
    )

    for column, table in LIST_TABLES.items():
        rows = [
            (int(movie_id), str(name), position)
            for movie_id, values in zip(frame["id"], frame[column])
            for position, name in enumerate(values if values is not None else ())
        ]
        connection.executemany(f"INSERT INTO {table} VALUES (?, ?, ?)", rows)

    connection.commit()
    return connection


def show(connection: sqlite3.Connection, sql: str, limit: int = 50) -> None:
    """Print a result set as a plain aligned table."""
    cursor = connection.execute(sql)
    headers = [d[0] for d in cursor.description]
    rows = [[("" if v is None else str(v)) for v in row] for row in cursor.fetchmany(limit)]
    widths = [
        min(45, max(len(h), *(len(r[i]) for r in rows)) if rows else len(h))
        for i, h in enumerate(headers)
    ]
    line = "  ".join(h[:w].ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(cell[:w].ljust(w) for cell, w in zip(row, widths)))
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=ROOT / "artifacts" / "movies.parquet")
    parser.add_argument("--db", type=Path, default=ROOT / "artifacts" / "movies.db")
    parser.add_argument("--query", type=str, default=None, help="Run one SQL statement.")
    parser.add_argument("--samples", action="store_true", help="Run the example queries.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild even if the db exists.")
    args = parser.parse_args()

    if args.query or args.samples:
        if not args.db.exists():
            print(f"no database at {args.db}; build it first", file=sys.stderr)
            return 1
        connection = sqlite3.connect(args.db)
    else:
        if not args.parquet.exists():
            print(
                f"missing {args.parquet}; run `python scripts/build_index.py` first",
                file=sys.stderr,
            )
            return 1
        if args.db.exists() and not args.rebuild:
            print(f"{args.db} already exists; pass --rebuild to replace it")
            return 0
        connection = build(args.parquet, args.db)
        counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("movies", *LIST_TABLES.values())
        }
        print(f"built {args.db} ({args.db.stat().st_size / 1e6:.1f} MB)")
        for table, count in counts.items():
            print(f"  {table:14} {count:>7,} rows")
        print("\nTry:  python scripts/build_sqlite.py --samples")

    if args.query:
        show(connection, args.query)
    if args.samples:
        for title, sql in SAMPLE_QUERIES:
            print(f"### {title}")
            show(connection, sql)

    connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
