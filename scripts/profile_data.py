"""Profile the raw TMDB 5000 CSVs and write a Markdown data-analysis report.

Run::

    python scripts/profile_data.py [--data-dir data] [--out artifacts/data_analysis.md]

This reads the *raw* CSVs in ``data/`` rather than the built parquet artifacts: the point
is to describe what the source actually contains -- column types, gaps, and the shape of
the embedded JSON blobs -- before preprocessing normalises any of it away.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# Cells in these columns hold serialised JSON rather than a scalar.
JSON_PREFIXES = ("[", "{")
SAMPLE_SIZE = 200
TOP_N = 10
EMPTY_TOKENS = {"", "[]", "{}", "nan", "None"}


# --------------------------------------------------------------------------- helpers


def parse_json_cell(value: object) -> object | None:
    """Return the decoded cell, or ``None`` when it is missing or unparseable."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def looks_like_json(series: pd.Series) -> bool:
    """True when most non-empty sampled cells decode as JSON containers."""
    sample = series.dropna().astype(str).str.strip()
    sample = sample[sample.str.startswith(JSON_PREFIXES)].head(SAMPLE_SIZE)
    if sample.empty:
        return False
    return sum(parse_json_cell(v) is not None for v in sample) >= 0.9 * len(sample)


def hashable(value: object) -> object:
    """Collapse nested containers to a stable string so they can live in a set."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return value


def missing_mask(series: pd.Series) -> pd.Series:
    """NaN, or a string that is blank / an empty JSON container.

    The dataset encodes "absent" three different ways, so counting NaN alone
    understates every text and JSON column.
    """
    if series.dtype == object:
        text = series.astype(str).str.strip()
        return series.isna() | text.isin(EMPTY_TOKENS)
    return series.isna()


def md_table(rows: list[list[object]], headers: list[str]) -> str:
    if not rows:
        return "_(none)_\n"
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = ["" if c is None else str(c).replace("|", "\\|") for c in row]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def fmt(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def truncate(value: object, width: int = 60) -> str:
    text = str(value).replace("\n", " ").replace("\r", " ")
    return text if len(text) <= width else text[:width] + "..."


# --------------------------------------------------------------------------- profiling


def profile_columns(df: pd.DataFrame, json_cols: set[str]) -> str:
    rows = []
    total = max(len(df), 1)
    for col in df.columns:
        series = df[col]
        na = int(series.isna().sum())
        blank = int(missing_mask(series).sum())
        uniques = series.astype(str) if col in json_cols else series
        nunique = int(uniques.nunique(dropna=True))
        present = series.dropna()
        sample = truncate(present.iloc[0]) if len(present) else ""
        rows.append(
            [
                f"`{col}`",
                "json (" + str(series.dtype) + ")" if col in json_cols else str(series.dtype),
                fmt(na),
                f"{na / total:.1%}",
                fmt(blank),
                f"{blank / total:.1%}",
                fmt(nunique),
                sample,
            ]
        )
    headers = [
        "column",
        "dtype",
        "nulls",
        "null %",
        "missing/empty",
        "missing %",
        "unique",
        "sample value",
    ]
    return md_table(rows, headers)


def profile_json_column(series: pd.Series) -> dict[str, object]:
    """Key inventory for one JSON column: which keys appear, and how varied their values are."""
    rows_with_key: Counter[str] = Counter()
    key_values: dict[str, set] = {}
    key_types: dict[str, Counter] = {}
    element_counts: list[int] = []
    shapes: Counter[str] = Counter()
    unparsed = 0

    for raw in series:
        decoded = parse_json_cell(raw)
        if decoded is None:
            unparsed += 1
            element_counts.append(0)
            continue
        records = decoded if isinstance(decoded, list) else [decoded]
        shapes["list of objects" if isinstance(decoded, list) else "single object"] += 1
        element_counts.append(len(records))
        seen_here: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                continue
            for key, value in record.items():
                seen_here.add(key)
                key_values.setdefault(key, set()).add(hashable(value))
                key_types.setdefault(key, Counter())[type(value).__name__] += 1
        for key in seen_here:
            rows_with_key[key] += 1

    total_rows = max(len(series), 1)
    total_elements = sum(element_counts)
    table_rows = []
    for key, n_rows in rows_with_key.most_common():
        types = "/".join(t for t, _ in key_types.get(key, Counter()).most_common())
        example = next(iter(sorted(map(str, key_values.get(key, set())))[:1]), "")
        table_rows.append(
            [
                f"`{key}`",
                types,
                fmt(len(key_values.get(key, set()))),
                fmt(n_rows),
                f"{n_rows / total_rows:.1%}",
                truncate(example, 40),
            ]
        )
    headers = ["key", "value type(s)", "unique values", "rows with key", "row coverage", "example value"]
    return {
        "table": md_table(table_rows, headers),
        "total_elements": total_elements,
        "empty_rows": sum(1 for c in element_counts if c == 0),
        "max_elements": max(element_counts) if element_counts else 0,
        "mean_elements": total_elements / len(element_counts) if element_counts else 0.0,
        "unparsed": unparsed,
        "shape": ", ".join(f"{k} ({fmt(v)} rows)" for k, v in shapes.most_common()) or "n/a",
    }


def crew_by_job(credits: pd.DataFrame, job: str) -> tuple[dict[object, set[str]], pd.Series]:
    """Distinct crew names holding ``job`` per ``movie_id``, pulled out of the crew blob.

    Names are de-duplicated per movie: TMDB sometimes lists the same person twice on one
    film under near-identical credits, and counting those as two people would inflate the
    "multiple X" tallies.
    """
    names: dict[object, set[str]] = {}
    for movie_id, crew in zip(credits["movie_id"], credits["crew"]):
        decoded = parse_json_cell(crew) or []
        names[movie_id] = {
            str(member.get("name", "")).strip()
            for member in decoded
            if isinstance(member, dict)
            and member.get("job") == job
            and str(member.get("name", "")).strip()
        }
    counts = pd.Series({k: len(v) for k, v in names.items()}, dtype="int64")
    return names, counts


# --------------------------------------------------------------------------- report


def build_report(movies: pd.DataFrame, credits: pd.DataFrame, paths: dict[str, Path]) -> str:
    parts: list[str] = []

    def add(text: str) -> None:
        parts.append(text if text.endswith("\n") else text + "\n")

    add("# TMDB 5000 - raw data analysis\n")
    add(
        f"Generated by `scripts/profile_data.py` from `{paths['movies'].name}` "
        f"and `{paths['credits'].name}`.\n"
    )

    add("## 1. Files\n")
    add(
        md_table(
            [
                [
                    f"`{p.name}`",
                    fmt(len(df)),
                    fmt(df.shape[1]),
                    fmt(p.stat().st_size // 1024) + " KB",
                    ", ".join(f"`{c}`" for c in df.columns),
                ]
                for p, df in ((paths["movies"], movies), (paths["credits"], credits))
            ],
            ["file", "rows", "columns", "size", "column names"],
        )
    )

    movie_json = {c for c in movies.columns if looks_like_json(movies[c])}
    credit_json = {c for c in credits.columns if looks_like_json(credits[c])}

    add("\n## 2. Column types and missing values\n")
    add(
        "`nulls` counts true NaN. `missing/empty` also counts blank strings and empty JSON "
        "containers (`[]`, `{}`), which is how this dataset actually encodes an absent value.\n"
    )
    add(f"\n### 2.1 `{paths['movies'].name}`\n")
    add(profile_columns(movies, movie_json))
    add(f"\n### 2.2 `{paths['credits'].name}`\n")
    add(profile_columns(credits, credit_json))

    add("\n## 3. JSON columns - keys and unique values per key\n")
    add(
        f"Detected JSON columns: "
        + ", ".join(f"`{c}`" for c in sorted(movie_json) + sorted(credit_json))
        + "\n"
    )
    for label, frame, cols in (
        (paths["movies"].name, movies, sorted(movie_json)),
        (paths["credits"].name, credits, sorted(credit_json)),
    ):
        for col in cols:
            info = profile_json_column(frame[col])
            add(f"\n### `{col}` ({label})\n")
            add(f"- Structure: {info['shape']}")
            add(
                f"- Nested records: {fmt(info['total_elements'])} total, "
                f"mean {info['mean_elements']:.2f} per movie, max {fmt(info['max_elements'])}"
            )
            add(f"- Rows with no records (empty or missing): {fmt(info['empty_rows'])}")
            if info["unparsed"]:
                add(f"- Rows that did not parse as JSON: {fmt(info['unparsed'])}")
            add("")
            add(info["table"])

    # -- multiple directors / producers ------------------------------------
    titles = credits.set_index("movie_id")["title"]
    popularity = (
        movies.set_index("id")["popularity"] if "popularity" in movies.columns else pd.Series(dtype=float)
    )

    for section, job, plural in ((4, "Director", "directors"), (5, "Producer", "producers")):
        name_map, counts = crew_by_job(credits, job)
        multi = counts[counts > 1]
        add(f"\n## {section}. Movies with multiple {plural}\n")
        add(
            f"- Movies with **more than one** {job.lower()}: **{fmt(int(len(multi)))}** "
            f"of {fmt(len(credits))} ({len(multi) / max(len(credits), 1):.1%})"
        )
        add(f"- Movies with exactly one {job.lower()}: {fmt(int((counts == 1).sum()))}")
        add(f"- Movies with no {job.lower()} credited: {fmt(int((counts == 0).sum()))}")
        add(
            f"- Most {plural} on a single movie: {fmt(int(counts.max()) if len(counts) else 0)}"
        )
        add(f"- Distinct {plural} across the dataset: "
            f"{fmt(len(set().union(*name_map.values())) if name_map else 0)}")
        add("")

        ranked = sorted(
            multi.index,
            key=lambda mid: (-int(multi[mid]), -float(popularity.get(mid, 0) or 0)),
        )[:TOP_N]
        add(f"Top {TOP_N} by {job.lower()} count (ties broken by popularity):\n")
        add(
            md_table(
                [
                    [
                        i,
                        titles.get(mid, ""),
                        mid,
                        fmt(int(multi[mid])),
                        truncate(", ".join(sorted(name_map[mid])), 200),
                    ]
                    for i, mid in enumerate(ranked, 1)
                ],
                ["#", "title", "movie_id", plural, "names"],
            )
        )

    # -- missing overview --------------------------------------------------
    add("\n## 6. Movies with a missing overview\n")
    missing_overview = movies[missing_mask(movies["overview"])]
    add(
        f"- Movies with no overview text: **{fmt(len(missing_overview))}** of {fmt(len(movies))} "
        f"({len(missing_overview) / max(len(movies), 1):.1%})"
    )
    add("")
    sort_col = "popularity" if "popularity" in missing_overview.columns else "vote_count"
    top_missing = missing_overview.sort_values(sort_col, ascending=False).head(TOP_N)
    add(f"Top {TOP_N} by popularity:\n")
    add(
        md_table(
            [
                [
                    i,
                    row.get("title", ""),
                    row.get("id", ""),
                    fmt(float(row.get("popularity", 0) or 0)),
                    fmt(int(row.get("vote_count", 0) or 0)),
                    row.get("release_date") if isinstance(row.get("release_date"), str) else "-",
                    row.get("status") if isinstance(row.get("status"), str) else "-",
                ]
                for i, (_, row) in enumerate(top_missing.iterrows(), 1)
            ],
            ["#", "title", "id", "popularity", "vote_count", "release_date", "status"],
        )
    )

    # -- localised titles --------------------------------------------------
    add("\n## 7. Rows where `title` differs from `original_title`\n")
    left = movies["title"].fillna("").astype(str).str.strip()
    right = movies["original_title"].fillna("").astype(str).str.strip()
    differs = left.ne(right)
    add(
        f"- Rows where the two differ: **{fmt(int(differs.sum()))}** of {fmt(len(movies))} "
        f"({differs.mean():.1%})"
    )
    add(
        "- Of those, differing only by case or surrounding whitespace: "
        f"{fmt(int((differs & left.str.casefold().eq(right.str.casefold())).sum()))}"
    )
    add("")

    by_language = (
        pd.DataFrame({"original_language": movies["original_language"], "differs": differs})
        .groupby("original_language", dropna=False)["differs"]
        .agg(["sum", "count"])
        .sort_values("sum", ascending=False)
    )
    by_language = by_language[by_language["sum"] > 0]
    add("Counted by `original_language` (languages with at least one differing row):\n")
    add(
        md_table(
            [
                [
                    lang if isinstance(lang, str) else "(missing)",
                    fmt(int(row["sum"])),
                    fmt(int(row["count"])),
                    f"{row['sum'] / max(int(row['count']), 1):.1%}",
                ]
                for lang, row in by_language.iterrows()
            ],
            ["original_language", "title != original_title", "movies in language", "share"],
        )
    )

    # -- status ------------------------------------------------------------
    add("\n## 8. Unique values for `status`\n")
    status_counts = movies["status"].fillna("(missing)").value_counts(dropna=False)
    add(f"- Distinct values: **{fmt(int(movies['status'].nunique(dropna=True)))}**")
    add("")
    add(
        md_table(
            [
                [f"`{value}`", fmt(int(n)), f"{n / max(len(movies), 1):.2%}"]
                for value, n in status_counts.items()
            ],
            ["status", "movies", "share"],
        )
    )

    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--out", type=Path, default=root / "artifacts" / "data_analysis.md")
    args = parser.parse_args()

    paths = {
        "movies": args.data_dir / "tmdb_5000_movies.csv",
        "credits": args.data_dir / "tmdb_5000_credits.csv",
    }
    for path in paths.values():
        if not path.exists():
            print(f"missing source file: {path}", file=sys.stderr)
            return 1

    movies = pd.read_csv(paths["movies"])
    credits = pd.read_csv(paths["credits"])

    report = build_report(movies, credits, paths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"wrote {args.out} ({len(report):,} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
