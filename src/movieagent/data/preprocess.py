"""CSV -> normalized frame (ADR-0005).

This module is the single home for preprocessing decisions (R-018). Every choice here
is documented in ADR-0005's decision table; the code below is that table executed.

Run once via ``scripts/build_index.py``; the result is written to Parquet, which is
what makes the null/date/unknown semantics survive a reload.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from movieagent.data import schema as S
from movieagent.logging import get_logger

log = get_logger("preprocess")

_ARTICLE_RE = re.compile(r"^(the|a|an)\s+", flags=re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- helpers


def normalize_title(value: str | None) -> str:
    """Normalize a title for fuzzy matching (ADR-0009, R-042).

    Casefold, strip diacritics, drop punctuation, collapse whitespace, and move a
    leading English article to the end so ``the dark knight`` and ``dark knight, the``
    converge.

    The article rule is English-centric and does nothing for ``Le``/``La``/``Der`` --
    a known limitation, recorded rather than silently accepted.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    match = _ARTICLE_RE.match(text)
    if match:
        text = f"{text[match.end():].strip()} {match.group(1).strip()}"
    return text.strip()


@dataclass
class PreprocessReport:
    """Counts worth surfacing rather than swallowing. Written into the manifest so the
    documentation can quote real numbers (R-018) instead of estimates."""

    movies_in: int = 0
    credits_in: int = 0
    joined: int = 0
    unmatched_movies: int = 0
    unmatched_credits: int = 0
    malformed_json: dict[str, int] = field(default_factory=dict)
    missing_release_date: int = 0
    missing_runtime: int = 0
    missing_overview: int = 0
    zero_budget: int = 0
    zero_revenue: int = 0
    no_director: int = 0
    no_keywords: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "movies_in": self.movies_in,
            "credits_in": self.credits_in,
            "joined": self.joined,
            "unmatched_movies": self.unmatched_movies,
            "unmatched_credits": self.unmatched_credits,
            "malformed_json": self.malformed_json,
            "missing_release_date": self.missing_release_date,
            "missing_runtime": self.missing_runtime,
            "missing_overview": self.missing_overview,
            "zero_budget_treated_as_unknown": self.zero_budget,
            "zero_revenue_treated_as_unknown": self.zero_revenue,
            "no_director": self.no_director,
            "no_keywords": self.no_keywords,
        }


def _parse_json_column(series: pd.Series, column: str, report: PreprocessReport) -> pd.Series:
    """Parse a JSON-encoded column into Python objects (R-012).

    Malformed JSON yields an empty list and is *counted*, never raised and never
    silently dropped -- the count lands in the manifest.
    """
    malformed = 0

    def parse(raw: object) -> list:
        nonlocal malformed
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return []
        if isinstance(raw, list):
            return raw
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            malformed += 1
            return []
        return value if isinstance(value, list) else []

    parsed = series.map(parse)
    if malformed:
        report.malformed_json[column] = malformed
        log.warning("%s: %d malformed JSON values coerced to empty lists", column, malformed)
    return parsed


def _names(entries: list) -> list[str]:
    """Pull the ``name`` field out of TMDB's ``[{"id":.., "name":..}]` shape."""
    out: list[str] = []
    for entry in entries or []:
        if isinstance(entry, dict):
            name = entry.get("name")
            if isinstance(name, str) and name.strip():
                out.append(name.strip())
    return out


def _top_cast(entries: list, limit: int = S.TOP_CAST_SIZE) -> list[str]:
    """Top-billed cast by ``order`` (ADR-0008 / OQ-014).

    Sorting by ``order`` rather than trusting file order: the CSV is usually already
    ordered, but "usually" is not a guarantee and billing order is the whole point.
    """
    people = [e for e in entries or [] if isinstance(e, dict) and e.get("name")]
    people.sort(key=lambda e: e.get("order", 10_000) if isinstance(e.get("order"), int) else 10_000)
    return [str(p["name"]).strip() for p in people[:limit]]


def _all_cast(entries: list) -> list[str]:
    return _names(entries)


def _directors(crew: list) -> list[str]:
    """Directors from the crew blob. A list, not a scalar -- co-directed films are real
    (the Coens, the Wachowskis) and collapsing them to one name loses data."""
    out: list[str] = []
    for entry in crew or []:
        if isinstance(entry, dict) and entry.get("job") == "Director":
            name = entry.get("name")
            if isinstance(name, str) and name.strip():
                out.append(name.strip())
    return out


def file_sha256(path: Path) -> str:
    """Hash a source file so the manifest can detect a stale artifact (ADR-0005)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------------------ entrypoint


def build_movies_frame(
    movies_csv: Path, credits_csv: Path
) -> tuple[pd.DataFrame, PreprocessReport]:
    """Read both CSVs and produce the processed frame plus a report.

    Order of operations matters: join first, then parse, then normalize, then apply the
    missing-value policy last so nothing upstream can reintroduce a zero.
    """
    report = PreprocessReport()

    movies = pd.read_csv(movies_csv)
    credits = pd.read_csv(credits_csv)
    report.movies_in = len(movies)
    report.credits_in = len(credits)

    # --- join (R-011). Inner join on id, with unmatched rows counted, not ignored.
    credits = credits.rename(columns={"movie_id": "id"}).drop(columns=["title"], errors="ignore")
    merged = movies.merge(credits, on="id", how="inner", validate="one_to_one")
    report.joined = len(merged)
    report.unmatched_movies = report.movies_in - report.joined
    report.unmatched_credits = report.credits_in - report.joined
    if report.unmatched_movies or report.unmatched_credits:
        log.warning(
            "join dropped rows: %d movies, %d credits unmatched",
            report.unmatched_movies,
            report.unmatched_credits,
        )

    df = pd.DataFrame(index=merged.index)
    df[S.ID] = merged["id"].astype("int64")

    # --- titles (R-013, R-017). Raw titles preserved; normalized forms added alongside.
    df[S.TITLE] = merged["title"].fillna("").astype("string")
    df[S.ORIGINAL_TITLE] = merged["original_title"].fillna("").astype("string")
    df[S.TITLE_NORM] = df[S.TITLE].map(normalize_title).astype("string")
    df[S.ORIGINAL_TITLE_NORM] = df[S.ORIGINAL_TITLE].map(normalize_title).astype("string")

    # --- generation text, verbatim (R-017). No normalization, ever.
    df[S.OVERVIEW] = merged["overview"].astype("string")
    df[S.TAGLINE] = merged["tagline"].astype("string")
    report.missing_overview = int(
        df[S.OVERVIEW].isna().sum() + (df[S.OVERVIEW].fillna("").str.strip() == "").sum()
    )

    # --- JSON columns (R-012) -> normalized name lists (R-013).
    genres = _parse_json_column(merged["genres"], "genres", report)
    keywords = _parse_json_column(merged["keywords"], "keywords", report)
    companies = _parse_json_column(merged["production_companies"], "production_companies", report)
    countries = _parse_json_column(merged["production_countries"], "production_countries", report)
    languages = _parse_json_column(merged["spoken_languages"], "spoken_languages", report)
    cast = _parse_json_column(merged["cast"], "cast", report)
    crew = _parse_json_column(merged["crew"], "crew", report)

    df[S.GENRES] = genres.map(_names)
    df[S.KEYWORDS] = keywords.map(_names)
    df[S.COMPANIES] = companies.map(_names)
    df[S.COUNTRIES] = countries.map(_names)
    df[S.LANGUAGES] = languages.map(_names)
    df[S.TOP_CAST] = cast.map(_top_cast)
    df[S.FULL_CAST] = cast.map(_all_cast)
    df[S.DIRECTORS] = crew.map(_directors)

    report.no_director = int((df[S.DIRECTORS].map(len) == 0).sum())
    report.no_keywords = int((df[S.KEYWORDS].map(len) == 0).sum())

    # --- numerics. Nullable dtypes throughout; no fillna(0) anywhere (R-014).
    df[S.VOTE_AVERAGE] = pd.to_numeric(merged["vote_average"], errors="coerce").astype("Float64")
    df[S.VOTE_COUNT] = pd.to_numeric(merged["vote_count"], errors="coerce").astype("Int64")
    df[S.POPULARITY] = pd.to_numeric(merged["popularity"], errors="coerce").astype("Float64")

    runtime = pd.to_numeric(merged["runtime"], errors="coerce")
    # A runtime of 0 is a data-entry artifact, not a zero-length film.
    runtime = runtime.mask(runtime <= 0)
    df[S.RUNTIME] = runtime.astype("Int64")
    report.missing_runtime = int(df[S.RUNTIME].isna().sum())

    # --- budget/revenue: 0 means unknown (R-016, OQ-010).
    # Roughly a fifth of the dataset has a zero here. Treating those as $0 would put
    # them at the bottom of every "lowest revenue" sort and corrupt every average, so
    # they become NA with a companion flag that lets the UI say "unknown" honestly.
    for column, source in ((S.BUDGET, "budget"), (S.REVENUE, "revenue")):
        raw = pd.to_numeric(merged[source], errors="coerce")
        zeros = int((raw == 0).sum())
        known = raw.notna() & (raw > 0)
        df[column] = raw.mask(~known).astype("Int64")
        df[f"{column}_known"] = known.astype("boolean")
        if column == S.BUDGET:
            report.zero_budget = zeros
        else:
            report.zero_revenue = zeros

    # --- dates (R-015). Unparseable/empty -> NaT, and NaT never satisfies a filter.
    release = pd.to_datetime(merged["release_date"], errors="coerce", format="mixed")
    df[S.RELEASE_DATE] = release.dt.date.astype("object")
    df[S.RELEASE_YEAR] = release.dt.year.astype("Int64")
    report.missing_release_date = int(release.isna().sum())

    df[S.ORIGINAL_LANGUAGE] = merged["original_language"].astype("string")
    df[S.STATUS] = merged["status"].astype("string")
    df[S.HOMEPAGE] = merged["homepage"].astype("string")

    df = df.reset_index(drop=True)
    log.info(
        "processed %d movies (%d missing release_date, %d unknown budget, %d unknown revenue)",
        len(df),
        report.missing_release_date,
        report.zero_budget,
        report.zero_revenue,
    )
    return df, report
