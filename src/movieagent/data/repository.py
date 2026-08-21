"""Read-only access to the processed dataset (ADR-0005).

This is the seam. Tools call ``repo.search(query)``; nothing above this module writes
pandas idioms, which is what would make a later swap to DuckDB a one-module change.

**Immutability is a hard constraint, not a style preference.** ADR-0014 shares this
object across Streamlit sessions and threads via ``@st.cache_resource``, so any lazy
cache or mutable attribute added here becomes a cross-user data race.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process as rf_process

from movieagent.data import schema as S
from movieagent.data.manifest import Manifest
from movieagent.data.query import (
    AggregateMetric,
    AggregateSpec,
    ComparisonOp,
    Condition,
    GroupBy,
    SearchQuery,
)
from movieagent.data.schema import MovieRef
from movieagent.errors import ArtifactError
from movieagent.logging import get_logger

log = get_logger("repository")

#: Columns shown in result tables. Full records come from `movie_details`.
SUMMARY_COLUMNS: tuple[str, ...] = (
    S.ID,
    S.TITLE,
    S.RELEASE_YEAR,
    S.GENRES,
    S.VOTE_AVERAGE,
    S.VOTE_COUNT,
    S.RUNTIME,
)

#: Which SearchQuery list-filter maps onto which dataset column.
_LIST_FILTERS: tuple[tuple[str, str], ...] = (
    ("genres", S.GENRES),
    ("keywords", S.KEYWORDS),
    ("companies", S.COMPANIES),
    ("countries", S.COUNTRIES),
    ("languages", S.LANGUAGES),
    ("directors", S.DIRECTORS),
)


def _normalize_list_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce list columns to real Python lists.

    Parquet round-trips list columns as ``numpy.ndarray``, and the CSV path produces
    ``list``. That difference is not cosmetic: ``value or []`` -- a natural way to write
    a default -- raises ``ValueError: truth value of an array is ambiguous`` on an
    ndarray, so identical code would work when built from CSV and crash when loaded from
    the artifact. Normalizing once here removes the whole class of bug rather than asking
    every call site to remember which shape it has.
    """
    out = frame.copy()
    for column in S.LIST_COLUMNS:
        if column in out.columns:
            out[column] = out[column].map(
                lambda v: [] if v is None else [str(x) for x in v]
            )
    return out


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A structured-search outcome.

    ``total`` is the true match count and ``rows`` is capped at ``query.limit``
    (ADR-0004 / OQ-007). Keeping both means a truncated view can never masquerade as
    the complete answer -- the tool always reports "312 match; showing 25".
    """

    rows: pd.DataFrame
    total: int
    refs: list[MovieRef]
    excluded_unknown: int = 0
    binding: list[str] | None = None


class MovieRepository:
    """Immutable, in-memory access to the processed frame."""

    def __init__(self, frame: pd.DataFrame, manifest: Manifest | None = None) -> None:
        self._df = _normalize_list_columns(frame.reset_index(drop=True))
        self._manifest = manifest
        # Exploded long-form views, built once. ADR-0005 sketched these as separate
        # artifacts; at 4,800 rows building them at load costs milliseconds and keeps
        # the artifact set to one file, so they are derived here instead.
        self._exploded: dict[str, pd.DataFrame] = {
            column: self._explode(column) for column in S.LIST_COLUMNS
        }
        # Lowercase term -> row positions, for exact membership filtering.
        self._term_index: dict[str, dict[str, np.ndarray]] = {
            column: self._build_term_index(column) for column in S.LIST_COLUMNS
        }
        self._id_to_pos: dict[int, int] = {
            int(mid): pos for pos, mid in enumerate(self._df[S.ID].to_numpy())
        }

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_parquet(cls, path: Path, manifest: Manifest | None = None) -> MovieRepository:
        if not path.exists():
            raise ArtifactError(
                f"No processed dataset at {path}.\n"
                "  Build it first:  python scripts/build_index.py"
            )
        return cls(pd.read_parquet(path), manifest)

    def _explode(self, column: str) -> pd.DataFrame:
        exploded = self._df[[S.ID, column, S.VOTE_AVERAGE, S.REVENUE]].explode(column)
        exploded = exploded[exploded[column].notna()]
        return exploded.rename(columns={column: "term"})

    def _build_term_index(self, column: str) -> dict[str, np.ndarray]:
        index: dict[str, list[int]] = {}
        for pos, values in enumerate(self._df[column]):
            for value in values if values is not None else ():
                index.setdefault(str(value).casefold(), []).append(pos)
        return {term: np.asarray(rows, dtype=np.int64) for term, rows in index.items()}

    # -------------------------------------------------------------------- properties

    @property
    def frame(self) -> pd.DataFrame:
        """The processed frame. Treat as read-only."""
        return self._df

    @property
    def manifest(self) -> Manifest | None:
        return self._manifest

    def __len__(self) -> int:
        return len(self._df)

    def vocabulary(self, column: str) -> list[str]:
        """Distinct values for a list column, most common first."""
        return list(self._exploded[column]["term"].value_counts().index)

    # ------------------------------------------------------------------- validation

    def validate(self, query: SearchQuery) -> list[str]:
        """Check a query's *values* against the dataset's real vocabularies (R-038).

        Pydantic validates structure; only the data knows whether "Sci-Fi" is a genre.
        That distinction matters because the failure that actually bites on this
        dataset is not a syntax error but a *plausible* term with the wrong wording --
        which SQL would have answered with zero rows and total confidence.

        Returns a list of human-readable problems, each with a suggestion where one
        exists. Empty list means the query is executable.
        """
        problems: list[str] = []
        checks = (
            (query.genres, S.GENRES, "genre"),
            (query.companies, S.COMPANIES, "production company"),
            (query.countries, S.COUNTRIES, "country"),
            (query.languages, S.LANGUAGES, "language"),
            (query.directors, S.DIRECTORS, "director"),
        )
        for values, column, label in checks:
            for value in values:
                if value.casefold() in self._term_index[column]:
                    continue
                suggestion = self._suggest(column, value)
                if suggestion:
                    problems.append(
                        f"unknown {label} {value!r}; closest match is {suggestion!r}"
                    )
                else:
                    problems.append(f"unknown {label} {value!r}")

        for value in query.people:
            if (
                value.casefold() in self._term_index[S.FULL_CAST]
                or value.casefold() in self._term_index[S.DIRECTORS]
            ):
                continue
            suggestion = self._suggest(S.FULL_CAST, value) or self._suggest(S.DIRECTORS, value)
            problems.append(
                f"unknown person {value!r}"
                + (f"; closest match is {suggestion!r}" if suggestion else "")
            )
        return problems

    @staticmethod
    def _suggest_score(query: str, choice: str, **_: object) -> float:
        """Combined scorer for vocabulary suggestions.

        ``partial_ratio`` carries the abbreviation cases that matter most here:
        "sci-fi" scores 67 against "science fiction" where ``WRatio`` gives 60 and
        ``token_set_ratio`` only 29, because the tokens genuinely differ.
        """
        return max(
            fuzz.WRatio(query, choice),
            fuzz.token_set_ratio(query, choice),
            fuzz.partial_ratio(query, choice),
        )

    def _suggest(self, column: str, value: str, cutoff: int = 65) -> str | None:
        """Best vocabulary suggestion for a near-miss term, or None.

        The cutoff is deliberately loose. This is advisory text inside an error the
        agent has already been told to correct, so a wrong suggestion costs one retry
        while a missing one costs the user an unexplained empty result.
        """
        match = rf_process.extractOne(
            value.casefold(),
            list(self._term_index[column].keys()),
            scorer=self._suggest_score,
            score_cutoff=cutoff,
        )
        if match is None:
            return None
        # Recover the original casing from the exploded view.
        term = match[0]
        rows = self._term_index[column][term]
        for original in self._df.iloc[rows[0]][column]:
            if str(original).casefold() == term:
                return str(original)
        return term

    # ------------------------------------------------------------------------- masks

    def mask_for(self, query: SearchQuery) -> np.ndarray:
        """Boolean row mask for a query (ADR-0004, and the hybrid pre-filter of ADR-0011).

        The same mask serves structured search and semantic pre-filtering, which is
        exactly why hybrid retrieval is correct by construction here: it is an array
        operation over known row order, not a database feature whose filter semantics
        we would have to verify.
        """
        mask = np.ones(len(self._df), dtype=bool)

        for condition in query.conditions:
            mask &= self._condition_mask(condition)

        if query.year_from is not None:
            mask &= self._numeric_mask(S.RELEASE_YEAR, ComparisonOp.GTE, query.year_from)
        if query.year_to is not None:
            mask &= self._numeric_mask(S.RELEASE_YEAR, ComparisonOp.LTE, query.year_to)

        for attribute, column in _LIST_FILTERS:
            values = getattr(query, attribute)
            if values:
                mask &= self._membership_mask(column, values)

        if query.people:
            cast_mask = self._membership_mask(S.FULL_CAST, query.people)
            director_mask = self._membership_mask(S.DIRECTORS, query.people)
            mask &= cast_mask | director_mask

        return mask

    def _condition_mask(self, condition: Condition) -> np.ndarray:
        column = condition.field_name.value
        if condition.op is ComparisonOp.BETWEEN:
            low, high = condition.value  # type: ignore[misc]
            return self._numeric_mask(column, ComparisonOp.GTE, low) & self._numeric_mask(
                column, ComparisonOp.LTE, high
            )
        return self._numeric_mask(column, condition.op, float(condition.value))  # type: ignore[arg-type]

    def _numeric_mask(self, column: str, op: ComparisonOp, value: float) -> np.ndarray:
        series = self._df[column]
        match op:
            case ComparisonOp.GT:
                result = series > value
            case ComparisonOp.GTE:
                result = series >= value
            case ComparisonOp.LT:
                result = series < value
            case ComparisonOp.LTE:
                result = series <= value
            case ComparisonOp.EQ:
                result = series == value
            case _:  # pragma: no cover - BETWEEN handled by the caller
                raise ValueError(f"unsupported operator {op}")
        # `fillna(False)` is R-014: a null never satisfies a comparison. It is also
        # R-016 for budget/revenue, whose zeros became NA in preprocessing -- so
        # "budget above $100M" is unaffected by unknown budgets rather than treating
        # them as $0.
        return result.fillna(False).to_numpy(dtype=bool)

    def _membership_mask(self, column: str, values: list[str]) -> np.ndarray:
        mask = np.zeros(len(self._df), dtype=bool)
        index = self._term_index[column]
        for value in values:
            rows = index.get(value.casefold())
            if rows is not None:
                mask[rows] = True
        return mask

    # ---------------------------------------------------------------------- querying

    def search(self, query: SearchQuery) -> SearchResult:
        """Execute a structured query. Deterministic; no LLM involved (R-030)."""
        mask = self.mask_for(query)
        matched = self._df[mask]
        excluded_unknown = 0

        if query.sort_by is not None:
            column = query.sort_by.value
            before = len(matched)
            # R-014/R-016: rows with an unknown value are *excluded* from a ranking
            # rather than sorted to one end. "Top 10 by revenue" must not be shaped by
            # films whose revenue we simply do not know.
            matched = matched[matched[column].notna()]
            excluded_unknown = before - len(matched)
            matched = matched.sort_values(
                column, ascending=not query.sort_desc, kind="mergesort"
            )

        total = len(matched)
        rows = matched.head(query.limit)
        return SearchResult(
            rows=rows[list(SUMMARY_COLUMNS)].copy(),
            total=total,
            refs=self.refs_from(rows),
            excluded_unknown=excluded_unknown,
            binding=query.describe() or None,
        )

    def aggregate(self, query: SearchQuery) -> pd.DataFrame:
        """Group-and-measure over the filtered subset (R-036).

        List fields are exploded first, so a film in three genres counts once per genre
        -- which is what "how many movies are there in each genre?" means.
        """
        spec = query.aggregate
        if spec is None:  # pragma: no cover - guarded by the caller
            raise ValueError("aggregate() requires query.aggregate")

        mask = self.mask_for(query)
        ids = set(self._df.loc[mask, S.ID].tolist())

        if spec.group_by is GroupBy.RELEASE_YEAR:
            source = self._df.loc[mask, [S.ID, S.RELEASE_YEAR, S.VOTE_AVERAGE, S.REVENUE]]
            source = source[source[S.RELEASE_YEAR].notna()]
            source = source.rename(columns={S.RELEASE_YEAR: "term"})
        else:
            source = self._exploded[spec.group_by.value]
            source = source[source[S.ID].isin(ids)]

        grouped = self._apply_metric(source, spec)
        grouped = grouped.sort_values(
            grouped.columns[1], ascending=not spec.sort_desc, kind="mergesort"
        )
        return grouped.head(spec.limit).reset_index(drop=True)

    @staticmethod
    def _apply_metric(source: pd.DataFrame, spec: AggregateSpec) -> pd.DataFrame:
        label = spec.group_by.value
        match spec.metric:
            case AggregateMetric.COUNT:
                out = source.groupby("term", dropna=True).size().reset_index(name="movie_count")
            case AggregateMetric.AVG_RATING:
                out = (
                    source.groupby("term", dropna=True)[S.VOTE_AVERAGE]
                    .mean()
                    .round(2)
                    .reset_index(name="avg_rating")
                )
            case AggregateMetric.SUM_REVENUE:
                # Unknown revenue is NA, so `sum` skips it rather than adding zero.
                out = (
                    source.groupby("term", dropna=True)[S.REVENUE]
                    .sum()
                    .reset_index(name="total_revenue")
                )
            case _:  # pragma: no cover
                raise ValueError(f"unsupported metric {spec.metric}")
        return out.rename(columns={"term": label})

    # ----------------------------------------------------------------------- lookups

    def get(self, movie_id: int) -> pd.Series | None:
        pos = self._id_to_pos.get(int(movie_id))
        return None if pos is None else self._df.iloc[pos]

    def position(self, movie_id: int) -> int | None:
        """Row position for a movie id -- the bridge to the vector index's row order."""
        return self._id_to_pos.get(int(movie_id))

    def ref(self, movie_id: int) -> MovieRef | None:
        row = self.get(movie_id)
        return None if row is None else self._ref_from_row(row)

    def refs_from(self, rows: pd.DataFrame) -> list[MovieRef]:
        return [self._ref_from_row(row) for _, row in rows.iterrows()]

    def refs_at(self, positions: np.ndarray) -> list[MovieRef]:
        return [self._ref_from_row(self._df.iloc[int(p)]) for p in positions]

    @staticmethod
    def _ref_from_row(row: pd.Series) -> MovieRef:
        year = row[S.RELEASE_YEAR]
        return MovieRef(
            movie_id=int(row[S.ID]),
            title=str(row[S.TITLE]),
            year=int(year) if pd.notna(year) else None,
        )

    def summary_rows(self, movie_ids: list[int]) -> pd.DataFrame:
        """Summary table for an explicit id list, preserving the given order."""
        positions = [self._id_to_pos[i] for i in movie_ids if i in self._id_to_pos]
        return self._df.iloc[positions][list(SUMMARY_COLUMNS)].copy()
