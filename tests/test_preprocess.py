"""Preprocessing invariants (R-011 – R-020).

These are the requirements most easily broken by a well-meaning refactor, because the
"fix" for a null is almost always `fillna(0)` -- which is exactly what R-014 and R-016
forbid.
"""

from __future__ import annotations

import pandas as pd
import pytest

from movieagent.data import schema as S
from movieagent.data.preprocess import normalize_title


class TestJoin:
    def test_join_is_one_to_one_and_lossless(self, frame: pd.DataFrame) -> None:
        """R-011: the join must not silently drop or duplicate movies."""
        assert len(frame) == 4803
        assert frame[S.ID].is_unique

    def test_credits_columns_are_present(self, frame: pd.DataFrame) -> None:
        """The join is only useful if cast and crew actually arrived."""
        assert frame[S.FULL_CAST].map(len).sum() > 100_000
        assert (frame[S.DIRECTORS].map(len) > 0).sum() > 4_000


class TestJsonParsing:
    def test_json_columns_became_sequences_of_strings(self, frame: pd.DataFrame) -> None:
        """R-012/R-013: parsed into usable structures, not left as JSON strings."""
        for column in S.LIST_COLUMNS:
            sample = frame[column].iloc[0]
            assert not isinstance(sample, str), f"{column} is still a JSON string"
            assert all(isinstance(v, str) for v in sample)

    def test_repository_normalizes_sequences_to_lists(self, repo) -> None:
        """Guards a real bug, not a style preference.

        Parquet round-trips list columns as ``numpy.ndarray`` while the CSV path yields
        ``list``. ``value or []`` -- a natural default -- raises "truth value of an array
        is ambiguous" on an ndarray, so the same code would pass when built from CSV and
        crash when loaded from the artifact. The repository normalizes at its boundary so
        no call site has to know which shape it got.
        """
        for column in S.LIST_COLUMNS:
            value = repo.frame[column].iloc[0]
            assert isinstance(value, list), f"{column} was not normalized to a list"
            assert (value or []) == value  # the expression that fails on an ndarray

    def test_genres_use_dataset_wording(self, repo) -> None:
        """The exact wording matters: "Sci-Fi" would silently match nothing."""
        genres = set(repo.vocabulary(S.GENRES))
        assert "Science Fiction" in genres
        assert "Sci-Fi" not in genres

    def test_top_cast_is_billing_ordered_and_capped(self, frame: pd.DataFrame) -> None:
        """ADR-0008/OQ-014: top-billed only in the document, full cast in metadata."""
        assert frame[S.TOP_CAST].map(len).max() <= S.TOP_CAST_SIZE
        assert frame[S.FULL_CAST].map(len).max() > S.TOP_CAST_SIZE


class TestMissingValues:
    def test_no_column_silently_zero_filled(self, frame: pd.DataFrame) -> None:
        """R-014: absent must be representable as absent."""
        assert frame[S.RUNTIME].isna().sum() > 0
        assert frame[S.RELEASE_YEAR].isna().sum() > 0

    def test_runtime_is_nullable_integer(self, frame: pd.DataFrame) -> None:
        assert str(frame[S.RUNTIME].dtype) == "Int64"

    def test_release_date_is_a_real_date(self, frame: pd.DataFrame) -> None:
        """R-015: a date type, not a string that sorts lexicographically."""
        value = frame[S.RELEASE_DATE].dropna().iloc[0]
        assert hasattr(value, "year")

    def test_zero_budget_and_revenue_became_unknown(self, frame: pd.DataFrame) -> None:
        """R-016 / OQ-010 -- the load-bearing one.

        About a fifth of this dataset has budget 0 and nearly a third revenue 0. Treating
        those as $0 would put them at the bottom of every ranking and corrupt every
        average, so they are NA with a companion flag.
        """
        assert (frame[S.BUDGET] == 0).sum() == 0
        assert (frame[S.REVENUE] == 0).sum() == 0
        assert frame[S.BUDGET].isna().sum() > 1_000
        assert frame[S.REVENUE].isna().sum() > 1_000
        # The flag must agree with the value, or the UI would say "unknown" for a
        # figure it actually has.
        assert (frame[S.BUDGET_KNOWN] == frame[S.BUDGET].notna()).all()
        assert (frame[S.REVENUE_KNOWN] == frame[S.REVENUE].notna()).all()

    def test_generation_text_is_preserved_verbatim(self, frame: pd.DataFrame) -> None:
        """R-017: normalization must not touch what generation reads."""
        overview = frame.loc[frame[S.TITLE] == "Interstellar", S.OVERVIEW].iloc[0]
        assert overview[0].isupper()
        assert "." in overview


class TestTitleNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("The Dark Knight Rises", "dark knight rises the"),
            ("Amélie", "amelie"),
            ("Spider-Man 3", "spider man 3"),
            ("  WALL·E  ", "wall e"),
            ("A Beautiful Mind", "beautiful mind a"),
            ("", ""),
            (None, ""),
        ],
    )
    def test_normalizer(self, raw: str | None, expected: str) -> None:
        """R-042: casefold, strip diacritics and punctuation, move leading articles."""
        assert normalize_title(raw) == expected
