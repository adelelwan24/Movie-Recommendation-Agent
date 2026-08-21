"""Fuzzy title matching with explicit confidence bands (ADR-0009).

The PDF's rule (p3 §2, restated p5 §10) is not "find the best match" -- it is
*"do not silently return an arbitrary result when confidence is low."* So the output is
a **banded decision**, not a winner:

======================  ==================================================
score                   outcome
======================  ==================================================
exact normalized match  ``MATCH`` (short-circuit, score 100)
>= accept (82)          ``MATCH``
ambiguous..accept       ``AMBIGUOUS`` -- return candidates and ask
< ambiguous (65)        ``NOT_FOUND``
======================  ==================================================

Plus a rule that matters more than the thresholds: **if the top two candidates are
within ``tie_margin`` points of each other, the result is ambiguous regardless of
score.** The dangerous failure is not a low score, it is returning film A at 90 when
film B scored 89.

That rule turns out to carry more weight than the thresholds do: in calibration it was
the near-tie check, not the floor, that caught franchise queries like "lord of the
rings" and "the matrix", where three films all score 100.

Scoring is ``ratio``, with ``token_set_ratio`` admitted **only when the query's
significant tokens are a subset of the candidate's** -- see ADR-0025, which supersedes
ADR-0009's original ``max(WRatio, token_set_ratio)``. That combination inflated garbage:
"qqqzzz nonexistent film" scored 86 against "An Alan Smithee Film: Burn, Hollywood, Burn"
because both contain "film", and ``WRatio``'s internal partial matching rewards a short
fragment inside a long title. Plain ``ratio`` does not, and the subset gate recovers the
partial-title case that ``token_set_ratio`` was introduced for.

Thresholds are calibrated against a measured 14-case match set and a 6-case refusal set
(ADR-0025), not the PDF's five examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process as rf_process

from movieagent.config import FuzzySettings
from movieagent.data import schema as S
from movieagent.data.preprocess import normalize_title
from movieagent.data.repository import MovieRepository
from movieagent.data.schema import FuzzyCandidate, MovieRef


#: Tokens too common to establish that a query is "contained in" a title. Without this,
#: a bare "the" is a subset of almost every title and scores 100 via `token_set_ratio`.
_GATE_STOPWORDS = frozenset(
    {"the", "a", "an", "of", "and", "in", "on", "to", "for", "part", "ii", "iii", "iv"}
)

#: Below this length, approximate matching is not attempted -- only exact matching.
#: `ratio("the", "they")` is 86 and correctly so: at three characters, one edit is most
#: of the string. Short *exact* titles ("Up", "Her") still resolve, because the exact
#: short-circuit runs first. This is a length guard, not a threshold problem.
MIN_FUZZY_LENGTH = 4


class MatchOutcome(StrEnum):
    MATCH = "match"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"


@dataclass(frozen=True, slots=True)
class TitleMatch:
    outcome: MatchOutcome
    query: str
    normalized: str
    best: FuzzyCandidate | None
    candidates: list[FuzzyCandidate]
    reason: str


class FuzzyTitleMatcher:
    """Matches user-typed titles against the dataset.

    Immutable after construction; safe to share across sessions (ADR-0014).
    """

    def __init__(self, repository: MovieRepository, settings: FuzzySettings) -> None:
        self._repo = repository
        self._settings = settings
        frame = repository.frame

        # Two parallel choice lists so a film can be found by either title. Non-English
        # films are often searched by their original name.
        self._choices: list[str] = []
        self._positions: list[int] = []
        self._sources: list[str] = []
        self._exact: dict[str, list[tuple[int, str]]] = {}

        for column, label in ((S.TITLE_NORM, "title"), (S.ORIGINAL_TITLE_NORM, "original_title")):
            for position, value in enumerate(frame[column]):
                text = "" if pd.isna(value) else str(value)
                if not text:
                    continue
                self._choices.append(text)
                self._positions.append(position)
                self._sources.append(label)
                # Carry the source through the exact path too. Hardcoding "title" here
                # would make the trace misreport which field actually matched -- and the
                # trace is what a user reads to judge whether the system did the right
                # thing.
                self._exact.setdefault(text, []).append((position, label))

    @staticmethod
    def _score(query: str, choice: str, **_: object) -> float:
        """Edit distance, with subset-gated token matching (ADR-0025).

        ``ratio`` alone would fail the partial-title case -- "lord of the rings" against
        the full official title is dominated by the unmatched remainder. ``token_set_ratio``
        fixes that but, applied unconditionally, scores any query sharing one common token
        with a long title far too highly. Gating it on *significant-token containment*
        keeps the partial-title behaviour and drops the false positives.
        """
        base = float(fuzz.ratio(query, choice))
        query_tokens = set(query.split()) - _GATE_STOPWORDS
        if query_tokens and query_tokens <= set(choice.split()):
            return max(base, float(fuzz.token_set_ratio(query, choice)))
        return base

    def match(self, title: str) -> TitleMatch:
        """Resolve a title string to a movie, or refuse to guess."""
        raw = (title or "").strip()
        normalized = normalize_title(raw)

        if not normalized:
            return TitleMatch(
                outcome=MatchOutcome.NOT_FOUND,
                query=raw,
                normalized=normalized,
                best=None,
                candidates=[],
                reason="empty title",
            )

        # Exact normalized match short-circuits -- but only when it is unambiguous.
        # Several TMDB films share a normalized title across remakes.
        exact = self._exact.get(normalized)
        if exact:
            # One *movie*, not one entry: a film whose title and original_title are
            # identical appears twice here and is still unambiguous.
            positions = {position for position, _ in exact}
            if len(positions) == 1:
                position, source = exact[0]
                ref = self._repo.refs_at(np.array([position]))[0]
                candidate = FuzzyCandidate(ref=ref, score=100.0, matched_on=source)
                return TitleMatch(
                    outcome=MatchOutcome.MATCH,
                    query=raw,
                    normalized=normalized,
                    best=candidate,
                    candidates=[candidate],
                    reason=f"exact match on {source}",
                )

        if len(normalized) < MIN_FUZZY_LENGTH:
            return TitleMatch(
                outcome=MatchOutcome.NOT_FOUND,
                query=raw,
                normalized=normalized,
                best=None,
                candidates=[],
                reason=(
                    f"{raw!r} is too short to match approximately; at this length an "
                    "edit-distance score says nothing. Give more of the title."
                ),
            )

        raw_hits = rf_process.extract(
            normalized,
            self._choices,
            scorer=self._score,
            limit=self._settings.max_candidates * 4,
        )

        # Collapse to one entry per movie: a film matched on both title and
        # original_title must not occupy two candidate slots.
        best_per_movie: dict[int, FuzzyCandidate] = {}
        for _choice, score, index in raw_hits:
            position = self._positions[index]
            ref = self._repo.refs_at(np.array([position]))[0]
            existing = best_per_movie.get(ref.movie_id)
            if existing is None or score > existing.score:
                best_per_movie[ref.movie_id] = FuzzyCandidate(
                    ref=ref, score=float(score), matched_on=self._sources[index]
                )

        candidates = sorted(best_per_movie.values(), key=lambda c: -c.score)
        candidates = candidates[: self._settings.max_candidates]

        if not candidates:
            return TitleMatch(
                outcome=MatchOutcome.NOT_FOUND,
                query=raw,
                normalized=normalized,
                best=None,
                candidates=[],
                reason="no title scored above zero",
            )

        best = candidates[0]
        runner_up = candidates[1] if len(candidates) > 1 else None

        if best.score < self._settings.ambiguous:
            return TitleMatch(
                outcome=MatchOutcome.NOT_FOUND,
                query=raw,
                normalized=normalized,
                best=best,
                candidates=candidates,
                reason=(
                    f"best score {best.score:.0f} is below the "
                    f"{self._settings.ambiguous} floor"
                ),
            )

        near_tie = (
            runner_up is not None
            and (best.score - runner_up.score) <= self._settings.tie_margin
        )
        if near_tie:
            return TitleMatch(
                outcome=MatchOutcome.AMBIGUOUS,
                query=raw,
                normalized=normalized,
                best=None,
                candidates=candidates,
                reason=(
                    f"{best.ref.title!r} ({best.score:.0f}) and "
                    f"{runner_up.ref.title!r} ({runner_up.score:.0f}) are within "  # type: ignore[union-attr]
                    f"{self._settings.tie_margin} points -- refusing to guess"
                ),
            )

        if best.score >= self._settings.accept:
            return TitleMatch(
                outcome=MatchOutcome.MATCH,
                query=raw,
                normalized=normalized,
                best=best,
                candidates=candidates,
                reason=f"matched at {best.score:.0f}",
            )

        return TitleMatch(
            outcome=MatchOutcome.AMBIGUOUS,
            query=raw,
            normalized=normalized,
            best=None,
            candidates=candidates,
            reason=(
                f"best score {best.score:.0f} is between the "
                f"{self._settings.ambiguous} and {self._settings.accept} bands"
            ),
        )

    def resolve_person(self, name: str, limit: int = 5) -> list[str]:
        """Fuzzy-resolve a person's name against cast and crew vocabularies.

        Supports R-084's "How many movies have Christopher Nolan as director?", where a
        misspelled or partial name must become an exact vocabulary term before the
        structured filter can use it.
        """
        vocabulary = set(self._repo.vocabulary(S.DIRECTORS)) | set(
            self._repo.vocabulary(S.FULL_CAST)
        )
        hits = rf_process.extract(
            name, sorted(vocabulary), scorer=fuzz.WRatio, limit=limit, score_cutoff=80
        )
        return [hit[0] for hit in hits]


def refs_of(candidates: list[FuzzyCandidate]) -> list[MovieRef]:
    return [candidate.ref for candidate in candidates]
