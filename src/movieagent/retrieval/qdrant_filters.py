"""``SearchQuery`` -> Qdrant payload filter (ADR-0004, ADR-0011).

This module is the whole risk of moving to a vector database. ``MovieRepository.mask_for``
is the definition of what a filter *means*; everything below is a second implementation
of that same meaning in another engine's language, and a second implementation is a
second thing that can be wrong. Three semantics have to survive the translation:

* **Null never satisfies a comparison** (R-014, R-016). pandas gets this from
  ``fillna(False)``; Qdrant gets it by the payload key being *absent* for a missing
  value, since a ``Range`` cannot match a key that is not there. This is why the writer
  omits null fields rather than storing ``None`` -- a stored null would still be absent
  for range purposes, but omitting says it on purpose.
* **Membership is case-folded exact match**, not substring. The repository's term index
  keys on ``str.casefold()``, so payload lists are written case-folded and the filter
  case-folds its values the same way.
* **``people`` matches cast OR director**, which is the one disjunction the DSL has. It
  becomes a nested filter with ``should``, ANDed into the outer ``must``.

Every condition is ANDed, matching ADR-0004's deliberate expressive ceiling. If the DSL
ever grows a predicate this cannot express, raise ``UntranslatableFilter`` -- the caller
falls back to an explicit id list, which is always exact.
"""

from __future__ import annotations

from qdrant_client import models as qm

from movieagent.data import schema as S
from movieagent.data.query import ComparisonOp, Condition, SearchQuery

#: SearchQuery list attribute -> payload key. Mirrors ``repository._LIST_FILTERS``.
LIST_FILTERS: tuple[tuple[str, str], ...] = (
    ("genres", S.GENRES),
    ("keywords", S.KEYWORDS),
    ("companies", S.COMPANIES),
    ("countries", S.COUNTRIES),
    ("languages", S.LANGUAGES),
    ("directors", S.DIRECTORS),
)

#: Payload keys holding case-folded string lists, indexed as keywords.
KEYWORD_PAYLOAD_KEYS: tuple[str, ...] = (
    S.GENRES,
    S.KEYWORDS,
    S.COMPANIES,
    S.COUNTRIES,
    S.LANGUAGES,
    S.FULL_CAST,
    S.DIRECTORS,
)

#: Payload keys holding numbers, indexed as floats so ranges work on all of them.
NUMERIC_PAYLOAD_KEYS: tuple[str, ...] = (
    S.RELEASE_YEAR,
    S.RUNTIME,
    S.VOTE_AVERAGE,
    S.VOTE_COUNT,
    S.POPULARITY,
    S.BUDGET,
    S.REVENUE,
)


class UntranslatableFilter(Exception):
    """A predicate with no payload-filter equivalent. Caller falls back to id lists."""


def _range(op: ComparisonOp, value: float) -> qm.Range:
    """One numeric comparison as a Qdrant range.

    Equality goes through ``Range`` rather than ``MatchValue`` on purpose: the numeric
    payload is float-typed, and ``MatchValue`` on a float is an exact-representation
    match that would miss ``7.5`` stored as ``7.5000001``. A degenerate range is the
    same predicate without that trap.
    """
    match op:
        case ComparisonOp.GT:
            return qm.Range(gt=value)
        case ComparisonOp.GTE:
            return qm.Range(gte=value)
        case ComparisonOp.LT:
            return qm.Range(lt=value)
        case ComparisonOp.LTE:
            return qm.Range(lte=value)
        case ComparisonOp.EQ:
            return qm.Range(gte=value, lte=value)
        case _:  # pragma: no cover - BETWEEN is expanded by the caller
            raise UntranslatableFilter(f"unsupported operator {op}")


def _condition_clauses(condition: Condition) -> list[qm.FieldCondition]:
    key = condition.field_name.value
    if condition.op is ComparisonOp.BETWEEN:
        low, high = condition.value  # type: ignore[misc]
        return [qm.FieldCondition(key=key, range=qm.Range(gte=float(low), lte=float(high)))]
    return [qm.FieldCondition(key=key, range=_range(condition.op, float(condition.value)))]  # type: ignore[arg-type]


def _membership(key: str, values: list[str]) -> qm.FieldCondition:
    """Case-folded exact membership, matching the repository's term index."""
    return qm.FieldCondition(
        key=key, match=qm.MatchAny(any=[str(v).casefold() for v in values])
    )


def to_qdrant_filter(query: SearchQuery | None) -> qm.Filter | None:
    """Translate a validated query. ``None`` when nothing constrains the result set."""
    if query is None or query.is_empty():
        return None

    must: list[qm.Condition] = []

    for condition in query.conditions:
        must.extend(_condition_clauses(condition))

    if query.year_from is not None:
        must.append(
            qm.FieldCondition(key=S.RELEASE_YEAR, range=qm.Range(gte=float(query.year_from)))
        )
    if query.year_to is not None:
        must.append(
            qm.FieldCondition(key=S.RELEASE_YEAR, range=qm.Range(lte=float(query.year_to)))
        )

    for attribute, key in LIST_FILTERS:
        values = getattr(query, attribute)
        if values:
            must.append(_membership(key, values))

    if query.people:
        # The DSL's only disjunction: a person matches as cast or as director.
        must.append(
            qm.Filter(
                should=[
                    _membership(S.FULL_CAST, query.people),
                    _membership(S.DIRECTORS, query.people),
                ]
            )
        )

    # An aggregate spec constrains nothing about *which* rows match; it describes what to
    # compute afterwards, and structured_search owns that path.
    return qm.Filter(must=must) if must else None
