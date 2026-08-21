"""Semantic document construction (ADR-0008).

The PDF forbids embedding raw CSV rows (R-053) and discourages chunking short overviews
(R-055), so this is one curated, labelled document per movie.

**What is embedded** -- title (+year), original title when it differs, tagline, genres,
keywords, director, top-billed cast, overview.

**What is deliberately *not* embedded** -- production companies, countries, spoken
languages, the full cast, and every numeric field. Those are kept as metadata because
they are better served *exactly* by structured filters (ADR-0004). That is the field
allocation principle behind the whole design: **embed what is fuzzy and thematic, filter
what is exact and categorical.** It is also the answer to R-054's "how do metadata
filters interact with vector retrieval" -- those metadata fields are precisely what
ADR-0011's pre-filter masks on.

Known costs, recorded rather than hidden:

* the labels repeat across all ~4,800 documents, which raises the *floor* of pairwise
  similarity and is why the low-confidence threshold has to be calibrated empirically
  rather than set to an intuitive round number;
* TMDB keyword coverage is uneven, so films with no keywords and a short overview
  produce thin documents that rarely retrieve -- a systematic recall gap that correlates
  with obscurity.
"""

from __future__ import annotations

import pandas as pd

from movieagent.data import schema as S

#: Bumped whenever the template changes. Stamped into the manifest, because a template
#: change invalidates every embedding built with the old one.
DOCUMENT_TEMPLATE_VERSION = "1"


def _clean(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if value is pd.NA:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def _joined(values: object, limit: int | None = None) -> str:
    if values is None:
        return ""
    items = [str(v).strip() for v in values if str(v).strip()]
    if limit is not None:
        items = items[:limit]
    return ", ".join(items)


def build_document(row: pd.Series) -> str:
    """Render one movie as its semantic document.

    Empty fields are omitted rather than emitted as ``Field:`` with nothing after --
    a blank label is noise that every document would share.
    """
    lines: list[str] = []

    title = _clean(row[S.TITLE])
    year = row[S.RELEASE_YEAR]
    header = f"{title} ({int(year)})" if pd.notna(year) else title
    lines.append(f"Title: {header}")

    original = _clean(row[S.ORIGINAL_TITLE])
    if original and original.casefold() != title.casefold():
        lines.append(f"Also known as: {original}")

    if tagline := _clean(row[S.TAGLINE]):
        lines.append(f"Tagline: {tagline}")

    if genres := _joined(row[S.GENRES]):
        lines.append(f"Genres: {genres}")

    # Keywords carry TMDB's own thematic vocabulary ("dystopia", "time travel", "loss
    # of loved one"), which is frequently absent from the overview prose. This is the
    # field that makes R-051's theme-shaped queries work, and the reason ADR-0008
    # rejected the cleaner overview-only option.
    if keywords := _joined(row[S.KEYWORDS], limit=20):
        lines.append(f"Keywords: {keywords}")

    if directors := _joined(row[S.DIRECTORS]):
        lines.append(f"Director: {directors}")

    if cast := _joined(row[S.TOP_CAST]):
        lines.append(f"Starring: {cast}")

    if overview := _clean(row[S.OVERVIEW]):
        lines.append(f"Overview: {overview}")

    return "\n".join(lines)


def build_documents(frame: pd.DataFrame) -> list[str]:
    """Documents for every row, in frame order.

    Row order is load-bearing: it is the same order the embedding matrix uses, so the
    index can map a row position straight back to a movie (ADR-0006).
    """
    return [build_document(row) for _, row in frame.iterrows()]
