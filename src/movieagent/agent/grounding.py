"""Post-hoc grounding check (ADR-0012, layer 3).

Layers 1 and 2 -- ``rag_answer``'s id-scoped context and the prompt constraints -- do
most of the work. This is the tripwire for the residual case where the model reaches
past its context anyway, and its real contribution is that it makes R-004 **testable**.
"What we asked the model nicely" is not something a traceability matrix can point at.

Scope, stated plainly rather than implied:

* It catches invented **entities** -- a movie named that was not in the payload.
* It does **not** catch invented **attributes**. If the payload says 169 minutes and the
  answer says 195, no title is unmatched and nothing fires. Layers 1 and 2 are the only
  defence there, and ADR-0012 names an LLM-judge pass as the remedy if that turns out to
  matter.
* Common-word titles (*Up*, *Her*, *Alien*) are largely invisible to it.
* False positives are **advisory**, shown alongside the answer rather than blocking it.
  The rate matters, though: ADR-0012 set a revisit trigger at ~1 in 10, and a live run
  breached it immediately (3 of 5 answers), which is why verification is now against the
  whole payload rather than titles alone.
"""

from __future__ import annotations

import re

from movieagent.data.preprocess import normalize_title
from movieagent.data.schema import MovieRef

#: Quoted or italicised spans, and runs of capitalised words -- the shapes a title
#: mention actually takes in prose.
_QUOTED = re.compile(r"[\"“”'‘’*_]{1,2}([^\"“”'‘’*_\n]{2,80})[\"“”'‘’*_]{1,2}")
#: Runs of capitalised words, confined to a single line: `\s` spans newlines, which
#: glued a trailing label to the next line's first word ("Overview\nThe").
_CAPITALISED_RUN = re.compile(
    r"\b((?:[A-Z][\w'’-]*)(?:[ ]+(?:of|the|and|a|an|in|to|for|de|le|la)[ ]+[A-Z\w'’-]+|"
    r"[ ]+[A-Z][\w'’-]*){1,7})\b"
)

#: Words that begin sentences or appear in stock phrases; capitalised, but never titles.
_STOPWORDS = {
    "the",
    "this",
    "that",
    "these",
    "those",
    "here",
    "there",
    "it",
    "they",
    "i",
    "you",
    "we",
    "if",
    "based",
    "however",
    "both",
    "each",
    "unknown",
    "note",
    "overall",
    "in",
    "from",
    "according",
    "dataset",
}


#: Generic metadata labels the model writes when formatting a record. They are never
#: movie titles, and they are synonyms rather than verbatim payload keys ("Rating" for
#: `vote_average`), so payload matching alone does not clear them.
_FIELD_LABELS = frozenset(
    {
        "rating", "ratings", "score", "votes", "vote", "runtime", "budget", "revenue",
        "genre", "genres", "director", "directors", "cast", "starring", "tagline",
        "overview", "title", "year", "release", "released", "language", "languages",
        "country", "countries", "company", "companies", "production", "popularity",
        "count", "movie", "movies", "film", "films", "details", "summary", "unknown",
        "status", "homepage", "keywords", "id",
    }
)


def _candidate_mentions(answer: str) -> set[str]:
    found: set[str] = set()
    for match in _QUOTED.finditer(answer):
        span = match.group(1).strip()
        # Straight quotes pair greedily across a sentence, so the text *between* two
        # quoted titles gets captured too. A real title starts with a letter or digit
        # and is capitalised; sentence fragments are not.
        if span and (span[0].isupper() or span[0].isdigit()):
            found.add(span)
    for match in _CAPITALISED_RUN.finditer(answer):
        phrase = match.group(1).strip()
        first = phrase.split()[0].casefold()
        if first in _STOPWORDS:
            continue
        found.add(phrase)
    # Drop phrases made entirely of generic field labels ("Rating", "Movie Count").
    return {
        f
        for f in found
        if len(f) > 2
        and not all(w.casefold() in _FIELD_LABELS for w in f.split())
    }


def check_answer(
    answer: str,
    allowed: list[MovieRef],
    extra_terms: list[str] | None = None,
    payload_text: str = "",
) -> list[str]:
    """Return advisory warnings for movie-like mentions absent from the payload.

    ``payload_text`` should be the *whole* tool payload for the turn. The rule is simply:
    **if a phrase appears anywhere in what the tools returned, it is grounded.** That is
    both more correct and far quieter than checking titles alone -- production companies,
    taglines, character names and field labels all appear in the payload and were
    previously flagged as fabrications.

    ``extra_terms`` carries structured values (people, genres, keywords) for the same
    reason.
    """
    if not answer.strip():
        return []

    allowed_norms = {normalize_title(ref.title) for ref in allowed}
    allowed_norms.update(normalize_title(f"{ref.title} {ref.year}") for ref in allowed)
    for term in extra_terms or []:
        allowed_norms.add(normalize_title(term))
    allowed_norms.discard("")

    # One normalized blob of everything the tools returned this turn. Underscores become
    # spaces first: payload keys are snake_case (`production_companies`) while the model
    # writes prose ("Production Companies"), and `normalize_title` treats `_` as a word
    # character -- so without this every field label looked like an invented entity.
    payload_blob = normalize_title(payload_text.replace("_", " ")) if payload_text else ""

    warnings: list[str] = []
    for mention in sorted(_candidate_mentions(answer)):
        normalized = normalize_title(mention)
        if not normalized or normalized in allowed_norms:
            continue
        # Present verbatim in the tool output -> grounded, whatever kind of thing it is.
        if payload_blob and normalized in payload_blob:
            continue
        # A mention contained in (or containing) an allowed title is the same film
        # phrased differently -- "Dark Knight" for "The Dark Knight".
        if any(
            normalized in allowed or allowed in normalized
            for allowed in allowed_norms
            if len(allowed) > 3
        ):
            continue
        warnings.append(
            f"{mention!r} appears in the answer but is not in the retrieved records"
        )

    return warnings[:10]
