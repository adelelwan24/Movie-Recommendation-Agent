"""Lexical coverage: the low-confidence signal that actually works (ADR-0026).

ADR-0011 specified an absolute cosine floor for R-056. Measurement killed it. Over this
corpus the top similarity for genuine queries spans 0.575-0.730 while pure gibberish spans
0.566-0.629 -- the populations **overlap**, so no threshold separates them. A z-score
against the corpus mean separates by only 0.1 sigma, which is not a signal either.

The cause is the one ADR-0008 predicted and understated: every document shares the same
field labels, so nothing in the corpus is ever truly dissimilar to anything else. Worse,
subword tokenizers map gibberish to *somewhere* plausible -- "flurble" lands near real
words -- so embedding geometry cannot tell nonsense from meaning.

What does separate them is orthogonal to the embedding entirely: **does the query consist
of words this corpus actually uses?** Gibberish scores 0.00-0.33 coverage; real queries
score 0.50-1.00.

Known limitation, stated rather than hidden: this catches *nonsense*, not *absence*.
"a movie about the 2022 invasion of Ukraine" is fluent English with full coverage and no
answer in a dataset that ends in 2016. Nothing here detects that, and the retrieved films
will look confident. That is the honest weak point of R-056 in this system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORD = re.compile(r"[a-z']{2,}")

#: Query scaffolding that says nothing about subject matter. Left in the query, these
#: inflate coverage for gibberish ("show me florb glorb") and mask it.
_QUERY_STOPWORDS = frozenset(
    """
    a an the of and or in on to for with is are was were be been am
    it its this that these those there here
    i we you my me us your
    want find show tell get give need looking look
    about something someone anything anyone somewhere
    movie movies film films picture pictures
    like similar related good great best nice
    please can could would should
    story stories plot theme
    """.split()
)


@dataclass(frozen=True, slots=True)
class Coverage:
    """How much of a query the corpus vocabulary recognises."""

    ratio: float
    known: tuple[str, ...]
    unknown: tuple[str, ...]

    @property
    def is_meaningful(self) -> bool:
        return not self.unknown or self.ratio >= 0.5


class CorpusVocabulary:
    """Every word appearing in the semantic documents.

    Built once at load (~33,500 words over 4,800 documents, a few hundred milliseconds)
    and immutable thereafter, so it is safe in the ``@st.cache_resource`` set (ADR-0014).
    """

    def __init__(self, documents: list[str]) -> None:
        words: set[str] = set()
        for document in documents:
            words.update(_WORD.findall(document.lower()))
        self._words = frozenset(words)

    def __len__(self) -> int:
        return len(self._words)

    def __contains__(self, word: str) -> bool:
        return word.lower() in self._words

    def coverage(self, query: str) -> Coverage:
        """Fraction of a query's content words that the corpus uses.

        Stopwords are removed first: without that, "show me florb glorb" scores 0.5 on
        scaffolding alone and gibberish passes.
        """
        tokens = [t for t in _WORD.findall(query.lower()) if t not in _QUERY_STOPWORDS]
        if not tokens:
            # Nothing but scaffolding, or a purely numeric query. Not evidence of
            # nonsense, so do not penalise it -- the similarity floor is the backstop.
            return Coverage(ratio=1.0, known=(), unknown=())

        known = tuple(t for t in tokens if t in self._words)
        unknown = tuple(t for t in tokens if t not in self._words)
        return Coverage(ratio=len(known) / len(tokens), known=known, unknown=unknown)
