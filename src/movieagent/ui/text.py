"""Answer-text presentation (R-107, ADR-0012).

The UI renders tool results itself, from the typed payload -- that is the whole point of
ADR-0012: numbers reach the screen from the data, not from the model's prose. So a
markdown table inside the answer text is *always* a duplicate of a table rendered a few
pixels below it, in the model's paraphrased column names rather than the real ones.

The prompts ask the model not to do this. Small models do it anyway, especially when the
result is a tidy ten-row aggregate that looks exactly like something to present. An
instruction the renderer can enforce should not be left to the model's cooperation, so
this strips them at the seam.

Deliberately narrow: it removes *tables*, not sentences, lists, or code. The raw text is
untouched in the ``Trace``, so what the model actually wrote is still there for debugging.

Kept free of the streamlit import so it can be tested as the pure function it is.
"""

from __future__ import annotations

import re

#: A markdown table's second line: `|---|---|`, with optional alignment colons.
_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")

_FENCE = re.compile(r"^\s*(```|~~~)")


def _is_row(line: str) -> bool:
    return line.count("|") >= 2


def strip_markdown_tables(text: str) -> str:
    """Remove markdown tables from answer prose, leaving everything else intact.

    Returns the original text when stripping would leave nothing: a bare table is a poor
    answer, but a blank message is a worse one, and the table below it still carries the
    data either way.
    """
    if not text or "|" not in text:
        return text

    lines = text.splitlines()
    kept: list[str] = []
    in_fence = False
    index = 0

    while index < len(lines):
        line = lines[index]

        if _FENCE.match(line):
            in_fence = not in_fence
            kept.append(line)
            index += 1
            continue

        # A table is a header row followed by a separator row. Anything else that happens
        # to contain a pipe -- prose, a code line, a filter expression -- is left alone.
        is_table = (
            not in_fence
            and _is_row(line)
            and index + 1 < len(lines)
            and _SEPARATOR.match(lines[index + 1])
        )
        if not is_table:
            kept.append(line)
            index += 1
            continue

        index += 2  # header and separator
        while index < len(lines) and _is_row(lines[index]) and not _FENCE.match(lines[index]):
            index += 1

    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    return cleaned if cleaned else text
