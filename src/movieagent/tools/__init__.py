"""The five tools (R-081) and the contract they share.

Boundaries, stated once and applied everywhere:

``structured_search``    filters, sorting, aggregations, counts, numeric comparisons
``fuzzy_movie_search``   *title string -> movie identity*
``semantic_search``      *meaning -> movies*, including hybrid metadata pre-filtering
``movie_details``        complete structured record for one movie
``rag_answer``           grounded natural-language answer over an explicit id list

Fuzzy and semantic are the pair most easily confused (the PDF files "similar to lord of
the rings" under both). The rule that separates them: **fuzzy maps strings to a movie;
semantic maps meaning to movies.** A query like "similar to lord of the rings" is a
two-tool chain, not one tool doing both jobs (ADR-0009, OQ-004).

Everything here is a pure function of its arguments and framework-free -- no LangChain,
no Streamlit. The ``@tool`` wrappers live in ``agent/tool_bindings.py``.
"""

from movieagent.tools.base import Outcome, ToolContext, ToolResult

__all__ = ["Outcome", "ToolContext", "ToolResult"]
