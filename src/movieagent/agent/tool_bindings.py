"""LangChain ``@tool`` wrappers over the pure tools (ADR-0021).

The tools themselves know nothing about LangChain (ADR-0019). This module is the only
place that does, and it exists to do one thing: map ``ToolResult`` onto
``response_format="content_and_artifact"``.

That mapping resolves a cost ADR-0003 originally accepted -- serializing full payloads
into the model's context:

* ``ToolMessage.content``  <- compact status JSON. What the model reasons about.
* ``ToolMessage.artifact`` <- the complete typed payload. Feeds UI tables and the trace,
  and never enters a prompt.

The conditional edge that implements R-043 reads the status off the artifact, which is
why the status enum had to be a first-class value rather than prose.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from movieagent.data.query import SearchQuery
from movieagent.logging import get_logger
from movieagent.tools import fuzzy_movie_search as t_fuzzy
from movieagent.tools import movie_details as t_details
from movieagent.tools import rag_answer as t_rag
from movieagent.tools import semantic_search as t_semantic
from movieagent.tools import structured_search as t_structured
from movieagent.tools.base import Outcome, ToolContext, ToolResult

log = get_logger("agent.tools")

ARTIFACT_STATUS = "status"


# ------------------------------------------------------------------- argument schemas


class StructuredSearchArgs(BaseModel):
    query: SearchQuery = Field(description="Filters, sorting and aggregation to apply.")


class FuzzySearchArgs(BaseModel):
    title: str = Field(description="The title as the user typed it, however misspelled.")


class SemanticSearchArgs(BaseModel):
    query: str | None = Field(
        default=None, description="Plot, theme or mood description to search by."
    )
    similar_to: int | None = Field(
        default=None, description="Movie id to find similar movies to."
    )
    filters: SearchQuery | None = Field(
        default=None, description="Hard constraints applied before ranking."
    )
    k: int | None = Field(default=None, description="How many movies to return.")


class MovieDetailsArgs(BaseModel):
    movie_id: int = Field(description="Numeric dataset id of the movie.")


class RagAnswerArgs(BaseModel):
    question: str = Field(description="The question to answer.")
    movie_ids: list[int] = Field(
        description="Ids of the movies to ground the answer in. Must not be empty."
    )


def _finish(result: ToolResult, tool: str, arguments: dict[str, Any]) -> tuple[str, dict]:
    log.info("%s -> %s (%s)", tool, result.status.value, result.message[:120])
    artifact = result.artifact()
    artifact["tool"] = tool
    artifact["arguments"] = arguments
    return result.summary_for_model(), artifact


def _guard(tool: str, arguments: dict[str, Any], exc: Exception) -> tuple[str, dict]:
    """Convert an unexpected failure into an envelope rather than a crashed turn.

    ADR-0003 warned that this can launder a genuine bug into a polite message, so the
    traceback is logged at ERROR level while the user-facing envelope stays sanitized.
    """
    log.exception("%s raised", tool)
    result = ToolResult.failure(f"{tool} failed: {exc}")
    return _finish(result, tool, arguments)


def build_tools(context: ToolContext) -> list[StructuredTool]:
    """Bind the five tools to a context. One registry, used by the graph and the tests."""

    def structured_search(query: SearchQuery) -> tuple[str, dict]:
        args = {"query": query.to_display()}
        try:
            return _finish(t_structured.run(context, query), t_structured.NAME, args)
        except Exception as exc:  # noqa: BLE001
            return _guard(t_structured.NAME, args, exc)

    def fuzzy_movie_search(title: str) -> tuple[str, dict]:
        args = {"title": title}
        try:
            return _finish(t_fuzzy.run(context, title), t_fuzzy.NAME, args)
        except Exception as exc:  # noqa: BLE001
            return _guard(t_fuzzy.NAME, args, exc)

    def semantic_search(
        query: str | None = None,
        similar_to: int | None = None,
        filters: SearchQuery | None = None,
        k: int | None = None,
    ) -> tuple[str, dict]:
        args = {
            "query": query,
            "similar_to": similar_to,
            "filters": filters.to_display() if filters else None,
            "k": k,
        }
        try:
            return _finish(
                t_semantic.run(context, query, similar_to, filters, k), t_semantic.NAME, args
            )
        except Exception as exc:  # noqa: BLE001
            return _guard(t_semantic.NAME, args, exc)

    def movie_details(movie_id: int) -> tuple[str, dict]:
        args = {"movie_id": movie_id}
        try:
            return _finish(t_details.run(context, movie_id), t_details.NAME, args)
        except Exception as exc:  # noqa: BLE001
            return _guard(t_details.NAME, args, exc)

    def rag_answer(question: str, movie_ids: list[int]) -> tuple[str, dict]:
        args = {"question": question, "movie_ids": movie_ids}
        try:
            return _finish(t_rag.run(context, question, movie_ids), t_rag.NAME, args)
        except Exception as exc:  # noqa: BLE001
            return _guard(t_rag.NAME, args, exc)

    specs = (
        (structured_search, t_structured.NAME, t_structured.DESCRIPTION, StructuredSearchArgs),
        (fuzzy_movie_search, t_fuzzy.NAME, t_fuzzy.DESCRIPTION, FuzzySearchArgs),
        (semantic_search, t_semantic.NAME, t_semantic.DESCRIPTION, SemanticSearchArgs),
        (movie_details, t_details.NAME, t_details.DESCRIPTION, MovieDetailsArgs),
        (rag_answer, t_rag.NAME, t_rag.DESCRIPTION, RagAnswerArgs),
    )

    return [
        StructuredTool.from_function(
            func=func,
            name=name,
            description=description,
            args_schema=schema,
            response_format="content_and_artifact",
        )
        for func, name, description, schema in specs
    ]


def artifact_status(artifact: dict[str, Any] | None) -> Outcome | None:
    """Read a tool outcome back off an artifact. Used by the graph's routing edges."""
    if not artifact:
        return None
    raw = artifact.get(ARTIFACT_STATUS)
    try:
        return Outcome(raw)
    except ValueError:
        return None
