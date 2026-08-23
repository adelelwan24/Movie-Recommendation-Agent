"""The plan produced before any tool runs (R-089, ADR-0002, ADR-0021).

Two things make this a schema rather than free text:

* **R-102** needs a *concise tool-selection explanation*. As a schema field it is always
  present, always short, and structurally incapable of carrying chain-of-thought --
  which is how R-104 is respected here even though ADR-0023 had to give up the stronger
  guarantee elsewhere.
* **R-093** needs "the first one" turned into a concrete movie id *before* execution, so
  tools never receive deictic arguments and stay independently testable.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from movieagent.data.query import SearchQuery

#: Hard cap on the rationale. R-102 asks for concise; a schema field that quietly grows
#: into paragraphs would be reasoning by another name.
RATIONALE_MAX_CHARS = 240


class ToolName(StrEnum):
    STRUCTURED_SEARCH = "structured_search"
    FUZZY_MOVIE_SEARCH = "fuzzy_movie_search"
    SEMANTIC_SEARCH = "semantic_search"
    MOVIE_DETAILS = "movie_details"
    RAG_ANSWER = "rag_answer"


class PlanStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool: ToolName
    why: str = Field(description="One short clause: why this tool, for this query.")


class Plan(BaseModel):
    """A statement of intent, rendered before tools finish running.

    The executor may deviate -- that is how ``AMBIGUOUS`` and ``EMPTY`` recovery works
    (ADR-0021) -- so the trace records plan *and* actual, and marks the difference.
    """

    model_config = ConfigDict(frozen=True)

    intent: str = Field(description="What the user is asking for, in one short sentence.")
    steps: list[PlanStep] = Field(
        default_factory=list, description="Tools to run, in order. Empty for chit-chat."
    )
    rationale: str = Field(
        default="",
        description="One sentence explaining the tool choice. Shown to the user.",
    )
    filters: SearchQuery | None = Field(
        default=None,
        description="Structured constraints extracted from the query, if any.",
    )
    resolved_movie_ids: list[int] = Field(
        default_factory=list,
        description=(
            "Movie ids that references like 'the first one' or 'that movie' resolve to, "
            "taken from the previous result set."
        ),
    )
    reference_note: str | None = Field(
        default=None,
        description="What a conversational reference was resolved to, for the trace.",
    )
    refines_previous: bool = Field(
        default=False,
        description=(
            "True ONLY when this message narrows or adjusts the previous result set "
            "instead of asking its own question -- 'only the ones above 7.5', 'just the "
            "ones after 2010', 'what about comedies'. Filters from earlier turns are "
            "carried forward and re-applied only when this is true. A question that "
            "names its own subject is not a refinement."
        ),
    )
    needs_tools: bool = Field(
        default=True,
        description="False for greetings, thanks, or questions unrelated to movies.",
    )

    @field_validator("rationale")
    @classmethod
    def _keep_it_short(cls, value: str) -> str:
        collapsed = " ".join(value.split())
        if len(collapsed) > RATIONALE_MAX_CHARS:
            collapsed = collapsed[: RATIONALE_MAX_CHARS - 1].rstrip() + "…"
        return collapsed

    def tool_sequence(self) -> list[str]:
        return [step.tool.value for step in self.steps]

    def to_display(self) -> dict[str, object]:
        return {
            "intent": self.intent,
            "tools": self.tool_sequence(),
            "rationale": self.rationale,
            "filters": self.filters.to_display() if self.filters else {},
            "resolved_movie_ids": self.resolved_movie_ids,
            "reference_note": self.reference_note,
            "refines_previous": self.refines_previous,
        }
