"""The tool contract: a typed result envelope with an explicit outcome (ADR-0003).

The PDF's error section (p5 §10) names eight conditions, and most of them are **not
errors** -- they are legitimate outcomes the *agent* must reason about. "This title is
ambiguous, here are five candidates" should make the agent ask a question (R-043), not
abort. That is why outcomes are values rather than exceptions: the model receives them
as data it can act on, and the graph routes on them (ADR-0021).

Under LangGraph the envelope splits two ways
(``response_format="content_and_artifact"``):

* ``summary_for_model()`` -- compact status JSON, becomes ``ToolMessage.content``;
* ``artifact()`` -- the full typed payload, becomes ``ToolMessage.artifact`` and feeds
  the UI tables and the trace.

That split resolves a cost ADR-0003 originally accepted: full result sets no longer
enter the prompt at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Protocol

from movieagent.config import Settings
from movieagent.data.repository import MovieRepository
from movieagent.data.schema import MovieRef
from movieagent.llm.embeddings import EmbeddingBackend
from movieagent.retrieval.coverage import CorpusVocabulary
from movieagent.retrieval.fuzzy import FuzzyTitleMatcher
from movieagent.retrieval.backend import SearchBackend


class Outcome(StrEnum):
    """Every state a tool can end in.

    These map one-to-one onto the PDF's p5 §10 conditions. Keeping them distinct is the
    whole point: with raw returns, "no movies match your filters" (R-039), "your filter
    was invalid" (R-038) and "I found three equally likely films and refuse to guess"
    (R-043) would all be an empty list.
    """

    OK = "ok"
    EMPTY = "empty"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    LOW_CONFIDENCE = "low_confidence"
    INVALID_INPUT = "invalid_input"
    ERROR = "error"


#: What the model should do with each outcome. Generated into the prompt from this
#: mapping rather than hand-written there, which is ADR-0003's stated mitigation for the
#: enum-to-prompt coupling it flagged as its main maintenance hazard.
OUTCOME_GUIDANCE: dict[Outcome, str] = {
    Outcome.OK: "The tool succeeded. Use only the data it returned.",
    Outcome.EMPTY: (
        "No records matched. Say so plainly, name which constraints were binding, and "
        "offer to relax one. Do not invent results and do not silently widen the query."
    ),
    Outcome.NOT_FOUND: (
        "The movie is not in this dataset. Say so. The dataset ends around 2016, so a "
        "recent film genuinely will not be there."
    ),
    Outcome.AMBIGUOUS: (
        "Several movies match and none is a clear winner. Ask the user which one they "
        "meant, listing the candidates. Never pick one yourself."
    ),
    Outcome.LOW_CONFIDENCE: (
        "The best matches are weak. You may show them, but say explicitly that you are "
        "not confident they are what the user meant."
    ),
    Outcome.INVALID_INPUT: (
        "The arguments were rejected. Read the errors, correct them, and call the tool "
        "once more. If it fails again, explain the limitation to the user."
    ),
    Outcome.ERROR: (
        "The tool failed for a technical reason. Apologise briefly, say what could not "
        "be done, and do not fabricate an answer."
    ),
}


@dataclass(frozen=True, slots=True)
class ToolResult:
    """A tool's outcome, its payload, and everything the trace needs."""

    status: Outcome
    message: str
    refs: list[MovieRef] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is Outcome.OK

    def summary_for_model(self) -> str:
        """Compact JSON for ``ToolMessage.content``.

        Capped at 25 references. The model needs to know *what* was found and *how
        much*; it does not need every row, and the artifact has them anyway.
        """
        body: dict[str, Any] = {
            "status": self.status.value,
            "message": self.message,
            "guidance": OUTCOME_GUIDANCE[self.status],
        }
        if self.refs:
            body["movies"] = [r.to_dict() for r in self.refs[:25]]
            if len(self.refs) > 25:
                body["movies_truncated_to"] = 25
        for key in (
            "total",
            "shown",
            "pool_size",
            "candidates",
            "aggregate",
            "record",
            "resolved",
        ):
            if key in self.payload:
                body[key] = self.payload[key]
        if self.meta:
            body["meta"] = {
                k: v for k, v in self.meta.items() if k not in {"rows", "documents"}
            }
        return json.dumps(body, default=str, ensure_ascii=False)

    def artifact(self) -> dict[str, Any]:
        """Full typed payload for the UI and the trace. Never enters a prompt."""
        return {
            "status": self.status.value,
            "message": self.message,
            "refs": [r.to_dict() for r in self.refs],
            "payload": self.payload,
            "meta": self.meta,
        }

    # ----------------------------------------------------------------- constructors

    @classmethod
    def ok_result(
        cls,
        message: str,
        *,
        refs: list[MovieRef] | None = None,
        payload: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> ToolResult:
        return cls(Outcome.OK, message, refs or [], payload or {}, meta or {})

    @classmethod
    def empty(cls, message: str, *, binding: list[str] | None = None) -> ToolResult:
        return cls(
            Outcome.EMPTY,
            message,
            payload={"binding_constraints": binding or []},
        )

    @classmethod
    def invalid(cls, problems: list[str]) -> ToolResult:
        return cls(
            Outcome.INVALID_INPUT,
            "The arguments were rejected: " + "; ".join(problems),
            payload={"errors": problems},
        )

    @classmethod
    def failure(cls, message: str) -> ToolResult:
        return cls(Outcome.ERROR, message)


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Everything the tools need, injected once.

    ``answer_fn`` is a plain ``(system, user) -> str`` callable rather than a chat
    model, which is what keeps this package free of LangChain imports (ADR-0019). The
    agent layer supplies the wrapper.
    """

    settings: Settings
    repository: MovieRepository
    matcher: FuzzyTitleMatcher
    index: SearchBackend
    embedder: EmbeddingBackend
    documents: list[str]
    vocabulary: CorpusVocabulary | None = None
    answer_fn: Callable[[str, str], str] | None = None


class Tool(Protocol):
    """A tool is a pure function of its context and arguments."""

    name: str
    description: str

    def __call__(self, context: ToolContext, **kwargs: Any) -> ToolResult: ...
