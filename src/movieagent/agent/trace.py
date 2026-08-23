"""The typed execution trace (ADR-0013, ADR-0023).

One object serves three consumers with no translation: the Streamlit trace panel
(R-102/R-103), the JSONL log, and the routing tests.

**There is deliberately no field for model reasoning content.** R-104 forbids exposing
chain-of-thought, and a type with nowhere to put it cannot leak it. Under LangGraph that
is no longer sufficient on its own -- provider reasoning rides in
``AIMessage.additional_kwargs`` -- so ADR-0023 adds a sanitizing wrapper at the model
boundary. This type is the second half of that defence, not the whole of it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from movieagent.agent.plan import Plan
from movieagent.data.schema import MovieRef


@dataclass(slots=True)
class ToolCallRecord:
    """One tool invocation, as it actually happened."""

    tool: str
    arguments: dict[str, Any]
    status: str
    message: str
    duration_ms: float = 0.0
    result_count: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
    artifact: dict[str, Any] | None = None

    def to_display(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "status": self.status,
            "message": self.message,
            "duration_ms": round(self.duration_ms, 1),
            "result_count": self.result_count,
        }


@dataclass(slots=True)
class Trace:
    """Everything the UI shows about one turn."""

    turn_index: int = 0
    question: str = ""
    plan: Plan | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    #: Filters inherited from *earlier* turns. Populated only when the planner read this
    #: message as a refinement -- on any other turn the active query is simply this
    #: turn's own filters, and labelling those "carried forward" made a working reset
    #: look like the accumulation bug it had just fixed.
    carried_forward: list[str] = field(default_factory=list)
    grounding_warnings: list[str] = field(default_factory=list)
    deviations: list[str] = field(default_factory=list)
    truncated: bool = False
    interrupted: bool = False
    error: str | None = None
    total_duration_ms: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)
    answer: str = ""
    result_refs: list[MovieRef] = field(default_factory=list)

    # -------------------------------------------------------------------- accessors

    @property
    def tools_used(self) -> list[str]:
        return [call.tool for call in self.tool_calls]

    def last_artifact(self, tool: str | None = None) -> dict[str, Any] | None:
        for call in reversed(self.tool_calls):
            if call.artifact and (tool is None or call.tool == tool):
                return call.artifact
        return None

    def artifacts(self, tool: str) -> list[dict[str, Any]]:
        return [c.artifact for c in self.tool_calls if c.tool == tool and c.artifact]

    def retrieved_documents(self) -> list[dict[str, Any]]:
        """Retrieved context for R-103, in the order it was retrieved."""
        docs: list[dict[str, Any]] = []
        for call in self.tool_calls:
            if call.tool == "semantic_search" and call.artifact:
                docs.extend(call.artifact.get("meta", {}).get("documents", []))
        return docs

    def record_deviation(self, note: str) -> None:
        """Note where execution diverged from the plan.

        ADR-0021 accepted that the displayed plan is a statement of intent rather than a
        guarantee. Recording divergence is what keeps that honest instead of quietly
        letting the UI imply otherwise.
        """
        if note not in self.deviations:
            self.deviations.append(note)

    def check_plan_adherence(self) -> None:
        planned = set(self.plan.tool_sequence()) if self.plan else set()
        actual = set(self.tools_used)
        for extra in sorted(actual - planned):
            self.record_deviation(f"ran {extra} which was not in the plan")
        for skipped in sorted(planned - actual):
            self.record_deviation(f"planned {skipped} but did not run it")

    # ---------------------------------------------------------------- serialization

    def to_dict(self) -> dict[str, Any]:
        """JSONL payload. Same object the UI renders -- one source of truth."""
        return {
            "turn_index": self.turn_index,
            "question": self.question,
            "plan": self.plan.to_display() if self.plan else None,
            "tool_calls": [call.to_display() for call in self.tool_calls],
            "carried_forward": self.carried_forward,
            "grounding_warnings": self.grounding_warnings,
            "deviations": self.deviations,
            "truncated": self.truncated,
            "interrupted": self.interrupted,
            "error": self.error,
            "total_duration_ms": round(self.total_duration_ms, 1),
            "token_usage": self.token_usage,
            "answer": self.answer,
            "result_refs": [asdict(r) for r in self.result_refs],
        }
