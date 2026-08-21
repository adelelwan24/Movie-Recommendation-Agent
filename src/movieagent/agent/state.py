"""Graph state and its reducers (ADR-0010's decision, ADR-0022's mechanism).

R-092 names what must persist -- filters, result sets, selected movies -- and all three
are **structured values**, not prose. Storing them as text and asking a language model to
recover them is a lossy round-trip. Storing them as typed channels makes
"tell me about the first one" an array index (R-091).

The merge rules live in one function, `resolve_active_query`, called by the plan node.
They were a channel *reducer* until a live run showed that unsound -- LangGraph skips the
reducer on the first write to an empty channel -- so the rules stayed in one place but
moved to the caller. See that function for the detail (ADR-0027).

Nothing large lives here. Result *rows*, retrieved document text and tool artifacts stay
in the turn's ``Trace``, because ``MemorySaver`` retains every checkpoint for the process
lifetime and anything put in state is retained with it.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from movieagent.config import get_settings
from movieagent.data.query import SearchQuery
from movieagent.data.schema import MovieRef



def trimmed_add_messages(
    current: list[AnyMessage] | None, update: list[AnyMessage] | AnyMessage
) -> list[AnyMessage]:
    """``add_messages`` with the window enforced at write time (ADR-0022).

    The window keeps conversational tone -- phrasing, follow-ups like "why that one?" --
    which the structured channels deliberately do not capture. Anything older than the
    window is gone unless it survives as structured state, and that failure is *visible*
    (the agent says it does not know what you mean) rather than silent.

    Tool-call pairing is preserved: an ``AIMessage`` carrying ``tool_calls`` must never
    be trimmed away from its ``ToolMessage`` replies, or the provider rejects the
    conversation.
    """
    merged = add_messages(current or [], update)
    window = get_settings().agent.message_window
    if len(merged) <= window:
        return merged

    # The window drops *previous* turns, never the turn in progress. A turn that makes
    # several tool calls easily exceeds the window on its own (7 calls = 15 messages),
    # and trimming those away silently truncated both the grounding payload and the
    # trace -- so the UI under-reported which tools had run (R-102/R-103).
    turn_start = 0
    for index in range(len(merged) - 1, -1, -1):
        message = merged[index]
        if getattr(message, "type", None) == "human" and not getattr(
            message, "additional_kwargs", {}
        ).get("clarification_reply"):
            turn_start = index
            break

    cut = min(len(merged) - window, turn_start)
    kept = merged[cut:]
    if len(kept) <= window:
        return kept
    # Walk backwards from the cut point and re-admit any AIMessage whose tool results
    # are inside the window.
    orphan_ids = {
        m.tool_call_id
        for m in kept
        if getattr(m, "type", None) == "tool" and getattr(m, "tool_call_id", None)
    }
    if orphan_ids:
        for message in reversed(merged[: -window or None]):
            call_ids = {c["id"] for c in getattr(message, "tool_calls", []) or []}
            if call_ids & orphan_ids:
                kept.insert(0, message)
                orphan_ids -= call_ids
            if not orphan_ids:
                break
    return kept


def resolve_active_query(
    current: SearchQuery | None,
    new_filters: SearchQuery | None,
    *,
    fresh_topic: bool,
) -> SearchQuery | None:
    """Decide the turn's carried filter set (R-148, the requester's OQ-006 decision).

    * ``fresh_topic`` -- the query changed subject; drop what was carried. Without a
      reset path filters accumulate forever and turn nine returns nothing.
    * new filters on an existing set -- layered per :meth:`SearchQuery.merged_with`.
    * nothing new -- carry the existing set unchanged.

    The requester chose *carry filters forward and re-query* over *filter the rows you
    displayed*, so what travels is the filter object rather than the rendered table --
    which is why a display cap can never silently truncate a follow-up.

    **This is a plain function called by the plan node, not a channel reducer.** It was
    a reducer until a live run proved that unsound: LangGraph's `BinaryOperatorAggregate`
    assigns `values[0]` directly when a channel is still `MISSING` and only applies the
    operator from the *second* write onward (`langgraph/channels/binop.py`). A
    first-turn write therefore bypassed the reducer entirely and put a raw sentinel
    string into `active_query`, which every reader then tried to call `.is_empty()` on.
    Computing the value in the node removes the hazard: what is written is already
    correct, so no reducer has to run for state to be valid.
    """
    if fresh_topic:
        return new_filters if (new_filters and not new_filters.is_empty()) else None
    if new_filters is None or new_filters.is_empty():
        return current
    if current is None:
        return new_filters
    return current.merged_with(new_filters)


def overwrite(current: Any, update: Any) -> Any:
    """Last write wins, **including a write of `None`**.

    Subtle but load-bearing: LangGraph only invokes a channel's reducer when a node
    actually writes that channel, so "unchanged" is already expressed by not writing.
    An earlier version returned `current` when `update` was `None`, which made a
    deliberate clear impossible -- a fresh-topic reset of `active_query` and the clearing
    of `pending_clarification` both silently did nothing.
    """
    return update


class AgentState(TypedDict, total=False):
    """The checkpointed conversation state, keyed by ``thread_id``."""

    messages: Annotated[list[AnyMessage], trimmed_add_messages]
    question: Annotated[str, overwrite]
    plan: Annotated[dict[str, Any] | None, overwrite]
    active_query: Annotated[SearchQuery | None, overwrite]
    last_results: Annotated[list[dict[str, Any]] | None, overwrite]
    last_result_total: Annotated[int | None, overwrite]
    selected_movie_id: Annotated[int | None, overwrite]
    pending_clarification: Annotated[dict[str, Any] | None, overwrite]
    answer: Annotated[str, overwrite]
    deviations: Annotated[list[str] | None, overwrite]
    tool_iterations: Annotated[int | None, overwrite]


# ------------------------------------------------------------------ reference helpers


def refs_from_state(state: AgentState) -> list[MovieRef]:
    return [MovieRef.from_dict(r) for r in (state.get("last_results") or [])]


def ordinal_ref(state: AgentState, index: int) -> MovieRef | None:
    """Resolve "the first one" / "the third" against the stored result set.

    ``index`` is zero-based. This is the array access that R-091's turn three reduces
    to -- no model recall involved, so it either works or returns ``None``.
    """
    refs = refs_from_state(state)
    if 0 <= index < len(refs):
        return refs[index]
    return None


def describe_state(state: AgentState) -> str:
    """Compact rendering of carried state for the planner prompt.

    Deliberately terse: the planner needs the *referents*, not a transcript.
    """
    lines: list[str] = []
    query = state.get("active_query")
    if query is not None and not query.is_empty():
        lines.append("Filters currently in effect: " + "; ".join(query.describe()))

    refs = refs_from_state(state)
    if refs:
        total = state.get("last_result_total")
        header = f"Previous result set ({len(refs)} shown"
        header += f" of {total} matching)" if total and total > len(refs) else ")"
        lines.append(header + ":")
        lines.extend(f"  {i + 1}. {ref.label()} [id={ref.movie_id}]" for i, ref in enumerate(refs))

    selected = state.get("selected_movie_id")
    if selected:
        lines.append(f"Movie currently under discussion: id={selected}")

    pending = state.get("pending_clarification")
    if pending:
        options = ", ".join(
            f"{i + 1}. {c['title']} [id={c['movie_id']}]"
            for i, c in enumerate(pending.get("candidates", []))
        )
        lines.append(
            "The previous turn asked the user to choose between: " + options
        )

    return "\n".join(lines) if lines else "(no prior context in this conversation)"
