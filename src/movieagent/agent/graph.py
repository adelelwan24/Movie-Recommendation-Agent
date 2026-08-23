"""The agent graph (ADR-0020, ADR-0021, ADR-0022).

::

    START -> plan -> agent <-> tools -> synthesize -> ground -> END
               |       |         |
               |       |         +-> clarify -(interrupt)-> resumed next turn -> agent
               |       |
               +-> smalltalk -> END

Why a custom ``StateGraph`` rather than ``create_react_agent``: the prebuilt has no
explicit planning stage (R-089), and -- the deciding reason -- it cannot cleanly
interrupt on a tool **result**. Our R-043 requirement is exactly that shape: run the
fuzzy matcher, and if it returns three candidates within three points of each other,
stop and ask. That is a conditional edge reading ``ToolResult.status``, and it only
exists if we own the edges.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from movieagent.agent import prompts
from movieagent.agent.grounding import check_answer
from movieagent.agent.plan import Plan
from movieagent.agent.state import AgentState, describe_state, resolve_active_query
from movieagent.agent.tool_bindings import artifact_status, build_tools
from movieagent.agent.trace import ToolCallRecord, Trace
from movieagent.config import Settings
from movieagent.data.schema import MovieRef
from movieagent.llm.models import make_answer_fn, message_text, token_usage
from movieagent.logging import get_logger
from movieagent.tools.base import Outcome, ToolContext

log = get_logger("agent.graph")

_ORDINALS = {
    "first": 0, "1st": 0, "one": 0, "1": 0,
    "second": 1, "2nd": 1, "two": 1, "2": 1,
    "third": 2, "3rd": 2, "three": 2, "3": 2,
    "fourth": 3, "4th": 3, "four": 3, "4": 3,
    "fifth": 4, "5th": 4, "five": 4, "5": 4,
    "last": -1,
}


@dataclass(slots=True)
class TurnResult:
    """One turn's outcome, as the UI consumes it."""

    answer: str
    trace: Trace
    interrupted: bool = False
    clarification: dict[str, Any] | None = None
    refs: list[MovieRef] = field(default_factory=list)


# --------------------------------------------------------------------------- helpers


def current_turn_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Messages belonging to the turn in progress.

    Walks back to the most recent human message. Clarification replies are injected as
    human messages too, so a resumed turn correctly keeps the tool results that led to
    the question.
    """
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, HumanMessage) and not getattr(
            message, "additional_kwargs", {}
        ).get("clarification_reply"):
            return list(messages[index:])
    return list(messages)


def turn_artifacts(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    return [
        m.artifact
        for m in current_turn_messages(messages)
        if isinstance(m, ToolMessage) and isinstance(getattr(m, "artifact", None), dict)
    ]


def latest_tool_artifacts(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    """Artifacts from the **most recent** ToolNode execution only.

    Routing must not rescan the whole turn. After a clarification interrupt resolves, the
    ``AMBIGUOUS`` result that triggered it is still in the turn's history -- scanning all
    of it would route straight back to ``clarify`` and ask the same question forever.
    Only the trailing run of ToolMessages describes what just happened.
    """
    trailing: list[dict[str, Any]] = []
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            break
        artifact = getattr(message, "artifact", None)
        if isinstance(artifact, dict):
            trailing.append(artifact)
    return list(reversed(trailing))


def _stringify(value: Any) -> str:
    """Flatten a nested payload into searchable text for the grounding check."""
    if isinstance(value, dict):
        return " ".join(f"{k} {_stringify(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(_stringify(v) for v in value)
    return "" if value is None else str(value)


def _refs_of(artifact: dict[str, Any]) -> list[MovieRef]:
    return [MovieRef.from_dict(r) for r in artifact.get("refs", [])]


def resolve_choice(reply: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Interpret a clarification reply: an ordinal, a number, or a title.

    Deterministic on purpose. Having asked the user which film they meant, resolving
    their answer with another model call would reintroduce exactly the guessing the
    question was asked to avoid.
    """
    if not candidates:
        return None
    text = (reply or "").strip().casefold()
    if not text:
        return None

    for word, index in _ORDINALS.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            try:
                return candidates[index]
            except IndexError:
                return None

    for candidate in candidates:
        if str(candidate.get("movie_id")) in text:
            return candidate

    from movieagent.data.preprocess import normalize_title

    normalized = normalize_title(text)
    best: tuple[int, dict[str, Any]] | None = None
    for candidate in candidates:
        title = normalize_title(str(candidate.get("title", "")))
        if not title:
            continue
        if title == normalized:
            return candidate
        if title in normalized or normalized in title:
            score = len(title)
            if best is None or score > best[0]:
                best = (score, candidate)
    return best[1] if best else None


# ----------------------------------------------------------------------------- agent


class MovieAgent:
    """Compiles and runs the graph.

    The compiled graph is immutable and joins the ``@st.cache_resource`` set (ADR-0014).
    The checkpointer is instantiated once and isolated per ``thread_id``, so it is not
    shared mutable state in the sense that rule forbids.
    """

    def __init__(
        self,
        settings: Settings,
        context: ToolContext,
        model: Any,
        checkpointer: Any | None = None,
    ) -> None:
        self._settings = settings
        self._context = context
        self._model = model
        self._tools = build_tools(context)
        self._tools_by_name = {t.name: t for t in self._tools}
        self._bound = model.bind_tools(self._tools)
        self._planner = model.with_structured_output(Plan)
        self._checkpointer = checkpointer if checkpointer is not None else MemorySaver()
        self.graph = self._build()

    # ------------------------------------------------------------------ construction

    def _build(self) -> Any:
        builder = StateGraph(AgentState)
        builder.add_node("plan", self._plan_node)
        builder.add_node("agent", self._agent_node)
        builder.add_node("tools", ToolNode(self._tools))
        builder.add_node("clarify", self._clarify_node)
        builder.add_node("synthesize", self._synthesize_node)
        builder.add_node("ground", self._ground_node)
        builder.add_node("smalltalk", self._smalltalk_node)

        builder.add_edge(START, "plan")
        builder.add_conditional_edges(
            "plan", self._after_plan, {"agent": "agent", "smalltalk": "smalltalk"}
        )
        builder.add_conditional_edges(
            "agent", self._after_agent, {"tools": "tools", "synthesize": "synthesize"}
        )
        builder.add_conditional_edges(
            "tools",
            self._after_tools,
            {"clarify": "clarify", "agent": "agent", "synthesize": "synthesize"},
        )
        builder.add_edge("clarify", "agent")
        builder.add_edge("synthesize", "ground")
        builder.add_edge("ground", END)
        builder.add_edge("smalltalk", END)
        return builder.compile(checkpointer=self._checkpointer)

    def mermaid(self) -> str:
        """Topology straight from the code, so ARCHITECTURE.md cannot drift (ADR-0021)."""
        return self.graph.get_graph().draw_mermaid()

    # ------------------------------------------------------------------------- nodes

    def _plan_node(self, state: AgentState) -> dict[str, Any]:
        """One structured call, before any tool runs (R-089).

        Also where conversational references become concrete ids (R-093), so tools never
        receive "the first one" and stay independently testable.
        """
        question = message_text(state["messages"][-1])
        rendered = describe_state(state)
        try:
            plan: Plan = self._planner.invoke(
                [
                    SystemMessage(content=prompts.PLANNER_SYSTEM),
                    HumanMessage(
                        content=prompts.PLANNER_USER.format(state=rendered, question=question)
                    ),
                ]
            )
        except Exception as exc:  # noqa: BLE001 - provider failure is expected (R-087)
            log.exception("planning failed")
            return {
                "question": question,
                "plan": None,
                "answer": (
                    "I could not reach the language model to plan a response. "
                    f"({type(exc).__name__}) Please try again."
                ),
                "deviations": ["planning failed"],
                "pending_clarification": None,
            }

        update: dict[str, Any] = {
            "question": question,
            "plan": plan.model_dump(mode="json"),
            "tool_iterations": 0,
            "deviations": [],
            # Turn-scoped channels must be cleared here. State persists across turns by
            # design (ADR-0022), so a stale `answer` from the previous turn would make
            # `_after_plan` believe this turn had already failed and route it to
            # smalltalk without running a single tool.
            "answer": "",
            "pending_clarification": None,
        }
        # Resolve the carried filter set here and write the finished value. The merge
        # rules still live in one function (`resolve_active_query`), but the node calls
        # it rather than a reducer -- see that function's docstring for why.
        update["active_query"] = resolve_active_query(
            state.get("active_query"),
            plan.filters,
            refines_previous=plan.refines_previous,
        )
        if plan.resolved_movie_ids:
            update["selected_movie_id"] = plan.resolved_movie_ids[0]
        return update

    @staticmethod
    def _after_plan(state: AgentState) -> Literal["agent", "smalltalk"]:
        plan = state.get("plan")
        if plan is None:
            # Planning failed and already produced a user-facing answer; `smalltalk`
            # passes it through rather than calling the model again.
            return "smalltalk"
        if not plan.get("needs_tools", True):
            return "smalltalk"
        return "agent"

    def _agent_node(self, state: AgentState) -> dict[str, Any]:
        """The tool-calling step, seeded by the plan."""
        plan = state.get("plan") or {}
        messages = self._agent_messages(state, plan)
        try:
            response = self._bound.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - R-087
            log.exception("agent step failed")
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I could not reach the language model. "
                            f"({type(exc).__name__})"
                        )
                    )
                ],
                "answer": (
                    "I could not reach the language model to answer that. "
                    "The dataset tools are unaffected -- please try again."
                ),
                "deviations": [*(state.get("deviations") or []), "agent step failed"],
            }
        return {"messages": [response]}

    def _agent_messages(self, state: AgentState, plan: dict[str, Any]) -> list[BaseMessage]:
        extras: list[str] = []
        if plan.get("filters"):
            extras.append(f"  extracted filters: {plan['filters']}")
        if plan.get("resolved_movie_ids"):
            extras.append(
                f"  resolved references: {plan.get('reference_note') or ''} "
                f"-> movie ids {plan['resolved_movie_ids']}"
            )
        active = state.get("active_query")
        if active is not None and not active.is_empty():
            extras.append("  filters carried from earlier turns: " + "; ".join(active.describe()))

        block = prompts.EXECUTOR_PLAN_BLOCK.format(
            intent=plan.get("intent", "answer the question"),
            tools=", ".join(plan.get("steps") and [s["tool"] for s in plan["steps"]] or ["(none)"]),
            rationale=plan.get("rationale", ""),
            extras="\n".join(extras),
        )
        return [
            SystemMessage(content=prompts.EXECUTOR_SYSTEM + "\n\n" + block),
            *current_turn_messages(state["messages"]),
        ]

    def _after_agent(self, state: AgentState) -> Literal["tools", "synthesize"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "synthesize"

    def _after_tools(self, state: AgentState) -> Literal["clarify", "agent", "synthesize"]:
        """Route on the tool *outcome* -- the reason this graph is custom (ADR-0021)."""
        iterations = (state.get("tool_iterations") or 0) + 1

        # Only what this batch returned -- see `latest_tool_artifacts`.
        for artifact in reversed(latest_tool_artifacts(list(state["messages"]))):
            if artifact_status(artifact) is Outcome.AMBIGUOUS:
                return "clarify"

        if iterations >= self._settings.agent.max_tool_iterations:
            log.warning("tool iteration cap reached (%d)", iterations)
            return "synthesize"
        return "agent"

    def _clarify_node(self, state: AgentState) -> dict[str, Any]:
        """Ask which movie was meant, and pause the turn (R-043).

        ``interrupt`` is what makes this a genuine pause-and-resume rather than a
        hand-managed flag: the checkpointer holds the state, and the user's next message
        resumes *this* node rather than starting a new turn.
        """
        ambiguous = next(
            (
                a
                for a in reversed(latest_tool_artifacts(list(state["messages"])))
                if artifact_status(a) is Outcome.AMBIGUOUS
            ),
            None,
        )
        candidates = (ambiguous or {}).get("payload", {}).get("candidates", [])
        options = "\n".join(
            f"{i + 1}. {c['title']}" + (f" ({c['year']})" if c.get("year") else "")
            for i, c in enumerate(candidates)
        )
        question = (
            "I found several movies that could match, and none is a clear winner. "
            "Which did you mean?\n\n" + options
        )

        reply = interrupt(
            {
                "type": "clarification",
                "question": question,
                "candidates": candidates,
            }
        )

        chosen = resolve_choice(str(reply), candidates)
        if chosen is None:
            return {
                "messages": [
                    HumanMessage(
                        content=str(reply),
                        additional_kwargs={"clarification_reply": True},
                    )
                ],
                "pending_clarification": None,
            }
        return {
            "messages": [
                HumanMessage(
                    content=(
                        f"I meant {chosen['title']} (movie id {chosen['movie_id']}). "
                        "Continue with that one."
                    ),
                    additional_kwargs={"clarification_reply": True},
                )
            ],
            "selected_movie_id": int(chosen["movie_id"]),
            "pending_clarification": None,
        }

    def _synthesize_node(self, state: AgentState) -> dict[str, Any]:
        """Final answer, with tools unbound so nothing else can be called."""
        if state.get("answer"):  # an upstream failure already produced one
            return {}

        messages = current_turn_messages(list(state["messages"]))
        last = messages[-1] if messages else None

        # If the agent already answered in prose without calling tools, keep it rather
        # than paying for a second call to rewrite it.
        if isinstance(last, AIMessage) and not last.tool_calls and message_text(last).strip():
            answer = message_text(last).strip()
        else:
            try:
                response = self._model.invoke(
                    [SystemMessage(content=prompts.SYNTHESIS_SYSTEM), *messages]
                )
                answer = message_text(response).strip()
            except Exception as exc:  # noqa: BLE001 - R-087
                log.exception("synthesis failed")
                answer = (
                    "I gathered the data but could not reach the language model to write "
                    f"the summary. ({type(exc).__name__}) The results below are still valid."
                )

        return {"answer": answer, **self._state_from_tools(state)}

    def _state_from_tools(self, state: AgentState) -> dict[str, Any]:
        """Carry this turn's result set into memory (R-092).

        Only references are stored -- never rows -- because ``MemorySaver`` retains every
        checkpoint for the process lifetime (ADR-0022).
        """
        artifacts = turn_artifacts(list(state["messages"]))
        preferred = ("structured_search", "semantic_search", "fuzzy_movie_search", "rag_answer")
        for tool in preferred:
            for artifact in reversed(artifacts):
                if artifact.get("tool") != tool or not artifact.get("refs"):
                    continue
                if artifact_status(artifact) not in (Outcome.OK, Outcome.LOW_CONFIDENCE):
                    continue
                refs = artifact["refs"]
                update: dict[str, Any] = {
                    "last_results": refs,
                    "last_result_total": artifact.get("payload", {}).get("total", len(refs)),
                }
                if len(refs) == 1:
                    update["selected_movie_id"] = int(refs[0]["movie_id"])
                return update
        return {}

    def _ground_node(self, state: AgentState) -> dict[str, Any]:
        """Terminal grounding check (ADR-0012, layer 3).

        A node rather than a helper so no code path can skip it. Advisory: it flags,
        it does not rewrite or suppress -- a silently edited answer would be worse than
        a flagged one.
        """
        answer = state.get("answer") or ""
        artifacts = turn_artifacts(list(state["messages"]))

        allowed: list[MovieRef] = []
        extra: list[str] = []
        for artifact in artifacts:
            allowed.extend(_refs_of(artifact))
            payload = artifact.get("payload", {})
            record = payload.get("record")
            if isinstance(record, dict):
                for key in ("director", "top_cast", "full_cast", "genres", "keywords"):
                    extra.extend(str(v) for v in record.get(key, []) or [])
                extra.append(str(record.get("original_title") or ""))
            for document in artifact.get("meta", {}).get("documents", []) or []:
                extra.append(str(document.get("title", "")))

        # Everything the tools returned this turn, as one text blob (see check_answer).
        payload_text = " ".join(
            str(part)
            for artifact in artifacts
            for part in (
                artifact.get("message", ""),
                _stringify(artifact.get("payload")),
                # `meta` holds the result rows -- genre names, cast, scores. Omitting it
                # made every genre in a results table look like an invented entity.
                _stringify(artifact.get("meta")),
            )
        )
        warnings = check_answer(answer, allowed, extra, payload_text)
        if warnings:
            log.warning("grounding warnings: %s", warnings)
        return {"deviations": [*(state.get("deviations") or []), *warnings]} if warnings else {}

    def _smalltalk_node(self, state: AgentState) -> dict[str, Any]:
        """No dataset lookup needed -- greetings, thanks, off-topic (R-086)."""
        if state.get("answer"):
            return {}
        try:
            response = self._model.invoke(
                [
                    SystemMessage(content=prompts.NO_TOOL_SYSTEM),
                    HumanMessage(content=state.get("question", "")),
                ]
            )
            return {"answer": message_text(response).strip()}
        except Exception as exc:  # noqa: BLE001
            log.exception("smalltalk failed")
            return {
                "answer": (
                    "I could not reach the language model. "
                    f"({type(exc).__name__}) Please try again."
                )
            }

    # --------------------------------------------------------------------- execution

    def _config(self, thread_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self._settings.agent.recursion_limit,
        }

    def run(self, question: str, thread_id: str) -> TurnResult:
        """Run one turn."""
        return self._execute({"messages": [HumanMessage(content=question)]}, thread_id, question)

    def resume(self, reply: str, thread_id: str) -> TurnResult:
        """Resume a turn paused at the clarification interrupt."""
        return self._execute(Command(resume=reply), thread_id, reply)

    def _execute(self, payload: Any, thread_id: str, question: str) -> TurnResult:
        started = time.perf_counter()
        config = self._config(thread_id)
        try:
            state = self.graph.invoke(payload, config=config)
        except Exception as exc:  # noqa: BLE001 - never let a turn crash the UI (R-105)
            log.exception("graph execution failed")
            trace = Trace(question=question, error=f"{type(exc).__name__}: {exc}")
            trace.total_duration_ms = (time.perf_counter() - started) * 1000
            return TurnResult(
                answer=(
                    "Something went wrong while answering that. The error has been "
                    f"logged: {type(exc).__name__}."
                ),
                trace=trace,
            )

        elapsed = (time.perf_counter() - started) * 1000
        interrupts = state.get("__interrupt__") or ()
        if interrupts:
            payload_value = getattr(interrupts[0], "value", {}) or {}
            trace = self._build_trace(thread_id, question, elapsed)
            trace.interrupted = True
            trace.answer = payload_value.get("question", "")
            return TurnResult(
                answer=payload_value.get("question", "Which movie did you mean?"),
                trace=trace,
                interrupted=True,
                clarification=payload_value,
            )

        trace = self._build_trace(thread_id, question, elapsed)
        return TurnResult(answer=trace.answer, trace=trace, refs=trace.result_refs)

    def _build_trace(self, thread_id: str, question: str, elapsed_ms: float) -> Trace:
        """Assemble the trace from graph state and tool artifacts (ADR-0023).

        Nothing is threaded through the nodes: the artifacts already carry the typed
        payloads, so the trace is read off the finished state.
        """
        snapshot = self.graph.get_state(self._config(thread_id))
        state: dict[str, Any] = snapshot.values or {}
        messages = list(state.get("messages") or [])
        turn = current_turn_messages(messages)

        trace = Trace(question=question, total_duration_ms=elapsed_ms)
        trace.answer = state.get("answer") or ""

        raw_plan = state.get("plan")
        if raw_plan:
            try:
                trace.plan = Plan.model_validate(raw_plan)
            except Exception:  # noqa: BLE001 - a bad plan must not break rendering
                trace.plan = None

        for message in turn:
            if not isinstance(message, ToolMessage):
                continue
            artifact = getattr(message, "artifact", None)
            if not isinstance(artifact, dict):
                continue
            trace.tool_calls.append(
                ToolCallRecord(
                    tool=artifact.get("tool", message.name or "unknown"),
                    arguments=artifact.get("arguments", {}),
                    status=artifact.get("status", "unknown"),
                    message=artifact.get("message", ""),
                    result_count=len(artifact.get("refs", [])),
                    meta={
                        k: v
                        for k, v in artifact.get("meta", {}).items()
                        if k not in {"rows", "documents"}
                    },
                    artifact=artifact,
                )
            )

        # Only a refinement inherits anything. On any other turn `active_query` holds
        # exactly the filters this message produced, which the panel already shows as
        # "filters extracted from this message".
        active = state.get("active_query")
        if (
            active is not None
            and not active.is_empty()
            and trace.plan is not None
            and trace.plan.refines_previous
        ):
            trace.carried_forward = active.describe()

        trace.result_refs = [MovieRef.from_dict(r) for r in (state.get("last_results") or [])]
        trace.token_usage = token_usage(turn)
        trace.truncated = (
            len(trace.tool_calls) >= self._settings.agent.max_tool_iterations
        )

        deviations = state.get("deviations") or []
        for note in deviations:
            trace.record_deviation(note)
        trace.grounding_warnings = [d for d in deviations if "not in the retrieved records" in d]
        trace.check_plan_adherence()
        return trace


def build_agent(
    settings: Settings,
    context: ToolContext,
    model: Any,
    checkpointer: Any | None = None,
) -> MovieAgent:
    """Construct the agent, wiring the model into ``rag_answer`` as a plain callable."""
    grounded_context = ToolContext(
        settings=context.settings,
        repository=context.repository,
        matcher=context.matcher,
        index=context.index,
        embedder=context.embedder,
        documents=context.documents,
        vocabulary=context.vocabulary,
        answer_fn=make_answer_fn(model),
    )
    return MovieAgent(settings, grounded_context, model, checkpointer)
