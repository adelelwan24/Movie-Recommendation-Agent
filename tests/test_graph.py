"""Agent contract tests (ADR-0024 tier 2).

Substituted at ``BaseChatModel``, so the **real graph** runs: reducers, conditional edges,
``ToolNode``, the checkpointer and the interrupt are all exercised. Only the model is fake.

Covers the four PDF flows (R-083), the documented worked examples (R-084), tool chaining
(R-082), the three-turn memory flow (R-091), filter carry-forward (R-148), the
clarification interrupt (R-043) and every provider failure path (R-086, R-087).

No network. A scripted plan proves the graph *handles* that plan, not that a real model
would produce it — the honest limit ADR-0024 records.
"""

from __future__ import annotations

import uuid

import pytest
from langchain_core.messages import AIMessage

from movieagent.agent.graph import build_agent, resolve_choice
from movieagent.agent.plan import Plan, PlanStep, ToolName
from movieagent.data.query import (
    ComparisonOp,
    Condition,
    NumericField,
    SearchQuery,
)
from movieagent.llm.fake import ScriptedChatModel, text, tool_call

INCEPTION = 27205
INTERSTELLAR = 157336


@pytest.fixture
def model() -> ScriptedChatModel:
    return ScriptedChatModel(responses=[], structured_queue=[])


@pytest.fixture
def agent(runtime, settings, model):
    return build_agent(settings, runtime.tool_context(), model)


@pytest.fixture
def thread() -> str:
    return str(uuid.uuid4())


def plan(*tools: ToolName, **kwargs) -> Plan:
    return Plan(
        intent=kwargs.pop("intent", "test"),
        steps=[PlanStep(tool=t, why="scripted") for t in tools],
        rationale=kwargs.pop("rationale", "scripted plan"),
        **kwargs,
    )


class TestPdfFlows:
    """R-083: the four flows the PDF documents on p4 §7."""

    def test_flow_a_count_is_structured_and_deterministic(self, agent, model, thread) -> None:
        """"How many movies were released after 2010?" -> structured -> count."""
        model.structured_queue = [
            plan(ToolName.STRUCTURED_SEARCH, filters=SearchQuery(year_from=2011))
        ]
        model.responses = [
            tool_call("structured_search", {"query": {"year_from": 2011}}),
            text("There are 1,101 movies released after 2010."),
        ]
        result = agent.run("How many movies were released after 2010?", thread)

        assert result.trace.tools_used == ["structured_search"]
        call = result.trace.tool_calls[0]
        assert call.status == "ok"
        # The count comes from pandas, not from the scripted prose.
        assert call.artifact["payload"]["total"] > 1_000

    def test_flow_b_fuzzy_then_details_chains(self, agent, model, thread) -> None:
        """R-082: "Tell me about Intersteler" -> fuzzy -> movie_details."""
        model.structured_queue = [plan(ToolName.FUZZY_MOVIE_SEARCH, ToolName.MOVIE_DETAILS)]
        model.responses = [
            tool_call("fuzzy_movie_search", {"title": "Intersteler"}),
            tool_call("movie_details", {"movie_id": INTERSTELLAR}),
            text("Interstellar (2014), directed by Christopher Nolan."),
        ]
        result = agent.run("Tell me about Intersteler", thread)

        assert result.trace.tools_used == ["fuzzy_movie_search", "movie_details"]
        assert result.trace.tool_calls[0].artifact["refs"][0]["title"] == "Interstellar"
        assert result.trace.tool_calls[1].artifact["payload"]["record"]["title"] == "Interstellar"

    def test_flow_c_semantic_then_rag(self, agent, model, thread) -> None:
        model.structured_queue = [plan(ToolName.SEMANTIC_SEARCH, ToolName.RAG_ANSWER)]
        model.responses = [
            tool_call("semantic_search", {"query": "losing grip on reality", "k": 5}),
            text("Several psychological thrillers fit that description."),
        ]
        result = agent.run("Find me a dark psychological movie", thread)

        assert "semantic_search" in result.trace.tools_used
        assert result.trace.retrieved_documents(), "R-103 needs retrieved context in the trace"

    def test_flow_d_hybrid(self, agent, model, thread) -> None:
        """R-070: structured filters + semantic intent in one call."""
        filters = SearchQuery(
            genres=["Science Fiction"],
            conditions=[Condition(field=NumericField.RUNTIME, op=ComparisonOp.LT, value=120)],
        )
        model.structured_queue = [plan(ToolName.SEMANTIC_SEARCH, filters=filters)]
        model.responses = [
            tool_call(
                "semantic_search",
                {
                    "query": "highly rated science fiction",
                    "filters": {"genres": ["Science Fiction"],
                                "conditions": [{"field": "runtime", "op": "lt", "value": 120}]},
                    "k": 5,
                },
            ),
            text("Here are some short science-fiction films."),
        ]
        result = agent.run("Find me a highly rated sci-fi movie under 120 minutes", thread)

        artifact = result.trace.tool_calls[0].artifact
        assert artifact["status"] == "ok"
        assert artifact["meta"]["pool_size"] < len(agent._context.repository)


class TestWorkedExamples:
    """R-084: the two examples the PDF spells out on p6 §2."""

    def test_nolan_director_aggregate(self, agent, model, thread) -> None:
        model.structured_queue = [
            plan(ToolName.STRUCTURED_SEARCH, filters=SearchQuery(directors=["Christopher Nolan"]))
        ]
        model.responses = [
            tool_call("structured_search", {"query": {"directors": ["Christopher Nolan"]}}),
            text("Christopher Nolan directed 8 movies in this dataset."),
        ]
        result = agent.run("How many movies have Christopher Nolan as director?", thread)
        assert result.trace.tool_calls[0].artifact["payload"]["total"] == 8


class TestMultiTurnMemory:
    """R-090 – R-093: the canonical three-turn flow from p4 §8."""

    def test_three_connected_turns(self, agent, model, thread) -> None:
        # --- Turn 1: sci-fi after 2010
        model.structured_queue = [
            plan(
                ToolName.STRUCTURED_SEARCH,
                filters=SearchQuery(genres=["Science Fiction"], year_from=2011),
            )
        ]
        model.responses = [
            tool_call(
                "structured_search",
                {"query": {"genres": ["Science Fiction"], "year_from": 2011}},
            ),
            text("Here are science fiction movies from after 2010."),
        ]
        turn1 = agent.run("Show me science fiction movies from after 2010", thread)
        total1 = turn1.trace.tool_calls[0].artifact["payload"]["total"]
        assert total1 > 25, "turn 1 must be capped so turn 2 can prove it re-queries"

        # --- Turn 2: refine. `refines_previous=True` is what carries turn 1 forward.
        model.structured_queue = [
            plan(
                ToolName.STRUCTURED_SEARCH,
                filters=SearchQuery(
                    conditions=[
                        Condition(field=NumericField.VOTE_AVERAGE, op=ComparisonOp.GT, value=7.5)
                    ]
                ),
                refines_previous=True,
            )
        ]
        model.responses = [
            tool_call(
                "structured_search",
                {
                    "query": {
                        "genres": ["Science Fiction"],
                        "year_from": 2011,
                        "conditions": [{"field": "vote_average", "op": "gt", "value": 7.5}],
                    }
                },
            ),
            text("Filtered to those rated above 7.5."),
        ]
        turn2 = agent.run("Only show ones rated above 7.5", thread)
        total2 = turn2.trace.tool_calls[0].artifact["payload"]["total"]

        assert 0 < total2 < total1, "turn 2 must narrow turn 1"
        # R-148 / OQ-006: carried forward, and *visible*.
        assert turn2.trace.carried_forward
        assert any("Science Fiction" in c for c in turn2.trace.carried_forward)

        # --- Turn 3: "the first one" resolves by index, not by model recall.
        first = turn2.refs[0]
        model.structured_queue = [
            plan(
                ToolName.MOVIE_DETAILS,
                resolved_movie_ids=[first.movie_id],
                reference_note=f"'the first one' -> {first.title}",
                refines_previous=True,
            )
        ]
        model.responses = [
            tool_call("movie_details", {"movie_id": first.movie_id}),
            text(f"{first.title} is..."),
        ]
        turn3 = agent.run("Tell me more about the first one", thread)

        record = turn3.trace.tool_calls[0].artifact["payload"]["record"]
        assert record["id"] == first.movie_id
        assert turn3.trace.plan.reference_note

    def test_a_new_topic_resets_carried_filters(self, agent, model, thread) -> None:
        """Without a reset path, filters accumulate forever and turn nine returns nothing."""
        model.structured_queue = [
            plan(ToolName.STRUCTURED_SEARCH, filters=SearchQuery(genres=["Horror"]))
        ]
        model.responses = [
            tool_call("structured_search", {"query": {"genres": ["Horror"]}}),
            text("Horror movies."),
        ]
        agent.run("Show me horror movies", thread)

        model.structured_queue = [
            plan(
                ToolName.STRUCTURED_SEARCH,
                filters=SearchQuery(genres=["Comedy"]),
            )
        ]
        model.responses = [
            tool_call("structured_search", {"query": {"genres": ["Comedy"]}}),
            text("Comedies."),
        ]
        second = agent.run("Actually, show me comedies", thread)

        carried = " ".join(second.trace.carried_forward)
        assert "Horror" not in carried


class TestClarificationInterrupt:
    """R-043: ambiguity pauses the turn instead of guessing (ADR-0021)."""

    def test_ambiguous_title_interrupts_and_resumes(self, agent, model, thread) -> None:
        model.structured_queue = [plan(ToolName.FUZZY_MOVIE_SEARCH, ToolName.MOVIE_DETAILS)]
        model.responses = [tool_call("fuzzy_movie_search", {"title": "lord of the rings"})]

        paused = agent.run("Tell me about lord of the rings", thread)

        assert paused.interrupted, "an ambiguous match must not resolve itself"
        assert paused.clarification is not None
        candidates = paused.clarification["candidates"]
        assert len(candidates) >= 2
        assert "Which did you mean" in paused.answer

        # Resume the *same* turn: the tool result that prompted the question is still
        # in scope, which is the whole point of using a real interrupt.
        model.responses = [
            tool_call("movie_details", {"movie_id": candidates[1]["movie_id"]}),
            text("Here are the details."),
        ]
        resumed = agent.resume("the second one", thread)

        assert not resumed.interrupted
        record = resumed.trace.last_artifact("movie_details")["payload"]["record"]
        assert record["id"] == candidates[1]["movie_id"]

    @pytest.mark.parametrize(
        ("reply", "expected_index"),
        [("the first", 0), ("second", 1), ("number 3", 2), ("3", 2)],
    )
    def test_choice_resolution_is_deterministic(self, reply: str, expected_index: int) -> None:
        """Having asked which film they meant, resolving the answer with another model
        call would reintroduce exactly the guessing the question avoided."""
        candidates = [
            {"movie_id": 1, "title": "The Fellowship of the Ring", "year": 2001},
            {"movie_id": 2, "title": "The Two Towers", "year": 2002},
            {"movie_id": 3, "title": "The Return of the King", "year": 2003},
        ]
        assert resolve_choice(reply, candidates) == candidates[expected_index]

    def test_choice_by_title(self) -> None:
        candidates = [
            {"movie_id": 1, "title": "The Two Towers", "year": 2002},
            {"movie_id": 2, "title": "The Return of the King", "year": 2003},
        ]
        assert resolve_choice("return of the king", candidates)["movie_id"] == 2

    def test_unresolvable_choice_returns_none(self) -> None:
        assert resolve_choice("neither of those", [{"movie_id": 1, "title": "X"}]) is None


class TestErrorPaths:
    """R-086, R-087: every failure surfaces as an answer, never a crash."""

    def test_provider_failure_during_planning_is_graceful(self, agent, model, thread) -> None:
        model.structured_queue = [RuntimeError("429 rate limited")]
        result = agent.run("How many action movies are there?", thread)

        assert "could not reach the language model" in result.answer.lower()
        assert result.trace.tool_calls == []

    def test_provider_failure_during_execution_is_graceful(self, agent, model, thread) -> None:
        model.structured_queue = [plan(ToolName.STRUCTURED_SEARCH)]
        model.responses = [TimeoutError("upstream timeout")]
        result = agent.run("Show me comedies", thread)
        assert "could not reach" in result.answer.lower()

    def test_invalid_filter_reaches_the_model_as_correctable_data(
        self, agent, model, thread
    ) -> None:
        """R-038: the agent gets the error and a suggestion, not an empty result."""
        model.structured_queue = [plan(ToolName.STRUCTURED_SEARCH)]
        model.responses = [
            tool_call("structured_search", {"query": {"genres": ["Sci-Fi"]}}),
            tool_call("structured_search", {"query": {"genres": ["Science Fiction"]}}),
            text("Here are science fiction movies."),
        ]
        result = agent.run("Show me sci-fi movies", thread)

        statuses = [c.status for c in result.trace.tool_calls]
        assert statuses == ["invalid_input", "ok"]
        assert "Science Fiction" in str(result.trace.tool_calls[0].artifact["payload"]["errors"])

    def test_empty_result_is_explained_not_faked(self, agent, model, thread) -> None:
        """R-039: empty must be distinguishable and must name what bound it."""
        model.structured_queue = [plan(ToolName.STRUCTURED_SEARCH)]
        model.responses = [
            tool_call(
                "structured_search",
                {"query": {"conditions": [{"field": "vote_average", "op": "gt", "value": 10}]}},
            ),
            text("Nothing is rated above 10."),
        ]
        result = agent.run("Show me movies rated above 10", thread)

        call = result.trace.tool_calls[0]
        assert call.status == "empty"
        assert call.artifact["payload"]["binding_constraints"]

    def test_offtopic_query_needs_no_tools(self, agent, model, thread) -> None:
        """R-086: a greeting must not trigger a dataset lookup."""
        model.structured_queue = [
            Plan(intent="greeting", steps=[], rationale="no lookup needed", needs_tools=False)
        ]
        model.responses = [text("Hello. Ask me about movies in the TMDB dataset.")]
        result = agent.run("hello there", thread)

        assert result.trace.tool_calls == []
        assert result.answer

    def test_tool_iteration_cap_terminates(self, agent, model, settings, thread) -> None:
        """The loop guard: a model that never stops calling tools must still terminate."""
        model.structured_queue = [plan(ToolName.STRUCTURED_SEARCH)]
        model.responses = [
            tool_call("structured_search", {"query": {"genres": ["Drama"]}}, call_id=f"c{i}")
            for i in range(20)
        ] + [text("Done.")]

        result = agent.run("Show me dramas", thread)
        assert len(result.trace.tool_calls) <= settings.agent.max_tool_iterations
        assert result.answer

    def test_unknown_movie_id_is_not_found(self, agent, model, thread) -> None:
        model.structured_queue = [plan(ToolName.MOVIE_DETAILS)]
        model.responses = [
            tool_call("movie_details", {"movie_id": 999999999}),
            text("That movie is not in the dataset."),
        ]
        result = agent.run("Tell me about movie 999999999", thread)
        assert result.trace.tool_calls[0].status == "not_found"


class TestTraceContract:
    """R-102, R-103, R-104."""

    def test_trace_carries_what_the_ui_must_show(self, agent, model, thread) -> None:
        model.structured_queue = [
            plan(
                ToolName.SEMANTIC_SEARCH,
                rationale="Semantic search, because this is a plot description.",
            )
        ]
        model.responses = [
            tool_call("semantic_search", {"query": "survive alone on another planet", "k": 5}),
            text("The Martian fits."),
        ]
        result = agent.run("a movie about surviving alone on another planet", thread)
        trace = result.trace

        assert trace.plan.rationale  # R-102: concise selection explanation
        assert trace.tools_used == ["semantic_search"]  # R-102: selected tool
        assert trace.retrieved_documents()  # R-103: retrieved context
        assert trace.tool_calls[0].arguments  # R-103: arguments/filters

    def test_trace_has_no_field_for_reasoning(self) -> None:
        """R-104 held structurally in the type, per ADR-0013.

        The runtime defence is ADR-0023's sanitizer; this asserts the second half --
        that there is nowhere in the trace to put reasoning even if it survived.
        """
        from movieagent.agent.trace import Trace

        fields = set(Trace.__dataclass_fields__)
        assert not {"reasoning", "thinking", "chain_of_thought", "scratchpad"} & fields

    def test_plan_deviation_is_recorded_not_hidden(self, agent, model, thread) -> None:
        """ADR-0021 accepted that the displayed plan is intent, not a guarantee."""
        model.structured_queue = [plan(ToolName.STRUCTURED_SEARCH)]
        model.responses = [
            tool_call("fuzzy_movie_search", {"title": "Avatar"}),
            text("Avatar."),
        ]
        result = agent.run("Show me Avatar", thread)
        assert any("fuzzy_movie_search" in d for d in result.trace.deviations)


class TestGraphTopology:
    """ADR-0024 tier 5: the edges that encode requirements must exist."""

    def test_requirement_bearing_nodes_are_present(self, agent) -> None:
        nodes = set(agent.graph.get_graph().nodes)
        assert {"plan", "agent", "tools", "clarify", "synthesize", "ground"} <= nodes

    def test_mermaid_renders_for_the_docs(self, agent) -> None:
        """ARCHITECTURE.md's diagram is generated from this, so it cannot drift."""
        diagram = agent.mermaid()
        assert "clarify" in diagram and "ground" in diagram


class TestFilterCarryDefaults:
    """A self-contained question must not inherit the previous question's filters.

    The live failure this pins: after "well-rated science fiction", asking "How many
    movies have Christopher Nolan as director?" carried `genre in ['Science Fiction']`
    and `vote_count >= 1000` into the count. The answer was confident, plausible and
    about a question nobody asked -- the worst failure shape this system has.

    The fix inverts the default: filters carry only when the planner marks the turn a
    refinement. Forgetting context is recoverable and visible; inheriting it is neither.
    """

    def _establish_filters(self, agent, model, thread) -> None:
        model.structured_queue = [
            plan(
                ToolName.STRUCTURED_SEARCH,
                filters=SearchQuery(
                    genres=["Science Fiction"],
                    conditions=[
                        Condition(
                            field=NumericField.VOTE_COUNT, op=ComparisonOp.GTE, value=1000
                        )
                    ],
                ),
            )
        ]
        model.responses = [
            tool_call(
                "structured_search",
                {
                    "query": {
                        "genres": ["Science Fiction"],
                        "conditions": [{"field": "vote_count", "op": "gte", "value": 1000}],
                    }
                },
            ),
            text("Well-known science fiction."),
        ]
        agent.run("Show me well-known science fiction movies", thread)

    def test_a_new_question_does_not_inherit_earlier_filters(
        self, agent, model, thread
    ) -> None:
        self._establish_filters(agent, model, thread)

        # A complete question with its own subject: the planner leaves refines_previous
        # at its default of False.
        model.structured_queue = [
            plan(ToolName.STRUCTURED_SEARCH, filters=SearchQuery(directors=["Christopher Nolan"]))
        ]
        model.responses = [
            tool_call("structured_search", {"query": {"directors": ["Christopher Nolan"]}}),
            text("Christopher Nolan directed 8 films in this dataset."),
        ]
        result = agent.run("How many movies have Christopher Nolan as director?", thread)

        carried = " ".join(result.trace.carried_forward)
        assert "Science Fiction" not in carried
        assert "vote_count" not in carried

        applied = result.trace.tool_calls[0].artifact["arguments"]["query"]
        assert applied.get("genres", []) == []
        assert applied.get("conditions", []) == []
        assert result.trace.tool_calls[0].artifact["payload"]["total"] == 8

    def test_a_refinement_still_carries(self, agent, model, thread) -> None:
        """The inverted default must not break R-148 -- refinements still layer."""
        self._establish_filters(agent, model, thread)

        model.structured_queue = [
            plan(
                ToolName.STRUCTURED_SEARCH,
                filters=SearchQuery(year_from=2010),
                refines_previous=True,
            )
        ]
        model.responses = [
            tool_call(
                "structured_search",
                {
                    "query": {
                        "genres": ["Science Fiction"],
                        "year_from": 2010,
                        "conditions": [{"field": "vote_count", "op": "gte", "value": 1000}],
                    }
                },
            ),
            text("Narrowed to 2010 and later."),
        ]
        result = agent.run("only the ones from 2010 onwards", thread)

        carried = " ".join(result.trace.carried_forward)
        assert "Science Fiction" in carried
        assert "release_year >= 2010" in carried
