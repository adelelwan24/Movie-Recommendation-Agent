"""Streamlit demo (Deliverable 1).

Run with::

    streamlit run app.py

Caching topology is ADR-0014's hard rule, and it is easy to get wrong in Streamlit in a
way that only shows up with a second concurrent user:

* ``@st.cache_resource`` -- the runtime and the compiled graph. Shared across **all**
  sessions and threads, therefore strictly read-only after construction.
* ``st.session_state`` -- the ``thread_id``, rendered chat history and traces. Never
  promoted to ``cache_resource``, however tempting the performance argument.
* Conversation state itself lives in neither: it is checkpointed by the graph, keyed by
  ``thread_id`` (ADR-0022).
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:  # run without an editable install
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402

from movieagent.config import get_settings  # noqa: E402
from movieagent.errors import ArtifactError, ConfigurationError  # noqa: E402
from movieagent.logging import configure_logging  # noqa: E402
from movieagent.runtime import build_default_agent, load_runtime  # noqa: E402
from movieagent.ui.components import render_results, render_trace  # noqa: E402
from movieagent.ui.text import strip_markdown_tables  # noqa: E402

st.set_page_config(page_title="TMDB Movie Agent", page_icon="🎬", layout="wide")

EXAMPLES = [
    "What are the 10 most common genres?",
    "Show the 10 highest-rated science fiction movies with at least 1,000 votes",
    "Tell me about Intersteler",
    "I want a movie about someone trying to survive alone on another planet",
    "Find me a funny science-fiction movie from after 2010 that is under 2 hours",
    "How many movies have Christopher Nolan as director?",
    "What movies are similar to lord of the rings?",
    "Show me science fiction movies from after 2010",
]


@st.cache_resource(show_spinner="Loading the dataset and vector index…")
def _runtime():
    return load_runtime(get_settings())


@st.cache_resource(show_spinner="Starting the agent…")
def _agent():
    return build_default_agent(_runtime(), get_settings())


def _init_session() -> None:
    st.session_state.setdefault("thread_id", str(uuid.uuid4()))
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("awaiting_clarification", False)


def _sidebar(settings) -> None:
    with st.sidebar:
        st.header("🎬 TMDB Movie Agent")
        st.caption(
            "Agentic discovery over the TMDB 5000 dataset. The model routes and "
            "narrates; every number comes from the data."
        )

        try:
            runtime = _runtime()
            manifest = runtime.manifest
            st.success(f"{len(runtime.repository):,} movies · {len(runtime.index):,} vectors")
            with st.expander("Dataset build", expanded=False):
                st.caption(f"Embedding model: `{manifest.embedding_model}`")
                st.caption(f"Built: {manifest.built_at[:19].replace('T', ' ')}")
                st.json(manifest.report, expanded=False)
        except ArtifactError as exc:
            st.error(str(exc))

        st.divider()
        st.caption(f"Chat model: `{settings.llm.model}`")
        st.caption(f"Endpoint: `{settings.llm.base_url}`")
        if not settings.llm.api_key:
            # ADR-0015's layered validation: the deterministic half works without a key,
            # so this is a warning rather than a refusal to start.
            st.warning("`LLM_API_KEY` is not set — the agent cannot call the model.")

        st.divider()
        st.markdown("**Try one**")
        for index, example in enumerate(EXAMPLES):
            if st.button(example, key=f"ex_{index}", width="stretch"):
                st.session_state["pending_input"] = example
                st.rerun()

        st.divider()
        if st.button("New conversation", width="stretch"):
            st.session_state["thread_id"] = str(uuid.uuid4())
            st.session_state["history"] = []
            st.session_state["awaiting_clarification"] = False
            st.rerun()
        st.caption(f"Thread `{st.session_state['thread_id'][:8]}`")


def _render_history() -> None:
    for entry in st.session_state["history"]:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(strip_markdown_tables(entry["answer"]))
            trace = entry.get("trace")
            if trace is not None:
                render_results(trace)
                with st.expander("How this answer was produced", expanded=False):
                    render_trace(trace)


def _handle(question: str) -> None:
    agent = _agent()
    thread_id = st.session_state["thread_id"]

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Planning and running tools…"):
            # A clarification reply *resumes* the paused turn rather than starting a new
            # one, so the tool results that prompted the question are still in scope
            # (ADR-0021's interrupt).
            if st.session_state["awaiting_clarification"]:
                result = agent.resume(question, thread_id)
            else:
                result = agent.run(question, thread_id)

        st.session_state["awaiting_clarification"] = result.interrupted
        # Tables in the answer text are always duplicates of what `render_results`
        # is about to draw from the payload, so the renderer drops them (ADR-0012).
        # `result.answer` itself is stored and traced unchanged.
        st.markdown(strip_markdown_tables(result.answer))
        render_results(result.trace)
        with st.expander("How this answer was produced", expanded=result.interrupted):
            render_trace(result.trace)

    st.session_state["history"].append(
        {"question": question, "answer": result.answer, "trace": result.trace}
    )


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_file)
    _init_session()
    _sidebar(settings)

    st.title("Movie discovery agent")
    st.caption(
        "Ask in plain language. Structured questions are answered from the data, "
        "conceptual ones by semantic retrieval, and the agent will ask rather than "
        "guess when a title is ambiguous."
    )

    _render_history()

    typed = st.chat_input("Ask about movies…")
    pending = st.session_state.pop("pending_input", None)
    question = typed or pending

    if question:
        try:
            _handle(question)
        except ArtifactError as exc:
            st.error(str(exc))
        except ConfigurationError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - R-105: never show a traceback
            st.error(
                f"Something went wrong: {type(exc).__name__}. The details are in the logs."
            )
            st.exception(exc) if settings.log_level.upper() == "DEBUG" else None

    # Last, deliberately. Streamlit renders top to bottom and the interrupt that raises
    # this flag happens inside `_handle`, so writing the notice any earlier shows either
    # the previous turn's state or places it above the exchange it refers to. Written
    # here it lands directly beneath the candidate list the agent just printed.
    if st.session_state["awaiting_clarification"]:
        st.info(
            "Waiting for you to choose which movie you meant — reply with a number, "
            "an ordinal, or the title."
        )


main()
