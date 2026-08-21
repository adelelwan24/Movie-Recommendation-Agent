"""Rendering for results and the execution trace (R-102, R-103, R-104, R-105).

The trace is a **product surface**, not a diagnostic: the PDF lists its required
contents and forbids one of them. Everything here renders from the typed ``Trace``
(ADR-0013), so there is no second source of truth and nothing to parse.

R-104 is respected in two ways. The ``Trace`` type has no field for model reasoning, and
whatever the provider sends was already stripped at the model boundary (ADR-0023). This
module therefore cannot display chain-of-thought even if it tried.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from movieagent.agent.trace import Trace
from movieagent.tools.base import Outcome

#: How each outcome is shown. Distinct rendering per status is the visible payoff of
#: ADR-0003's envelope -- "no movies matched" must not look like "something broke".
_STATUS_STYLE: dict[str, tuple[str, str]] = {
    Outcome.OK.value: ("✅", "ok"),
    Outcome.EMPTY.value: ("∅", "no matches"),
    Outcome.NOT_FOUND.value: ("🔍", "not found"),
    Outcome.AMBIGUOUS.value: ("❓", "ambiguous — asked the user"),
    Outcome.LOW_CONFIDENCE.value: ("⚠️", "low confidence"),
    Outcome.INVALID_INPUT.value: ("🚫", "invalid arguments"),
    Outcome.ERROR.value: ("💥", "error"),
}


def render_results(trace: Trace) -> None:
    """Tables for structured results and records (R-037, R-107).

    Numbers are rendered from the tool payload rather than from the model's prose --
    which removes the most damaging class of fabrication from the model's remit
    entirely (ADR-0012).
    """
    for call in trace.tool_calls:
        artifact = call.artifact or {}
        payload = artifact.get("payload", {})
        meta = artifact.get("meta", {})

        if aggregate := payload.get("aggregate"):
            st.caption(f"**{call.tool}** — {call.message}")
            st.dataframe(pd.DataFrame(aggregate), width="stretch", hide_index=True)

        elif rows := meta.get("rows"):
            st.caption(f"**{call.tool}** — {call.message}")
            st.dataframe(_display_frame(rows), width="stretch", hide_index=True)

        elif record := payload.get("record"):
            _render_record(record)


def _display_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if "genre_names" in frame.columns:
        frame["genre_names"] = frame["genre_names"].map(
            lambda v: ", ".join(v) if isinstance(v, (list, tuple)) else v
        )
    renames = {
        "id": "ID",
        "title": "Title",
        "release_year": "Year",
        "genre_names": "Genres",
        "vote_average": "Rating",
        "vote_count": "Votes",
        "runtime": "Runtime",
    }
    return frame.rename(columns={k: v for k, v in renames.items() if k in frame.columns})


def _money(value: Any, known: bool) -> str:
    """Unknown is unknown -- never ``$0`` (R-016, R-061)."""
    if not known or value in (None, ""):
        return "unknown"
    return f"${int(value):,}"


def _render_record(record: dict[str, Any]) -> None:
    title = record.get("title") or "Unknown title"
    year = record.get("release_year")
    st.markdown(f"#### {title}" + (f" ({year})" if year else ""))

    if tagline := record.get("tagline"):
        st.caption(f"_{tagline}_")

    columns = st.columns(4)
    rating = record.get("vote_average")
    votes = record.get("vote_count")
    runtime = record.get("runtime_minutes")
    columns[0].metric("Rating", f"{rating}" if rating is not None else "unknown")
    columns[1].metric("Votes", f"{votes:,}" if votes is not None else "unknown")
    columns[2].metric("Runtime", f"{runtime} min" if runtime is not None else "unknown")
    columns[3].metric(
        "Revenue", _money(record.get("revenue_usd"), bool(record.get("revenue_known")))
    )

    facts = {
        "Director": ", ".join(record.get("director") or []) or "unknown",
        "Starring": ", ".join(record.get("top_cast") or []) or "unknown",
        "Genres": ", ".join(record.get("genres") or []) or "unknown",
        "Released": record.get("release_date") or "unknown",
        "Budget": _money(record.get("budget_usd"), bool(record.get("budget_known"))),
        "Language": record.get("original_language") or "unknown",
        "Companies": ", ".join(record.get("production_companies") or []) or "unknown",
    }
    st.table(pd.DataFrame({"": list(facts.values())}, index=list(facts.keys())))

    if overview := record.get("overview"):
        st.markdown(f"**Overview** — {overview}")


def render_trace(trace: Trace) -> None:
    """The execution trace panel (R-102, R-103)."""
    if trace.error:
        st.error(f"Execution failed: {trace.error}")

    if trace.grounding_warnings:
        # Advisory, not blocking. ADR-0012 flags rather than rewrites, because silently
        # editing an answer would be worse than showing a caveat -- and this check is
        # heuristic, so it produces false positives on character and person names.
        st.warning(
            "**Grounding check** — these look like movie references that were not in "
            "the retrieved records:\n\n"
            + "\n".join(f"- {w}" for w in trace.grounding_warnings)
        )

    if trace.plan:
        st.markdown("**Plan**")
        st.markdown(f"> {trace.plan.rationale or trace.plan.intent}")
        tools = trace.plan.tool_sequence()
        st.markdown("Tools selected: " + (" → ".join(f"`{t}`" for t in tools) or "_none_"))
        if trace.plan.reference_note:
            st.caption(f"Reference resolved: {trace.plan.reference_note}")
        if filters := trace.plan.to_display().get("filters"):
            st.caption("Filters extracted from this message:")
            st.json(filters, expanded=False)

    if trace.carried_forward:
        # R-148: state what was carried, so a follow-up can never silently inherit
        # constraints the user has forgotten about.
        st.info("**Carried forward from earlier turns:** " + "; ".join(trace.carried_forward))

    if trace.tool_calls:
        st.markdown("**Execution**")
        for index, call in enumerate(trace.tool_calls, start=1):
            icon, label = _STATUS_STYLE.get(call.status, ("•", call.status))
            with st.expander(f"{icon} {index}. `{call.tool}` — {label}", expanded=False):
                st.caption(call.message)
                if call.arguments:
                    st.markdown("_Arguments_")
                    st.json(call.arguments, expanded=False)
                _render_call_meta(call.meta)

    if documents := trace.retrieved_documents():
        st.markdown("**Retrieved context**")
        st.caption(
            "Exactly the documents that were embedded and handed to the model. "
            "Nothing else about these films entered the prompt."
        )
        for document in documents:
            year = f" ({document['year']})" if document.get("year") else ""
            with st.expander(
                f"{document['title']}{year} — similarity {document['score']:.3f}",
                expanded=False,
            ):
                st.code(document.get("document", ""), language=None)

    if trace.deviations:
        non_grounding = [d for d in trace.deviations if d not in trace.grounding_warnings]
        if non_grounding:
            st.caption("**Plan vs actual:** " + "; ".join(non_grounding))

    footer = []
    if trace.total_duration_ms:
        footer.append(f"{trace.total_duration_ms / 1000:.1f}s")
    if usage := trace.token_usage:
        footer.append(f"{usage.get('total_tokens', 0):,} tokens")
    if trace.truncated:
        footer.append("hit the tool-iteration cap")
    if footer:
        st.caption(" · ".join(footer))


def _render_call_meta(meta: dict[str, Any]) -> None:
    """Show the numbers that make a trace explanatory rather than decorative."""
    if not meta:
        return
    if constraints := meta.get("constraints"):
        st.markdown("_Filters applied_")
        st.markdown("\n".join(f"- `{c}`" for c in constraints))
    bits: list[str] = []
    if (pool := meta.get("pool_size")) is not None:
        corpus = meta.get("corpus_size")
        bits.append(f"ranked within {pool} of {corpus} movies" if corpus else f"pool {pool}")
    if (score := meta.get("score")) is not None:
        bits.append(f"match score {score:.0f}/100")
    if scores := meta.get("scores"):
        bits.append(f"similarity {scores[0]:.3f}–{scores[-1]:.3f}")
    if unknown := meta.get("unknown_fields"):
        bits.append(f"{len(unknown)} fields unknown in the dataset")
    if reason := meta.get("reason"):
        bits.append(reason)
    if bits:
        st.caption(" · ".join(bits))
