"""Agentic movie discovery and analysis over the TMDB 5000 dataset.

Layering (ADR-0019), dependencies point one way only:

    config -> data / llm -> retrieval -> tools -> agent -> ui

Two import rules are enforced by ``tests/test_layering.py``:

* ``streamlit`` may only be imported under ``ui/`` and ``app.py``.
* ``langgraph`` / ``langchain*`` may only be imported under ``agent/`` and ``llm/``.

That is what keeps ``data/``, ``retrieval/`` and ``tools/`` framework-free, and it is
why superseding ADR-0001 with ADR-0020 cost ``agent/`` rather than the whole system.
"""

__version__ = "0.1.0"
