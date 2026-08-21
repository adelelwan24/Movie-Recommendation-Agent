"""The LangGraph agent (ADR-0020, ADR-0021).

Topology::

    START -> plan -> agent <-> tools -> synthesize -> ground -> END
                       |
                       +-> clarify (interrupt) -> END, resumed next turn

This package and ``llm/`` are the only two allowed to import ``langgraph`` /
``langchain*``. Everything below stays framework-free, which is what kept superseding
ADR-0001 with ADR-0020 a change to ``agent/`` rather than to the whole system.
"""
