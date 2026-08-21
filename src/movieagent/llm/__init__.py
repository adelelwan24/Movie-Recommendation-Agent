"""Model access: chat (LangChain) and embeddings (our own protocol).

This package and ``agent/`` are the only two allowed to import ``langchain*`` /
``langgraph`` -- enforced by ``tests/test_layering.py`` (ADR-0019, ADR-0020).
"""
