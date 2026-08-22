"""Retrieval: semantic documents, the vector backends, and fuzzy title matching.

Vector search sits behind ``backend.SearchBackend``, with two implementations: the
in-process numpy index (default) and a Qdrant collection. Both enforce ADR-0011's
pre-filter guarantee; ``scripts/benchmark_vector_backends.py`` asserts they agree.
"""
