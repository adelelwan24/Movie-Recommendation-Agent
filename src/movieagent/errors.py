"""Exceptions for *programmer* and *environment* errors.

Deliberately narrow. Per ADR-0003, expected business outcomes -- movie not found,
ambiguous title, empty result, invalid filter -- are **not** exceptions. They are
``ToolResult`` statuses, because the agent has to reason about them and ask the user
a question rather than abort. Exceptions here mean something is genuinely wrong with
the installation, the artifacts, or the configuration.
"""

from __future__ import annotations


class MovieAgentError(Exception):
    """Base class for everything this package raises deliberately."""


class ArtifactError(MovieAgentError):
    """The built artifacts are missing, unreadable, or stale.

    ADR-0005: a stale artifact fails *silently and plausibly*, which is the worst
    kind of failure, so the manifest guard turns it into a loud one.
    """


class ArtifactStaleError(ArtifactError):
    """Artifacts do not match the source CSVs, the preprocessing version, or the
    embedding model they were built with (ADR-0005, ADR-0007)."""


class ConfigurationError(MovieAgentError):
    """Configuration is structurally invalid, or a credential is missing at the point
    of use.

    ADR-0015 validates in layers: structure at import, credentials at first use, so
    that the deterministic half of the system (preprocessing, structured search, fuzzy
    matching, local embeddings) runs with no API key at all.
    """


class EmbeddingBackendError(MovieAgentError):
    """The selected embedding backend could not be constructed or failed to embed."""
