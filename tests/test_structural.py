"""Structural guarantees (ADR-0024 tier 5).

These assert properties the architecture *claims*, so that a claim cannot quietly stop
being true. Each one corresponds to an ADR that would otherwise rest on discipline alone.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from movieagent.agent.grounding import check_answer
from movieagent.agent.trace import Trace
from movieagent.data.schema import MovieRef
from movieagent.llm.models import REASONING_KEYS, sanitize_message

SRC = Path(__file__).resolve().parents[1] / "src" / "movieagent"


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def package_of(path: Path) -> str:
    relative = path.relative_to(SRC)
    return relative.parts[0] if len(relative.parts) > 1 else ""


class TestLayering:
    """ADR-0019: dependencies point one way, and the rule is checkable.

    This is the test that made superseding ADR-0001 with ADR-0020 cost `agent/` rather
    than the whole system.
    """

    ORDER = ["config", "errors", "logging", "data", "llm", "retrieval", "tools", "agent", "ui"]

    def test_imports_never_point_upward(self) -> None:
        rank = {name: i for i, name in enumerate(self.ORDER)}
        violations: list[str] = []

        for path in SRC.rglob("*.py"):
            package = package_of(path)
            if package not in rank:
                continue
            for module in imported_modules(path):
                if not module.startswith("movieagent."):
                    continue
                target = module.split(".")[1]
                if target not in rank:
                    continue
                if rank[target] > rank[package]:
                    violations.append(
                        f"{path.relative_to(SRC)} imports {module} "
                        f"({package} -> {target} points upward)"
                    )
        assert not violations, "\n".join(violations)

    def test_only_ui_imports_streamlit(self) -> None:
        """ADR-0014: the core must stay runnable outside Streamlit.

        This is what keeps a future REST or MCP transport a wrapper rather than a
        refactor (ADR-0017).
        """
        offenders = [
            str(path.relative_to(SRC))
            for path in SRC.rglob("*.py")
            if package_of(path) != "ui"
            and any(m.split(".")[0] == "streamlit" for m in imported_modules(path))
        ]
        assert not offenders, f"streamlit imported outside ui/: {offenders}"

    def test_only_agent_and_llm_import_the_framework(self) -> None:
        """ADR-0020's containment: adopting LangGraph must not leak downward.

        `data/`, `retrieval/` and `tools/` stay framework-free, which is why the runtime
        reversal cost one package.
        """
        allowed = {"agent", "llm"}
        offenders: list[str] = []
        for path in SRC.rglob("*.py"):
            package = package_of(path)
            if package in allowed:
                continue
            for module in imported_modules(path):
                root = module.split(".")[0]
                if root in {"langgraph", "langchain", "langchain_core", "langchain_openai"}:
                    offenders.append(f"{path.relative_to(SRC)} imports {module}")
        assert not offenders, "\n".join(offenders)

    def test_tools_do_not_import_each_other(self) -> None:
        """ADR-0003: tools are pure functions, not a call graph."""
        offenders: list[str] = []
        for path in (SRC / "tools").glob("*.py"):
            if path.name in {"__init__.py", "base.py"}:
                continue
            for module in imported_modules(path):
                if module.startswith("movieagent.tools.") and not module.endswith(".base"):
                    offenders.append(f"{path.name} imports {module}")
        assert not offenders, "\n".join(offenders)


class TestReasoningNeverLeaks:
    """R-104, defended by ADR-0023's sanitizer plus ADR-0013's type.

    ADR-0020 traded away the structural guarantee -- reasoning now *can* arrive from the
    provider -- so this is the check that replaces it. A failure here is an R-104 defect,
    not a cosmetic bug.
    """

    def test_known_reasoning_keys_are_stripped(self) -> None:
        message = AIMessage(
            content="Interstellar (2014).",
            additional_kwargs={
                "reasoning_content": "First I should consider...",
                "reasoning": "...then conclude...",
                "thinking": "hmm",
                "tool_calls": [],
            },
            response_metadata={"reasoning": "leaked", "model_name": "test", "finish_reason": "stop"},
        )
        sanitize_message(message)

        assert not REASONING_KEYS & set(message.additional_kwargs)
        assert not REASONING_KEYS & set(message.response_metadata)
        assert "tool_calls" in message.additional_kwargs
        assert message.response_metadata["model_name"] == "test"

    def test_unknown_provider_fields_are_dropped_by_default(self) -> None:
        """The allowlist is what makes this survive a provider we have never seen.

        A denylist would keep working until someone shipped a new key name, then fail
        silently -- the worst outcome for a MUST requirement.
        """
        message = AIMessage(
            content="ok",
            additional_kwargs={"some_future_reasoning_field": "secret deliberation"},
        )
        sanitize_message(message)
        assert message.additional_kwargs == {}

    def test_reasoning_content_blocks_are_removed(self) -> None:
        message = AIMessage(
            content=[
                {"type": "reasoning", "text": "internal deliberation"},
                {"type": "text", "text": "The Martian (2015)."},
            ]
        )
        sanitize_message(message)
        assert message.content == [{"type": "text", "text": "The Martian (2015)."}]

    def test_trace_type_has_nowhere_to_put_reasoning(self) -> None:
        """ADR-0013's original guarantee, still holding as the second layer."""
        fields = set(Trace.__dataclass_fields__)
        assert not REASONING_KEYS & fields
        assert not {"scratchpad", "deliberation", "internal"} & fields


class TestGroundingCheck:
    """ADR-0012 layer 3. Advisory, heuristic, and honest about both."""

    ALLOWED = [
        MovieRef(157336, "Interstellar", 2014),
        MovieRef(27205, "Inception", 2010),
    ]

    def test_flags_a_movie_that_was_not_retrieved(self) -> None:
        warnings = check_answer(
            'You might also enjoy "The Prestige", which is similar.', self.ALLOWED
        )
        assert any("The Prestige" in w for w in warnings)

    def test_allows_the_retrieved_movies(self) -> None:
        warnings = check_answer(
            "Interstellar (2014) and Inception are both Nolan films.",
            self.ALLOWED,
            extra_terms=["Christopher Nolan"],
        )
        assert warnings == []

    def test_people_from_the_payload_are_not_flagged(self) -> None:
        """False positives on cast and director names would train users to ignore the
        warning, which is worse than not showing one."""
        warnings = check_answer(
            "Interstellar stars Matthew McConaughey and Anne Hathaway.",
            self.ALLOWED,
            extra_terms=["Matthew McConaughey", "Anne Hathaway"],
        )
        assert warnings == []

    def test_title_variants_are_tolerated(self) -> None:
        warnings = check_answer(
            "The Dark Knight is a great film.", [MovieRef(155, "The Dark Knight", 2008)]
        )
        assert warnings == []

    def test_does_not_catch_invented_attributes(self) -> None:
        """The documented blind spot (ADR-0012).

        The runtime is wrong -- Interstellar is 169 minutes -- and nothing fires, because
        no title is unmatched. Pinned as a test so the limitation is visible in the suite
        rather than only in prose.
        """
        warnings = check_answer("Interstellar runs for 240 minutes.", self.ALLOWED)
        assert warnings == []

    def test_empty_answer_is_not_flagged(self) -> None:
        assert check_answer("", self.ALLOWED) == []


class TestOutcomePromptCoupling:
    """ADR-0003 named the enum-to-prompt coupling as its main maintenance hazard.

    The mitigation was to generate the prompt text from the enum. This asserts the
    generation actually covers every member, so adding a status cannot leave the model
    uninstructed about it.
    """

    def test_every_outcome_has_guidance_in_the_prompt(self) -> None:
        from movieagent.agent.prompts import EXECUTOR_SYSTEM
        from movieagent.tools.base import OUTCOME_GUIDANCE, Outcome

        for outcome in Outcome:
            assert outcome in OUTCOME_GUIDANCE, f"{outcome} has no guidance"
            assert outcome.value in EXECUTOR_SYSTEM, f"{outcome} never reaches the prompt"


class TestConfigContract:
    """ADR-0015: layered validation, and the env surface stays documented."""

    def test_deterministic_half_works_without_an_api_key(self, repo, matcher) -> None:
        """The reason validation is layered rather than done at startup."""
        assert len(repo) > 4_000
        assert matcher.match("Avatar").best is not None

    def test_missing_key_raises_a_useful_error(self) -> None:
        from movieagent.config import LLMSettings
        from movieagent.errors import ConfigurationError

        with pytest.raises(ConfigurationError, match="LLM_API_KEY"):
            LLMSettings(api_key=None).require_key()

    def test_openai_compatible_embeddings_require_their_own_base_url(self) -> None:
        """OpenRouter serves no /v1/embeddings, so this must not silently inherit
        `LLM_BASE_URL` (ADR-0007)."""
        from movieagent.config import EmbeddingSettings

        with pytest.raises(ValueError, match="EMBEDDING_BASE_URL"):
            EmbeddingSettings(provider="openai_compatible", base_url=None)

    def test_blank_env_values_mean_unset(self) -> None:
        """Regression: `LOG_FILE=` in a .env crashed the app on startup.

        dotenv reads a bare `NAME=` as `""`, not as missing. For a `Path | None` field
        pydantic coerces `""` to `Path(".")`, which resolves to the project directory --
        so `logging.FileHandler` was handed a directory and raised `PermissionError`.
        """
        from movieagent.config import Settings

        settings = Settings(
            _env_file=None,
            log_file="",
            llm_api_key="",
            embedding_base_url="",
            embedding_api_key="",
        )
        assert settings.log_file is None
        assert settings.llm_api_key is None
        assert settings.embedding_base_url is None
        assert settings.embedding_api_key is None

    def test_generated_env_example_never_emits_a_blank_optional(self) -> None:
        """The generator must not reintroduce the line that caused the crash.

        `LLM_API_KEY=` is the deliberate exception -- it is a prompt to fill in, and an
        empty string there is harmless because it is only ever truth-tested.
        """
        import re

        example = (Path(__file__).resolve().parents[1] / ".env.example").read_text("utf-8")
        blanks = re.findall(r"^([A-Z_]+)=\s*$", example, flags=re.MULTILINE)
        assert blanks == ["LLM_API_KEY"], f"blank optional settings emitted: {blanks}"

    def test_misconfigured_log_file_degrades_to_stderr(self, tmp_path) -> None:
        """Defence in depth: a bad LOG_FILE must not take the app down."""
        import logging as stdlib_logging

        import movieagent.logging as ml

        ml._CONFIGURED = False
        try:
            ml.configure_logging("INFO", tmp_path)  # a directory, not a file
        finally:
            ml._CONFIGURED = False
            stdlib_logging.getLogger("movieagent").handlers.clear()

    def test_env_example_documents_every_setting(self) -> None:
        """R-112. Generated from the model, so it cannot drift -- this proves it."""
        from movieagent.config import Settings

        example = (Path(__file__).resolve().parents[1] / ".env.example").read_text("utf-8")
        for name in Settings.model_fields:
            assert name.upper() in example, f"{name.upper()} missing from .env.example"
