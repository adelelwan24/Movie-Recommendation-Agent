"""Chat model construction and the reasoning sanitizer (ADR-0020, ADR-0023).

ADR-0013's strongest property was structural: the ``Trace`` type had no field for model
reasoning, so R-104 could not be violated by accident. Adopting LangGraph traded that
away -- provider reasoning arrives in ``AIMessage.additional_kwargs`` /
``response_metadata`` and flows into state, streams and checkpoints, where any future
debug view over ``state["messages"]`` would render it.

So the defence moved to the model boundary. Everything is stripped **before it enters
graph state**, which means no downstream consumer has to know the rule. It is a filter
rather than a shape, and that is genuinely weaker -- ADR-0023 says so rather than
pretending otherwise. Two things make it as strong as a filter can be:

* an **allowlist**, so a provider field we have never heard of is dropped by default
  rather than leaked until someone notices;
* a test (``tests/test_no_reasoning_leak.py``) asserting nothing survives.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI

from movieagent.config import LLMSettings
from movieagent.logging import get_logger

log = get_logger("llm.models")

#: Keys known to carry chain-of-thought. Used by the leak test; the *runtime* defence is
#: the allowlist below, which does not depend on this list being complete.
REASONING_KEYS: frozenset[str] = frozenset(
    {
        "reasoning",
        "reasoning_content",
        "reasoning_details",
        "thinking",
        "thought",
        "thoughts",
        "chain_of_thought",
    }
)

#: The only `additional_kwargs` we keep. Anything else -- including provider reasoning --
#: is discarded.
#:
#: Every entry is here because the runtime *breaks* without it, not for convenience:
#:
#: * ``tool_calls`` / ``function_call`` -- the agent loop cannot dispatch without them.
#: * ``refusal`` -- ``langchain_openai``'s structured-output parser branches on it.
#: * ``parsed`` -- ``langchain_openai`` stows the validated Pydantic object here
#:   (``base.py:1320``) and reads it back in ``_oai_structured_outputs_parser``
#:   (``base.py:3419``). Stripping it made every planning call fail with
#:   "Structured Output response does not have a 'parsed' field". It carries our own
#:   ``Plan`` model rebuilt from the response JSON -- structured output, never
#:   chain-of-thought -- so keeping it does not weaken R-104.
#:
#: This is the allowlist cost ADR-0023 predicted: it fails safe, but "safe" can mean
#: breaking a feature, so additions must be justified rather than convenient.
_ALLOWED_ADDITIONAL_KWARGS: frozenset[str] = frozenset(
    {"tool_calls", "function_call", "refusal", "parsed"}
)

#: The only `response_metadata` we keep: enough for cost and debugging, nothing generative.
_ALLOWED_RESPONSE_METADATA: frozenset[str] = frozenset(
    {"model_name", "model", "finish_reason", "token_usage", "system_fingerprint", "id"}
)


def sanitize_message(message: BaseMessage) -> BaseMessage:
    """Strip reasoning content from a message in place (R-104).

    Also handles the block-style content some providers return, where reasoning arrives
    as a typed block inside a list rather than as a metadata key.
    """
    if isinstance(message.additional_kwargs, dict):
        for key in list(message.additional_kwargs):
            if key not in _ALLOWED_ADDITIONAL_KWARGS:
                message.additional_kwargs.pop(key, None)

    if isinstance(getattr(message, "response_metadata", None), dict):
        for key in list(message.response_metadata):
            if key not in _ALLOWED_RESPONSE_METADATA:
                message.response_metadata.pop(key, None)

    if isinstance(message.content, list):
        message.content = [
            block
            for block in message.content
            if not (
                isinstance(block, dict)
                and str(block.get("type", "")).lower()
                in {"reasoning", "thinking", "redacted_thinking"}
            )
        ]

    return message


class SanitizedChatOpenAI(ChatOpenAI):
    """``ChatOpenAI`` that never lets reasoning content past the boundary.

    Subclassed rather than composed with a ``RunnableLambda`` so that ``bind_tools`` and
    ``with_structured_output`` keep working -- both return bindings that wrap *this*
    object, so sanitization stays in force through the whole graph.
    """

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        for generation in result.generations:
            sanitize_message(generation.message)
        return result


def build_chat_model(settings: LLMSettings, **overrides: Any) -> SanitizedChatOpenAI:
    """Construct the chat model.

    OpenRouter by default; a vLLM endpoint is a base-URL change with no code edit
    (R-117). Retries are the library's (``max_retries`` covers 429/5xx/timeouts), which
    is one of the things adopting the framework bought.
    """
    return SanitizedChatOpenAI(
        base_url=settings.base_url,
        api_key=settings.require_key(),
        model=settings.model,
        temperature=overrides.pop("temperature", settings.temperature),
        timeout=settings.timeout_s,
        max_retries=settings.max_retries,
        **overrides,
    )


def make_answer_fn(model: Any) -> Callable[[str, str], str]:
    """Adapt a chat model to the plain callable ``rag_answer`` expects.

    ``tools/`` must not import LangChain (ADR-0019), so the tool takes a
    ``(system, user) -> str`` function and the agent layer supplies the wrapper.
    """

    def answer(system: str, user: str) -> str:
        response = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        sanitize_message(response)
        content = response.content
        if isinstance(content, list):
            return "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        return str(content)

    return answer


def message_text(message: BaseMessage) -> str:
    """Flatten message content to text, tolerating block-style responses."""
    content = message.content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def token_usage(messages: Sequence[BaseMessage]) -> dict[str, int]:
    """Sum usage across the AI messages of a turn, for the trace."""
    totals: dict[str, int] = {}
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        usage = message.usage_metadata or {}
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            if key in usage:
                totals[key] = totals.get(key, 0) + int(usage[key])
    return totals
