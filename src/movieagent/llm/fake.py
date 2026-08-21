"""A scripted chat model for tests (ADR-0024).

Substitutes at the same seam the real model uses, so the graph, ``ToolNode``, reducers,
conditional edges and the recursion limit are all exercised for real -- only the model is
fake. That is better coverage than ADR-0016's original design, which stubbed our own
client and therefore exercised no runtime at all.

The deciding capability is **fault injection**. R-087 requires graceful handling of
provider failures, and that is only testable if failures happen on demand -- no real
endpoint will rate-limit you on the third call, and no upstream fake raises at all.

Known weakness, stated in ADR-0024: this fixture now encodes LangChain's message
contract as well as our own assumptions, so a version bump could break the integration
while the suite stays green. Run ``pytest -m live`` after dependency upgrades.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Callable, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

#: A scripted turn: an AIMessage, or a callable raising to simulate a provider failure.
Script = Sequence[AIMessage | BaseException | Callable[[], AIMessage]]


def tool_call(name: str, args: dict[str, Any], call_id: str | None = None) -> AIMessage:
    """Build an AIMessage carrying one tool call, in the shape ``ToolNode`` expects."""
    identifier = call_id or f"call_{name}_{abs(hash(str(args))) % 10_000}"
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": identifier, "type": "tool_call"}],
    )


def text(content: str) -> AIMessage:
    return AIMessage(content=content)


class ScriptedChatModel(BaseChatModel):
    """Replays a fixed sequence of responses.

    Structured output is served from ``structured_queue`` so a planner call and the
    tool-calling turns can be scripted independently.
    """

    responses: list[Any] = []
    structured_queue: list[Any] = []
    calls: list[list[BaseMessage]] = []
    bound_tools: list[Any] = []
    default_response: str = "Done."

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _next(self) -> AIMessage:
        if not self.responses:
            return AIMessage(content=self.default_response)
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callable(item) and not isinstance(item, BaseMessage):
            return item()
        return item

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        message = self._next()
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedChatModel:
        self.bound_tools = list(tools)
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        """Return a runnable yielding the next queued structured object.

        Reads ``structured_queue`` off the model **at call time**, not at binding time.
        The agent binds this once in its constructor while tests assign the queue
        afterwards; capturing the list object here would freeze the empty list the
        fixture was built with, and every planned turn would look like a provider
        failure.
        """
        model = self

        class _Structured:
            def invoke(self, _input: Any, config: Any = None, **_: Any) -> Any:
                queue = model.structured_queue
                if not queue:
                    raise AssertionError(
                        "ScriptedChatModel.structured_queue is empty -- the test did not "
                        "script a plan for this call"
                    )
                item = queue.pop(0)
                if isinstance(item, BaseException):
                    raise item
                return item

            def __or__(self, other: Any) -> Any:  # pragma: no cover - composition guard
                raise NotImplementedError

        return _Structured()

    def stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:  # pragma: no cover
        yield self._generate(list(args[0]) if args else []).generations[0].message
