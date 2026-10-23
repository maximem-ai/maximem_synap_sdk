"""Synap's compaction in place of an agent's earlier turns, for ``create_agent``.

LangChain v1 agents keep the whole conversation in ``state["messages"]`` and
shrink it with ``SummarizationMiddleware`` when it gets long. Left alone, that
summary is the framework's and Synap's compaction never reaches the model.

:class:`SynapHistoryMiddleware` sits on ``wrap_model_call``, the one hook that
rewrites what the model is sent without rewriting what the graph stores. Once
Synap has compacted the conversation, every turn before the current one is
replaced by Synap's compaction and the turns after it. Until then, and whenever
Synap cannot answer, the messages go through untouched. Keep
``SummarizationMiddleware`` in the list: it stays the fallback, and with a short
history it never fires.

The compaction goes into the request's system message, not the message list:
Anthropic's chat model rejects a system message that is not the first.

Recording is not this middleware's job. ``SynapCallbackHandler`` records the
turns (on the gRPC stream, opened on the first one) and this reads them back.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    fetch_compacted_history,
    is_synap_history,
    run_async,
    splice_history,
)

from synap_langchain._messages import message_text, role_of

logger = logging.getLogger(__name__)

_HISTORY_OPEN = "<synap_conversation_history>"


def _make_message(role: str, text: str) -> Any:
    if role == "system":
        return SystemMessage(content=text)
    return AIMessage(content=text) if role == "assistant" else HumanMessage(content=text)


class SynapHistoryMiddleware(AgentMiddleware):
    """Replace an agent's earlier turns with Synap's compaction.

    Example::

        from langchain.agents import create_agent

        agent = create_agent(
            model="anthropic:claude-sonnet-5",
            tools=[...],
            middleware=[SynapHistoryMiddleware(sdk, conversation_id="conv-123")],
        )

    Args:
        sdk: An initialised :class:`~maximem_synap.MaximemSynapSDK`.
        conversation_id: Synap conversation id. Required, and never inferred
            from a thread id: the two namespaces can diverge.
    """

    def __init__(self, sdk: MaximemSynapSDK, conversation_id: str) -> None:
        if sdk is None:
            raise ValueError("SynapHistoryMiddleware requires a non-None sdk")
        if not conversation_id or not str(conversation_id).strip():
            raise ValueError("SynapHistoryMiddleware requires a non-empty conversation_id")
        super().__init__()
        self.sdk = sdk
        self.conversation_id = conversation_id

    async def _splice(self, request: ModelRequest) -> ModelRequest:
        """The request with Synap's history in place of the earlier turns."""
        if request.system_message is not None and _HISTORY_OPEN in message_text(request.system_message):
            return request  # already spliced by an outer layer
        history = await fetch_compacted_history(self.sdk, self.conversation_id)
        if history is None:
            return request
        try:
            spliced = splice_history(
                request.messages, history,
                role_of=role_of, text_of=message_text, make_message=_make_message,
            )
        except Exception:  # noqa: BLE001 — the agent's own messages still stand
            logger.warning("synap: could not splice the compacted history", exc_info=True)
            return request
        # The preamble is the one system message splice adds; the system
        # prompt is the request's own, so it joins that rather than the list.
        preamble = next((m for m in spliced if isinstance(m, SystemMessage)
                         and is_synap_history(message_text(m))), None)
        if preamble is None:
            return request
        spliced = [m for m in spliced if m is not preamble]
        base = message_text(request.system_message) if request.system_message is not None else ""
        system = SystemMessage(content=f"{base}\n\n{message_text(preamble)}" if base else message_text(preamble))
        return request.override(messages=spliced, system_message=system)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(await self._splice(request))

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(run_async(self._splice(request)))


__all__ = ["SynapHistoryMiddleware"]
