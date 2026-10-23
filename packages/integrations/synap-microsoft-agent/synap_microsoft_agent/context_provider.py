"""SynapContextProvider — MAF ContextProvider backed by Synap.

This is the file that owns the turn lifecycle. It is the only place in the
package that sees a whole turn: the question on the way in, every tool the
agent reaches for while answering, and the answer on the way out.

Pattern mirrors ``agent_framework.mem0.Mem0ContextProvider``:

- ``before_run`` builds a query from ``context.input_messages`` text, calls
  ``sdk.fetch(...)``, and appends the formatted result as instructions via
  ``context.extend_instructions(source_id, ...)``. Read failures degrade
  gracefully — we log + skip rather than breaking the agent turn.

- ``after_run`` records the input + response messages to Synap. Write failures
  are logged but never re-raised, following MAF's "context providers should
  not crash the agent" contract (same as LangChain's
  ``SynapCallbackHandler``).

The live stream
---------------

**Stream first, REST only as a fallback, never both.** The server persists
``user_message`` and ``assistant_message`` from the stream itself
(``grpc/servicer.py::_persist_message``), so reporting a turn *and* calling
``sdk.conversation.record_message`` writes it twice and extracts it twice.
``report_message_turn`` answers whether the turn actually went out, which is
what makes the choice decidable rather than guessed.

The turn reports in two halves, in the order the turn happened:

- the **question** in ``before_run``, before the fetch, so the anticipation
  agent has it before the tool calls it caused;
- the agent's **reasoning**, then the **answer**, in ``after_run``.

Anything the stream did not take is written over REST in ``after_run``.
The first report opens the stream if the caller has not
(``SYNAP_AUTO_LISTEN=false`` stops that and everything goes over REST).

Tool calls and their results come from a function middleware added to the
invocation in ``before_run``; see ``stream.SynapToolReporter``.

Synap's history
---------------

``before_run`` also adds :class:`~synap_microsoft_agent.compaction.SynapCompactionMiddleware`
to the invocation, so once Synap has compacted the conversation the model sees
Synap's compaction instead of the earlier turns the agent's history provider
holds. When the agent's history provider is itself a Synap one
(:class:`~synap_microsoft_agent.history_provider.SynapHistoryProvider`, the
harness memory provider) the history already comes from Synap and the turns are
written by that provider: this provider then neither adds the middleware nor
writes the same turns a second time.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Optional

from agent_framework import ContextProvider
from maximem_synap import MaximemSynapSDK

from synap_microsoft_agent.compaction import SynapCompactionMiddleware
from synap_microsoft_agent.stream import (
    already_reported,
    install_tool_reporter,
    mark_reported,
    report_message_reasoning,
    report_message_turn,
)

logger = logging.getLogger(__name__)


class SynapContextProvider(ContextProvider):
    """Inject Synap context + record turns back to Synap."""

    DEFAULT_SOURCE_ID: ClassVar[str] = "synap"
    DEFAULT_CONTEXT_PROMPT: ClassVar[str] = (
        "## User Memory Context\n"
        "Consider the following context about the user and their history "
        "when answering. Treat it as background knowledge, not direct input."
    )

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        user_id: str,
        customer_id: str = "",
        conversation_id: Optional[str] = None,
        *,
        source_id: str = DEFAULT_SOURCE_ID,
        mode: str = "accurate",
        max_results: int = 20,
        context_prompt: Optional[str] = None,
        include_scope_labels: bool = False,
    ) -> None:
        if sdk is None:
            raise ValueError("SynapContextProvider requires a non-None sdk")
        if not user_id:
            raise ValueError("SynapContextProvider requires a non-empty user_id")

        super().__init__(source_id)
        self.sdk = sdk
        self.user_id = user_id
        self.customer_id = customer_id
        self.conversation_id = conversation_id
        self.mode = mode
        self.max_results = max_results
        self.context_prompt = context_prompt or self.DEFAULT_CONTEXT_PROMPT
        self.include_scope_labels = include_scope_labels

    async def before_run(
        self,
        *,
        agent: Any,
        session: Any,
        context: Any,
        state: dict[str, Any],
    ) -> None:
        """Fetch Synap context and append it to the agent's instructions.

        Also the front half of the turn report: the tool reporter goes on the
        invocation and the question goes on the stream, both before the fetch
        and both before any early return, because a turn Synap has nothing
        stored for is still a turn worth anticipating.
        """
        conv_id = self._resolve_conversation_id(context)
        install_tool_reporter(
            context,
            self.sdk,
            source_id=self.source_id,
            conversation_id=conv_id or "",
            user_id=self.user_id,
            customer_id=self.customer_id,
        )
        synap_history = _synap_history_providers(agent)
        extend = getattr(context, "extend_middleware", None)
        if callable(extend) and not any(p.SYNAP_LOADS_HISTORY for p in synap_history):
            try:
                extend(
                    self.source_id,
                    SynapCompactionMiddleware(
                        self.sdk, lambda: self._resolve_conversation_id(context)),
                )
            except Exception:  # noqa: BLE001 — a hook must never break the agent run
                logger.warning("synap: could not add the compaction middleware, so the "
                               "model keeps the history provider's own turns", exc_info=True)
        # On the stream before the fetch, whoever writes the turn afterwards: a
        # Synap history provider skips what is already marked reported.
        await self._report_input_turns(context, conv_id)

        input_text = self._concat_text(context.input_messages)
        if not input_text:
            return

        try:
            response = await self.sdk.fetch(
                conversation_id=conv_id,
                user_id=self.user_id,
                customer_id=self.customer_id or None,
                search_query=[input_text],
                max_results=self.max_results,
                mode=self.mode,
                include_conversation_context=False,
                include_scope_labels=self.include_scope_labels,
            )
        except Exception as exc:  # noqa: BLE001 — read-side degrades gracefully
            logger.error(
                "SynapContextProvider.before_run: sdk.fetch failed "
                "user_id=%s conversation_id=%s error=%s",
                self.user_id,
                conv_id,
                exc,
                exc_info=True,
            )
            return

        formatted = (response.formatted_context or "").strip()
        if not formatted:
            return

        context.extend_instructions(
            self.source_id,
            f"{self.context_prompt}\n{formatted}",
        )

    async def after_run(
        self,
        *,
        agent: Any,
        session: Any,
        context: Any,
        state: dict[str, Any],
    ) -> None:
        """Record input + response messages to Synap conversation history.

        In turn order: any input the stream did not already take, then the
        agent's reasoning, then the answer. Reasoning sits between the
        question and the answer because that is where it happened, and
        because it is the part that says what the agent is about to need.
        """
        conv_id = self._resolve_conversation_id(context)
        if not conv_id:
            # Without a conversation id we can't attribute the turn — skip.
            return

        # A Synap history provider on the agent writes the same turns in its own
        # `after_run`: writing them here as well stores each one twice.
        synap_history = _synap_history_providers(agent)
        write_inputs = not any(p.store_inputs for p in synap_history)
        write_outputs = not any(p.store_outputs for p in synap_history)

        for message in (context.input_messages or []) if write_inputs else []:
            # `before_run` puts the question on the stream. Recording it again
            # here is the exact double-write the stream-first rule exists to
            # prevent, so only what the stream refused reaches this path.
            if already_reported(context, self.source_id, message):
                continue
            role, text = self._message_to_role_text(message)
            if role and text:
                await self._record(conv_id, role, text)

        response = context.response
        if response is None or not getattr(response, "messages", None):
            return

        step = 0
        for message in response.messages:
            step += await report_message_reasoning(
                self.sdk,
                message,
                conversation_id=conv_id,
                user_id=self.user_id,
                customer_id=self.customer_id,
                first_step=step,
            )
            role, text = self._message_to_role_text(message)
            if role and text and write_outputs:
                await self._record(conv_id, role, text)

    async def _record(self, conv_id: str, role: str, text: str) -> None:
        """One turn to Synap: the stream if there is one, REST if there is not.

        ⚠ Never both. A ``record_message`` on top of a delivered stream event
        stores the turn twice and extracts it twice.
        """
        if await report_message_turn(
            self.sdk,
            role=role,
            content=text,
            conversation_id=conv_id,
            user_id=self.user_id,
            customer_id=self.customer_id,
        ):
            return
        try:
            await self.sdk.conversation.record_message(
                conversation_id=conv_id,
                role=role,
                content=text,
                user_id=self.user_id,
                customer_id=self.customer_id,
            )
        except Exception as exc:  # noqa: BLE001 — must not crash the agent
            logger.error(
                "SynapContextProvider.after_run: record_message failed "
                "role=%s conversation_id=%s error=%s",
                role,
                conv_id,
                exc,
                exc_info=True,
            )

    async def _report_input_turns(
        self, context: Any, conv_id: Optional[str]
    ) -> None:
        """Put the question on the stream, if there is a stream and an id.

        Stream only. The REST write stays in ``after_run`` where it has always
        been, so a caller with no stream open sees exactly the behaviour they
        had before: nothing is written until the turn completes.
        """
        if not conv_id:
            return
        for message in context.input_messages or []:
            role, text = self._message_to_role_text(message)
            if not role or not text:
                continue
            if await report_message_turn(
                self.sdk,
                role=role,
                content=text,
                conversation_id=conv_id,
                user_id=self.user_id,
                customer_id=self.customer_id,
            ):
                mark_reported(context, self.source_id, message)

    # -- helpers --------------------------------------------------------------

    def _resolve_conversation_id(self, context: Any) -> Optional[str]:
        if self.conversation_id:
            return self.conversation_id
        sid = getattr(context, "session_id", None)
        return sid if sid else None

    @staticmethod
    def _concat_text(messages: Any) -> str:
        if not messages:
            return ""
        parts: list[str] = []
        for msg in messages:
            text = getattr(msg, "text", None)
            if text and text.strip():
                parts.append(text.strip())
        return "\n".join(parts)

    @staticmethod
    def _message_to_role_text(message: Any) -> tuple[Optional[str], str]:
        role_raw = getattr(message, "role", None)
        if role_raw is None:
            return None, ""
        role = role_raw.value if hasattr(role_raw, "value") else str(role_raw)
        # `record_message` takes "user" or "assistant" only: a system message
        # raises ValueError in the SDK and would be dropped with an error.
        if role not in {"user", "assistant"}:
            return None, ""
        text = getattr(message, "text", "") or ""
        if not text.strip():
            return None, ""
        return role, text


def _synap_history_providers(agent: Any) -> list:
    """The providers on ``agent`` that keep the conversation in Synap."""
    try:
        providers = list(getattr(agent, "context_providers", None) or [])
    except TypeError:
        return []
    return [p for p in providers if getattr(p, "SYNAP_HISTORY", False) is True]
