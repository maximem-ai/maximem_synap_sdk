"""Wiring for the MAF harness memory subsystem: the provider and the factory.

Two things live here, and both exist to stop the same mistake.

``create_harness_agent`` takes exactly **one** ``history_provider``. MAF's
``MemoryContextProvider`` is a ``HistoryProvider``. So is our
:class:`~synap_microsoft_agent.history_provider.SynapHistoryProvider`. Wire both
and you silently get whichever was passed last, with no error and no warning —
the agent simply loses half its memory. It is the sharpest footgun in this
integration and it is invisible at runtime.

:func:`create_synap_harness_memory` is the answer: one call that returns a
correctly-built provider, so the common path is hard to get wrong.

The two configurations, stated plainly:

``history_provider=SynapHistoryProvider(...)``
    Synap owns the conversation transcript. The harness's topic-memory
    subsystem — ``MEMORY.md``, topic files, extraction, consolidation — is off.
    Right when you want durable conversation history and nothing more.

``history_provider=create_synap_harness_memory(sdk, ...)``
    The harness owns extraction, merging, topic selection, and consolidation.
    Synap is the durable memory beneath it, and transcripts go to Synap too.
    Right when you want the harness's memory reasoning with real recall under
    it. This is the one most people want.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, List, Optional, Sequence

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import wrap_sdk_errors_async

from synap_microsoft_agent._harness_compat import require_harness

require_harness()

from agent_framework import (  # noqa: E402 — must follow the version guard
    MemoryContextProvider,
    Message,
)

from synap_microsoft_agent.memory_store import (  # noqa: E402
    DEFAULT_OWNER_STATE_KEY,
    SynapMemoryStore,
    TopicRecordStore,
)

logger = logging.getLogger(__name__)

# Fixed namespace for deriving conversation UUIDs from MAF session ids.
# Synap validates ``conversation_id`` as a UUID client-side, and a MAF session
# id is an arbitrary string, so the two have to be bridged. UUID5 makes the
# mapping deterministic: the same session id always reaches the same
# conversation, across processes and across restarts.
_SESSION_NAMESPACE = uuid.UUID("9c8b8f5e-2d4a-5f11-9a3c-7e6b1d0f4a22")


def session_conversation_id(session_id: Optional[str], *, owner: str = "") -> str:
    """Return the Synap conversation UUID for a MAF session id.

    Deterministic, so a transcript written in one process is readable in the
    next. ``owner`` is folded in so the same session id under two owners
    cannot collide.
    """
    return str(uuid.uuid5(_SESSION_NAMESPACE, f"{owner}::{session_id or 'default'}"))


class SynapMemoryContextProvider(MemoryContextProvider):
    """``MemoryContextProvider`` with transcripts in Synap instead of on disk.

    The stock provider reaches the filesystem in exactly three places, all via
    ``store.get_transcripts_directory``: the private ``_create_history_provider``
    and the two public methods that are its only callers, ``get_messages`` and
    ``save_messages``. Overriding those two removes every path that touches
    disk, and the private one becomes unreachable.

    Everything else — extraction, topic selection, merging, consolidation,
    prompt assembly — is inherited untouched. That is deliberate: MAF's
    reasoning about memory is good, and replacing it would be the
    augment-don't-replace mistake this codebase has made before.

    Args:
        sdk: An initialised :class:`~maximem_synap.MaximemSynapSDK`.
        store: The :class:`~synap_microsoft_agent.memory_store.SynapMemoryStore`
            (or any ``MemoryStore``) holding topics.
        user_id: External user ID for transcript writes.
        customer_id: External customer ID for transcript writes.
        owner_state_key: Session-state key carrying the owner, used to keep
            transcripts from colliding across owners.
        **kwargs: Passed straight to ``MemoryContextProvider`` — ``recent_turns``,
            ``selection_limit``, ``consolidation_interval``, and the rest.
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        *,
        store: Any,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        owner_state_key: str = DEFAULT_OWNER_STATE_KEY,
        **kwargs: Any,
    ) -> None:
        if sdk is None:
            raise ValueError("SynapMemoryContextProvider requires a non-None sdk")
        super().__init__(store=store, **kwargs)
        self.sdk = sdk
        self.user_id = user_id
        self.customer_id = customer_id
        self.owner_state_key = owner_state_key

    async def before_run(
        self, *, agent: Any, session: Any, context: Any, state: Any
    ) -> None:
        """Scope this turn's recall to the question, then run MAF's assembly.

        ``MemoryStore.get_index_text`` receives a line limit and a line length
        and nothing else — MAF never hands it the user's question, because the
        file store it was designed against has no use for one. That leaves the
        recall block as an unqueried scope fetch, which returns whatever the
        scope finds generally salient. On the conformance bench, "when do we
        send invoices" came back with the user's code-review preferences.

        So the question is stashed for the duration of the turn and cleared
        afterwards. Everything else is inherited: this override reads
        ``context.input_messages`` and delegates. Same move as
        ``SynapMemoryMiddleware`` in the deepagents integration, for the same
        reason.
        """
        token = None
        store = self.store
        if hasattr(store, "set_recall_query"):
            token = store.set_recall_query(_query_from(context))
        try:
            await super().before_run(
                agent=agent, session=session, context=context, state=state
            )
        finally:
            if token is not None:
                store.reset_recall_query(token)

    def _owner_from_state(self, state: Optional[dict]) -> str:
        if isinstance(state, dict):
            owner = state.get(self.owner_state_key)
            if owner:
                return str(owner)
        return ""

    async def get_messages(
        self,
        session_id: Optional[str],
        *,
        state: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Message]:
        """Load this session's transcript from Synap.

        Degrades to ``[]`` on any failure. A transcript read sits inside
        ``before_run``, so raising here would end the agent turn over
        something the agent can continue without.
        """
        del kwargs
        conversation_id = session_conversation_id(
            session_id, owner=self._owner_from_state(state)
        )
        try:
            response = await self.sdk.conversation.context.get_context_for_prompt(
                conversation_id=conversation_id,
            )
        except Exception as exc:  # noqa: BLE001 — read-side degrades
            # A conversation with no turns yet returns "not found", and that is
            # the normal state of every session's first turn. Logging it at
            # ERROR with a traceback made a clean first run look broken, so it
            # is reported at DEBUG and only genuine failures reach ERROR.
            if _is_absent(exc):
                logger.debug(
                    "SynapMemoryContextProvider.get_messages: no transcript yet "
                    "for conversation_id=%s",
                    conversation_id,
                )
            else:
                logger.error(
                    "SynapMemoryContextProvider.get_messages failed "
                    "conversation_id=%s error=%s",
                    conversation_id,
                    exc,
                    exc_info=True,
                )
            return []

        messages: List[Message] = []
        for recorded in getattr(response, "recent_messages", None) or []:
            role = getattr(recorded, "role", None)
            content = getattr(recorded, "content", None)
            if not role or not content:
                continue
            messages.append(Message(role=role, contents=[content]))
        return messages

    async def save_messages(
        self,
        session_id: Optional[str],
        messages: Sequence[Message],
        *,
        state: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """Persist this session's transcript to Synap.

        Honours the inherited ``history_message_filter`` so a caller that
        redacts messages on the file store gets the same redaction here.
        """
        del kwargs
        if not messages:
            return
        if not self.user_id and not self.customer_id:
            # Nothing to attribute the turn to. Recording it would put the
            # transcript in a scope nobody can read back.
            logger.warning(
                "SynapMemoryContextProvider.save_messages: no user_id or "
                "customer_id, skipping transcript write for session_id=%s",
                session_id,
            )
            return

        conversation_id = session_conversation_id(
            session_id, owner=self._owner_from_state(state)
        )

        async with wrap_sdk_errors_async(
            "microsoft_agent.harness_memory.save_messages",
            logger,
            conversation_id=conversation_id,
            user_id=self.user_id,
        ):
            for message in messages:
                if self.history_message_filter is not None:
                    filtered = self.history_message_filter(message)
                    if filtered is None:
                        continue
                    message = filtered
                role, text = _role_and_text(message)
                if not role or not text:
                    continue
                await self.sdk.conversation.record_message(
                    conversation_id=conversation_id,
                    role=role,
                    content=text,
                    user_id=self.user_id,
                    customer_id=self.customer_id or "",
                )


def create_synap_harness_memory(
    sdk: MaximemSynapSDK,
    *,
    user_id: Optional[str] = None,
    customer_id: Optional[str] = None,
    record_store: Optional[TopicRecordStore] = None,
    include_recall: bool = True,
    store: Optional[Any] = None,
    **kwargs: Any,
) -> SynapMemoryContextProvider:
    """Build the harness memory provider, wired correctly.

    Returns a ``HistoryProvider`` — pass it as ``create_harness_agent(...,
    history_provider=...)`` and do **not** also pass ``SynapHistoryProvider``
    there. There is only one slot; the second one silently wins.

        from agent_framework import create_harness_agent
        from synap_microsoft_agent import create_synap_harness_memory

        agent = create_harness_agent(
            client,
            history_provider=create_synap_harness_memory(
                sdk, user_id="alice", customer_id="acme",
            ),
        )

    Args:
        sdk: An initialised :class:`~maximem_synap.MaximemSynapSDK`.
        user_id: External user ID.
        customer_id: External customer ID.
        record_store: Exact storage for topic records. Defaults to
            in-process — read ``memory_store``'s docstring on what that means
            after a restart.
        include_recall: Append the Synap recall block to ``MEMORY.md``.
        store: A pre-built ``MemoryStore``, if you need to configure one
            beyond these arguments. Mutually exclusive with ``user_id`` /
            ``customer_id`` / ``record_store`` / ``include_recall``.
        **kwargs: Passed to ``MemoryContextProvider`` — ``recent_turns``,
            ``selection_limit``, ``consolidation_interval``, and the rest.

    Raises:
        ValueError: If ``store`` is combined with the store-building arguments.
    """
    if store is not None:
        conflicting = [
            name
            for name, value in (
                ("user_id", user_id),
                ("customer_id", customer_id),
                ("record_store", record_store),
            )
            if value is not None
        ]
        if conflicting:
            raise ValueError(
                "create_synap_harness_memory got both `store` and "
                f"{conflicting} — pass one or the other. When you build the "
                "store yourself, configure it there."
            )
        memory_store = store
    else:
        memory_store = SynapMemoryStore(
            sdk,
            user_id=user_id,
            customer_id=customer_id,
            record_store=record_store,
            include_recall=include_recall,
        )

    return SynapMemoryContextProvider(
        sdk,
        store=memory_store,
        user_id=user_id,
        customer_id=customer_id,
        **kwargs,
    )


def _query_from(context: Any) -> str:
    """Return the turn's question, as a retrieval query.

    Only the pending input messages, not the whole transcript: a query built
    from the full history retrieves the history's topics rather than the one
    being asked about.
    """
    parts: List[str] = []
    for message in getattr(context, "input_messages", None) or []:
        text = getattr(message, "text", "") or ""
        if text.strip():
            parts.append(text.strip())
    return "\n".join(parts)


def _is_absent(exc: BaseException) -> bool:
    """Is this exception "there is nothing here yet" rather than a failure?

    Matched on the message rather than a type, because the SDK reports a
    missing conversation as a generic transient transport error carrying
    ``HTTP 403: Conversation not found``. Narrow enough that a real permission
    failure — which says ``access denied`` without ``not found`` — still
    reaches ERROR.
    """
    text = str(exc).lower()
    return "not found" in text or "404" in text


def _role_and_text(message: Any) -> tuple[Optional[str], str]:
    """Return ``(role, text)`` for a message Synap will accept.

    ``record_message`` takes ``"user"`` or ``"assistant"`` only — a ``system``
    or ``tool`` message raises ``ValueError`` inside the SDK. Those are dropped
    rather than coerced: relabelling a tool result as user speech would poison
    the extracted memory with text the user never said.
    """
    role_raw = getattr(message, "role", None)
    if role_raw is None:
        return None, ""
    role = role_raw.value if hasattr(role_raw, "value") else str(role_raw)
    if role not in {"user", "assistant"}:
        return None, ""
    text = getattr(message, "text", "") or ""
    if not text.strip():
        return None, ""
    return role, text


__all__ = [
    "SynapMemoryContextProvider",
    "create_synap_harness_memory",
    "session_conversation_id",
]
