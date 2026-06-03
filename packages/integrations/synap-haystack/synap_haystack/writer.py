"""Synap memory writer component for Haystack pipelines.

Records conversation messages to Synap for server-side extraction. A thin
wrapper over :class:`SynapMemoryStore`, which owns the SDK interaction and the
failure policy (see ``store.py``).

Semantics:

- Accepts ``Document``s where ``content`` is the message text and
  ``meta["role"]`` is ``"user"`` or ``"assistant"``.
- Tracks ``written_count`` / ``failed_count`` / ``skipped_count`` separately and
  exposes all three plus ``first_error`` so downstream components can branch on
  partial failures.
- Documents with an unrecognized role are skipped (``skipped_count``); they never
  reach the store.
- If **every** recordable document fails, :class:`SynapIntegrationError` is
  raised by the store and propagates — a 100% failure rate is a broken pipeline,
  not a partial result, and should stop loudly.
"""

import logging
from typing import Dict, List, Optional

from haystack import Document, component
from haystack.dataclasses import ChatMessage

from maximem_synap import MaximemSynapSDK
from synap_haystack.store import SynapMemoryStore

logger = logging.getLogger(__name__)

_VALID_ROLES = frozenset(("user", "assistant"))


@component
class SynapMemoryWriter:
    """Haystack component that writes conversation turns to Synap.

    Example::

        writer = SynapMemoryWriter(sdk=sdk, conversation_id="c1", user_id="u1")
        pipeline.add_component("memory_writer", writer)

        # or share an existing store:
        writer = SynapMemoryWriter(store=store, conversation_id="c1")
    """

    def __init__(
        self,
        sdk: Optional[MaximemSynapSDK] = None,
        conversation_id: str = "",
        user_id: str = "",
        customer_id: str = "",
        *,
        store: Optional[SynapMemoryStore] = None,
    ):
        if not conversation_id and (store is None or store.conversation_id is None):
            raise ValueError(
                "SynapMemoryWriter requires a non-empty conversation_id "
                "(pass one here or set it on the store)"
            )

        if store is not None:
            self.store = store
        else:
            if sdk is None:
                raise ValueError("SynapMemoryWriter requires either a store or an sdk")
            if not user_id and not customer_id:
                raise ValueError(
                    "SynapMemoryWriter requires a non-empty user_id or customer_id"
                )
            self.store = SynapMemoryStore(
                sdk=sdk,
                user_id=user_id or None,
                customer_id=customer_id,
                conversation_id=conversation_id,
            )
        # Per-run conversation id: explicit arg wins, else the store's default.
        self.conversation_id = conversation_id or self.store.conversation_id

    @component.output_types(
        written_count=int,
        failed_count=int,
        skipped_count=int,
        first_error=Optional[str],
    )
    def run(self, documents: List[Document]) -> Dict[str, object]:
        messages: List[ChatMessage] = []
        skipped = 0
        for doc in documents:
            role = doc.meta.get("role", "user")
            if role not in _VALID_ROLES:
                skipped += 1
                logger.info(
                    "SynapMemoryWriter: skipping document with unsupported "
                    "role=%r (expected one of %s)",
                    role, sorted(_VALID_ROLES),
                )
                continue
            if role == "user":
                messages.append(ChatMessage.from_user(doc.content or ""))
            else:
                messages.append(ChatMessage.from_assistant(doc.content or ""))

        # add_memories raises SynapIntegrationError on 100% failure — propagate.
        results = (
            self.store.add_memories(
                messages=messages, conversation_id=self.conversation_id
            )
            if messages
            else []
        )

        written = sum(1 for r in results if r.get("status") == "written")
        failed = sum(1 for r in results if r.get("status") == "failed")
        first_error = next(
            (r.get("error") for r in results if r.get("status") == "failed"), None
        )

        return {
            "written_count": written,
            "failed_count": failed,
            "skipped_count": skipped,
            "first_error": first_error,
        }
