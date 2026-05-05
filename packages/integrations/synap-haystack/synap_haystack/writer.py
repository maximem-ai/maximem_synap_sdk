"""Synap memory writer component for Haystack pipelines.

Records conversation messages to Synap for server-side extraction.

Previously this component caught every ingestion exception and demoted
it to a ``warning`` log line, then reported whatever happened to succeed
via ``written_count``. Hidden ingestion failures are dangerous: a
Haystack pipeline that looks green is not the same as a pipeline whose
memory ingestion actually worked.

New semantics:

- Track ``written_count`` and ``failed_count`` separately and expose
  both as component outputs, so downstream components can branch on
  partial failures.
- Also emit ``first_error`` (string) when at least one document fails,
  so operators can see *what* went wrong without grepping logs.
- If **every** document fails, raise :class:`SynapIntegrationError`.
  A 100% failure rate is not "partial" — it's a broken pipeline and
  should stop.
- Per-document failures are logged at ``ERROR`` (not ``WARNING``) with
  structured context.
- Documents with an unrecognized role are skipped and surfaced via
  ``skipped_count``.
"""

import logging
from typing import Dict, List, Optional

from haystack import Document, component

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import SynapIntegrationError, run_async

logger = logging.getLogger(__name__)

_VALID_ROLES = frozenset(("user", "assistant"))


@component
class SynapMemoryWriter:
    """Haystack component that writes conversation turns to Synap.

    Accepts Documents where content is the message text and
    meta["role"] is "user" or "assistant".

    Example::

        writer = SynapMemoryWriter(sdk=sdk, conversation_id="c1", user_id="u1")
        pipeline.add_component("memory_writer", writer)
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        conversation_id: str,
        user_id: str,
        customer_id: str = "",
    ):
        if sdk is None:
            raise ValueError("SynapMemoryWriter requires a non-None sdk")
        if not user_id:
            raise ValueError("SynapMemoryWriter requires a non-empty user_id")
        if not conversation_id:
            raise ValueError(
                "SynapMemoryWriter requires a non-empty conversation_id"
            )

        self.sdk = sdk
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.customer_id = customer_id

    @component.output_types(
        written_count=int,
        failed_count=int,
        skipped_count=int,
        first_error=Optional[str],
    )
    def run(self, documents: List[Document]) -> Dict[str, object]:
        return run_async(self._arun(documents))

    async def _arun(self, documents: List[Document]) -> Dict[str, object]:
        written = 0
        failed = 0
        skipped = 0
        first_error: Optional[str] = None

        for doc in documents:
            role = doc.meta.get("role", "user")
            if role not in _VALID_ROLES:
                skipped += 1
                logger.info(
                    "SynapMemoryWriter: skipping document with unsupported "
                    "role=%r (expected one of %s)",
                    role,
                    sorted(_VALID_ROLES),
                )
                continue

            try:
                await self.sdk.conversation.record_message(
                    conversation_id=self.conversation_id,
                    role=role,
                    content=doc.content,
                    user_id=self.user_id,
                    customer_id=self.customer_id,
                )
                written += 1
            except Exception as exc:  # noqa: BLE001 — boundary
                failed += 1
                logger.error(
                    "SynapMemoryWriter: record_message failed "
                    "conversation_id=%s role=%s error=%s",
                    self.conversation_id,
                    role,
                    exc,
                    exc_info=True,
                )
                if first_error is None:
                    first_error = f"{type(exc).__name__}: {exc}"

        processed = written + failed
        if processed > 0 and written == 0:
            # Every single attempt failed — this is a broken pipeline,
            # not a partial result. Stop loudly.
            raise SynapIntegrationError(
                "haystack.SynapMemoryWriter.run",
                f"all {failed} record_message attempts failed; "
                f"first error: {first_error}",
                {"conversation_id": self.conversation_id, "failed": failed},
            )

        return {
            "written_count": written,
            "failed_count": failed,
            "skipped_count": skipped,
            "first_error": first_error,
        }
