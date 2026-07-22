"""Synap turn recorder for Smolagents.

`create_synap_recorder` returns a callback for
``step_callbacks={ActionStep: create_synap_recorder(...)}``. Smolagents fires it at
the end of every completed step; the recorder ingests the step's output into Synap
so future sessions can recall what the agent did.

Two behaviours are load-bearing:

- **Per-step ``document_id``.** Each step is its own memory document
  (``smolagents-{conversation_id}-{step_number}``); a single stable id would clobber
  each step down to the last.
- **Log, never raise.** The recorder runs inside the agent loop, in a ``finally``
  with no surrounding try/except, so a raising callback would abort the whole run. A
  failed ingest is logged at ERROR and swallowed — same contract as the Strands
  stream hook.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import run_async

logger = logging.getLogger(__name__)

_MARKER = "smolagents_step"


def _as_text(output: Any) -> str:
    """Coerce an ``ActionStep.model_output`` (``str | list[dict] | None``) to text."""
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts = []
        for item in output:
            if isinstance(item, dict):
                parts.append(item.get("text") or item.get("content") or json.dumps(item))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(output)


def create_synap_recorder(
    sdk: MaximemSynapSDK,
    user_id: str,
    conversation_id: str,
    *,
    customer_id: str = "",
) -> Callable[..., None]:
    """Build a Smolagents step callback that records each action into Synap.

    Args:
        sdk: Configured :class:`MaximemSynapSDK`.
        user_id: Synap user scope. **Required.**
        conversation_id: Conversation id; seeds each step's document id. **Required.**
        customer_id: Optional customer/org scope; empty means customer-less.

    Returns:
        A callable ``recorder(memory_step, agent=None)`` for ``step_callbacks``.
    """
    if sdk is None:
        raise ValueError("create_synap_recorder requires a non-None sdk")
    if not user_id or not str(user_id).strip():
        raise ValueError("create_synap_recorder requires a non-empty user_id")
    if not conversation_id or not str(conversation_id).strip():
        raise ValueError("create_synap_recorder requires a non-empty conversation_id")

    def recorder(memory_step: Any, agent: Optional[Any] = None) -> None:
        # Skip failed steps — the callback fires for them too.
        if getattr(memory_step, "error", None) is not None:
            return
        text = _as_text(getattr(memory_step, "model_output", None)).strip()
        if not text:
            return
        step_number = getattr(memory_step, "step_number", 0)
        try:
            run_async(
                sdk.memories.create(
                    document=text,
                    user_id=user_id,
                    customer_id=customer_id or None,
                    document_type="ai-chat-conversation",
                    document_id=f"smolagents-{conversation_id}-{step_number}",
                    metadata={_MARKER: True, "step": step_number},
                )
            )
        except Exception as exc:  # noqa: BLE001 — in-loop callback: log, never raise
            logger.error(
                "SynapStepRecorder: ingest failed step=%s error=%s",
                step_number,
                exc,
                exc_info=True,
            )

    return recorder
