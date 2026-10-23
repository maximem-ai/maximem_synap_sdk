"""Put Synap's compaction where the Agents SDK builds the model's input.

The Agents SDK keeps a conversation as a list of input items (the caller's own
list, or a ``Session``) and sends all of it on every model call. Nothing in
``Agent`` or ``RunHooks`` can change what is sent: ``on_llm_start`` receives the
input and its return value is ignored. The one place that can is
``RunConfig.call_model_input_filter``, which runs after the SDK has built the
input and just before the model call.

:func:`synap_call_model_input_filter` is that filter. Once Synap has compacted
the conversation, every item before the current turn is replaced by Synap's
compaction and the turns after it; until then, and whenever Synap cannot
answer, the SDK's items are sent untouched. It works the same with a ``Session``
or a plain input list. It is not for a run that keeps its history on OpenAI's
side (``conversation_id=`` / ``previous_response_id=``): there the input holds
only the new items and the SDK refuses a filter that rewrites them.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Dict, Optional

from synap_integrations_common import (
    COMPACTION_PREFIX,
    fetch_compacted_history,
    splice_history,
)

logger = logging.getLogger(__name__)


def _field(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def item_role(item: Any) -> str:
    """``system`` / ``user`` / ``assistant`` for a message item, else ``other``.

    A function call, its output and a reasoning item carry no role (or a type
    other than ``message``), so they are never mistaken for a turn.
    """
    kind = _field(item, "type")
    if kind not in (None, "message"):
        return "other"
    role = _field(item, "role")
    return role if role in ("system", "user", "assistant") else "other"


def item_text(item: Any) -> str:
    """The text of a message item: a string, or the text parts of a list."""
    content = _field(item, "content")
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        text = _field(part, "text")
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def make_item(role: str, text: str) -> Dict[str, Any]:
    return {"role": role, "content": text}


def synap_call_model_input_filter(
    sdk: Any,
    conversation_id: str,
    *,
    then: Optional[Callable[[Any], Any]] = None,
) -> Callable[[Any], Any]:
    """A ``RunConfig(call_model_input_filter=...)`` that sends Synap's history.

    ``then`` is the caller's own filter, run on the result, so installing this
    does not displace one they already had.
    """

    async def _filter(data: Any) -> Any:
        from agents.run import ModelInputData

        model_data = data.model_data
        history = await fetch_compacted_history(sdk, conversation_id)
        items = list(model_data.input)
        if history is not None:
            items = splice_history(
                items, history, role_of=item_role, text_of=item_text, make_message=make_item,
            )
            # `synap_st_instructions` already put the compaction in the system
            # prompt: do not send it twice.
            if COMPACTION_PREFIX in (model_data.instructions or ""):
                items = [i for i in items if not (
                    item_role(i) == "system" and item_text(i) == history.preamble)]
        updated = ModelInputData(input=items, instructions=model_data.instructions)
        if then is None:
            return updated
        result = then(dataclasses.replace(data, model_data=updated))
        if hasattr(result, "__await__"):
            result = await result
        return result

    return _filter


__all__ = ["synap_call_model_input_filter"]
