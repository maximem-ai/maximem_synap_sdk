"""Which model the agent runs on, sent with each conversation turn.

The server sizes compaction to the agent's context window: it compacts at
about 11.7% of the window (15K tokens on a 128K model) and caps the
compacted record at half that. It reads the window from the latest turn's
metadata, under ``agent_model`` and ``agent_context_window``. A model name is
enough: the server has a model -> window table. Give ``context_window`` only
for a model the table would not know, or to override it.

Precedence, highest first: the value passed on the call, the value set on
``SDKConfig`` (``agent_model`` / ``agent_context_window``), a value already in
the caller's own metadata. With none of them the server assumes 128K.
"""
from typing import Any, Dict, Optional

AGENT_MODEL_KEY = "agent_model"
AGENT_CONTEXT_WINDOW_KEY = "agent_context_window"


def with_agent_model(
    metadata: Optional[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    context_window: Optional[int] = None,
    config: Any = None,
    as_strings: bool = False,
) -> Optional[Dict[str, Any]]:
    """``metadata`` with the agent's model folded in.

    ``as_strings`` for the gRPC stream, whose metadata is a string map.
    Returns the input unchanged (None stays None) when there is nothing to add.
    """
    model = model or getattr(config, "agent_model", None)
    window = context_window or getattr(config, "agent_context_window", None)
    if not model and not window:
        return metadata
    out = dict(metadata or {})
    if model:
        out[AGENT_MODEL_KEY] = str(model)
    if window:
        out[AGENT_CONTEXT_WINDOW_KEY] = str(int(window)) if as_strings else int(window)
    return out
