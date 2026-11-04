"""Synap memory integration for Google ADK.

Deeper scope ladders: if your account has levels beyond customer and user, pass
``scope_path`` (level key to value) to ``create_synap_tools`` and
``create_synap_callbacks``. It is sent on every read, write and turn only when
set, and is closed over, never a tool parameter, so the model cannot choose a
level::

    conv_id = "conv-123"
    path = {"customer": "acme", "team": "payments", "user": "dana"}
    tools = create_synap_tools(sdk, user_id="dana", customer_id="acme", scope_path=path)
    callbacks = create_synap_callbacks(sdk, user_id="dana", customer_id="acme",
                                       conversation_id=conv_id, scope_path=path)

Needs a ``maximem-synap`` release whose ``fetch`` accepts ``scope_path``.
"""

from synap_google_adk.callbacks import create_synap_callbacks, report_user_turn
from synap_google_adk.short_term import synap_st_instruction
from synap_google_adk.tools import create_synap_tools

__all__ = [
    "create_synap_tools",
    "create_synap_callbacks",
    "report_user_turn",
    "synap_st_instruction",
]
