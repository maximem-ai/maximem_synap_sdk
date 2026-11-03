"""Synap memory integration for AutoGen / AG2.

Deeper scope ladders
--------------------
If your account's scope ladder has more levels than customer and user, pass the
address of the level to read and write at. It is sent only when set, and the
two tools close it over, so the model never sees or chooses it::

    path = {"customer": "acme", "team": "payments", "user": "dana"}
    tools = [SynapSearchTool(sdk, "dana", "acme", scope_path=path),
             SynapStoreTool(sdk, "dana", "acme", scope_path=path)]
    ctx = SynapStreamingChatContext(sdk, "conv-1", "dana", "acme", scope_path=path)
"""

from synap_autogen.short_term import SynapShortTermChatContext
from synap_autogen.stream import SynapStreamingChatContext
from synap_autogen.tools import SynapSearchTool, SynapStoreTool

__all__ = [
    "SynapSearchTool",
    "SynapStoreTool",
    "SynapShortTermChatContext",
    "SynapStreamingChatContext",
]
