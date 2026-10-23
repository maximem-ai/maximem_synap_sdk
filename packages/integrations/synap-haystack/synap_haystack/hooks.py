"""Synap's compaction in place of a Haystack agent's earlier turns, via hooks.

Haystack 3's ``Agent`` has no chat-message store. It takes ``hooks``: objects run
at points in its loop that mutate the live ``State`` in place. That is where its
own compaction lives (``haystack.hooks.compaction.CompactionHook``), and it is
where Synap's goes:

- :class:`SynapHistoryHook` on ``before_llm``: once Synap has compacted the
  conversation, every turn before the current one in ``state["messages"]`` is
  replaced by Synap's compaction (a system message) and the turns after it.
  Until then, and whenever Synap cannot answer, the messages are left alone.
  The agent's own compaction hooks can stay installed: with a short history they
  do not fire, and they are the fallback.
- :class:`SynapRecordHook` on ``after_run``: records the run's turns on Synap's
  gRPC stream (opened by the first one) and reports its tool calls, results and
  reasoning, through :meth:`SynapMemoryStore.write_messages`.

::

    store = SynapMemoryStore(sdk, user_id="alice")
    agent = Agent(
        chat_generator=...,
        hooks={
            "before_llm": [SynapHistoryHook(store, "conv-1")],
            "after_run": [SynapRecordHook(store, "conv-1")],
        },
    )

On haystack-ai 2.x use ``SynapMemoryStore`` as the experimental ``Agent``'s
``chat_message_store`` instead; this module needs the 3.x ``Agent.hooks``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, ChatRole

from synap_integrations_common import (
    fetch_compacted_history,
    run_async,
    splice_history,
)

from synap_haystack.store import RECORDED_KEY, SynapMemoryStore

logger = logging.getLogger(__name__)

_ROLES = {ChatRole.USER: "user", ChatRole.ASSISTANT: "assistant", ChatRole.SYSTEM: "system"}


def _role_of(message: ChatMessage) -> str:
    return _ROLES.get(message.role, "other")


def _text_of(message: ChatMessage) -> str:
    return message.text or ""


def _replace(_current: Any, new: Any) -> Any:
    return new


class _StoreHook:
    """Shared construction and serialization for the two hooks."""

    def __init__(self, store: SynapMemoryStore, chat_history_id: str) -> None:
        if store is None:
            raise ValueError(f"{type(self).__name__} requires a store")
        if not chat_history_id or not str(chat_history_id).strip():
            raise ValueError(f"{type(self).__name__} requires a non-empty chat_history_id")
        self.store = store
        self.chat_history_id = chat_history_id

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(self, store=self.store.to_dict(), chat_history_id=self.chat_history_id)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "_StoreHook":
        init = dict(data.get("init_parameters", {}))
        init["store"] = SynapMemoryStore.from_dict(init["store"])
        return default_from_dict(cls, {**data, "init_parameters": init})


class SynapHistoryHook(_StoreHook):
    """``before_llm`` hook: Synap's compaction in place of the earlier turns."""

    def run(self, state: Any) -> None:
        run_async(self.run_async(state))

    async def run_async(self, state: Any) -> None:
        messages = list(state.get("messages") or [])
        if not any(_role_of(m) == "user" for m in messages):
            return
        history = await fetch_compacted_history(self.store.sdk, self.chat_history_id)
        if history is None:
            return

        def make(role: str, text: str) -> ChatMessage:
            if role == "system":
                return ChatMessage.from_system(text)
            meta = {RECORDED_KEY: self.chat_history_id}  # Synap already has it
            return (ChatMessage.from_assistant(text, meta=meta) if role == "assistant"
                    else ChatMessage.from_user(text, meta=meta))

        try:
            spliced = splice_history(messages, history, role_of=_role_of, text_of=_text_of,
                                     make_message=make)
        except Exception:  # noqa: BLE001 — the agent's own messages still stand
            logger.warning("synap: could not splice the compacted history", exc_info=True)
            return
        state.set("messages", spliced, handler_override=_replace)


class SynapRecordHook(_StoreHook):
    """``after_run`` hook: record the run on Synap, each message once."""

    def run(self, state: Any) -> None:
        self.store.write_messages(self.chat_history_id, list(state.get("messages") or []))

    async def run_async(self, state: Any) -> None:
        await self.store.awrite_messages(self.chat_history_id, list(state.get("messages") or []))


__all__ = ["SynapHistoryHook", "SynapRecordHook"]
