"""A model client that records what it was asked, and an SDK whose stream opens."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from autogen_core import FunctionCall
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    ModelInfo,
    RequestUsage,
)


class RecordingClient(ChatCompletionClient):
    """Replays scripted results; ``seen`` holds the messages of every call."""

    def __init__(self, results):
        self._results = list(results)
        self._i = 0
        self.seen = []

    async def create(self, messages, **kwargs):
        self.seen.append(list(messages))
        result = self._results[self._i]
        self._i += 1
        return result

    def create_stream(self, *a, **k):
        raise NotImplementedError

    async def close(self):
        pass

    def actual_usage(self):
        return RequestUsage(0, 0)

    def total_usage(self):
        return RequestUsage(0, 0)

    def count_tokens(self, messages, **k):
        return 0

    def remaining_tokens(self, messages, **k):
        return 1000

    @property
    def capabilities(self):
        return self.model_info

    @property
    def model_info(self):
        return ModelInfo(
            vision=False, function_calling=True, json_output=False,
            family="unknown", structured_output=False,
        )


def text(content):
    return CreateResult(finish_reason="stop", content=content,
                        usage=RequestUsage(0, 0), cached=False)


def tool_call(name="lookup", args=None, call_id="call_1"):
    return CreateResult(
        finish_reason="function_calls",
        content=[FunctionCall(id=call_id, name=name, arguments=json.dumps(args or {}))],
        usage=RequestUsage(0, 0), cached=False,
    )


def lookup(pnr: str) -> str:
    return "shipped"


def sdk_with_closed_stream(compaction=None, recent=()):
    """An SDK whose stream is closed until ``listen()`` is awaited, like a real one.

    ``compaction`` is the context block Synap returns (None: not compacted).
    """
    sdk = MagicMock()
    sdk.instance.is_listening = False

    async def listen():
        sdk.instance.is_listening = True

    sdk.instance.listen = AsyncMock(side_effect=listen)
    for name in ("send_message", "record_thinking", "record_tool_call", "record_tool_result"):
        setattr(sdk.instance, name, AsyncMock())
    sdk.conversation.record_message = AsyncMock()
    resp = MagicMock()
    resp.context_block = compaction
    resp.history_block = None
    resp.recent_messages = [MagicMock(role=r, content=c) for r, c in recent]
    resp.compaction_id = "cmp-1" if compaction else None
    sdk.conversation.context.get_context_for_prompt = AsyncMock(return_value=resp)
    return sdk
