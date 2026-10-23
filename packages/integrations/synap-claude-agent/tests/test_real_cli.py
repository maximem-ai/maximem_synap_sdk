"""The hooks and the session store, proved against the real Claude Code CLI.

Claude Code is a separate program. Whether it sends `last_assistant_message` to
the Stop hook, whether a resumed session really starts from what
`SessionStore.load()` returns, and whether a PreCompact hook's output reaches
its summarizer are facts about that program, not about this package, so these
tests run the CLI bundled in `claude-agent-sdk` against a mock Messages API
(`_mock_anthropic.py`) and look at what reaches Synap and what reaches the model.
No network and no key are needed.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

claude_agent_sdk = pytest.importorskip("claude_agent_sdk")
from claude_agent_sdk import ClaudeAgentOptions, query  # noqa: E402
from claude_agent_sdk.types import ResultMessage  # noqa: E402

from synap_claude_agent import create_synap_hooks, create_synap_session_store  # noqa: E402
from synap_integrations_common import set_auto_listen  # noqa: E402
from tests._mock_anthropic import MockAnthropic  # noqa: E402

_CLI = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
pytestmark = pytest.mark.skipif(not _CLI.exists(), reason="no bundled Claude Code CLI")

BLOCK = "## Conversation so far (compacted 2026-10-22)\nThe user is planning a trip to Lisbon."


def make_sdk(*, compaction: bool = False, recent: Optional[list] = None, listening: bool = False):
    s = MagicMock()
    s.instance = MagicMock()
    s.instance.is_listening = listening

    async def listen():
        s.instance.is_listening = True

    s.instance.listen = AsyncMock(side_effect=listen)
    for name in ("send_message", "record_tool_call", "record_tool_result", "end_session"):
        setattr(s.instance, name, AsyncMock())
    s.conversation = MagicMock()
    s.conversation.record_message = AsyncMock()
    s.fetch = AsyncMock(return_value=SimpleNamespace(formatted_context=""))
    set_compaction(s, compaction, recent)
    return s


def set_compaction(s, compaction: bool, recent: Optional[list] = None) -> None:
    turns = [SimpleNamespace(role=r, content=c) for r, c in (recent or [])]
    s.conversation.context.get_context_for_prompt = AsyncMock(return_value=SimpleNamespace(
        available=True, compaction_id="cmp_1" if compaction else None,
        context_block=BLOCK if compaction else None, history_block=None,
        recent_messages=turns,
        formatted_context=(BLOCK + "\n" if compaction else "") + "\n".join(
            f"{r}: {c}" for r, c in (recent or [])),
    ))


class Cli:
    """One mock Messages API and one config dir, shared by the queries of a test."""

    def __init__(self, reply: Any = "Booked.") -> None:
        self.mock = MockAnthropic(reply)
        self.cfg = tempfile.mkdtemp()
        self.env = {
            "ANTHROPIC_BASE_URL": self.mock.url, "ANTHROPIC_API_KEY": "sk-ant-fake",
            "CLAUDE_CONFIG_DIR": self.cfg, "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }

    async def turn(self, prompt: str, *, resume: Optional[str] = None, **options: Any) -> str:
        sid = ""
        opts = ClaudeAgentOptions(env=self.env, cwd=self.cfg, resume=resume, **options)
        async for message in query(prompt=prompt, options=opts):
            if isinstance(message, ResultMessage):
                assert not message.is_error, message.result
                sid = message.session_id
        return sid

    def model_requests(self) -> List[dict]:
        return [r["body"] for r in self.mock.requests
                if r["path"].startswith("/v1/messages") and r["body"].get("stream")]

    def close(self) -> None:
        self.mock.close()


@pytest.fixture
def cli():
    c = Cli()
    yield c
    c.close()


def flat(body: dict) -> str:
    return json.dumps(body["messages"], ensure_ascii=False)


def sent(s) -> List[tuple]:
    return [(c.kwargs["role"], c.kwargs["content"]) for c in s.instance.send_message.await_args_list]


@pytest.mark.asyncio
async def test_a_run_with_no_listen_opens_the_stream_and_sends_both_turns_on_it(cli):
    """The user's prompt and the Stop hook's last_assistant_message, from the real CLI."""
    s = make_sdk()
    sid = await cli.turn("book the first one", hooks=create_synap_hooks(s, user_id="u1"))
    s.instance.listen.assert_awaited_once()
    assert sent(s) == [("user", "book the first one"), ("assistant", "Booked.")]
    assert s.instance.send_message.await_args_list[0].kwargs["conversation_id"] == sid
    s.instance.end_session.assert_awaited_once_with(sid)
    s.conversation.record_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_with_auto_listen_off_both_turns_go_over_rest(cli):
    s = make_sdk()
    set_auto_listen(s, False)
    await cli.turn("book it", hooks=create_synap_hooks(s, user_id="u1"))
    calls = s.conversation.record_message.await_args_list
    assert [(c.kwargs["role"], c.kwargs["content"]) for c in calls] == [
        ("user", "book it"), ("assistant", "Booked.")]
    s.instance.listen.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_resumed_session_starts_from_synaps_compaction_not_the_stored_transcript(cli):
    s = make_sdk()
    store = create_synap_session_store(s)
    sid = await cli.turn("flights to Lisbon?", session_store=store)
    await cli.turn("and hotels?", resume=sid, session_store=store)
    set_compaction(s, True, [("user", "and hotels?"), ("assistant", "Three near the river.")])
    await cli.turn("book the first one", resume=sid, session_store=store)

    before, after = cli.model_requests()[1], cli.model_requests()[2]
    assert "flights to Lisbon?" in flat(before)  # no compaction yet: the transcript, unchanged
    assert "planning a trip to Lisbon" not in flat(before)
    text = flat(after)
    assert "flights to Lisbon?" not in text
    assert "planning a trip to Lisbon" in text
    assert text.index("planning a trip") < text.index("Three near the river.") < text.index("book the first one")


@pytest.mark.asyncio
async def test_the_session_store_falls_back_to_the_mirrored_transcript_when_synap_errors(cli):
    s = make_sdk()
    store = create_synap_session_store(s)
    sid = await cli.turn("flights to Lisbon?", session_store=store)
    s.conversation.context.get_context_for_prompt = AsyncMock(side_effect=RuntimeError("down"))
    await cli.turn("and hotels?", resume=sid, session_store=store)
    assert "flights to Lisbon?" in flat(cli.model_requests()[1])


@pytest.mark.asyncio
async def test_a_fresh_query_per_turn_gets_the_whole_conversation_from_synap(cli):
    """No transcript, so Synap is the only record: the recent turns come too."""
    s = make_sdk()
    set_compaction(s, True, [("user", "and hotels?"), ("assistant", "Three near the river.")])
    await cli.turn("book the first one", hooks=create_synap_hooks(s, user_id="u1", conversation_id="c1"))
    text = flat(cli.model_requests()[0])
    assert "planning a trip to Lisbon" in text and "Three near the river." in text


@pytest.mark.asyncio
async def test_a_resumed_transcript_gets_the_compaction_once_and_no_recent_turns(cli):
    s = make_sdk()
    hooks = create_synap_hooks(s, user_id="u1", conversation_id="c1")
    sid = await cli.turn("flights to Lisbon?", hooks=hooks)
    set_compaction(s, True, [("user", "flights to Lisbon?"), ("assistant", "Booked.")])
    await cli.turn("and hotels?", resume=sid, hooks=hooks)
    await cli.turn("any deals?", resume=sid, hooks=hooks)

    second, third = cli.model_requests()[1], cli.model_requests()[2]
    assert flat(second).count("planning a trip to Lisbon") == 1
    # The transcript already holds the block, and the hook leaves it there.
    assert flat(third).count("planning a trip to Lisbon") == 1
    # Synap's recent turns are the transcript's own turns: not sent a second time.
    assert "user: flights to Lisbon?" not in flat(second)


@pytest.mark.asyncio
async def test_claude_codes_summarizer_is_given_synaps_compaction(cli):
    s = make_sdk(compaction=True)
    hooks = create_synap_hooks(s, user_id="u1", conversation_id="c1")
    sid = await cli.turn("flights to Lisbon?", hooks=hooks)
    await cli.turn("/compact", resume=sid, hooks=hooks)
    summarizer = [b for b in cli.model_requests() if "Additional Instructions" in flat(b)]
    assert summarizer, "the PreCompact hook's output never reached the summarizer"
    assert "planning a trip to Lisbon" in flat(summarizer[0])
