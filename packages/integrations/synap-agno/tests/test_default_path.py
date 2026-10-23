"""The default path of the hooks, driven by a real Agno agent.

Two defaults are proved here, against Agno's own run loop and a local
OpenAI-compatible server (so the request body is exactly what a model gets):

- no ``listen()`` is ever called by the user, and the first run opens the
  stream and sends every turn on it;
- once Synap has compacted the conversation the model is sent Synap's
  compaction in place of the earlier turns Agno stored; until then, and when
  Synap cannot answer, Agno's history is sent untouched.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("openai")

from agno.agent import Agent  # noqa: E402
from agno.db.base import SessionType  # noqa: E402
from agno.models.openai import OpenAIChat  # noqa: E402

from synap_agno import (  # noqa: E402
    AsyncSynapDb,
    SynapDb,
    create_synap_async_hooks,
    create_synap_hooks,
    synap_st_instructions,
)
from synap_integrations_common import set_auto_listen  # noqa: E402
from tests._fake_openai import Fake  # noqa: E402

COMPACTION = "## Conversation so far (compacted, 40 turns)\nthe user likes tea"


def _sdk(compaction=COMPACTION, recent=(("user", "post q"), ("assistant", "post a"))):
    sdk = MagicMock()
    sdk.instance.is_listening = False

    async def listen():
        sdk.instance.is_listening = True

    sdk.instance.listen = AsyncMock(side_effect=listen)
    for name in ("send_message", "record_thinking", "record_tool_call", "record_tool_result"):
        setattr(sdk.instance, name, AsyncMock())
    sdk.conversation.record_message = AsyncMock()
    ctx = NS(
        context_block=compaction,
        history_block=None,
        compaction_id="c1" if compaction else None,
        recent_messages=[NS(role=r, content=c) for r, c in recent],
    )
    sdk.conversation.context.get_context_for_prompt = AsyncMock(return_value=ctx)
    return sdk


def _agent(sdk, fake, *, is_async=False, instructions="BE NICE"):
    hooks = (create_synap_async_hooks if is_async else create_synap_hooks)(sdk)
    db = (AsyncSynapDb if is_async else SynapDb)(sdk)
    return Agent(
        db=db,
        model=OpenAIChat(id="fake", api_key="x", base_url=fake.url),
        add_history_to_context=True,
        instructions=instructions,
        user_id="u",
        **hooks,
    )


def _sent_to_model(fake, i):
    return [(m["role"], m["content"]) for m in fake.requests[i]["messages"]]


def _texts(fake, i):
    return " | ".join(c for _, c in _sent_to_model(fake, i))


# --------------------------------------------------------------------------
# gRPC by default
# --------------------------------------------------------------------------


@pytest.mark.parametrize("is_async", [False, True])
def test_first_run_opens_the_stream_and_every_turn_goes_on_it(is_async):
    fake, sdk = Fake(), _sdk(compaction=None)
    agent = _agent(sdk, fake, is_async=is_async)
    assert sdk.instance.is_listening is False  # the user never called listen()

    if is_async:
        async def go():
            await agent.arun("one", session_id="s1")
            await agent.arun("two", session_id="s1")
        asyncio.run(go())
    else:
        agent.run("one", session_id="s1")
        agent.run("two", session_id="s1")

    assert sdk.instance.listen.await_count == 1
    sent = [(c.kwargs["role"], c.kwargs["content"]) for c in sdk.instance.send_message.await_args_list]
    assert sent == [("user", "one"), ("assistant", "reply 1"), ("user", "two"), ("assistant", "reply 2")]
    sdk.conversation.record_message.assert_not_awaited()


def test_auto_listen_off_writes_turns_over_rest_and_never_listens():
    fake, sdk = Fake(), _sdk(compaction=None)
    set_auto_listen(sdk, False)
    _agent(sdk, fake).run("one", session_id="s1")
    sdk.instance.listen.assert_not_awaited()
    sdk.instance.send_message.assert_not_awaited()
    roles = [c.kwargs["role"] for c in sdk.conversation.record_message.await_args_list]
    assert roles == ["user", "assistant"]


# --------------------------------------------------------------------------
# Synap's compaction replaces Agno's history
# --------------------------------------------------------------------------


@pytest.mark.parametrize("is_async", [False, True])
def test_model_sees_synaps_compaction_instead_of_the_earlier_turns(is_async):
    fake, sdk = Fake(), _sdk()
    agent = _agent(sdk, fake, is_async=is_async)

    if is_async:
        async def go():
            for q in ("one", "two", "three"):
                await agent.arun(q, session_id="s1")
        asyncio.run(go())
    else:
        for q in ("one", "two", "three"):
            agent.run(q, session_id="s1")

    sent = _sent_to_model(fake, 2)
    assert sent[0] == ("developer", "BE NICE")  # Agno's own system prompt stays first
    assert "<synap_conversation_history>" in sent[1][1] and "the user likes tea" in sent[1][1]
    assert sent[2:] == [("user", "post q"), ("assistant", "post a"), ("user", "three")]
    # The turns Agno stored are not sent: Synap's compaction covers them.
    assert "one" not in _texts(fake, 2).replace("compacted", "")
    assert "reply 1" not in _texts(fake, 2) and "reply 2" not in _texts(fake, 2)


def test_without_a_compaction_agnos_history_is_sent_untouched():
    fake, sdk = Fake(), _sdk(compaction=None)
    agent = _agent(sdk, fake)
    for q in ("one", "two", "three"):
        agent.run(q, session_id="s1")
    assert _sent_to_model(fake, 2) == [
        ("developer", "BE NICE"), ("user", "one"), ("assistant", "reply 1"),
        ("user", "two"), ("assistant", "reply 2"), ("user", "three"),
    ]


def test_a_failing_synap_leaves_agnos_history_untouched(caplog):
    fake, sdk = Fake(), _sdk()
    sdk.conversation.context.get_context_for_prompt = AsyncMock(side_effect=RuntimeError("down"))
    agent = _agent(sdk, fake)
    for q in ("one", "two"):
        agent.run(q, session_id="s1")
    assert _sent_to_model(fake, 1) == [
        ("developer", "BE NICE"), ("user", "one"), ("assistant", "reply 1"), ("user", "two"),
    ]


def test_the_stored_session_keeps_every_turn_and_loses_the_wrapper():
    fake, sdk = Fake(), _sdk()
    agent = _agent(sdk, fake)
    for q in ("one", "two", "three"):
        agent.run(q, session_id="s1")
    stored = agent.db.get_session("s1", SessionType.AGENT)
    assert "get_messages" not in vars(stored)
    contents = [m.content for r in stored.runs for m in (r.messages or []) if m.role in ("user", "assistant")]
    assert contents == ["one", "reply 1", "two", "reply 2", "three", "reply 3"]


def test_the_current_user_turn_already_on_synap_is_not_sent_twice():
    """The user turn is written before the model runs, so Synap often has it."""
    fake = Fake()
    sdk = _sdk(recent=(("user", "post q"), ("assistant", "post a"), ("user", "one")))
    _agent(sdk, fake).run("one", session_id="s1")
    sent = _sent_to_model(fake, 0)
    assert [c for r, c in sent].count("one") == 1


def test_the_st_helper_adds_no_second_copy_next_to_the_hooks():
    fake, sdk = Fake(), _sdk()
    agent = _agent(sdk, fake, is_async=True, instructions=synap_st_instructions(sdk, "s1", instructions="BE NICE"))

    async def go():
        for q in ("one", "two"):
            await agent.arun(q, session_id="s1")
    asyncio.run(go())
    assert _sent_to_model(fake, 1)[0] == ("developer", "BE NICE")
    assert _texts(fake, 1).count("the user likes tea") == 1


# --------------------------------------------------------------------------
# deletes
# --------------------------------------------------------------------------


def test_delete_user_memory_deletes_the_synap_memory_by_id():
    from uuid import uuid4
    sdk = _sdk()
    sdk.memories.delete = AsyncMock(return_value={})
    mid = uuid4()
    SynapDb(sdk).delete_user_memory(str(mid))
    SynapDb(sdk).delete_user_memories([str(mid)])
    assert [c.args[0] for c in sdk.memories.delete.await_args_list] == [mid, mid]


@pytest.mark.asyncio
async def test_async_delete_user_memory_deletes_by_id_and_skips_foreign_ids(caplog):
    from uuid import uuid4
    sdk = _sdk()
    sdk.memories.delete = AsyncMock(return_value={})
    mid = uuid4()
    db = AsyncSynapDb(sdk)
    with caplog.at_level(logging.WARNING):
        await db.delete_user_memories([str(mid), "agno-made-id"])
    sdk.memories.delete.assert_awaited_once_with(mid)
    assert "no public delete" in caplog.text


def test_a_failed_delete_is_logged_not_raised(caplog):
    from uuid import uuid4
    sdk = _sdk()
    sdk.memories.delete = AsyncMock(side_effect=RuntimeError("404"))
    with caplog.at_level(logging.ERROR):
        SynapDb(sdk).delete_user_memory(str(uuid4()))
    assert "memories.delete failed" in caplog.text
