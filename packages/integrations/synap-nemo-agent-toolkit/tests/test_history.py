"""Synap's compaction reaches a NAT agent through function middleware, and the
turns the toolkit records are not ingested a second time.

The middleware tests build a real NAT workflow (``WorkflowBuilder``) whose
function records the input it is handed, so they prove NAT calls the middleware
on its default path. The stubbed ``nat`` modules of ``test_short_term.py`` are
dropped first, as in ``test_stream.py``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

for _name in [n for n in list(sys.modules) if n == "nat" or n.startswith("nat.")]:
    if getattr(sys.modules[_name], "__file__", None) is None:
        del sys.modules[_name]

from nat.builder.context import ContextState  # noqa: E402
from nat.builder.function_info import FunctionInfo  # noqa: E402
from nat.builder.workflow_builder import WorkflowBuilder  # noqa: E402
from nat.cli.register_workflow import register_function  # noqa: E402
from nat.data_models.api_server import ChatRequest, Message  # noqa: E402
from nat.data_models.config import Config  # noqa: E402
from nat.data_models.function import FunctionBaseConfig  # noqa: E402
from nat.data_models.intermediate_step import (  # noqa: E402
    IntermediateStep, IntermediateStepPayload, IntermediateStepType, StreamEventData,
)
from nat.data_models.invocation_node import InvocationNode  # noqa: E402
from nat.memory.models import MemoryItem  # noqa: E402

import synap_nemo_agent_toolkit._sdk as sdk_module  # noqa: E402
import synap_nemo_agent_toolkit.register  # noqa: E402,F401
from synap_integrations_common import set_auto_listen  # noqa: E402
from synap_nemo_agent_toolkit import _ledger  # noqa: E402
from synap_nemo_agent_toolkit.editor import SynapMemoryEditor  # noqa: E402
from synap_nemo_agent_toolkit.history import SynapHistoryMiddlewareConfig  # noqa: E402
from synap_nemo_agent_toolkit.stream import SynapStreamExporter  # noqa: E402

BLOCK = "## Conversation so far (compacted 2026-10-22)\nThe user is planning a trip to Lisbon."
seen: list = []


class EchoConfig(FunctionBaseConfig, name="synap_history_probe"):
    ...


@register_function(config_type=EchoConfig)
async def echo(config, builder):
    async def _fn(req: ChatRequest) -> str:
        seen.append([(m.role.value, m.content) for m in req.messages])
        return "ok"

    yield FunctionInfo.from_fn(_fn, description="records its input")


class FakeInstance:
    """Behaves like sdk.instance: listen() flips is_listening on."""

    def __init__(self):
        self.is_listening = False
        self.listen_calls = 0
        self.send_message = AsyncMock()
        self.record_tool_call = AsyncMock()
        self.record_tool_result = AsyncMock()
        self.record_thinking = AsyncMock()
        self.stop_listening = AsyncMock()
        self._controller = SimpleNamespace(_transport=object())

    async def listen(self):
        self.listen_calls += 1
        self.is_listening = True


def make_sdk(compaction=True):
    ctx = SimpleNamespace(context_block=None, history_block=None, recent_messages=[], compaction_id=None)
    if compaction:
        ctx = SimpleNamespace(
            context_block=BLOCK, history_block=None, compaction_id="cmp_1",
            recent_messages=[SimpleNamespace(role="user", content="flights?"),
                             SimpleNamespace(role="assistant", content="TAP at 9am")],
        )
    sdk = MagicMock()
    sdk._initialized = True
    sdk.instance = FakeInstance()
    sdk.initialize = AsyncMock()
    sdk.shutdown = AsyncMock()
    sdk.conversation.record_message = AsyncMock()
    sdk.conversation.context.get_context_for_prompt = AsyncMock(return_value=ctx)
    sdk.memories.create = AsyncMock()
    return sdk


@pytest.fixture
def sdk_factory(monkeypatch):
    holder = {}

    def use(sdk):
        holder["sdk"] = sdk
        monkeypatch.setattr(sdk_module, "MaximemSynapSDK", lambda **kw: sdk)

    monkeypatch.setenv("SYNAP_API_KEY", "sk-test")
    monkeypatch.delenv("SYNAP_AUTO_LISTEN", raising=False)
    sdk_module._registry.clear()
    _ledger._written.clear()
    seen.clear()
    yield use
    sdk_module._registry.clear()
    _ledger._written.clear()


def conversation():
    return ChatRequest(messages=[
        Message(role="user", content="I want to visit Portugal"),
        Message(role="assistant", content="Great, when?"),
        Message(role="user", content="In May, for a week"),
        Message(role="assistant", content="Lisbon is lovely in May."),
        Message(role="user", content="and the hotel?"),
    ])


async def run_agent(request, **mw_kwargs):
    cfg = Config(
        middleware={"synap_history": SynapHistoryMiddlewareConfig(**mw_kwargs)},
        workflow=EchoConfig(middleware=["synap_history"]),
    )
    async with WorkflowBuilder.from_config(cfg) as builder:
        workflow = await builder.build()
        async with workflow.run(request) as runner:
            return await runner.result()


# ---------------------------------------------------------------------------
# The seam, driven by the real framework
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_agent_is_handed_the_compaction_in_place_of_the_earlier_turns(sdk_factory):
    sdk_factory(make_sdk())
    assert await run_agent(conversation(), conversation_id="conv-1") == "ok"
    roles = [r for r, _ in seen[0]]
    texts = " ".join(str(t) for _, t in seen[0])
    assert roles[0] == "system" and "<synap_conversation_history>" in seen[0][0][1]
    assert "The user is planning a trip to Lisbon" in texts
    assert "I want to visit Portugal" not in texts and "In May, for a week" not in texts
    assert ("assistant", "TAP at 9am") in seen[0]
    assert seen[0][-1] == ("user", "and the hotel?")


@pytest.mark.asyncio
async def test_the_input_is_untouched_when_synap_has_no_compaction(sdk_factory):
    sdk_factory(make_sdk(compaction=False))
    await run_agent(conversation(), conversation_id="conv-1")
    assert [t for _, t in seen[0]][:2] == ["I want to visit Portugal", "Great, when?"]
    assert len(seen[0]) == 5


@pytest.mark.asyncio
async def test_the_input_is_untouched_when_the_sdk_fails(sdk_factory):
    sdk = make_sdk()
    sdk.conversation.context.get_context_for_prompt = AsyncMock(side_effect=RuntimeError("down"))
    sdk_factory(sdk)
    await run_agent(conversation(), conversation_id="conv-1")
    assert len(seen[0]) == 5


@pytest.mark.asyncio
async def test_with_no_conversation_id_nothing_is_fetched(sdk_factory):
    sdk = make_sdk()
    sdk_factory(sdk)
    await run_agent(conversation())
    assert len(seen[0]) == 5
    sdk.conversation.context.get_context_for_prompt.assert_not_awaited()


@pytest.mark.asyncio
async def test_nats_own_conversation_id_wins_over_the_configured_one(sdk_factory):
    sdk = make_sdk()
    sdk_factory(sdk)
    token = ContextState.get().conversation_id.set("nat-conv")
    try:
        await run_agent(conversation(), conversation_id="configured")
    finally:
        ContextState.get().conversation_id.reset(token)
    assert sdk.conversation.context.get_context_for_prompt.await_args.kwargs["conversation_id"] == "nat-conv"


# ---------------------------------------------------------------------------
# The editor does not ingest, as a document, turns that were recorded
# ---------------------------------------------------------------------------


def _item(role, text, user="u1"):
    return MemoryItem(user_id=user, conversation=[{"role": role, "content": text}], tags=[], metadata={})


@pytest.mark.asyncio
async def test_a_conversation_item_is_recorded_as_a_turn_not_ingested_as_a_document(sdk_factory):
    sdk = make_sdk(compaction=False)
    set_auto_listen(sdk, False)
    token = ContextState.get().conversation_id.set("nat-conv")
    try:
        await SynapMemoryEditor(sdk=sdk, customer_id="acme").add_items([_item("user", "hello")])
    finally:
        ContextState.get().conversation_id.reset(token)
    kwargs = sdk.conversation.record_message.await_args.kwargs
    assert (kwargs["role"], kwargs["content"], kwargs["conversation_id"]) == ("user", "hello", "nat-conv")
    sdk.memories.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_fact_item_is_still_a_document(sdk_factory):
    sdk = make_sdk(compaction=False)
    await SynapMemoryEditor(sdk=sdk).add_items([MemoryItem(user_id="u1", memory="likes tea", tags=[], metadata={})])
    sdk.memories.create.assert_awaited_once()
    sdk.conversation.record_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_turn_that_cannot_be_recorded_falls_back_to_the_document(sdk_factory):
    sdk = make_sdk(compaction=False)
    set_auto_listen(sdk, False)  # no stream, and no conversation id for REST
    await SynapMemoryEditor(sdk=sdk).add_items([_item("user", "hello")])
    sdk.memories.create.assert_awaited_once()
    assert "user: hello" in sdk.memories.create.await_args.kwargs["document"]


@pytest.mark.asyncio
async def test_turns_the_exporter_wrote_are_skipped_one_for_one(sdk_factory):
    sdk = make_sdk(compaction=False)
    exporter = SynapStreamExporter(sdk=sdk, conversation_id="conv-1", user_id="u1")
    anc = InvocationNode(function_id="f", function_name="agent")

    def step(kind, **data):
        return IntermediateStep(parent_id="root", function_ancestry=anc, payload=IntermediateStepPayload(
            UUID="s", event_type=kind, data=StreamEventData(**data)))

    await exporter._report(step(IntermediateStepType.WORKFLOW_START, input="hello"))
    await exporter._report(step(IntermediateStepType.WORKFLOW_END, output="hi there"))
    assert sdk.instance.send_message.await_count == 2

    editor = SynapMemoryEditor(sdk=sdk)
    await editor.add_items([_item("user", "hello"), _item("assistant", "hi there"), _item("user", "next")])

    sent = [c.kwargs["content"] for c in sdk.instance.send_message.await_args_list]
    assert sent == ["hello", "hi there", "next"]  # only "next" was new
    sdk.memories.create.assert_not_awaited()


# ---------------------------------------------------------------------------
# The stream is the default path: nobody called listen()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_first_turn_opens_the_stream_and_both_turns_go_on_it():
    sdk = make_sdk(compaction=False)
    assert sdk.instance.is_listening is False
    exporter = SynapStreamExporter(sdk=sdk, conversation_id="conv-1", user_id="u1")
    anc = InvocationNode(function_id="f", function_name="agent")
    for kind, data in ((IntermediateStepType.WORKFLOW_START, dict(input="where is my order")),
                       (IntermediateStepType.WORKFLOW_END, dict(output="tomorrow"))):
        await exporter._report(IntermediateStep(
            parent_id="root", function_ancestry=anc,
            payload=IntermediateStepPayload(UUID="s", event_type=kind, data=StreamEventData(**data))))
    assert sdk.instance.listen_calls == 1
    assert [(c.kwargs["event_type"], c.kwargs["content"]) for c in sdk.instance.send_message.await_args_list] == [
        ("user_message", "where is my order"), ("assistant_message", "tomorrow")]
    sdk.conversation.record_message.assert_not_awaited()
