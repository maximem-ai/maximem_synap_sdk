"""Tests for the live gRPC stream wiring.

The rule these pin is **stream first, REST only as a fallback, never both**.
The server persists ``user_message`` and ``assistant_message`` from the stream
itself (``grpc/servicer.py::_persist_message``), so a ``record_message`` on top
of a delivered stream event writes the same turn twice and extracts it twice.

Half of this file drives a **real MAF agent run** rather than calling the hooks
by hand. That is deliberate. The first version of ``SynapToolReporter`` was not
a ``FunctionMiddleware`` subclass, so MAF sorted it into *agent* middleware,
``extend_middleware`` refused it, the refusal was caught, and every tool call
and tool result in every run was dropped. Every hand-driven test passed.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from agent_framework import (
    Agent,
    BaseChatClient,
    ChatMiddlewareLayer,
    ChatResponse,
    Content,
    FunctionInvocationLayer,
    Message,
)

from synap_microsoft_agent.context_provider import SynapContextProvider
from synap_microsoft_agent.history_provider import SynapHistoryProvider
from synap_microsoft_agent.stream import (
    SynapToolReporter,
    install_tool_reporter,
    report_message_turn,
    tool_arguments,
    tool_result_payload,
)

from tests._harness import requires_harness


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


def streaming_sdk(is_listening: bool = True, order: list | None = None):
    """An SDK whose stream is open (or not), with every reporting call wired."""
    sdk = MagicMock()
    sdk.instance.is_listening = is_listening

    def _log(name):
        async def _call(*args, **kwargs):
            if order is not None:
                order.append(name)

        return _call

    sdk.instance.send_message = AsyncMock(side_effect=_log("send_message"))
    sdk.instance.record_thinking = AsyncMock(side_effect=_log("record_thinking"))
    sdk.instance.record_tool_call = AsyncMock(side_effect=_log("record_tool_call"))
    sdk.instance.record_tool_result = AsyncMock(side_effect=_log("record_tool_result"))
    sdk.conversation.record_message = AsyncMock(side_effect=_log("record_message"))

    fetched = MagicMock()
    fetched.formatted_context = "known: the user flies a lot"
    sdk.fetch = AsyncMock(side_effect=_log_fetch(order, fetched))
    return sdk


def _log_fetch(order, result):
    async def _call(*args, **kwargs):
        if order is not None:
            order.append("fetch")
        return result

    return _call


class FakeChatClient(FunctionInvocationLayer, ChatMiddlewareLayer, BaseChatClient):
    """A chat client that replays scripted assistant messages.

    Enough of a real client for MAF to run its own tool loop over: the
    function-invocation layer sees the ``function_call`` content, invokes the
    tool, and calls back for the next message.
    """

    def __init__(self, scripted, **kwargs):
        super().__init__(**kwargs)
        self._scripted = list(scripted)

    def _inner_get_response(self, *, messages, stream, options, **kwargs):
        async def _go():
            await self._validate_options(options)
            return ChatResponse(messages=[self._scripted.pop(0)], response_id="r")

        return _go()


def lookup_order(pnr: str) -> dict:
    """Look up an order by PNR."""
    return {"pnr": pnr, "status": "confirmed"}


def a_tool_turn():
    """The two model replies of a one-tool turn: reason, call, then answer."""
    return [
        Message(
            role="assistant",
            contents=[
                Content.from_text_reasoning(text="Look the booking up first."),
                Content.from_function_call(
                    "call-abc-1", "lookup_order", arguments={"pnr": "QX41RT"}
                ),
            ],
        ),
        Message(role="assistant", contents=[Content.from_text("QX41RT is confirmed.")]),
    ]


async def run_a_turn(sdk, question="where is my order QX41RT?"):
    """Drive one real agent turn through ``SynapContextProvider``."""
    provider = SynapContextProvider(
        sdk=sdk, user_id="alice", customer_id="acme", conversation_id="conv-1"
    )
    agent = Agent(
        FakeChatClient(a_tool_turn()),
        instructions="help",
        tools=[lookup_order],
        context_providers=[provider],
    )
    return await agent.run(question, session=agent.get_session("sess-1"))


def sent(sdk, event_type):
    return [
        c.kwargs
        for c in sdk.instance.send_message.await_args_list
        if c.kwargs.get("event_type") == event_type
    ]


# ---------------------------------------------------------------------------
# report_message_turn — which roles have a stream event at all
# ---------------------------------------------------------------------------


class TestWhichTurnsTheStreamTakes:
    @pytest.mark.asyncio
    async def test_a_user_turn_goes_out(self):
        sdk = streaming_sdk(True)
        assert await report_message_turn(
            sdk, role="user", content="hi", conversation_id="c", user_id="u"
        )
        assert sdk.instance.send_message.await_args.kwargs["event_type"] == "user_message"

    @pytest.mark.asyncio
    async def test_an_assistant_turn_is_the_anticipation_moment(self):
        sdk = streaming_sdk(True)
        await report_message_turn(
            sdk, role="assistant", content="hello", conversation_id="c", user_id="u"
        )
        kw = sdk.instance.send_message.await_args.kwargs
        assert kw["event_type"] == "assistant_message"
        assert kw["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_a_system_message_is_refused(self):
        """`report_turn` maps anything that is not "user" onto
        `assistant_message`. A system prompt put through it would be filed as
        something the assistant said, so it keeps the REST path instead."""
        sdk = streaming_sdk(True)
        assert not await report_message_turn(
            sdk, role="system", content="you are helpful",
            conversation_id="c", user_id="u",
        )
        sdk.instance.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_with_no_stream_nothing_goes_out(self):
        sdk = streaming_sdk(False)
        assert not await report_message_turn(
            sdk, role="user", content="hi", conversation_id="c", user_id="u"
        )

    @pytest.mark.asyncio
    async def test_a_turn_with_no_conversation_id_is_refused(self):
        sdk = streaming_sdk(True)
        assert not await report_message_turn(
            sdk, role="user", content="hi", conversation_id="", user_id="u"
        )
        sdk.instance.send_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# tool_arguments — the parsed structure, never a repr
# ---------------------------------------------------------------------------


class TestToolArgumentsAreRealStructure:
    def test_a_mapping_comes_through_as_a_dict(self):
        assert tool_arguments({"pnr": "QX41RT"}) == {"pnr": "QX41RT"}

    def test_a_pydantic_model_is_unwrapped(self):
        """⚠ `report_tool_call` drops anything that is not a dict
        (`tool_args if isinstance(tool_args, dict) else None`), and MAF
        documents `FunctionInvocationContext.arguments` as BaseModel *or*
        Mapping. Unwrapped, or the call reports its name with no arguments."""
        from pydantic import BaseModel

        class Args(BaseModel):
            pnr: str

        assert tool_arguments(Args(pnr="QX41RT")) == {"pnr": "QX41RT"}

    def test_the_unwrapped_arguments_survive_json_dumps(self):
        from pydantic import BaseModel

        class Args(BaseModel):
            pnr: str

        json.dumps(tool_arguments(Args(pnr="QX41RT")))

    def test_something_that_is_neither_is_dropped_rather_than_guessed(self):
        assert tool_arguments("pnr=QX41RT") is None
        assert tool_arguments(None) is None


# ---------------------------------------------------------------------------
# tool_result_payload — the silent drop
# ---------------------------------------------------------------------------


class TestTheToolResultSurvivesSerialisation:
    """The SDK does `json.dumps` on the result and MAF hands function
    middleware a `list[Content]`, which is not JSON serialisable. Measured:
    `TypeError: Object of type Content is not JSON serializable`.
    `stream_events._send` swallows every exception by design, so passing it
    through would report no tool result at all, silently."""

    def test_a_raw_maf_result_is_not_json_serialisable(self):
        """The premise. If this ever stops being true the unwrap can go."""
        with pytest.raises(TypeError):
            json.dumps([Content.from_text("shipped")])

    def test_a_list_of_content_is_unwrapped_to_its_text(self):
        assert tool_result_payload([Content.from_text("shipped")]) == "shipped"

    def test_the_unwrapped_result_survives_json_dumps(self):
        json.dumps(tool_result_payload([Content.from_text("shipped")]))

    def test_several_contents_all_come_through(self):
        payload = tool_result_payload(
            [Content.from_text("one"), Content.from_text("two")]
        )
        assert payload == ["one", "two"]

    def test_a_function_result_content_yields_its_result(self):
        payload = tool_result_payload([Content.from_function_result("c1", result="ok")])
        assert payload == "ok"

    def test_a_plain_result_is_left_alone(self):
        assert tool_result_payload({"status": "ok"}) == {"status": "ok"}

    def test_an_absent_result_reports_nothing(self):
        assert tool_result_payload(None) is None
        assert tool_result_payload([]) is None

    def test_a_result_nothing_can_serialise_is_still_reported(self):
        """The last line of defence. An integration cannot know every shape a
        tool can return, but it can guarantee what it hands the SDK
        serialises, rather than letting the send be swallowed."""
        payload = tool_result_payload(object())
        json.dumps(payload)
        assert isinstance(payload, str)


# ---------------------------------------------------------------------------
# The reporter has to be middleware MAF will accept
# ---------------------------------------------------------------------------


class TestTheReporterIsAcceptedAsFunctionMiddleware:
    def test_maf_categorises_it_as_function_middleware(self):
        """⚠ The bug that cost every tool event. `categorize_middleware` sorts
        an object it does not recognise into *agent* middleware, and
        `SessionContext.extend_middleware` refuses agent middleware from a
        context provider."""
        from agent_framework._middleware import categorize_middleware

        reporter = SynapToolReporter(
            MagicMock(), conversation_id="c", user_id="u", customer_id=""
        )
        assert categorize_middleware([reporter])["function"] == [reporter]
        assert categorize_middleware([reporter])["agent"] == []

    def test_installing_it_on_a_context_actually_takes(self):
        from agent_framework import SessionContext

        context = SessionContext(input_messages=[])
        assert install_tool_reporter(
            context, MagicMock(), source_id="synap",
            conversation_id="c", user_id="u",
        )
        installed = context.get_middleware()
        assert len(installed) == 1
        assert isinstance(installed[0], SynapToolReporter)

    def test_a_second_provider_does_not_install_a_second_reporter(self):
        """Two Synap providers on one agent share a SessionContext, and two
        reporters would report every tool twice."""
        from agent_framework import SessionContext

        context = SessionContext(input_messages=[])
        install_tool_reporter(
            context, MagicMock(), source_id="synap",
            conversation_id="c", user_id="u",
        )
        assert not install_tool_reporter(
            context, MagicMock(), source_id="synap_harness",
            conversation_id="c", user_id="u",
        )
        assert len(context.get_middleware()) == 1

    def test_no_conversation_id_means_no_reporter(self):
        from agent_framework import SessionContext

        context = SessionContext(input_messages=[])
        assert not install_tool_reporter(
            context, MagicMock(), source_id="synap",
            conversation_id="", user_id="u",
        )
        assert context.get_middleware() == []


# ---------------------------------------------------------------------------
# A real agent turn, end to end
# ---------------------------------------------------------------------------


class TestARealTurnReportsTheWholeTurn:
    @pytest.mark.asyncio
    async def test_the_question_is_reported(self):
        sdk = streaming_sdk(True)
        await run_a_turn(sdk)
        assert [m["content"] for m in sent(sdk, "user_message")] == [
            "where is my order QX41RT?"
        ]

    @pytest.mark.asyncio
    async def test_the_answer_is_reported(self):
        sdk = streaming_sdk(True)
        await run_a_turn(sdk)
        assert [m["content"] for m in sent(sdk, "assistant_message")] == [
            "QX41RT is confirmed."
        ]

    @pytest.mark.asyncio
    async def test_the_reasoning_is_reported(self):
        sdk = streaming_sdk(True)
        await run_a_turn(sdk)
        sdk.instance.record_thinking.assert_awaited_once()
        assert (
            sdk.instance.record_thinking.await_args.args[0]
            == "Look the booking up first."
        )

    @pytest.mark.asyncio
    async def test_the_tool_call_is_reported_with_its_parsed_arguments(self):
        sdk = streaming_sdk(True)
        await run_a_turn(sdk)
        sdk.instance.record_tool_call.assert_awaited_once()
        args = sdk.instance.record_tool_call.await_args.args
        assert args[0] == "lookup_order"
        assert args[1] == {"pnr": "QX41RT"}

    @pytest.mark.asyncio
    async def test_the_tool_result_is_reported(self):
        sdk = streaming_sdk(True)
        await run_a_turn(sdk)
        sdk.instance.record_tool_result.assert_awaited_once()
        json.dumps(sdk.instance.record_tool_result.await_args.args[0])

    @pytest.mark.asyncio
    async def test_the_call_and_its_result_share_the_frameworks_own_id(self):
        """Without this the anticipation agent sees a call and a result and
        cannot tell they are the same invocation."""
        sdk = streaming_sdk(True)
        response = await run_a_turn(sdk)
        call = sdk.instance.record_tool_call.await_args.kwargs["tool_call_id"]
        result = sdk.instance.record_tool_result.await_args.kwargs["tool_call_id"]

        # ⚠ This used to assert the literal `"call-abc-1"`, the id the fixture
        # puts on the FunctionCallContent above. It failed on the installed
        # agent-framework, and the BEHAVIOUR WAS CORRECT: MAF only exposes its
        # own call id to function middleware on versions above this package's
        # floor, so on older ones `SynapToolReporter` mints `maf-<hex>` and
        # uses it for both halves. The literal pinned one framework version
        # rather than the contract, which is why it broke on another.
        #
        # What holds on every supported version is that the two halves PAIR.
        # That is what lets the anticipation agent follow a turn with several
        # tools in flight. A minted id pairs with nothing outside this process;
        # two calls both minting `''` would pair with each other's results.
        assert call and call == result, (
            f"the call and its result do not pair: {call!r} vs {result!r}"
        )
        assert (sdk.instance.record_tool_call.await_count == 1
                and sdk.instance.record_tool_result.await_count == 1), (
            "one tool ran, so exactly one call and one result should be "
            "reported; a second would mean two paths are both reporting it"
        )
        # Whether that id is MAF's own or one we minted is version-dependent,
        # and is pinned separately against the middleware itself, below.
        framework_ids = {
            c.call_id
            for message in response.messages
            for c in getattr(message, "contents", [])
            if getattr(c, "call_id", None)
        }
        assert framework_ids, "the fixture no longer produces a function call"

    @pytest.mark.asyncio
    async def test_the_turn_is_NOT_also_written_over_rest(self):
        """The double-write the stream-first rule exists to prevent."""
        sdk = streaming_sdk(True)
        await run_a_turn(sdk)
        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_question_is_reported_exactly_once(self):
        """`before_run` puts it on the stream; `after_run` must not repeat it."""
        sdk = streaming_sdk(True)
        await run_a_turn(sdk)
        assert len(sent(sdk, "user_message")) == 1

    @pytest.mark.asyncio
    async def test_the_events_arrive_in_the_order_the_turn_happened(self):
        order: list = []
        sdk = streaming_sdk(True, order=order)
        await run_a_turn(sdk)
        stream_only = [c for c in order if c != "fetch"]
        assert stream_only == [
            "send_message",  # the question
            "record_tool_call",
            "record_tool_result",
            "record_thinking",
            "send_message",  # the answer
        ]

    @pytest.mark.asyncio
    async def test_the_question_is_reported_before_the_fetch_it_caused(self):
        order: list = []
        sdk = streaming_sdk(True, order=order)
        await run_a_turn(sdk)
        assert order.index("send_message") < order.index("fetch")


class TestARealTurnWithNoStream:
    @pytest.mark.asyncio
    async def test_it_falls_back_to_rest(self):
        sdk = streaming_sdk(False)
        await run_a_turn(sdk)
        recorded = [
            (c.kwargs["role"], c.kwargs["content"])
            for c in sdk.conversation.record_message.await_args_list
        ]
        assert recorded == [
            ("user", "where is my order QX41RT?"),
            ("assistant", "QX41RT is confirmed."),
        ]

    @pytest.mark.asyncio
    async def test_and_nothing_at_all_goes_on_the_stream(self):
        sdk = streaming_sdk(False)
        await run_a_turn(sdk)
        sdk.instance.send_message.assert_not_awaited()
        sdk.instance.record_tool_call.assert_not_awaited()
        sdk.instance.record_tool_result.assert_not_awaited()
        sdk.instance.record_thinking.assert_not_awaited()


class TestNoneOfItBreaksTheAgentRun:
    """These hooks sit inside somebody's agent turn. A telemetry call that
    throws there ends their run, and no context is worth that."""

    @pytest.mark.asyncio
    async def test_a_stream_that_raises_on_every_call_still_answers(self):
        sdk = streaming_sdk(True)
        boom = RuntimeError("stream boom")
        sdk.instance.send_message = AsyncMock(side_effect=boom)
        sdk.instance.record_thinking = AsyncMock(side_effect=boom)
        sdk.instance.record_tool_call = AsyncMock(side_effect=boom)
        sdk.instance.record_tool_result = AsyncMock(side_effect=boom)
        response = await run_a_turn(sdk)
        assert response.text == "QX41RT is confirmed."

    @pytest.mark.asyncio
    async def test_and_the_turn_still_reaches_rest(self):
        """A send that failed did not go out, so the REST fallback owns it."""
        sdk = streaming_sdk(True)
        sdk.instance.send_message = AsyncMock(side_effect=RuntimeError("boom"))
        await run_a_turn(sdk)
        assert sdk.conversation.record_message.await_count == 2

    @pytest.mark.asyncio
    async def test_an_sdk_with_no_instance_namespace_still_answers(self):
        """An older SDK, or a caller's own double. Every report must no-op."""
        sdk = MagicMock()
        del sdk.instance
        sdk.conversation.record_message = AsyncMock()
        fetched = MagicMock()
        fetched.formatted_context = ""
        sdk.fetch = AsyncMock(return_value=fetched)
        response = await run_a_turn(sdk)
        assert response.text == "QX41RT is confirmed."


# ---------------------------------------------------------------------------
# SynapHistoryProvider
# ---------------------------------------------------------------------------


class TestTheHistoryProviderIsStreamFirst:
    @pytest.mark.asyncio
    async def test_a_turn_goes_on_the_stream_and_not_over_rest(self):
        sdk = streaming_sdk(True)
        provider = SynapHistoryProvider(sdk=sdk, user_id="alice", customer_id="acme")
        await provider.save_messages(
            "conv-1", [Message(role="user", contents=["hello"])]
        )
        sdk.instance.send_message.assert_awaited_once()
        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_with_no_stream_it_falls_back_to_rest(self):
        sdk = streaming_sdk(False)
        provider = SynapHistoryProvider(sdk=sdk, user_id="alice", customer_id="acme")
        await provider.save_messages(
            "conv-1", [Message(role="user", contents=["hello"])]
        )
        sdk.conversation.record_message.assert_awaited_once()
        sdk.instance.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_system_message_keeps_the_rest_path(self):
        """It has no stream event, and reporting it would file the system
        prompt as something the assistant said."""
        sdk = streaming_sdk(True)
        provider = SynapHistoryProvider(sdk=sdk, user_id="alice", customer_id="acme")
        await provider.save_messages(
            "conv-1", [Message(role="system", contents=["be helpful"])]
        )
        sdk.instance.send_message.assert_not_awaited()
        assert sdk.conversation.record_message.await_args.kwargs["role"] == "system"


# ---------------------------------------------------------------------------
# The harness memory provider
# ---------------------------------------------------------------------------


@requires_harness
class TestTheHarnessTranscriptIsStreamFirst:
    def _provider(self, sdk):
        from synap_microsoft_agent.harness_memory import create_synap_harness_memory

        return create_synap_harness_memory(sdk, user_id="alice", customer_id="acme")

    @pytest.mark.asyncio
    async def test_a_turn_goes_on_the_stream_and_not_over_rest(self):
        sdk = streaming_sdk(True)
        await self._provider(sdk).save_messages(
            "sess-1", [Message(role="user", contents=["hello"])], state={}
        )
        sdk.instance.send_message.assert_awaited_once()
        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_with_no_stream_it_falls_back_to_rest(self):
        sdk = streaming_sdk(False)
        await self._provider(sdk).save_messages(
            "sess-1", [Message(role="user", contents=["hello"])], state={}
        )
        sdk.conversation.record_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_the_stream_turn_uses_the_same_conversation_id_as_rest(self):
        """A transcript split across two ids is a transcript nobody can read
        back whole."""
        streamed = streaming_sdk(True)
        resting = streaming_sdk(False)
        message = Message(role="user", contents=["hello"])
        await self._provider(streamed).save_messages("sess-1", [message], state={})
        await self._provider(resting).save_messages("sess-1", [message], state={})
        assert (
            streamed.instance.send_message.await_args.kwargs["conversation_id"]
            == resting.conversation.record_message.await_args.kwargs["conversation_id"]
        )

    @pytest.mark.asyncio
    async def test_reasoning_in_the_transcript_is_reported(self):
        sdk = streaming_sdk(True)
        message = Message(
            role="assistant",
            contents=[
                Content.from_text_reasoning(text="checking the booking"),
                Content.from_text("all set"),
            ],
        )
        await self._provider(sdk).save_messages("sess-1", [message], state={})
        sdk.instance.record_thinking.assert_awaited_once()
        assert (
            sdk.instance.record_thinking.await_args.args[0] == "checking the booking"
        )

    @pytest.mark.asyncio
    async def test_before_run_installs_the_tool_reporter(self):
        """A harness agent reaches for tools too, and the harness memory
        provider is the only Synap provider in that wiring."""
        from agent_framework import AgentSession, SessionContext

        sdk = streaming_sdk(True)
        context = SessionContext(session_id="sess-1", input_messages=[])
        await self._provider(sdk).before_run(
            agent=MagicMock(),
            session=AgentSession(session_id="sess-1"),
            context=context,
            state={},
        )
        installed = context.get_middleware()
        assert len(installed) == 1
        assert isinstance(installed[0], SynapToolReporter)

    @pytest.mark.asyncio
    async def test_its_reporter_uses_the_transcript_conversation_id(self):
        """A tool event filed under a different id than the transcript is a
        tool event nobody can line up with the turn."""
        from agent_framework import AgentSession, SessionContext

        sdk = streaming_sdk(True)
        provider = self._provider(sdk)
        context = SessionContext(session_id="sess-1", input_messages=[])
        await provider.before_run(
            agent=MagicMock(),
            session=AgentSession(session_id="sess-1"),
            context=context,
            state={},
        )
        await provider.save_messages(
            "sess-1", [Message(role="user", contents=["hello"])], state={}
        )
        assert (
            context.get_middleware()[0].conversation_id
            == sdk.instance.send_message.await_args.kwargs["conversation_id"]
        )

    @pytest.mark.asyncio
    async def test_reasoning_is_not_also_sent_as_the_turn_text(self):
        """MAF leaves reasoning off `Message.text`, so the two never overlap."""
        sdk = streaming_sdk(True)
        message = Message(
            role="assistant",
            contents=[
                Content.from_text_reasoning(text="checking the booking"),
                Content.from_text("all set"),
            ],
        )
        await self._provider(sdk).save_messages("sess-1", [message], state={})
        assert [m["content"] for m in sent(sdk, "assistant_message")] == ["all set"]


class TestWhichToolCallIdWeReport:
    """MAF's own id when it offers one, a minted one when it does not.

    ⚠ This branch had no test, and the only test that touched the id at all
    asserted a literal from one framework version, so it broke on another
    while the code was right. What matters is the rule, not the value:

      - MAF puts the invocation's id on ``context.metadata["call_id"]``, but
        only on versions above this package's floor. Use it when it is there,
        because an id shared with the framework is the one a reader can
        correlate against MAF's own record of the turn.
      - When it is absent, mint one. The pairing between the call and its
        result is what anticipation needs, and it must not be lost just
        because the framework is older.
      - A minted id must be fresh per invocation. Two concurrent tools sharing
        one id pair with each other's results.
    """

    @staticmethod
    def _context(metadata, name="lookup_order"):
        ctx = MagicMock()
        ctx.metadata = metadata
        ctx.function.name = name
        ctx.arguments = {"pnr": "QX41RT"}
        ctx.result = {"status": "confirmed"}
        return ctx

    def _reporter(self, sdk):
        return SynapToolReporter(
            sdk=sdk, conversation_id="conv-1", user_id="alice", customer_id="acme"
        )

    @pytest.mark.asyncio
    async def test_mafs_own_id_is_used_when_maf_offers_one(self):
        sdk = streaming_sdk(True)
        await self._reporter(sdk).process(
            self._context({"call_id": "call-abc-1"}), AsyncMock()
        )
        assert sdk.instance.record_tool_call.await_args.kwargs["tool_call_id"] == "call-abc-1"
        assert sdk.instance.record_tool_result.await_args.kwargs["tool_call_id"] == "call-abc-1"

    @pytest.mark.asyncio
    async def test_an_id_is_minted_when_maf_offers_none(self):
        sdk = streaming_sdk(True)
        await self._reporter(sdk).process(self._context({}), AsyncMock())
        call = sdk.instance.record_tool_call.await_args.kwargs["tool_call_id"]
        result = sdk.instance.record_tool_result.await_args.kwargs["tool_call_id"]
        assert call.startswith("maf-") and call == result

    @pytest.mark.asyncio
    async def test_an_empty_maf_id_is_treated_as_absent(self):
        """Two calls both reporting '' would pair with each other's results."""
        sdk = streaming_sdk(True)
        await self._reporter(sdk).process(self._context({"call_id": ""}), AsyncMock())
        assert sdk.instance.record_tool_call.await_args.kwargs["tool_call_id"] != ""

    @pytest.mark.asyncio
    async def test_a_minted_id_is_fresh_per_invocation(self):
        sdk = streaming_sdk(True)
        reporter = self._reporter(sdk)
        await reporter.process(self._context({}), AsyncMock())
        first = sdk.instance.record_tool_call.await_args.kwargs["tool_call_id"]
        await reporter.process(self._context({}), AsyncMock())
        second = sdk.instance.record_tool_call.await_args.kwargs["tool_call_id"]
        assert first != second, (
            "two invocations share one id, so their results pair with the "
            "wrong call"
        )

    @pytest.mark.asyncio
    async def test_metadata_that_raises_does_not_stop_the_report(self):
        """A context object shaped differently must cost the id, not the run."""
        sdk = streaming_sdk(True)
        ctx = self._context({})
        type(ctx).metadata = property(lambda self: (_ for _ in ()).throw(RuntimeError()))
        await self._reporter(sdk).process(ctx, AsyncMock())
        assert sdk.instance.record_tool_call.await_count == 1

    @pytest.mark.asyncio
    async def test_the_tool_still_runs_when_there_is_no_stream(self):
        sdk = streaming_sdk(False)
        call_next = AsyncMock()
        await self._reporter(sdk).process(self._context({}), call_next)
        call_next.assert_awaited_once()
        sdk.instance.record_tool_call.assert_not_awaited()
