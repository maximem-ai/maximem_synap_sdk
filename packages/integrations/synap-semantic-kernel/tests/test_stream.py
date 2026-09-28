"""Tests for synap_semantic_kernel.stream — turns, tool calls, tool results.

Semantic Kernel had NO turn recording at all before this: the plugin let the
model search and store memory, and short_term put a context block in the
prompt, but nothing ever told Synap what was said. Anticipation saw an empty
conversation.

The rule these pin is "stream first, REST only as a fallback, never both".
The server persists `user_message` and `assistant_message` from the stream
itself, so a `record_message` on top of a delivered stream event writes the
same turn twice and extracts it twice.

These use the REAL Semantic Kernel models and a REAL `Kernel` filter call
stack, not doubles. Every bug worth catching here is a wrong assumption about
their shape.
"""

from __future__ import annotations

import json
from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock

import pytest

from semantic_kernel import Kernel
from semantic_kernel.contents import (
    AuthorRole,
    ChatHistory,
    ChatMessageContent,
    FunctionCallContent,
    FunctionResultContent,
    ReasoningContent,
    TextContent,
)
from semantic_kernel.filters import AutoFunctionInvocationContext, FilterTypes
from semantic_kernel.filters.kernel_filters_extension import (
    _rebuild_auto_function_invocation_context,
)
from semantic_kernel.functions import (
    FunctionResult,
    KernelArguments,
    KernelFunctionMetadata,
    kernel_function,
)

from synap_semantic_kernel.stream import (
    SynapAgentThread,
    _tool_args,
    _tool_output,
    attach_synap_filters,
)

# The filter context has forward references Semantic Kernel resolves lazily;
# this is the framework's own hook for doing it.
_rebuild_auto_function_invocation_context()


# ---------------------------------------------------------------------------
# Test doubles and builders
# ---------------------------------------------------------------------------


def _streaming_sdk(is_listening: bool = True):
    sdk = MagicMock()
    sdk.instance.is_listening = is_listening
    sdk.instance.send_message = AsyncMock()
    sdk.instance.record_tool_call = AsyncMock()
    sdk.instance.record_tool_result = AsyncMock()
    sdk.instance.record_thinking = AsyncMock()
    sdk.instance.end_session = AsyncMock()
    sdk.conversation.record_message = AsyncMock()
    return sdk


def _thread(sdk, **kwargs):
    return SynapAgentThread(
        sdk, conversation_id="conv-1", user_id="u1", customer_id="c1", **kwargs
    )


def _metadata():
    return KernelFunctionMetadata(
        name="lookup", plugin_name="orders", description="",
        parameters=[], is_prompt=False,
    )


def _kernel_with_tool():
    kernel = Kernel()

    @kernel_function(name="lookup", description="look up an order")
    def lookup(pnr: str) -> str:  # pragma: no cover — never actually invoked
        return "shipped"

    plugin = kernel.add_plugin(
        type("Orders", (), {"lookup": lookup})(), plugin_name="orders"
    )
    return kernel, plugin["lookup"]


def _context(kernel, function, *, call_id="call_1", arguments='{"pnr": "QX41RT"}'):
    return AutoFunctionInvocationContext(
        function=function,
        kernel=kernel,
        arguments=KernelArguments(),
        function_call_content=FunctionCallContent(
            id=call_id, name="orders-lookup", arguments=arguments
        ),
    )


async def _run_filter(kernel, context, inner):
    stack = kernel.construct_call_stack(
        FilterTypes.AUTO_FUNCTION_INVOCATION, inner
    )
    await stack(context)


def _returns(value):
    async def _inner(context):
        context.function_result = FunctionResult(function=_metadata(), value=value)
    return _inner


async def _returns_nothing(context):
    return None


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_public_surface_exported(self):
        import synap_semantic_kernel

        assert synap_semantic_kernel.SynapAgentThread is SynapAgentThread
        assert "attach_synap_filters" in synap_semantic_kernel.__all__

    def test_thread_requires_non_none_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            SynapAgentThread(None, conversation_id="c", user_id="u")

    def test_thread_requires_non_empty_conversation_id(self):
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            SynapAgentThread(_streaming_sdk(), conversation_id="", user_id="u")

    def test_thread_requires_non_empty_user_id(self):
        with pytest.raises(ValueError, match="non-empty user_id"):
            SynapAgentThread(_streaming_sdk(), conversation_id="c", user_id="")

    def test_it_is_still_a_chat_history_thread(self):
        """ChatCompletionAgent refuses a thread that is not one of its own."""
        from semantic_kernel.agents import ChatHistoryAgentThread

        assert isinstance(_thread(_streaming_sdk()), ChatHistoryAgentThread)


# ---------------------------------------------------------------------------
# Stream first, REST only as a fallback, never both
# ---------------------------------------------------------------------------


class TestStreamFirstRestFallback:
    @pytest.mark.asyncio
    async def test_a_user_turn_goes_out_on_the_stream(self):
        sdk = _streaming_sdk(True)
        await _thread(sdk).on_new_message(
            ChatMessageContent(role=AuthorRole.USER, content="where is my order")
        )
        sdk.instance.send_message.assert_awaited_once()
        kw = sdk.instance.send_message.await_args.kwargs
        assert kw["event_type"] == "user_message"
        assert kw["content"] == "where is my order"

    @pytest.mark.asyncio
    async def test_and_is_NOT_also_written_over_rest(self):
        """The double-write this rule exists to prevent."""
        sdk = _streaming_sdk(True)
        await _thread(sdk).on_new_message(
            ChatMessageContent(role=AuthorRole.USER, content="where is my order")
        )
        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_with_no_stream_it_falls_back_to_rest(self):
        sdk = _streaming_sdk(False)
        await _thread(sdk).on_new_message(
            ChatMessageContent(role=AuthorRole.ASSISTANT, content="it ships tomorrow")
        )
        sdk.conversation.record_message.assert_awaited_once()
        sdk.instance.send_message.assert_not_awaited()
        assert sdk.conversation.record_message.await_args.kwargs["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_the_assistant_turn_is_the_anticipation_moment(self):
        sdk = _streaming_sdk(True)
        await _thread(sdk).on_new_message(
            ChatMessageContent(role=AuthorRole.ASSISTANT, content="it ships tomorrow")
        )
        kw = sdk.instance.send_message.await_args.kwargs
        assert kw["event_type"] == "assistant_message"
        assert kw["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_a_bare_string_is_a_user_turn(self):
        """`Agent.invoke(messages="...")` reaches the thread as a plain str."""
        sdk = _streaming_sdk(True)
        await _thread(sdk).on_new_message("hello")
        assert sdk.instance.send_message.await_args.kwargs["event_type"] == (
            "user_message"
        )

    @pytest.mark.asyncio
    async def test_the_ids_travel_with_the_turn(self):
        sdk = _streaming_sdk(True)
        await _thread(sdk).on_new_message("hello")
        kw = sdk.instance.send_message.await_args.kwargs
        assert kw["conversation_id"] == "conv-1"
        assert kw["user_id"] == "u1"
        assert kw["customer_id"] == "c1"


# ---------------------------------------------------------------------------
# The history is never the casualty
# ---------------------------------------------------------------------------


class TestTheHistoryIsNeverTheCasualty:
    @pytest.mark.asyncio
    async def test_the_message_still_reaches_the_history_when_synap_fails(self):
        """`on_new_message` is awaited inside the caller's agent run. A Synap
        outage must cost them telemetry, not a message."""
        sdk = _streaming_sdk(False)
        sdk.conversation.record_message = AsyncMock(side_effect=RuntimeError("boom"))
        history = ChatHistory()
        thread = _thread(sdk, chat_history=history)
        await thread.on_new_message(
            ChatMessageContent(role=AuthorRole.USER, content="hello")
        )
        assert [m.content for m in history.messages] == ["hello"]

    @pytest.mark.asyncio
    async def test_a_synap_outage_does_not_raise_into_the_agent_run(self):
        sdk = _streaming_sdk(False)
        sdk.conversation.record_message = AsyncMock(side_effect=RuntimeError("boom"))
        await _thread(sdk).on_new_message(
            ChatMessageContent(role=AuthorRole.USER, content="hello")
        )

    @pytest.mark.asyncio
    async def test_an_sdk_with_no_instance_namespace_still_records_over_rest(self):
        """An older SDK, or a test double. Every stream call must no-op."""
        sdk = MagicMock(spec=["conversation"])
        sdk.conversation.record_message = AsyncMock()
        thread = SynapAgentThread(sdk, conversation_id="c", user_id="u")
        await thread.on_new_message(
            ChatMessageContent(role=AuthorRole.USER, content="hello")
        )
        sdk.conversation.record_message.assert_awaited_once()


# ---------------------------------------------------------------------------
# What is not a turn
# ---------------------------------------------------------------------------


class TestWhatIsNotATurn:
    @pytest.mark.asyncio
    async def test_a_tool_message_is_not_a_turn(self):
        """The tool result is reported by the filter, which has the call id
        to tie it to its call. This hook does not.

        Both shapes: the FunctionResultContent one Semantic Kernel builds,
        and the plain-text TOOL message some connectors produce."""
        sdk = _streaming_sdk(True)
        thread = _thread(sdk)
        await thread.on_new_message(
            ChatMessageContent(
                role=AuthorRole.TOOL,
                items=[FunctionResultContent(
                    id="call_1", name="orders-lookup", result="shipped"
                )],
            )
        )
        await thread.on_new_message(
            ChatMessageContent(
                role=AuthorRole.TOOL, items=[TextContent(text="shipped")]
            )
        )
        sdk.instance.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_assistant_lap_carrying_a_tool_call_is_not_a_turn(self):
        """Semantic Kernel streams an intermediate message's text into the
        joined message that closes the turn, so reporting the lap as well
        would extract the same words twice."""
        sdk = _streaming_sdk(True)
        await _thread(sdk).on_new_message(
            ChatMessageContent(
                role=AuthorRole.ASSISTANT,
                items=[
                    TextContent(text="let me check that"),
                    FunctionCallContent(
                        id="call_1", name="orders-lookup", arguments="{}"
                    ),
                ],
            )
        )
        sdk.instance.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_system_message_is_not_a_turn(self):
        sdk = _streaming_sdk(True)
        await _thread(sdk).on_new_message(
            ChatMessageContent(role=AuthorRole.SYSTEM, content="you are helpful")
        )
        sdk.instance.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_message_with_no_text_is_not_a_turn(self):
        sdk = _streaming_sdk(True)
        await _thread(sdk).on_new_message(
            ChatMessageContent(role=AuthorRole.ASSISTANT, items=[])
        )
        sdk.instance.send_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# The whole reply, not the first block of it
# ---------------------------------------------------------------------------


class TestTheWholeReplyIsReported:
    @pytest.mark.asyncio
    async def test_every_text_block_is_reported_not_just_the_first(self):
        """`ChatMessageContent.content` returns "the first TextContent's text"
        and drops the rest, so reading it would truncate a split reply."""
        sdk = _streaming_sdk(True)
        await _thread(sdk).on_new_message(
            ChatMessageContent(
                role=AuthorRole.ASSISTANT,
                items=[TextContent(text="It ships tomorrow. "),
                       TextContent(text="Tracking follows.")],
            )
        )
        assert sdk.instance.send_message.await_args.kwargs["content"] == (
            "It ships tomorrow. Tracking follows."
        )


# ---------------------------------------------------------------------------
# The tool call and its result
# ---------------------------------------------------------------------------


class TestTheToolCallAndItsResult:
    def test_the_filter_is_registered_on_the_kernel(self):
        kernel, _ = _kernel_with_tool()
        attach_synap_filters(
            kernel, _streaming_sdk(), conversation_id="conv-1", user_id="u1"
        )
        assert len(kernel.auto_function_invocation_filters) == 1

    def test_attach_rejects_something_that_is_not_a_kernel(self):
        with pytest.raises(ValueError, match="add_filter"):
            attach_synap_filters(
                object(), _streaming_sdk(), conversation_id="c", user_id="u"
            )

    @pytest.mark.asyncio
    async def test_the_call_is_reported_before_the_tool_runs(self):
        """A tool call is what the agent is ABOUT to need. Reporting it only
        after the tool returns throws away the whole window."""
        seen = []
        sdk = _streaming_sdk(True)
        sdk.instance.record_tool_call = AsyncMock(
            side_effect=lambda *a, **k: seen.append("call")
        )
        kernel, fn = _kernel_with_tool()
        attach_synap_filters(
            kernel, sdk, conversation_id="conv-1", user_id="u1", customer_id="c1"
        )

        async def inner(context):
            seen.append("tool")
            context.function_result = FunctionResult(
                function=_metadata(), value="shipped"
            )

        await _run_filter(kernel, _context(kernel, fn), inner)
        assert seen == ["call", "tool"]

    @pytest.mark.asyncio
    async def test_the_result_is_reported(self):
        sdk = _streaming_sdk(True)
        kernel, fn = _kernel_with_tool()
        attach_synap_filters(
            kernel, sdk, conversation_id="conv-1", user_id="u1", customer_id="c1"
        )
        await _run_filter(kernel, _context(kernel, fn), _returns("shipped"))
        sdk.instance.record_tool_result.assert_awaited_once()
        assert sdk.instance.record_tool_result.await_args.args[0] == "shipped"

    @pytest.mark.asyncio
    async def test_the_call_and_its_result_share_a_call_id(self):
        """Without this the anticipation agent sees a call and a result and
        cannot tell they are the same tool invocation."""
        sdk = _streaming_sdk(True)
        kernel, fn = _kernel_with_tool()
        attach_synap_filters(
            kernel, sdk, conversation_id="conv-1", user_id="u1", customer_id="c1"
        )
        await _run_filter(
            kernel, _context(kernel, fn, call_id="call_9"), _returns("shipped")
        )
        assert (sdk.instance.record_tool_call.await_args.kwargs["tool_call_id"]
                == sdk.instance.record_tool_result.await_args.kwargs["tool_call_id"]
                == "call_9")

    @pytest.mark.asyncio
    async def test_the_tool_name_is_reported(self):
        sdk = _streaming_sdk(True)
        kernel, fn = _kernel_with_tool()
        attach_synap_filters(
            kernel, sdk, conversation_id="conv-1", user_id="u1", customer_id="c1"
        )
        await _run_filter(kernel, _context(kernel, fn), _returns("shipped"))
        assert sdk.instance.record_tool_call.await_args.args[0] == "orders-lookup"

    @pytest.mark.asyncio
    async def test_no_result_means_no_result_event(self):
        """An integration passes through what the framework gives it and does
        not invent a result nobody produced."""
        sdk = _streaming_sdk(True)
        kernel, fn = _kernel_with_tool()
        attach_synap_filters(
            kernel, sdk, conversation_id="conv-1", user_id="u1", customer_id="c1"
        )
        await _run_filter(kernel, _context(kernel, fn), _returns_nothing)
        sdk.instance.record_tool_call.assert_awaited_once()
        sdk.instance.record_tool_result.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_raising_tool_still_reports_the_result_it_produced(self):
        """A filter further in can set the result and then blow up on the way
        out; the exception is the caller's to handle, the report is ours."""
        sdk = _streaming_sdk(True)
        kernel, fn = _kernel_with_tool()
        attach_synap_filters(
            kernel, sdk, conversation_id="conv-1", user_id="u1", customer_id="c1"
        )

        async def inner(context):
            context.function_result = FunctionResult(
                function=_metadata(), value="partial"
            )
            raise RuntimeError("tool blew up")

        with pytest.raises(RuntimeError, match="tool blew up"):
            await _run_filter(kernel, _context(kernel, fn), inner)
        assert sdk.instance.record_tool_result.await_args.args[0] == "partial"

    @pytest.mark.asyncio
    async def test_a_dead_stream_does_not_break_the_function_calling_loop(self):
        sdk = _streaming_sdk(True)
        sdk.instance.record_tool_call = AsyncMock(side_effect=RuntimeError("boom"))
        sdk.instance.record_tool_result = AsyncMock(side_effect=RuntimeError("boom"))
        kernel, fn = _kernel_with_tool()
        attach_synap_filters(
            kernel, sdk, conversation_id="conv-1", user_id="u1", customer_id="c1"
        )
        await _run_filter(kernel, _context(kernel, fn), _returns("shipped"))

    @pytest.mark.asyncio
    async def test_the_tool_still_runs_with_no_stream_open(self):
        sdk = _streaming_sdk(False)
        kernel, fn = _kernel_with_tool()
        attach_synap_filters(
            kernel, sdk, conversation_id="conv-1", user_id="u1", customer_id="c1"
        )
        ran = []

        async def inner(context):
            ran.append(True)

        await _run_filter(kernel, _context(kernel, fn), inner)
        assert ran == [True]
        sdk.instance.record_tool_call.assert_not_awaited()


# ---------------------------------------------------------------------------
# Tool args are the parsed dict
# ---------------------------------------------------------------------------


class TestToolArgsAreTheParsedDict:
    """`FunctionCallContent.arguments` is `str | Mapping | None`: a string
    whenever the model produced it. `report_tool_call` forwards only a `dict`,
    so the string would have arrived as no arguments at all."""

    @pytest.mark.asyncio
    async def test_the_json_the_model_emitted_is_parsed(self):
        sdk = _streaming_sdk(True)
        kernel, fn = _kernel_with_tool()
        attach_synap_filters(
            kernel, sdk, conversation_id="conv-1", user_id="u1", customer_id="c1"
        )
        await _run_filter(
            kernel,
            _context(kernel, fn, arguments='{"pnr": "QX41RT"}'),
            _returns("shipped"),
        )
        assert sdk.instance.record_tool_call.await_args.args[1] == {"pnr": "QX41RT"}

    @pytest.mark.asyncio
    async def test_the_reported_args_are_json_serialisable(self):
        sdk = _streaming_sdk(True)
        kernel, fn = _kernel_with_tool()
        attach_synap_filters(
            kernel, sdk, conversation_id="conv-1", user_id="u1", customer_id="c1"
        )
        await _run_filter(kernel, _context(kernel, fn), _returns("shipped"))
        json.dumps(sdk.instance.record_tool_call.await_args.args[1])

    def test_a_mapping_that_is_not_a_dict_is_converted(self):
        """`parse_arguments` is typed to return a `Mapping`, and
        `report_tool_call` drops anything that is not a `dict`."""
        stub = MagicMock()
        stub.arguments = "ignored"
        stub.parse_arguments = lambda: MappingProxyType({"pnr": "QX41RT"})
        args = _tool_args(stub)
        assert type(args) is dict
        assert args == {"pnr": "QX41RT"}

    def test_arguments_that_are_not_json_fall_back_to_the_raw_text(self):
        assert _tool_args(
            FunctionCallContent(id="c", name="orders-lookup", arguments="not json")
        ) == {"input": "not json"}

    def test_a_tool_taking_no_arguments_reports_an_empty_dict(self):
        assert _tool_args(
            FunctionCallContent(id="c", name="orders-lookup", arguments="")
        ) == {}



# ---------------------------------------------------------------------------
# The tool result survives serialisation
# ---------------------------------------------------------------------------


class TestTheToolResultSurvivesSerialisation:
    """The SDK does `json.dumps` on the result and `stream_events._send`
    swallows every exception by design, so a result it cannot serialise is
    not an error anybody sees — it is a result that silently never arrives.
    Semantic Kernel hands the filter a `FunctionResult`, a pydantic model
    carrying `KernelFunctionMetadata`, and `json.dumps` cannot touch it."""

    def test_the_raw_FunctionResult_really_is_unserialisable(self):
        """The premise, pinned. If this ever stops being true the unwrapping
        below is dead weight rather than a fix."""
        with pytest.raises(TypeError):
            json.dumps(FunctionResult(function=_metadata(), value="shipped"))

    def test_a_FunctionResult_is_unwrapped_to_its_value(self):
        assert _tool_output(
            FunctionResult(function=_metadata(), value="shipped")
        ) == "shipped"

    @pytest.mark.asyncio
    async def test_the_reported_result_is_json_serialisable(self):
        sdk = _streaming_sdk(True)
        kernel, fn = _kernel_with_tool()
        attach_synap_filters(
            kernel, sdk, conversation_id="conv-1", user_id="u1", customer_id="c1"
        )
        await _run_filter(kernel, _context(kernel, fn), _returns("shipped"))
        json.dumps(sdk.instance.record_tool_result.await_args.args[0])

    @pytest.mark.asyncio
    async def test_a_list_of_kernel_content_is_unwrapped_to_text(self):
        sdk = _streaming_sdk(True)
        kernel, fn = _kernel_with_tool()
        attach_synap_filters(
            kernel, sdk, conversation_id="conv-1", user_id="u1", customer_id="c1"
        )
        await _run_filter(
            kernel, _context(kernel, fn),
            _returns([TextContent(text="shipped")]),
        )
        sent = sdk.instance.record_tool_result.await_args.args[0]
        json.dumps(sent)
        assert sent == ["shipped"]

    @pytest.mark.asyncio
    async def test_a_value_json_cannot_touch_is_stringified_not_dropped(self):
        class NotSerialisable:
            def __str__(self):
                return "opaque-result"

        sdk = _streaming_sdk(True)
        kernel, fn = _kernel_with_tool()
        attach_synap_filters(
            kernel, sdk, conversation_id="conv-1", user_id="u1", customer_id="c1"
        )
        await _run_filter(
            kernel, _context(kernel, fn), _returns(NotSerialisable())
        )
        sent = sdk.instance.record_tool_result.await_args.args[0]
        json.dumps(sent)
        assert sent == "opaque-result"

    @pytest.mark.asyncio
    async def test_a_dict_result_is_left_alone(self):
        sdk = _streaming_sdk(True)
        kernel, fn = _kernel_with_tool()
        attach_synap_filters(
            kernel, sdk, conversation_id="conv-1", user_id="u1", customer_id="c1"
        )
        await _run_filter(
            kernel, _context(kernel, fn), _returns({"status": "shipped"})
        )
        assert sdk.instance.record_tool_result.await_args.args[0] == {
            "status": "shipped"
        }


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------


class TestReasoning:
    @pytest.mark.asyncio
    async def test_a_reasoning_block_is_reported(self):
        sdk = _streaming_sdk(True)
        await _thread(sdk).on_new_message(
            ChatMessageContent(
                role=AuthorRole.ASSISTANT,
                items=[ReasoningContent(text="check the booking first"),
                       TextContent(text="one moment")],
            )
        )
        sdk.instance.record_thinking.assert_awaited_once()
        assert sdk.instance.record_thinking.await_args.args[0] == (
            "check the booking first"
        )

    @pytest.mark.asyncio
    async def test_an_empty_reasoning_block_reports_nothing(self):
        sdk = _streaming_sdk(True)
        await _thread(sdk).on_new_message(
            ChatMessageContent(
                role=AuthorRole.ASSISTANT,
                items=[ReasoningContent(text="   "), TextContent(text="hi")],
            )
        )
        sdk.instance.record_thinking.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_turn_with_no_reasoning_is_an_ordinary_turn(self):
        sdk = _streaming_sdk(True)
        await _thread(sdk).on_new_message(
            ChatMessageContent(role=AuthorRole.ASSISTANT, content="hi")
        )
        sdk.instance.record_thinking.assert_not_awaited()
        sdk.instance.send_message.assert_awaited_once()


# ---------------------------------------------------------------------------
# The end of the thread
# ---------------------------------------------------------------------------


class TestTheEndOfTheThread:
    @pytest.mark.asyncio
    async def test_deleting_the_thread_ends_the_synap_session(self):
        sdk = _streaming_sdk(True)
        thread = _thread(sdk)
        await thread.create()
        await thread.delete()
        sdk.instance.end_session.assert_awaited_once_with("conv-1")

    @pytest.mark.asyncio
    async def test_deleting_still_clears_the_history(self):
        sdk = _streaming_sdk(True)
        history = ChatHistory()
        thread = _thread(sdk, chat_history=history)
        await thread.on_new_message(
            ChatMessageContent(role=AuthorRole.USER, content="hello")
        )
        await thread.delete()
        assert history.messages == []
