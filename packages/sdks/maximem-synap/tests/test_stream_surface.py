"""The stream surface: sessions, typed tool events, and the role trap.

Three things this pins, all of which were wrong or missing:

1. Neither SDK ever sent `session_control`. The server runs its warm-up
   prefetch on `session_start`, so the first turn of every conversation was a
   cold fetch by construction, and `session_end` is what writes the turn's
   telemetry row — without it the last turn of a conversation never produced
   one. The developer should not have to remember a lifecycle call, so the SDK
   opens and closes sessions itself.

2. `role` and `event_type` have to agree and nothing checked. `record_thinking`
   sent role="assistant", which the server read before the event type, so every
   reasoning step was filed as the assistant's reply. The typed methods set
   both, and reasoning now names no role at all: correct against a server that
   has the classifier fix and one that does not.

3. `event_type` was a free string. A typo travelled to the server, fell through
   every classifier into UNKNOWN and was acted on by nothing.
"""
from __future__ import annotations

import asyncio

import pytest

from maximem_synap import sdk as sdk_module
from maximem_synap.sdk import KNOWN_EVENT_TYPES, InstanceInterface


class FakeTransport:
    def __init__(self):
        self.sent = []
        self.controls = []
        self.session_control_fails = False

    async def send(self, payload):
        self.sent.append(payload)

    async def send_session_control(self, **kwargs):
        self.controls.append(kwargs)
        return not self.session_control_fails


class FakeController:
    def __init__(self):
        self._transport = FakeTransport()
        self.is_listening = True
        self.stopped = False

    async def stop(self):
        self.stopped = True


def make_instance() -> InstanceInterface:
    """An InstanceInterface with the transport replaced, nothing else stubbed."""
    inst = object.__new__(InstanceInterface)
    inst._sdk = type("FakeSDK", (), {
        "_check_customer_id": lambda *a, **k: None,
        "_st_store_active": lambda self=None: False,
        "_grpc_transport": None,
    })()
    inst._controller = FakeController()
    inst._open_sessions = {}
    return inst


def controls(inst):
    return [(c["action"], c["conversation_id"]) for c in inst._controller._transport.controls]


def sent(inst):
    return inst._controller._transport.sent


class TestTheSessionOpensItself:
    @pytest.mark.asyncio
    async def test_the_first_event_opens_a_session(self):
        inst = make_instance()
        await inst.send_message("hello", conversation_id="c1", user_id="u1")
        assert controls(inst) == [("start", "c1")]

    @pytest.mark.asyncio
    async def test_the_second_event_does_not_open_another(self):
        inst = make_instance()
        await inst.send_message("hello", conversation_id="c1", user_id="u1")
        await inst.send_message("again", conversation_id="c1", user_id="u1")
        assert controls(inst) == [("start", "c1")]

    @pytest.mark.asyncio
    async def test_a_second_conversation_gets_its_own(self):
        inst = make_instance()
        await inst.send_message("a", conversation_id="c1", user_id="u1")
        await inst.send_message("b", conversation_id="c2", user_id="u1")
        assert controls(inst) == [("start", "c1"), ("start", "c2")]

    @pytest.mark.asyncio
    async def test_the_session_id_rides_on_the_events(self):
        inst = make_instance()
        await inst.send_message("hello", conversation_id="c1", user_id="u1")
        assert sent(inst)[0]["session_id"] == inst._open_sessions["c1"]

    @pytest.mark.asyncio
    async def test_a_caller_supplied_session_id_wins(self):
        inst = make_instance()
        await inst.send_message(
            "hello", conversation_id="c1", user_id="u1", session_id="mine",
        )
        assert sent(inst)[0]["session_id"] == "mine"

    @pytest.mark.asyncio
    async def test_an_event_with_no_conversation_opens_nothing(self):
        inst = make_instance()
        await inst.send_message("hello", user_id="u1")
        assert controls(inst) == []

    @pytest.mark.asyncio
    async def test_a_failed_open_does_not_stop_the_event(self):
        """The turn still goes out; the next event tries the session again."""
        inst = make_instance()
        inst._controller._transport.session_control_fails = True
        await inst.send_message("hello", conversation_id="c1", user_id="u1")
        assert len(sent(inst)) == 1
        assert "c1" not in inst._open_sessions
        await inst.send_message("again", conversation_id="c1", user_id="u1")
        assert controls(inst) == [("start", "c1"), ("start", "c1")]

    @pytest.mark.asyncio
    async def test_stop_listening_closes_what_it_opened(self):
        inst = make_instance()
        await inst.send_message("hello", conversation_id="c1", user_id="u1")
        await inst.stop_listening()
        assert controls(inst) == [("start", "c1"), ("end", "c1")]
        assert inst._controller.stopped is True

    @pytest.mark.asyncio
    async def test_end_session_closes_one_conversation(self):
        inst = make_instance()
        await inst.send_message("a", conversation_id="c1", user_id="u1")
        await inst.send_message("b", conversation_id="c2", user_id="u1")
        await inst.end_session("c1")
        assert controls(inst)[-1] == ("end", "c1")
        assert "c2" in inst._open_sessions

    @pytest.mark.asyncio
    async def test_ending_an_unopened_session_is_a_no_op(self):
        inst = make_instance()
        await inst.end_session("never-started")
        assert controls(inst) == []


class TestTheTypedToolMethods:
    @pytest.mark.asyncio
    async def test_a_tool_call_names_the_role_for_you(self):
        inst = make_instance()
        await inst.record_tool_call(
            "lookup_order", {"id": 42}, tool_call_id="call_1",
            conversation_id="c1", user_id="u1",
        )
        payload = sent(inst)[0]
        assert payload["event_type"] == "tool_call"
        assert payload["role"] == "assistant"
        assert payload["tool_name"] == "lookup_order"
        assert payload["tool_args_json"] == '{"id": 42}'
        assert payload["tool_call_id"] == "call_1"

    @pytest.mark.asyncio
    async def test_a_tool_result_travels_in_its_own_field(self):
        inst = make_instance()
        await inst.record_tool_result(
            {"status": "shipped"}, tool_name="lookup_order",
            tool_call_id="call_1", conversation_id="c1", user_id="u1",
        )
        payload = sent(inst)[0]
        assert payload["event_type"] == "tool_result"
        assert payload["role"] == "tool"
        assert payload["tool_result_json"] == '{"status": "shipped"}'
        assert payload["content"] == "", "a result is not something a person said"

    @pytest.mark.asyncio
    async def test_a_plain_text_result_keeps_its_quotes_off(self):
        inst = make_instance()
        await inst.record_tool_result("shipped", conversation_id="c1", user_id="u1")
        assert sent(inst)[0]["tool_result_json"] == "shipped"

    @pytest.mark.asyncio
    async def test_a_call_and_a_result_can_be_tied_together(self):
        inst = make_instance()
        await inst.record_tool_call("lookup", tool_call_id="call_1",
                                    conversation_id="c1", user_id="u1")
        await inst.record_tool_result("ok", tool_call_id="call_1",
                                      conversation_id="c1", user_id="u1")
        assert sent(inst)[0]["tool_call_id"] == sent(inst)[1]["tool_call_id"]


class TestReasoningNamesNoRole:
    @pytest.mark.asyncio
    async def test_it_does_not_claim_to_be_the_assistants_reply(self):
        """role="assistant" is what the server read before the event type, so
        every reasoning step was filed as the final answer to the user."""
        inst = make_instance()
        await inst.record_thinking("plan the lookup", conversation_id="c1", user_id="u1")
        payload = sent(inst)[0]
        assert payload["event_type"] == "agent_thinking"
        assert payload["role"] == ""

    @pytest.mark.asyncio
    async def test_the_two_metadata_keys_still_travel(self):
        inst = make_instance()
        await inst.record_thinking(
            "plan", conversation_id="c1", user_id="u1",
            step_index=3, thought_type="self_correction",
        )
        md = sent(inst)[0]["metadata"]
        assert md["step_index"] == "3"
        assert md["thought_type"] == "self_correction"

    @pytest.mark.asyncio
    async def test_a_caller_metadata_key_is_not_overwritten(self):
        inst = make_instance()
        await inst.record_thinking(
            "plan", conversation_id="c1", user_id="u1", step_index=3,
            metadata={"step_index": "mine"},
        )
        assert sent(inst)[0]["metadata"]["step_index"] == "mine"


class TestAnUnknownEventTypeIsRefused:
    @pytest.mark.asyncio
    async def test_a_typo_raises_rather_than_travelling(self):
        inst = make_instance()
        with pytest.raises(ValueError) as excinfo:
            await inst.send_message("x", event_type="tool_reslt", conversation_id="c1")
        assert "tool_reslt" in str(excinfo.value)
        assert "tool_result" in str(excinfo.value), "the message should list the real ones"
        assert sent(inst) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("event_type", sorted(KNOWN_EVENT_TYPES))
    async def test_every_known_type_is_accepted(self, event_type):
        inst = make_instance()
        await inst.send_message("x", event_type=event_type, conversation_id="c1")
        assert sent(inst)[0]["event_type"] == event_type

    def test_the_typed_methods_only_send_known_types(self):
        assert {"tool_call", "tool_result", "agent_thinking"} <= KNOWN_EVENT_TYPES


class TestASessionIsOnlyOpenedWhenItCanBeAccepted:
    """The server refuses a session_start with no user_id, and the transport
    only reports whether the message was WRITTEN. Opening one on an event that
    has no user_id marked the conversation as open, never retried, and left
    `end_session` naming a session the server never had."""

    @pytest.mark.asyncio
    async def test_no_user_id_opens_nothing(self):
        inst = make_instance()
        await inst.send_message("hello", conversation_id="c1")
        assert controls(inst) == []
        assert len(sent(inst)) == 1, "the event itself must still go out"

    @pytest.mark.asyncio
    async def test_a_later_event_with_the_ids_opens_it(self):
        inst = make_instance()
        await inst.send_message("hello", conversation_id="c1")
        await inst.send_message("again", conversation_id="c1", user_id="u1")
        assert controls(inst) == [("start", "c1")]

    @pytest.mark.asyncio
    async def test_a_tool_event_without_ids_does_not_open_a_bogus_session(self):
        """The shape that hits this: a framework hook that knows the tool but
        not the user, firing before the turn's first user message."""
        inst = make_instance()
        await inst.record_tool_call("lookup", conversation_id="c1")
        assert controls(inst) == []
