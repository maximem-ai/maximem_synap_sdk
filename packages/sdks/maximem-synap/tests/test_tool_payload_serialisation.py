"""A tool result the caller's framework returned must still reach the server.

The bug this pins was silent twice over. `send_message` serialised
`tool_result` with a bare `json.dumps`, which raises on anything that is not
made of dicts, lists, strings and numbers. Every integration reports through
`synap_integrations_common.stream_events._send`, which swallows exceptions by
design so that a telemetry call can never break somebody's agent loop. So the
`TypeError` went nowhere, nothing logged above debug, and the whole
`tool_result` event was never sent.

Found under LangGraph, whose `ToolNode` hands back a `ToolMessage`: every tool
result in the run vanished and the anticipation agent never learned what any
tool returned. A `ToolMessage` is only the common case. A pydantic model, a
dataclass, a `datetime`, a numpy array and a set all do the same thing, and so
does any object holding a reference cycle.

The assertions here are on **what the transport received**, never on whether
the call raised. The broken version did not raise either.
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass

import pytest

from maximem_synap.sdk import InstanceInterface, _json_or_text


class FakeTransport:
    def __init__(self):
        self.sent = []

    async def send(self, payload):
        self.sent.append(payload)

    async def send_session_control(self, **kwargs):
        return True


class FakeController:
    def __init__(self):
        self._transport = FakeTransport()
        self.is_listening = True

    async def stop(self):
        pass


def make_instance() -> InstanceInterface:
    inst = object.__new__(InstanceInterface)
    inst._sdk = type("FakeSDK", (), {
        "_check_customer_id": lambda *a, **k: None,
        "_st_store_active": lambda self=None: False,
        "_grpc_transport": None,
    })()
    inst._controller = FakeController()
    inst._open_sessions = {}
    return inst


def sent(inst):
    return inst._controller._transport.sent


# The shapes a real framework actually hands back. Each one raises under a bare
# json.dumps, and each one used to take its whole event down with it.

class ToolMessageLike:
    """LangChain's `ToolMessage`, near enough: content plus an id, no dict."""

    def __init__(self, content):
        self.content = content
        self.tool_call_id = "call_1"


@dataclass
class DataclassResult:
    rows: int
    label: str


class PydanticLike:
    """An object with a `__dict__` and nothing json knows about."""

    def __init__(self):
        self.field = "value"


def cyclic():
    """A cycle raises ValueError, which `default=` cannot rescue."""
    d = {"name": "root"}
    d["self"] = d
    return d


UNSERIALISABLE = [
    pytest.param(ToolMessageLike("the answer"), id="langchain ToolMessage"),
    pytest.param(DataclassResult(rows=3, label="x"), id="dataclass"),
    pytest.param(PydanticLike(), id="pydantic-like object"),
    pytest.param(_dt.datetime(2026, 9, 26, 12, 0), id="datetime"),
    pytest.param({"seen"}, id="set"),
    pytest.param(_dt.timedelta(seconds=90), id="timedelta"),
    pytest.param(cyclic(), id="reference cycle"),
    pytest.param(b"\x00bytes", id="bytes"),
]


class TestTheEventSurvivesWhateverTheToolReturned:
    @pytest.mark.parametrize("result", UNSERIALISABLE)
    @pytest.mark.asyncio
    async def test_an_unserialisable_result_is_still_sent(self, result):
        inst = make_instance()
        await inst.record_tool_result(
            result, tool_name="search", conversation_id="c1", user_id="u1")

        # The event exists at all. This is the whole bug: it did not.
        assert len(sent(inst)) == 1, "the event was dropped, which is the bug"
        assert sent(inst)[0]["event_type"] == "tool_result"

    @pytest.mark.parametrize("result", UNSERIALISABLE)
    @pytest.mark.asyncio
    async def test_and_it_carries_something_a_reader_can_use(self, result):
        inst = make_instance()
        await inst.record_tool_result(result, conversation_id="c1", user_id="u1")

        body = sent(inst)[0]["tool_result_json"]
        assert isinstance(body, str) and body != ""

    @pytest.mark.parametrize("args", [
        pytest.param({"when": _dt.datetime(2026, 9, 26)}, id="datetime in args"),
        pytest.param({"obj": PydanticLike()}, id="object in args"),
        pytest.param(cyclic(), id="cycle in args"),
    ])
    @pytest.mark.asyncio
    async def test_a_tool_call_with_unserialisable_args_is_still_sent(self, args):
        """`tool_args_json` had exactly the same bare dumps. A tool called with
        a `datetime` argument lost its tool_call event, so the tool_result that
        followed had nothing to pair with."""
        inst = make_instance()
        await inst.record_tool_call(
            "search", args, conversation_id="c1", user_id="u1")

        assert len(sent(inst)) == 1
        assert sent(inst)[0]["event_type"] == "tool_call"
        assert sent(inst)[0]["tool_args_json"] != ""


class TestNothingThatAlreadyWorkedChanged:
    """The degrade must not touch the ordinary path. A tool result is read by a
    model, so a dict turning into its `str()` would quietly swap JSON for
    Python repr on every well-behaved tool in the fleet."""

    @pytest.mark.asyncio
    async def test_a_dict_result_is_still_json(self):
        inst = make_instance()
        await inst.record_tool_result(
            {"rows": 3, "ok": True}, conversation_id="c1", user_id="u1")

        body = sent(inst)[0]["tool_result_json"]
        assert json.loads(body) == {"rows": 3, "ok": True}
        assert "'" not in body, "this is Python repr, not JSON"

    @pytest.mark.asyncio
    async def test_a_string_result_still_travels_unquoted(self):
        """json.dumps would wrap a tool's plain-text answer in quotes and the
        agent would read the quotes as part of the result."""
        inst = make_instance()
        await inst.record_tool_result(
            "plain text", conversation_id="c1", user_id="u1")

        assert sent(inst)[0]["tool_result_json"] == "plain text"

    @pytest.mark.asyncio
    async def test_a_list_result_is_still_json(self):
        inst = make_instance()
        await inst.record_tool_result([1, 2, 3], conversation_id="c1", user_id="u1")
        assert json.loads(sent(inst)[0]["tool_result_json"]) == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_ordinary_tool_args_are_still_json(self):
        inst = make_instance()
        await inst.record_tool_call(
            "search", {"q": "hello"}, conversation_id="c1", user_id="u1")
        assert json.loads(sent(inst)[0]["tool_args_json"]) == {"q": "hello"}


class TestTheHelperItself:
    """`_json_or_text` is the one place the degrade happens, so pin its rungs.
    Each rung exists because the one above it cannot handle that input."""

    def test_plain_data_takes_the_first_rung_unchanged(self):
        assert _json_or_text({"a": [1, None, True]}) == '{"a": [1, null, true]}'

    def test_an_unserialisable_leaf_takes_the_default_str_rung(self):
        # Still JSON, with the leaf stringified. Better than str() of the whole
        # object, which would lose the structure around it.
        out = json.loads(_json_or_text({"when": _dt.datetime(2026, 9, 26)}))
        assert out == {"when": "2026-09-26 00:00:00"}

    def test_a_cycle_takes_the_last_rung(self):
        # `default=` never sees a cycle; json raises ValueError before calling
        # it. Without the outer fallback this rung raises and the event is lost.
        out = _json_or_text(cyclic())
        assert isinstance(out, str) and "root" in out

    def test_it_never_raises_on_anything_we_could_think_of(self):
        for value in [*[p.values[0] for p in UNSERIALISABLE],
                      None, object(), lambda: None, float("nan")]:
            assert isinstance(_json_or_text(value), str)
