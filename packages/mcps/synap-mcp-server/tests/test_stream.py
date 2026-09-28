"""Reporting a logged exchange as a conversation turn (synap_mcp_server.stream).

An MCP server is not an agent: it exposes tools to somebody else's agent and
is handed one stateless HTTP call at a time. Four of the five things an
integration normally reports — tool calls, tool results, reasoning, session
end — never reach this process at all, so nothing here reports them.

What does reach it is the turn: `log_exchange` is handed the user's message
and the assistant's reply as arguments. These tests pin what happens to it.

Contract under test:

- the turn goes to ``POST /v1/events/batch``, the HTTP door onto the same
  listening path the gRPC stream feeds, because a stateless multi-tenant
  server has nowhere to hold a stream;
- ``log_exchange`` still writes its long-range document, because that is a
  different store and the ``ingestion_id`` half of the tool depends on it;
- it is off unless the deployment switched it on;
- the user turn goes out before the assistant turn it answers;
- a failure never changes what ``log_exchange`` answers the agent;
- a refusal is logged with the server's own reason, never swallowed.
"""

import dataclasses
import json
import logging

import httpx
import pytest
import respx

from synap_mcp_server import stream
from synap_mcp_server.client import SynapAPIError
from tests.conftest import API_BASE

pytestmark = pytest.mark.asyncio

EVENTS_URL = f"{API_BASE}/v1/events/batch"


@pytest.fixture
def reporting_on(monkeypatch):
    """Switch the turn report on for the duration of a test."""
    monkeypatch.setattr(
        stream, "settings", dataclasses.replace(stream.settings, stream_events=True)
    )
    return True


def _accepted(n=2):
    return httpx.Response(
        200,
        json={
            "accepted": n,
            "duplicate": 0,
            "rejected": 0,
            "results": [{"event_id": "", "status": "accepted"} for _ in range(n)],
        },
    )


def _sent(route):
    return json.loads(route.calls.last.request.read().decode())["events"]


async def _report(**overrides):
    kwargs = {
        "user_message": "where is my order",
        "assistant_message": "it ships tomorrow",
        "conversation_id": "conv-1",
        "user_id": "u1",
        "customer_id": "cus1",
        "ingestion_id": "ing_1",
    }
    kwargs.update(overrides)
    return await stream.report_exchange(**kwargs)


# ---------------------------------------------------------------------------
# It is off unless a deployment turns it on
# ---------------------------------------------------------------------------


class TestItIsOffByDefault:
    @respx.mock
    async def test_nothing_is_reported_with_the_switch_off(self, with_token):
        route = respx.post(EVENTS_URL).mock(return_value=_accepted())
        assert await _report() is False
        assert not route.called

    @respx.mock
    async def test_the_switch_turns_it_on(self, with_token, reporting_on):
        route = respx.post(EVENTS_URL).mock(return_value=_accepted())
        assert await _report() is True
        assert route.called

    async def test_the_default_setting_is_off(self):
        """A behaviour and cost change for every no-code caller on a hosted
        server is a deployment's decision, not a default."""
        from synap_mcp_server.config import Settings

        assert Settings().stream_events is False


# ---------------------------------------------------------------------------
# The turn itself
# ---------------------------------------------------------------------------


class TestTheTurn:
    @respx.mock
    async def test_both_halves_go_out_as_one_batch(self, with_token, reporting_on):
        route = respx.post(EVENTS_URL).mock(return_value=_accepted())
        await _report()
        events = _sent(route)
        assert [e["event_type"] for e in events] == [
            "user_message",
            "assistant_message",
        ]

    @respx.mock
    async def test_the_assistant_turn_comes_last(self, with_token, reporting_on):
        """It is the moment anticipation acts on, and it has to arrive after
        the message it answers."""
        route = respx.post(EVENTS_URL).mock(return_value=_accepted())
        await _report()
        assert _sent(route)[-1]["event_type"] == "assistant_message"

    @respx.mock
    async def test_the_content_is_what_the_agent_forwarded(
        self, with_token, reporting_on
    ):
        route = respx.post(EVENTS_URL).mock(return_value=_accepted())
        await _report()
        events = _sent(route)
        assert events[0]["content"] == "where is my order"
        assert events[0]["role"] == "user"
        assert events[1]["content"] == "it ships tomorrow"
        assert events[1]["role"] == "assistant"

    @respx.mock
    async def test_a_log_with_no_reply_yet_reports_only_the_user_turn(
        self, with_token, reporting_on
    ):
        route = respx.post(EVENTS_URL).mock(return_value=_accepted(1))
        await _report(assistant_message="")
        events = _sent(route)
        assert [e["event_type"] for e in events] == ["user_message"]

    @respx.mock
    async def test_an_empty_exchange_is_not_reported(self, with_token, reporting_on):
        route = respx.post(EVENTS_URL).mock(return_value=_accepted(0))
        assert await _report(user_message="", assistant_message="") is False
        assert not route.called

    @respx.mock
    async def test_the_two_halves_carry_different_dedupe_ids(
        self, with_token, reporting_on
    ):
        """One id for both would make the server call the assistant turn a
        duplicate of the user turn and drop it."""
        route = respx.post(EVENTS_URL).mock(return_value=_accepted())
        await _report()
        events = _sent(route)
        assert events[0]["event_id"] != events[1]["event_id"]

    @respx.mock
    async def test_a_retry_of_the_same_exchange_reuses_its_ids(
        self, with_token, reporting_on
    ):
        """Keyed off the ingestion the write returned, so the server dedupes a
        replay instead of doubling the turn."""
        route = respx.post(EVENTS_URL).mock(return_value=_accepted())
        await _report()
        first = [e["event_id"] for e in _sent(route)]
        await _report()
        assert [e["event_id"] for e in _sent(route)] == first


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


class TestScope:
    @respx.mock
    async def test_the_ids_are_carried_on_every_event(self, with_token, reporting_on):
        route = respx.post(EVENTS_URL).mock(return_value=_accepted())
        await _report()
        for event in _sent(route):
            assert event["conversation_id"] == "conv-1"
            assert event["user_id"] == "u1"
            assert event["customer_id"] == "cus1"

    @respx.mock
    async def test_an_absent_customer_id_is_left_out_not_sent_empty(
        self, with_token, reporting_on
    ):
        """A B2C instance refuses a customer_id outright."""
        route = respx.post(EVENTS_URL).mock(return_value=_accepted())
        await _report(customer_id=None)
        for event in _sent(route):
            assert "customer_id" not in event

    @respx.mock
    async def test_the_single_user_shape_is_skipped_not_sent_to_be_refused(
        self, with_token, reporting_on
    ):
        """`log_exchange` documents omitting user_id when every conversation
        is the same person. The events route rejects that, so it is not sent."""
        route = respx.post(EVENTS_URL).mock(return_value=_accepted())
        assert await _report(user_id=None) is False
        assert not route.called

    @respx.mock
    async def test_no_conversation_id_is_skipped_too(self, with_token, reporting_on):
        route = respx.post(EVENTS_URL).mock(return_value=_accepted())
        assert await _report(conversation_id=None) is False
        assert not route.called


# ---------------------------------------------------------------------------
# Best effort
# ---------------------------------------------------------------------------


class TestItIsBestEffort:
    @respx.mock
    async def test_a_server_error_does_not_raise(self, with_token, reporting_on):
        respx.post(EVENTS_URL).mock(return_value=httpx.Response(500, text="boom"))
        assert await _report() is False

    @respx.mock
    async def test_a_server_error_is_logged_in_the_servers_own_words(
        self, with_token, reporting_on, caplog
    ):
        """"Reporting failed" with a stack trace is what you write for a bug
        in this file. An upstream status is a different thing and has to read
        like one, or nobody can tell which end is broken."""
        respx.post(EVENTS_URL).mock(return_value=httpx.Response(500, text="boom"))
        with caplog.at_level(logging.WARNING, logger="synap-mcp"):
            await _report()
        assert "could not report the turn" in caplog.text
        assert "500" in caplog.text

    @respx.mock
    async def test_a_network_failure_does_not_raise(self, with_token, reporting_on):
        respx.post(EVENTS_URL).mock(side_effect=httpx.ConnectError("no route"))
        assert await _report() is False

    async def test_a_missing_token_does_not_raise(self, no_token, reporting_on):
        assert await _report() is False

    @respx.mock
    async def test_an_unexpected_failure_does_not_raise(
        self, with_token, reporting_on, monkeypatch
    ):
        async def _explode(_events):
            raise RuntimeError("something nobody predicted")

        monkeypatch.setattr(stream, "send_events", _explode)
        assert await _report() is False


class TestARefusalIsNotSwallowed:
    @respx.mock
    async def test_the_servers_own_reason_is_logged(
        self, with_token, reporting_on, caplog
    ):
        """A refusal is a caller-shape problem, and it looks exactly like
        'anticipation is just quiet' until somebody goes looking."""
        respx.post(EVENTS_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "accepted": 0,
                    "duplicate": 0,
                    "rejected": 1,
                    "results": [
                        {
                            "event_id": "mcp:ing_1:user",
                            "status": "rejected",
                            "reason": "customer_id is required on a B2B instance",
                        }
                    ],
                },
            )
        )
        with caplog.at_level(logging.WARNING, logger="synap-mcp"):
            await _report()
        assert "customer_id is required on a B2B instance" in caplog.text

    @respx.mock
    async def test_an_accepted_batch_logs_no_warning(
        self, with_token, reporting_on, caplog
    ):
        respx.post(EVENTS_URL).mock(return_value=_accepted())
        with caplog.at_level(logging.WARNING, logger="synap-mcp"):
            await _report()
        assert caplog.text == ""

    @respx.mock
    async def test_a_duplicate_is_not_reported_as_a_problem(
        self, with_token, reporting_on, caplog
    ):
        """A replay is the dedupe working, not a failure."""
        respx.post(EVENTS_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "accepted": 0,
                    "duplicate": 2,
                    "rejected": 0,
                    "results": [{"event_id": "a", "status": "duplicate"}],
                },
            )
        )
        with caplog.at_level(logging.WARNING, logger="synap-mcp"):
            await _report()
        assert caplog.text == ""


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


class TestTheRoute:
    @respx.mock
    async def test_it_posts_to_the_events_batch_route(self, with_token, reporting_on):
        """The HTTP door onto the listening path. There is no gRPC stream to
        use: this server is stateless and every request is a different key."""
        route = respx.post(EVENTS_URL).mock(return_value=_accepted())
        await _report()
        assert route.called

    @respx.mock
    async def test_it_forwards_the_callers_bearer_token(
        self, with_token, reporting_on
    ):
        route = respx.post(EVENTS_URL).mock(return_value=_accepted())
        await _report()
        assert (
            route.calls.last.request.headers["authorization"]
            == f"Bearer {with_token}"
        )

    @respx.mock
    async def test_every_event_is_stamped_as_coming_from_this_server(
        self, with_token, reporting_on
    ):
        route = respx.post(EVENTS_URL).mock(return_value=_accepted())
        await _report()
        for event in _sent(route):
            assert event["metadata"]["source"] == "mcp-server"


# ---------------------------------------------------------------------------
# log_exchange end to end
# ---------------------------------------------------------------------------


def _text(result):
    """Flatten a call_tool result to text (see tests/test_protocol.py)."""
    blocks = result[0] if isinstance(result, tuple) else result
    return "".join(getattr(b, "text", "") for b in blocks)


class TestLogExchangeReportsTheTurn:
    @respx.mock
    async def test_the_turn_is_reported_after_the_write(
        self, with_token, reporting_on
    ):
        from synap_mcp_server.server import mcp

        respx.post(f"{API_BASE}/api/v1/memories/create").mock(
            return_value=httpx.Response(200, json={"ingestion_id": "ing_7"})
        )
        events = respx.post(EVENTS_URL).mock(return_value=_accepted())
        await mcp.call_tool(
            "log_exchange",
            {
                "user_message": "where is my order",
                "assistant_message": "it ships tomorrow",
                "conversation_id": "conv-1",
                "user_id": "u1",
            },
        )
        assert events.called
        assert _sent(events)[0]["event_id"] == "mcp:ing_7:user"

    @respx.mock
    async def test_the_long_range_write_still_happens(self, with_token, reporting_on):
        """The two are different stores. Dropping the create would leave
        check_memory_status and wait_for_processing with no ingestion to talk
        about, and nothing would be extracted at all."""
        from synap_mcp_server.server import mcp

        create = respx.post(f"{API_BASE}/api/v1/memories/create").mock(
            return_value=httpx.Response(200, json={"ingestion_id": "ing_7"})
        )
        respx.post(EVENTS_URL).mock(return_value=_accepted())
        text = _text(
            await mcp.call_tool(
                "log_exchange",
                {
                    "user_message": "hi",
                    "conversation_id": "conv-1",
                    "user_id": "u1",
                },
            )
        )
        assert create.called
        assert "ing_7" in text

    @respx.mock
    async def test_a_failed_report_does_not_change_what_the_agent_is_told(
        self, with_token, reporting_on
    ):
        """The write already succeeded. A telemetry miss is not the agent's
        problem and must not read as one."""
        from synap_mcp_server.server import mcp

        respx.post(f"{API_BASE}/api/v1/memories/create").mock(
            return_value=httpx.Response(200, json={"ingestion_id": "ing_7"})
        )
        respx.post(EVENTS_URL).mock(return_value=httpx.Response(500, text="boom"))
        text = _text(
            await mcp.call_tool(
                "log_exchange",
                {
                    "user_message": "hi",
                    "conversation_id": "conv-1",
                    "user_id": "u1",
                },
            )
        )
        assert text.startswith("Logged to memory")
        assert "ERROR" not in text

    @respx.mock
    async def test_a_rejected_write_reports_no_turn(self, with_token, reporting_on):
        """Nothing was stored, so there is no turn to tell anticipation about."""
        from synap_mcp_server.server import mcp

        respx.post(f"{API_BASE}/api/v1/memories/create").mock(
            return_value=httpx.Response(402, text="out of credits")
        )
        events = respx.post(EVENTS_URL).mock(return_value=_accepted())
        await mcp.call_tool(
            "log_exchange",
            {
                "user_message": "hi",
                "conversation_id": "conv-1",
                "user_id": "u1",
            },
        )
        assert not events.called


class TestTheClientCall:
    @respx.mock
    async def test_send_events_raises_a_synap_api_error_upward(self, with_token):
        """The stream layer catches it; the client layer must still speak the
        one error type every other call here speaks."""
        from synap_mcp_server import client

        respx.post(EVENTS_URL).mock(return_value=httpx.Response(429, text="slow down"))
        with pytest.raises(SynapAPIError) as exc:
            await client.send_events([{"event_type": "user_message"}])
        assert exc.value.status == 429
