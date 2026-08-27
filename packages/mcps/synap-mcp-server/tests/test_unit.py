"""Unit tests for pure functions — no HTTP calls, no async I/O.

Covers:
  - client.scope_for
  - client.CustomerIdNotAcceptedError (message content)
  - client.get_isolation (no-token short circuit)
  - tools._server_message (400 body -> the fix text)
  - client.SynapAPIError  (constructor fields, __str__, retry_after)
  - tools._format_context (all buckets, empty items, None guard)
  - tools._summarize_status (all status branches)
  - tools._describe_api_error (every status branch)
  - tools._soft_recall_error (all branches)
  - context.set_token / get_token isolation
  - context.MissingTokenError  (is an Exception)
  - client._auth_headers (Bearer prefix, Content-Type, missing-token raises)
"""

import pytest

from synap_mcp_server.client import (
    B2C_ISOLATION,
    NETWORK_STATUS,
    TIMEOUT_STATUS,
    CustomerIdNotAcceptedError,
    SynapAPIError,
    get_isolation,
    scope_for,
    _auth_headers,
)
from synap_mcp_server.context import (
    MissingTokenError,
    get_token,
    set_token,
)
from synap_mcp_server.tools import (
    _describe_api_error,
    _format_context,
    _server_message,
    _soft_recall_error,
    _summarize_status,
)


# ---------------------------------------------------------------------------
# context: set_token / get_token / MissingTokenError
# ---------------------------------------------------------------------------


def test_set_and_get_token_roundtrip():
    """set_token followed by get_token returns the same value."""
    set_token("synap_abc123")
    try:
        assert get_token() == "synap_abc123"
    finally:
        set_token(None)


def test_set_none_clears_token():
    """set_token(None) results in get_token() returning None."""
    set_token("synap_abc123")
    set_token(None)
    assert get_token() is None


def test_missing_token_error_is_exception():
    """MissingTokenError is a subclass of Exception and carries its message."""
    exc = MissingTokenError("no token here")
    assert isinstance(exc, Exception)
    assert "no token" in str(exc)


# ---------------------------------------------------------------------------
# client.scope_for
# ---------------------------------------------------------------------------


def test_scope_for_no_ids_returns_client():
    assert scope_for(None, None) == "client"


def test_scope_for_user_id_returns_user():
    assert scope_for("u1", None) == "user"


def test_scope_for_customer_id_only_returns_customer():
    assert scope_for(None, "c1") == "customer"


def test_scope_for_both_ids_user_takes_priority():
    """When both user_id and customer_id are supplied, user scope wins."""
    assert scope_for("u1", "c1") == "user"


# ---------------------------------------------------------------------------
# client.SynapAPIError
# ---------------------------------------------------------------------------


def test_synapapierror_fields():
    """Constructor stores status, detail, retry_after correctly."""
    exc = SynapAPIError(429, "too many requests", retry_after="30")
    assert exc.status == 429
    assert exc.detail == "too many requests"
    assert exc.retry_after == "30"


def test_synapapierror_str_contains_status_and_detail():
    exc = SynapAPIError(500, "server blow up")
    assert "500" in str(exc)
    assert "server blow up" in str(exc)


def test_synapapierror_retry_after_defaults_none():
    exc = SynapAPIError(401, "unauthorized")
    assert exc.retry_after is None


def test_synapapierror_is_exception():
    exc = SynapAPIError(503, "down")
    assert isinstance(exc, Exception)


# ---------------------------------------------------------------------------
# client._auth_headers
# ---------------------------------------------------------------------------


def test_auth_headers_returns_bearer_and_content_type():
    set_token("synap_mykey")
    try:
        h = _auth_headers()
        assert h["Authorization"] == "Bearer synap_mykey"
        assert h["Content-Type"] == "application/json"
    finally:
        set_token(None)


def test_auth_headers_raises_missing_token_error_when_no_token():
    set_token(None)
    with pytest.raises(MissingTokenError):
        _auth_headers()


# ---------------------------------------------------------------------------
# tools._summarize_status
# ---------------------------------------------------------------------------


def test_summarize_status_completed_with_count():
    result = _summarize_status({"status": "completed", "memories_created": 3})
    assert "complete" in result.lower()
    assert "3" in result


def test_summarize_status_completed_one_memory():
    """Singular 'memory' for count=1."""
    result = _summarize_status({"status": "completed", "memories_created": 1})
    assert "1" in result
    assert "memory" in result


def test_summarize_status_complete_alias():
    """'complete' is a valid terminal status (alias for 'completed')."""
    result = _summarize_status({"status": "complete", "memories_created": 2})
    assert "complete" in result.lower()


def test_summarize_status_success_alias():
    """'success' terminal alias."""
    result = _summarize_status({"status": "success", "memories_created": 0})
    assert "complete" in result.lower()


def test_summarize_status_completed_none_count():
    """memories_created=None -> 'Processing complete.' without a count."""
    result = _summarize_status({"status": "completed", "memories_created": None})
    assert "complete" in result.lower()
    # Should not crash and should not include "None"
    assert "None" not in result


def test_summarize_status_partial_success():
    result = _summarize_status({"status": "partial_success", "memories_created": 1})
    assert "partial" in result.lower()
    assert "1" in result


def test_summarize_status_partial_success_no_count():
    result = _summarize_status({"status": "partial_success"})
    assert "partial" in result.lower()
    # memories_created defaults to 0
    assert "0" in result


def test_summarize_status_failed_with_error_message():
    result = _summarize_status({"status": "failed", "error_message": "disk full"})
    assert "fail" in result.lower()
    assert "disk full" in result


def test_summarize_status_error_alias():
    """'error' status is treated as failed."""
    result = _summarize_status({"status": "error", "error_message": "oops"})
    assert "fail" in result.lower()


def test_summarize_status_failed_no_error_message():
    """Missing error_message falls back to 'see logs'."""
    result = _summarize_status({"status": "failed"})
    assert "fail" in result.lower()
    assert "see logs" in result


def test_summarize_status_in_progress():
    """Non-terminal status is reported as still processing."""
    result = _summarize_status({"status": "processing"})
    assert "processing" in result.lower()


def test_summarize_status_unknown():
    """Unknown/absent status returns a stable placeholder."""
    result = _summarize_status({})
    assert "unknown" in result.lower()


def test_summarize_status_empty_dict():
    """Empty dict doesn't crash; 'unknown' is included."""
    result = _summarize_status({})
    assert result  # non-empty string
    assert "unknown" in result.lower()


def test_summarize_status_none_dict():
    """None payload — defensive guard via `(data or {})`."""
    result = _summarize_status(None)
    assert result


# ---------------------------------------------------------------------------
# tools._describe_api_error
# ---------------------------------------------------------------------------


def test_describe_api_error_401():
    exc = SynapAPIError(401, "unauthorized")
    msg = _describe_api_error(exc, "saving to memory")
    assert "ERROR" in msg
    assert "token" in msg.lower()
    assert "401" in msg


def test_describe_api_error_403():
    exc = SynapAPIError(403, "forbidden")
    msg = _describe_api_error(exc, "saving")
    assert "ERROR" in msg
    assert "token" in msg.lower()
    assert "403" in msg


def test_describe_api_error_402():
    exc = SynapAPIError(402, "no credits")
    msg = _describe_api_error(exc, "saving")
    assert "ERROR" in msg
    assert "credit" in msg.lower()


def test_describe_api_error_429_with_retry_after():
    exc = SynapAPIError(429, "slow down", retry_after="7")
    msg = _describe_api_error(exc, "saving")
    assert "ERROR" in msg
    assert "rate limit" in msg.lower()
    assert "7" in msg


def test_describe_api_error_429_without_retry_after():
    exc = SynapAPIError(429, "slow down")
    msg = _describe_api_error(exc, "saving")
    assert "ERROR" in msg
    assert "rate limit" in msg.lower()
    # No explicit retry-after hint in output
    assert "Retry after" not in msg


def test_describe_api_error_timeout():
    exc = SynapAPIError(TIMEOUT_STATUS, "timed out")
    msg = _describe_api_error(exc, "reading")
    assert "ERROR" in msg
    assert "timed out" in msg.lower()


def test_describe_api_error_network():
    exc = SynapAPIError(NETWORK_STATUS, "connection refused")
    msg = _describe_api_error(exc, "saving")
    assert "ERROR" in msg
    assert "reach" in msg.lower() or "network" in msg.lower()


def test_describe_api_error_500():
    exc = SynapAPIError(500, "internal error")
    msg = _describe_api_error(exc, "saving")
    assert "ERROR" in msg
    assert "500" in msg


def test_describe_api_error_503_is_network_status():
    """503 is NETWORK_STATUS — maps to the network-error branch, not generic 5xx."""
    exc = SynapAPIError(NETWORK_STATUS, "connection refused")  # NETWORK_STATUS == 503
    msg = _describe_api_error(exc, "saving")
    assert "ERROR" in msg
    assert "reach" in msg.lower() or "network" in msg.lower() or "memory service" in msg.lower()


def test_describe_api_error_502():
    """502 is a real 5xx (not NETWORK_STATUS=503) — generic 5xx branch."""
    exc = SynapAPIError(502, "bad gateway")
    msg = _describe_api_error(exc, "saving")
    assert "ERROR" in msg
    assert "502" in msg


def test_describe_api_error_unknown_4xx():
    """An unrecognised 4xx (e.g. 418) falls through to generic message."""
    exc = SynapAPIError(418, "I'm a teapot")
    msg = _describe_api_error(exc, "doing something")
    assert "ERROR" in msg
    assert "418" in msg


# ---------------------------------------------------------------------------
# tools._soft_recall_error
# ---------------------------------------------------------------------------


def test_soft_recall_error_429():
    exc = SynapAPIError(429, "slow down")
    msg = _soft_recall_error(exc)
    assert "rate limited" in msg.lower()


def test_soft_recall_error_402():
    exc = SynapAPIError(402, "no credits")
    msg = _soft_recall_error(exc)
    assert "credit" in msg.lower()
    # Must be a benign string, not an ERROR prefix
    assert "ERROR" not in msg


def test_soft_recall_error_500():
    exc = SynapAPIError(500, "boom")
    msg = _soft_recall_error(exc)
    # Falls to generic benign message
    assert msg  # non-empty
    assert "ERROR" not in msg


def test_soft_recall_error_401():
    exc = SynapAPIError(401, "unauthorized")
    msg = _soft_recall_error(exc)
    # Generic fallback; still benign
    assert "ERROR" not in msg


# ---------------------------------------------------------------------------
# tools._format_context
# ---------------------------------------------------------------------------


def test_format_context_returns_fact():
    data = {"context": {"facts": [{"content": "User is an engineer"}]}}
    result = _format_context(data)
    assert "(fact)" in result
    assert "User is an engineer" in result


def test_format_context_returns_preference():
    data = {"context": {"preferences": [{"content": "Prefers dark mode"}]}}
    result = _format_context(data)
    assert "(preference)" in result
    assert "Prefers dark mode" in result


def test_format_context_returns_episode():
    data = {"context": {"episodes": [{"content": "Had a support call"}]}}
    result = _format_context(data)
    assert "(episode)" in result


def test_format_context_returns_emotion():
    data = {"context": {"emotions": [{"content": "Felt frustrated"}]}}
    result = _format_context(data)
    assert "(emotion)" in result


def test_format_context_returns_temporal_events():
    data = {"context": {"temporal_events": [{"content": "Trial expires April 15"}]}}
    result = _format_context(data)
    assert "(temporal_event)" in result
    assert "Trial expires April 15" in result


def test_format_context_multiple_buckets():
    """Facts and preferences are both formatted."""
    data = {
        "context": {
            "facts": [{"content": "engineer"}],
            "preferences": [{"content": "dark mode"}],
        }
    }
    result = _format_context(data)
    assert "(fact)" in result
    assert "(preference)" in result
    assert "engineer" in result
    assert "dark mode" in result


def test_format_context_skips_items_without_content():
    """Items whose 'content' is absent or falsy are excluded."""
    data = {
        "context": {
            "facts": [{"content": ""}, {"content": "real fact"}],
        }
    }
    result = _format_context(data)
    assert result.count("- (fact)") == 1
    assert "real fact" in result


def test_format_context_handles_string_items():
    """Items that are plain strings (not dicts) are included."""
    data = {"context": {"facts": ["plain string fact"]}}
    result = _format_context(data)
    assert "plain string fact" in result


def test_format_context_empty_context():
    """All-empty buckets returns empty string."""
    data = {"context": {}}
    result = _format_context(data)
    assert result == ""


def test_format_context_missing_context_key():
    """'context' key absent: treated as empty -> returns empty string."""
    result = _format_context({})
    assert result == ""


def test_format_context_none_payload():
    """None payload: defensive `(data or {})` guard -> returns empty string."""
    result = _format_context(None)
    assert result == ""


def test_format_context_context_is_none():
    """context value is None: treated as {} -> returns empty string."""
    result = _format_context({"context": None})
    assert result == ""


# ---------------------------------------------------------------------------
# The identifier contract
# ---------------------------------------------------------------------------

_REJECTION_BODY = (
    '{"error": "customer_id_not_accepted_on_b2c", '
    '"message": "customer_id is not accepted on a B2C instance.", '
    '"detail": {"code": "customer_id_not_accepted_on_b2c", '
    '"message": "customer_id is not accepted on a B2C instance.", '
    '"where": "POST /v1/context/user/fetch", '
    '"fix": "Send user_id only and omit customer_id entirely."}}'
)


def test_customer_id_rejection_names_the_value_the_mode_and_the_fix():
    """All three or the message is not actionable: which value was refused, why
    the instance refuses it, and what to send instead."""
    exc = CustomerIdNotAcceptedError("POST /api/v1/memories/create", "acme")
    msg = str(exc)
    assert "'acme'" in msg
    assert B2C_ISOLATION in msg
    assert "user_id" in msg
    assert "POST /api/v1/memories/create" in msg


def test_customer_id_rejection_appends_the_route_specific_hint():
    """A customer-only recall gets one extra sentence, because for that call the
    whole ROUTE is gone, not just the field."""
    exc = CustomerIdNotAcceptedError(
        "POST /v1/context/customer/fetch", "acme",
        extra="Customer-scoped retrieval does not exist on a B2C instance at all.",
    )
    assert "does not exist on a B2C instance" in str(exc)


def test_customer_id_rejection_is_not_an_api_error():
    """It is raised before anything is sent, so it must not be mistaken for an
    upstream failure by a caller branching on SynapAPIError."""
    exc = CustomerIdNotAcceptedError("where", "acme")
    assert isinstance(exc, Exception)
    assert not isinstance(exc, SynapAPIError)


async def test_get_isolation_without_a_token_returns_none_without_a_request():
    """No token means no whoami to make; returning None keeps the check inert
    rather than turning a missing token into a scoping decision."""
    set_token(None)
    assert await get_isolation() is None


def test_server_message_pulls_message_and_fix_out_of_a_rejection_body():
    out = _server_message(_REJECTION_BODY)
    assert "not accepted on a B2C instance" in out
    assert "Send user_id only" in out
    assert "{" not in out, "raw JSON must not be handed to the model"


def test_server_message_falls_back_to_raw_text():
    """Not every 400 is the contract. A plain-text body is better than nothing."""
    assert "boom" in _server_message("boom")


def test_server_message_on_json_without_the_expected_keys():
    """A dict that carries neither message nor fix falls back rather than
    returning an empty string, which would read as 'no explanation given'."""
    out = _server_message('{"unexpected": true}')
    assert out


def test_server_message_on_empty_body():
    assert _server_message("") == ""


def test_describe_api_error_400_carries_the_fix_not_the_status():
    """The old mapping printed 'status 400' and dropped the body, which turned the
    one error that explains itself back into a bare status code."""
    exc = SynapAPIError(400, _REJECTION_BODY)
    msg = _describe_api_error(exc, "saving to memory")
    assert "ERROR" in msg
    assert "Send user_id only" in msg


def test_soft_recall_error_400_is_loud_not_benign():
    """Recall degrades quietly on outages and must NOT degrade quietly on a
    rejected shape: 'no memory available' for a refused call is the exact failure
    the contract exists to remove."""
    exc = SynapAPIError(400, _REJECTION_BODY)
    msg = _soft_recall_error(exc)
    assert "ERROR" in msg
    assert "Send user_id only" in msg
    assert "No memory available" not in msg
