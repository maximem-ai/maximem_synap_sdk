"""``client.credits.*`` hands the transport everything it needs.

Every credits call used to reach ``HTTPTransport.get``/``post`` without the
required ``auth_context`` (and the two POSTs passed ``body=``, which the
transport does not take), so each one raised ``TypeError`` before any request
was made. JS twin paths and bodies: js/src/credits/interface.ts.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from maximem_synap.auth.models import AuthContext
from maximem_synap.credits import CreditsInterface
from maximem_synap.transport.http_client import HTTPTransport

AUTH = AuthContext(client_id="c", instance_id="inst_1", api_key="k")


class _Recorder(HTTPTransport):
    """The real transport methods, with the network call replaced."""

    def __init__(self):
        super().__init__(instance_id="inst_1", base_url="http://127.0.0.1:9")
        self.calls = []
        self.reply = {}

    async def request(self, method, path, auth_context, json=None, params=None, correlation_id=None):
        self.calls.append((method, path, auth_context, json, params))
        return self.reply


def _credits():
    transport = _Recorder()
    sdk = MagicMock()
    sdk._http_transport = transport

    async def _auth(correlation_id=None):
        return AUTH

    sdk._get_auth_context = _auth
    return CreditsInterface(sdk), transport


def test_get_balance_sends_the_caller_auth():
    credits, transport = _credits()
    transport.reply = {"client_id": "c", "balance_credits": 5, "warning_low": False, "buckets": []}
    balance = asyncio.run(credits.get_balance())
    assert balance.balance_credits == 5.0
    assert transport.calls == [("GET", "/v1/credits/balance", AUTH, None, None)]


def test_get_ledger_sends_the_caller_auth_and_the_paging_params():
    credits, transport = _credits()
    transport.reply = {"entries": [], "total": 0, "limit": 10, "offset": 20}
    asyncio.run(credits.get_ledger(limit=10, offset=20))
    method, path, auth, body, params = transport.calls[0]
    assert (method, path, auth, body) == ("GET", "/v1/credits/ledger", AUTH, None)
    assert params == {"limit": 10, "offset": 20}


def test_estimate_posts_its_body_as_json():
    credits, transport = _credits()
    transport.reply = {"credits_estimate": 2.5}
    result = asyncio.run(credits.estimate(metric_type="ingest", units=3))
    assert result.credits_estimate == 2.5
    method, path, auth, body, params = transport.calls[0]
    assert (method, path, auth, params) == ("POST", "/v1/credits/estimate", AUTH, None)
    assert body["metric_type"] == "ingest" and body["units"] == 3


def test_redeem_posts_the_code_as_json():
    credits, transport = _credits()
    transport.reply = {"redemption_id": "r", "credits_granted": 10, "new_balance_credits": 15}
    asyncio.run(credits.redeem("ABC"))
    assert transport.calls == [("POST", "/v1/credits/redeem", AUTH, {"code": "ABC"}, None)]
