"""P22: the four navigation tools (synap_describe, synap_query, synap_document_index,
synap_document_read; D166, D167, T95). Each maps to one REST route and forwards the
caller's Bearer token. They exist only where the deployment turned on
SYNAP_DOCS_NAVIGATION (default off); a 404 from the API reads as "not enabled for this
instance". The existing tools are untouched.
"""

import importlib
import json

import httpx
import pytest
import respx
from mcp.server.fastmcp import FastMCP

from synap_mcp_server import tools
from tests.conftest import API_BASE, B2B, B2C, mock_whoami

pytestmark = pytest.mark.asyncio

KNOW = {"synap_describe", "synap_query"}
DOCS = {"synap_document_index", "synap_document_read"}
NAV = KNOW | DOCS
OLD = {"log_exchange", "recall_context", "list_recent_memories", "check_memory_status"}
KEY = "6f1c0b0e-0000-4000-8000-000000000001"


def _text(result) -> str:
    if isinstance(result, tuple):
        content, structured = result
        if isinstance(structured, dict) and "result" in structured:
            return str(structured["result"])
        result = content
    parts = [getattr(b, "text", None) for b in result]
    return "\n".join(p for p in parts if p is not None)


def build(navigation=True, knowledge=True) -> FastMCP:
    m = FastMCP("t")
    tools.register(m)
    if knowledge:
        tools.register_knowledge(m)
    if navigation:
        tools.register_navigation(m)
    return m


async def call(name, args, m=None):
    return _text(await (m or build()).call_tool(name, args))


async def test_the_tools_are_listed_with_their_arguments():
    listed = {t.name: t for t in await build().list_tools()}
    assert NAV <= set(listed) and OLD <= set(listed)
    q = listed["synap_query"].inputSchema["properties"]
    for field in ("target", "where", "related", "as_of", "document_ids", "aggregate", "sort",
                  "limit", "cursor", "user_id", "customer_id", "scope"):
        assert field in q
    assert listed["synap_query"].inputSchema["required"] == ["target"]
    r = listed["synap_document_read"].inputSchema["properties"]
    assert {"document_key", "section_key", "pages", "user_id", "customer_id"} <= set(r)
    assert listed["synap_document_index"].inputSchema["required"] == ["document_key"]


async def test_the_descriptions_tell_the_agent_how_to_use_them():
    listed = {t.name: (t.description or "").lower() for t in await build().list_tools()}
    assert "first" in listed["synap_describe"]
    assert "coverage" in listed["synap_query"] and "complete" in listed["synap_query"]
    assert "recall_context" in listed["synap_query"]       # recall stays the search tool
    assert "synap_document_index" in listed["synap_document_read"]
    assert "20 pages" in listed["synap_document_read"]


async def test_the_existing_tools_are_unchanged_by_the_new_ones():
    plain = {t.name: t for t in await build(navigation=False, knowledge=False).list_tools()}
    both = {t.name: t for t in await build().list_tools()}
    assert set(plain) == OLD
    for name in OLD:
        assert plain[name].description == both[name].description
        assert plain[name].inputSchema == both[name].inputSchema


async def test_each_pair_is_registered_only_under_its_own_flag(monkeypatch):
    """describe and query sit under SYNAP_KNOWLEDGE_QUERY, index and read under
    SYNAP_DOCS_NAVIGATION (T98); both default off and are parsed the same way."""
    from synap_mcp_server import config, server

    async def listed(**env):
        for name in ("SYNAP_KNOWLEDGE_QUERY", "SYNAP_DOCS_NAVIGATION"):
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        importlib.reload(config)
        importlib.reload(server)
        return {t.name for t in await server.mcp.list_tools()}

    try:
        assert await listed() == OLD
        assert await listed(SYNAP_KNOWLEDGE_QUERY="true") == OLD | KNOW
        assert await listed(SYNAP_DOCS_NAVIGATION="1") == OLD | DOCS
        assert await listed(SYNAP_KNOWLEDGE_QUERY="yes", SYNAP_DOCS_NAVIGATION="true") == OLD | NAV
        assert await listed(SYNAP_KNOWLEDGE_QUERY="false", SYNAP_DOCS_NAVIGATION="no") == OLD
    finally:
        for name in ("SYNAP_KNOWLEDGE_QUERY", "SYNAP_DOCS_NAVIGATION"):
            monkeypatch.delenv(name, raising=False)
        importlib.reload(config)
        importlib.reload(server)


@respx.mock
async def test_describe_is_a_get_with_the_scope(with_token):
    route = respx.get(f"{API_BASE}/api/v1/knowledge/describe").mock(
        return_value=httpx.Response(200, json={"fact_types": ["Obligation"]}))
    out = await call("synap_describe", {"user_id": "dana"})
    assert route.calls.last.request.headers["authorization"] == "Bearer synap_testkey"
    assert dict(route.calls.last.request.url.params) == {"user_id": "dana"}
    assert json.loads(out) == {"fact_types": ["Obligation"]}


@respx.mock
async def test_query_posts_the_structured_body_and_returns_it_with_coverage(with_token):
    reply = {"items": [{"id": "f1"}], "total": 1, "complete": True,
             "coverage": {"documents_in_scope": 3, "unprocessed": 1, "light_only": 0, "failed": 0},
             "next_cursor": None, "groups": None}
    mock_whoami(B2B)
    route = respx.post(f"{API_BASE}/api/v1/knowledge/query").mock(
        return_value=httpx.Response(200, json=reply))
    out = await call("synap_query", {
        "target": "Obligation", "where": {"status": "open"}, "limit": 20,
        "customer_id": "acme", "scope": {"customer": "acme"}})
    sent = json.loads(route.calls.last.request.read())
    assert sent == {"target": "Obligation", "where": {"status": "open"}, "limit": 20,
                    "customer_id": "acme", "scope": {"customer": "acme"}}
    assert json.loads(out) == reply


@respx.mock
async def test_index_and_read_map_to_their_routes(with_token):
    mock_whoami(B2B)
    index = respx.get(f"{API_BASE}/api/v1/documents/{KEY}/index").mock(
        return_value=httpx.Response(200, json={"sections": []}))
    read = respx.get(f"{API_BASE}/api/v1/documents/{KEY}/read").mock(
        return_value=httpx.Response(200, json={"text": "hello", "citation": {"page": 3}}))
    await call("synap_document_index", {"document_key": KEY, "version": 2, "customer_id": "acme"})
    assert dict(index.calls.last.request.url.params) == {"version": "2", "customer_id": "acme"}
    out = await call("synap_document_read", {"document_key": KEY, "pages": "3-5",
                                             "customer_id": "acme"})
    assert dict(read.calls.last.request.url.params) == {"pages": "3-5", "customer_id": "acme"}
    assert json.loads(out)["text"] == "hello"
    await call("synap_document_read", {"document_key": KEY, "section_key": "s4"})
    assert dict(read.calls.last.request.url.params) == {"section_key": "s4"}


@respx.mock
async def test_a_read_needs_exactly_one_of_section_or_pages_and_sends_nothing(with_token):
    read = respx.get(f"{API_BASE}/api/v1/documents/{KEY}/read").mock(
        return_value=httpx.Response(200, json={}))
    for args in ({}, {"section_key": "s", "pages": "1-2"}, {"pages": "3-1"}, {"pages": "x"}):
        out = await call("synap_document_read", {"document_key": KEY, **args})
        assert out.startswith("ERROR:")
    assert not read.called


@respx.mock
async def test_a_404_reads_as_not_enabled_or_not_visible(with_token):
    respx.get(f"{API_BASE}/api/v1/knowledge/describe").mock(
        return_value=httpx.Response(404, json={"detail": {"error": "not_found"}}))
    out = await call("synap_describe", {})
    assert out.startswith("ERROR:") and "not enabled for this instance" in out
    assert "SYNAP_KNOWLEDGE_QUERY" in out and "SYNAP_DOCS_NAVIGATION" not in out
    respx.get(f"{API_BASE}/api/v1/documents/{KEY}/index").mock(
        return_value=httpx.Response(404, json={"detail": {"error": "not_found"}}))
    out = await call("synap_document_index", {"document_key": KEY})
    assert "not enabled for this instance" in out and "cannot see" in out
    assert "SYNAP_DOCS_NAVIGATION" in out and "SYNAP_KNOWLEDGE_QUERY" not in out


@respx.mock
async def test_a_document_still_processing_says_so(with_token):
    respx.get(f"{API_BASE}/api/v1/documents/{KEY}/read").mock(return_value=httpx.Response(
        404, json={"detail": {"error": "document_not_ready",
                              "message": "The document is still being processed."}}))
    out = await call("synap_document_read", {"document_key": KEY, "pages": "1-1"})
    assert "still being processed" in out and "not enabled" not in out


@respx.mock
async def test_read_too_large_and_plan_and_price_errors_are_explained(with_token):
    respx.get(f"{API_BASE}/api/v1/documents/{KEY}/read").mock(return_value=httpx.Response(
        422, json={"detail": {"error": "read_too_large",
                              "message": "A read returns at most 20 pages; ask for a smaller range."}}))
    out = await call("synap_document_read", {"document_key": KEY, "pages": "1-30"})
    assert out.startswith("ERROR:") and "at most 20 pages" in out
    respx.post(f"{API_BASE}/api/v1/knowledge/query").mock(return_value=httpx.Response(
        403, json={"detail": {"error": "feature_not_in_plan", "feature": "knowledge_query",
                              "message": "Your plan does not include knowledge_query."}}))
    out = await call("synap_query", {"target": "Obligation"})
    assert "plan" in out.lower() and "token rejected" not in out
    respx.post(f"{API_BASE}/api/v1/knowledge/query").mock(return_value=httpx.Response(
        503, json={"detail": {"error": "pricing_unavailable", "message": "no price"}}))
    assert "unavailable" in (await call("synap_query", {"target": "O"})).lower()
    respx.post(f"{API_BASE}/api/v1/knowledge/query").mock(return_value=httpx.Response(
        504, json={"detail": {"error": "query_timeout", "message": "too slow"}}))
    out = await call("synap_query", {"target": "O"})
    assert "timed out" in out.lower() or "too slow" in out


@respx.mock
async def test_customer_id_follows_the_identifier_contract(with_token):
    """B2C refuses a customer_id before the request; B2B sends it."""
    mock_whoami(B2C)
    route = respx.get(f"{API_BASE}/api/v1/knowledge/describe").mock(
        return_value=httpx.Response(200, json={}))
    out = await call("synap_describe", {"customer_id": "acme", "user_id": "u"})
    assert out.startswith("ERROR:") and "user_id" in out and not route.called
    from synap_mcp_server import client

    client._isolation_cache.clear()
    mock_whoami(B2B)
    await call("synap_describe", {"customer_id": "acme", "user_id": "u"})
    assert route.called


async def test_a_missing_token_is_an_error_not_a_crash(no_token):
    out = await call("synap_describe", {})
    assert out.startswith("ERROR:") and "token" in out.lower()
