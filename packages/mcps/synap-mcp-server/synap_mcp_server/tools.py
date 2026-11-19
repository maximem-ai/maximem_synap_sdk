"""The MCP tools. The descriptions ARE the product UI: the model decides when to call
each tool from the description text alone, which is how we enforce the "forward
everything, Synap decides what matters" model without asking the no-code user to
configure anything.

Because a coding agent reads these descriptions as the whole interface, they also
carry the identifier contract the REST API enforces (production, 2026-08-26):

    B2C (user_context_isolation = "equals_customer")   send user_id, nothing else.
                                                       customer_id -> HTTP 400.
    B2B (user_context_isolation = "strict")            customer_id is REQUIRED.
                                                       user_id alone is an error.

A description that says "optional" for a field that is refused in one mode and
mandatory in the other is not a small inaccuracy: it is the instruction the model
follows, and it produced writes and reads that addressed different buckets.

Error policy:
  - log_exchange   -> hard-fail (return ERROR) so a write failure is visible to the agent.
  - recall_context -> soft-fail (return a benign message) so a slow/failed read never
                      blocks the agent's reply. One exception: a rejected request
                      shape (400, or the local contract check) is reported as an
                      ERROR, because a caller bug that reads as "nothing remembered
                      yet" is exactly the silent failure this contract removes.
"""

import asyncio
import json as _json
import re
from typing import Annotated, Any, Optional

from pydantic import Field

from .client import (
    NETWORK_STATUS,
    TIMEOUT_STATUS,
    CustomerIdNotAcceptedError,
    SynapAPIError,
    create_memory,
    document_index,
    document_read,
    fetch_context,
    get_ingestion_status,
    knowledge_describe,
    knowledge_query,
    scope_for,
)
from .config import settings
from .context import MissingTokenError
from .stream import report_exchange

# Ingestion is async (long-range extraction). Terminal states reported by /status.
_TERMINAL_STATUSES = {
    "completed",
    "complete",
    "success",
    "partial_success",
    "failed",
    "error",
    "done",
}
# Cap how long we will block while polling, so platforms don't surface a timeout.
_POLL_MAX_SECONDS = 12.0
_POLL_INTERVAL_SECONDS = 2.0

# The contract, in one sentence, appended to every tool that takes scope ids. Kept as
# one constant so the four tools cannot drift into saying four different things.
SCOPE_CONTRACT = (
    "SCOPING: customer_id is B2B ONLY. On a B2B instance "
    "(user_context_isolation=strict) it is REQUIRED and a user_id on its own is an "
    "error. On a B2C instance (user_context_isolation=equals_customer) it is NOT "
    "accepted: the API rejects it with HTTP 400, customer-scoped retrieval does not "
    "exist there, and user_id is the whole identity. If you do not know which mode "
    "this instance is in, send user_id only and no customer_id; "
    "GET /api/v1/auth/whoami reports the mode as user_context_isolation."
)

LOG_DESC = (
    "After each user message, send the exchange here so it can be remembered. "
    "You do not need to decide what is important, just forward the user message "
    "(and your reply, if you have one) and Synap will keep what matters. "
    "If your app serves more than one end-user, pass that person's stable id as "
    "user_id so each person's memory stays separate; if every conversation is the "
    "same single user, omit it. "
    + SCOPE_CONTRACT
    + " Logging is fire-and-forget by default; set wait_for_processing=true only when "
    "you need to confirm the memory finished extracting before continuing."
)
STATUS_DESC = (
    "Check whether a logged exchange has finished processing. Pass the ingestion_id "
    "returned by log_exchange. Returns the processing status and how many memories "
    "were extracted. Useful to confirm a save completed (extraction is asynchronous). "
    "Takes no user_id or customer_id: an ingestion_id already identifies the write, "
    "so the identifier contract does not apply here."
)
RECALL_DESC = (
    "Before replying, call this to recall anything already known about this user from "
    "past conversations. Use the user's latest message as the query. If you serve "
    "multiple end-users, pass the same user_id you log with so you recall the right "
    "person's memory. "
    + SCOPE_CONTRACT
)
LIST_DESC = (
    "List recent things remembered about this user. Useful for debugging or to confirm "
    "that memory is working. Pass user_id to scope to one person. "
    + SCOPE_CONTRACT
)

# Per-parameter help. FastMCP publishes these in the tool's inputSchema, which is
# where a coding agent looks when it is deciding what to put in a field, so the
# contract is repeated here rather than left only in the tool description.
USER_ID_HELP = (
    "Stable id of the end-user this call belongs to. On a B2C instance this is the "
    "ONLY identifier you send and it is the whole identity. On a B2B instance send "
    "it together with customer_id. Omit both if every conversation is the same "
    "single user."
)
CUSTOMER_ID_HELP = (
    "B2B ONLY. Required on a B2B instance (user_context_isolation=strict), where a "
    "user_id on its own is an error. NOT accepted on a B2C instance "
    "(user_context_isolation=equals_customer): the API rejects it with HTTP 400 and "
    "there is no customer scope to read from. When in doubt leave this out and send "
    "user_id alone. GET /api/v1/auth/whoami reports the mode."
)

# Human-readable scope labels for the echo we append to confirmations, so a
# misrouted call (wrong/missing user_id => unexpected scope) is visible to the agent.
_SCOPE_LABELS = {
    "user": "this user",
    "customer": "this customer/org (B2B only)",
    "client": "the shared (account-wide) memory",
}


def _server_message(detail: str) -> str:
    """Pull the human-readable half out of an API error body.

    The contract rejection answers with {"message": ..., "detail": {"fix": ...}}, and
    the `fix` field is the only part that tells a caller what to send instead. Dumping
    raw JSON at a model, or dropping the body and printing the status, both throw that
    away. Falls back to the raw text when the body is not the shape we expect, because
    a partial answer beats "status 400".
    """
    try:
        body = _json.loads(detail)
    except (ValueError, TypeError):
        return (detail or "").strip()[:400]
    if not isinstance(body, dict):
        return (detail or "").strip()[:400]
    inner = body.get("detail")
    if not isinstance(inner, dict):
        inner = {}
    parts = [
        str(p).strip()
        for p in (body.get("message") or inner.get("message"), inner.get("fix"))
        if p
    ]
    if parts:
        return " ".join(parts)[:800]
    return (detail or "").strip()[:400]


def _describe_api_error(exc: SynapAPIError, action: str) -> str:
    """Map an upstream status to a clear, actionable tool error. The REST API returns
    distinct codes (400 rejected shape, 401 auth, 402 credits, 429 rate/credit cap,
    5xx), so surface them: a failure must never look like "memory silently did
    nothing"."""
    s = exc.status
    if s == 400:
        # Almost always the identifier contract: a customer_id sent to a B2C
        # instance, or a customer-scoped route called there. The body says which
        # and what to send instead, so it is repeated verbatim.
        return (
            f"ERROR: the memory service rejected this request while {action}. "
            f"{_server_message(exc.detail)}"
        )
    if s in (401, 403):
        return f"ERROR: token rejected, check the Bearer token or its scope (status {s})."
    if s == 402:
        return "ERROR: out of credits, top up the workspace to keep using memory."
    if s == 429:
        hint = f" Retry after {exc.retry_after}s." if exc.retry_after else " Slow down and retry."
        return f"ERROR: rate limit or credit cap reached while {action}.{hint}"
    if s == TIMEOUT_STATUS:
        return f"ERROR: the memory service timed out while {action}. Try again."
    if s == NETWORK_STATUS:
        return f"ERROR: couldn't reach the memory service while {action}. Try again shortly."
    if s >= 500:
        return f"ERROR: memory service unavailable (status {s}). Try again shortly."
    return f"ERROR: could not {action} (status {s})."


def _soft_recall_error(exc: SynapAPIError) -> str:
    """Recall is non-blocking, so failures stay benign, with two exceptions worth
    naming: a rate-limit or credit stop explains why memory went quiet, and a 400
    means the CALL was wrong rather than the service.

    The 400 case is deliberately not benign. Answering "no memory available" to a
    request the server refused is how a B2C instance served 4,634 consecutive empty
    reads without anyone noticing. A wrong shape has to look like a wrong shape.
    """
    if exc.status == 400:
        return f"ERROR: recall request rejected. {_server_message(exc.detail)}"
    if exc.status == 429:
        return "Memory is rate limited right now, replying without it."
    if exc.status == 402:
        return "Memory is paused (out of credits), replying without it."
    return "No memory available right now."


def _summarize_status(data: dict) -> str:
    status = str((data or {}).get("status", "unknown")).lower()
    created = (data or {}).get("memories_created")
    if status in ("completed", "complete", "success"):
        if created is None:
            return "Processing complete."
        return f"Processing complete, {created} memor{'y' if created == 1 else 'ies'} stored."
    if status == "partial_success":
        return f"Processing finished with partial success, {created or 0} stored."
    if status in ("failed", "error"):
        msg = (data or {}).get("error_message") or "see logs"
        return f"Processing failed ({msg})."
    return f"Still processing (status={status})."


async def _poll_until_terminal(ingestion_id: str) -> dict:
    """Poll /status until terminal or the cap elapses. Best-effort; returns the last
    payload seen (possibly non-terminal if it timed out)."""
    elapsed = 0.0
    last: dict = {"status": "processing"}
    while elapsed < _POLL_MAX_SECONDS:
        last = await get_ingestion_status(ingestion_id)
        if str(last.get("status", "")).lower() in _TERMINAL_STATUSES:
            return last
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        elapsed += _POLL_INTERVAL_SECONDS
    return last


def _format_context(data: dict) -> str:
    ctx = (data or {}).get("context", {}) or {}
    lines: list[str] = []
    for bucket in ("facts", "preferences", "episodes", "emotions", "temporal_events"):
        for item in ctx.get(bucket, []) or []:
            content = item.get("content") if isinstance(item, dict) else str(item)
            if content:
                # bucket[:-1]: facts -> fact, preferences -> preference, etc.
                lines.append(f"- ({bucket[:-1]}) {content}")
    # D166/D167: document items, each with the citation an agent opens it by
    # (synap_document_index / synap_document_read). Absent for a chat-only
    # instance, so its lines are exactly the ones above.
    for bucket, label in (("passages", "passage"), ("records", "record"), ("fetched", "fetched")):
        for item in ctx.get(bucket, []) or []:
            content = item.get("content") if isinstance(item, dict) else str(item)
            if content:
                cite = _format_citation(item.get("citation") if isinstance(item, dict) else None)
                lines.append(f"- ({label}) {content}{cite}")
    return "\n".join(lines)


def _format_citation(citation) -> str:
    if not isinstance(citation, dict):
        return ""
    parts = []
    if citation.get("document_key"):
        parts.append(f"document_key: {citation['document_key']}")
    if citation.get("title"):
        parts.append(f"title: {citation['title']}")
    if citation.get("section_path"):
        parts.append("section: " + " > ".join(str(s) for s in citation["section_path"]))
    if citation.get("page") is not None:
        parts.append(f"page: {citation['page']}")
    return f" [{'; '.join(parts)}]" if parts else ""


# Annotated aliases, so all four tools describe the same field the same way.
UserIdArg = Annotated[Optional[str], Field(description=USER_ID_HELP)]
CustomerIdArg = Annotated[Optional[str], Field(description=CUSTOMER_ID_HELP)]


def register(mcp) -> None:
    """Attach the tools to a FastMCP instance."""

    @mcp.tool(description=LOG_DESC)
    async def log_exchange(
        user_message: str,
        assistant_message: str = "",
        conversation_id: str | None = None,
        user_id: UserIdArg = None,
        customer_id: CustomerIdArg = None,
        wait_for_processing: bool = False,
    ) -> str:
        document = f"User: {user_message}"
        if assistant_message:
            document += f"\nAssistant: {assistant_message}"
        metadata = {"conversation_id": conversation_id} if conversation_id else None
        try:
            res = await create_memory(
                document,
                metadata=metadata,
                user_id=user_id,
                customer_id=customer_id,
            )
        except MissingTokenError as exc:
            return f"ERROR: {exc}"
        except CustomerIdNotAcceptedError as exc:
            # Hard error, not a silent drop. Writing under a scope the caller did
            # not ask for is how the two sides came to disagree in the first place.
            return f"ERROR: {exc} Retry this call with user_id only."
        except SynapAPIError as exc:
            return _describe_api_error(exc, "saving to memory")

        ingestion_id = res.get("ingestion_id")

        # Also report the exchange as a conversation turn, so the anticipation
        # agent sees the turn and not only the memory that comes out of it.
        # Off unless the deployment switched it on, best effort, and it cannot
        # change what this tool answers: the write above has already
        # succeeded and a telemetry miss is not the agent's problem. See
        # stream.py for why this is the only one of the five events an MCP
        # server can report at all.
        await report_exchange(
            user_message=user_message,
            assistant_message=assistant_message,
            conversation_id=conversation_id,
            user_id=user_id,
            customer_id=customer_id,
            ingestion_id=ingestion_id,
        )

        scope_note = f"scope: {_SCOPE_LABELS[scope_for(user_id, customer_id)]}"
        if wait_for_processing and ingestion_id:
            try:
                final = await _poll_until_terminal(ingestion_id)
                return (
                    f"Logged to memory (ingestion_id={ingestion_id}, {scope_note}). "
                    f"{_summarize_status(final)}"
                )
            except SynapAPIError:
                # Saving succeeded; only the status check failed, don't hard-fail.
                return (
                    f"Logged to memory (ingestion_id={ingestion_id}, {scope_note}). "
                    "Could not read processing status."
                )
        return f"Logged to memory (ingestion_id={ingestion_id}, {scope_note})."

    @mcp.tool(description=STATUS_DESC)
    async def check_memory_status(ingestion_id: str) -> str:
        try:
            data = await get_ingestion_status(ingestion_id)
        except MissingTokenError as exc:
            return f"ERROR: {exc}"
        except SynapAPIError as exc:
            if exc.status == 404:
                return "Unknown ingestion_id (it may have expired or never existed)."
            return _describe_api_error(exc, "reading status")
        return _summarize_status(data)

    @mcp.tool(description=RECALL_DESC)
    async def recall_context(
        query: str,
        max_results: int | None = None,
        user_id: UserIdArg = None,
        customer_id: CustomerIdArg = None,
    ) -> str:
        try:
            data = await fetch_context(
                [query],
                max_results=max_results or settings.default_max_results,
                user_id=user_id,
                customer_id=customer_id,
            )
        except MissingTokenError as exc:
            return f"ERROR: {exc}"
        except CustomerIdNotAcceptedError as exc:
            return f"ERROR: {exc} Retry this call with user_id only."
        except SynapAPIError as exc:
            return _soft_recall_error(exc)
        scope_label = _SCOPE_LABELS[scope_for(user_id, customer_id)]
        return _format_context(data) or f"Nothing remembered yet for {scope_label}."

    @mcp.tool(description=LIST_DESC)
    async def list_recent_memories(
        max_results: int = 10,
        user_id: UserIdArg = None,
        customer_id: CustomerIdArg = None,
    ) -> str:
        try:
            data = await fetch_context(
                None, max_results=max_results, user_id=user_id, customer_id=customer_id
            )
        except MissingTokenError as exc:
            return f"ERROR: {exc}"
        except CustomerIdNotAcceptedError as exc:
            return f"ERROR: {exc} Retry this call with user_id only."
        except SynapAPIError as exc:
            return _soft_recall_error(exc)
        return _format_context(data) or "No memories yet."


# ---------------------------------------------------------------- navigation (D166, D167)
#
# Four more tools. synap_describe and synap_query are registered only where the
# deployment set SYNAP_KNOWLEDGE_QUERY (config.knowledge_query), synap_document_index
# and synap_document_read only where it set SYNAP_DOCS_NAVIGATION
# (config.docs_navigation): the API's own flags are per instance, so the deployment
# switches a pair on only where the API has it on (T98). `recall_context` stays the search tool; these are for the
# questions search answers badly: a complete filtered set (describe, then query) and
# a document opened section by section (index, then read). Errors are hard (ERROR:),
# never a benign empty answer: an agent told "nothing found" when the call was refused
# would report a wrong "none" to its user.

DESCRIBE_DESC = (
    "Call this FIRST, before synap_query: it lists what Synap knows in this scope. It "
    "returns the sources, document counts by type, and the entity and fact types "
    "(with their typed properties and relations) that synap_query accepts as target, "
    "filters and relations. Use recall_context to search; use this to learn what can "
    "be asked as a structured question. " + SCOPE_CONTRACT
)
QUERY_DESC = (
    "Ask a structured question over the facts and entities extracted from documents "
    "and get a COMPLETE filtered set, a count, or counts grouped by a property, not a "
    "ranked top-k: use it for 'list every', 'how many' and filter questions (every "
    "contract with Acme expiring before March). Call synap_describe first for the "
    "valid target, property and relation names. Send a structured request, never "
    "question text. Every answer carries total, complete (false means the row cap cut "
    "the set: page with cursor) and coverage (documents in scope not yet processed or "
    "only lightly extracted): ALWAYS report coverage to the user, for example '14 "
    "found; 3 contracts still processing', and never present a result as complete "
    "when complete is false or coverage shows gaps. For open-ended search use "
    "recall_context instead. " + SCOPE_CONTRACT
)
INDEX_DESC = (
    "Open a document's index: its title, summary and section tree, each section with "
    "section_key, heading, level and page range. Use it to check a citation or to find "
    "where to read; then call synap_document_read with a section_key. Takes the "
    "document_key from a recall_context citation or a synap_query result. Pass the "
    "same user_id and customer_id the document is read at: a document you cannot see "
    "is reported as not found. " + SCOPE_CONTRACT
)
READ_DESC = (
    "Read one section or one page range of a document, with its citation. Pass exactly "
    "one of section_key (from synap_document_index) or pages ('3-5'). At most 20 pages "
    "or 8,000 tokens per call; a larger request is refused, so ask for less. If the "
    "document is still being processed, try again shortly. Pass the same user_id and "
    "customer_id the document is read at. " + SCOPE_CONTRACT
)

KNOWLEDGE_FLAG = "SYNAP_KNOWLEDGE_QUERY"
NAVIGATION_FLAG = "SYNAP_DOCS_NAVIGATION"


def _error_code(detail: str) -> str:
    """The server's machine code from an error body, or ''."""
    try:
        body = _json.loads(detail)
    except (ValueError, TypeError):
        return ""
    if not isinstance(body, dict):
        return ""
    inner = body.get("detail")
    inner = inner if isinstance(inner, dict) else {}
    return str(body.get("error") or inner.get("code") or inner.get("error") or "")


def _describe_navigation_error(exc: SynapAPIError, action: str, *, document: bool,
                               flag: str) -> str:
    """Navigation errors are never soft. A 404 is either the feature's flag being off
    for this instance or, for a document, one the caller cannot see: the server answers
    the same on purpose (D93), so the message names both, and the flag, so the agent
    does not guess."""
    not_enabled = f"this feature ({flag}) is not enabled for this instance"
    code = _error_code(exc.detail)
    if exc.status == 404:
        if code == "document_not_ready":
            return (
                "ERROR: the document is still being processed. Try again shortly."
            )
        if document:
            return (
                f"ERROR: not found while {action}. Either the document does not exist or "
                "you cannot see it in this scope (check user_id and customer_id), or "
                f"{not_enabled}."
            )
        return f"ERROR: {not_enabled} (status 404)."
    if exc.status == 403 and code == "feature_not_in_plan":
        return (
            f"ERROR: this workspace's plan does not include this feature while {action} "
            f"(Pro and up). {_server_message(exc.detail)}"
        )
    if exc.status == 422:
        return f"ERROR: the request was rejected while {action}. {_server_message(exc.detail)}"
    if exc.status == 503 and code == "pricing_unavailable":
        return f"ERROR: this feature is unavailable while {action}: no price is configured."
    if exc.status == 504:
        return (
            f"ERROR: the query timed out while {action}; nothing partial is returned. "
            "Narrow it with a filter and try again."
        )
    return _describe_api_error(exc, action)


_PAGES = re.compile(r"^(\d+)-(\d+)$")


ScopeArg = Annotated[
    Optional[dict[str, str]],
    Field(description="The scope path as level to id, e.g. {\"customer\": \"acme\"}. "
                      "Usually leave out; user_id and customer_id are enough."),
]


async def _run(coro, action: str, *, document: bool, flag: str) -> str:
    try:
        data = await coro
    except MissingTokenError as exc:
        return f"ERROR: {exc}"
    except CustomerIdNotAcceptedError as exc:
        return f"ERROR: {exc} Retry this call with user_id only."
    except SynapAPIError as exc:
        return _describe_navigation_error(exc, action, document=document, flag=flag)
    return _json.dumps(data, ensure_ascii=False)


def register_knowledge(mcp) -> None:
    """Attach synap_describe and synap_query (behind SYNAP_KNOWLEDGE_QUERY)."""

    @mcp.tool(description=DESCRIBE_DESC)
    async def synap_describe(
        user_id: UserIdArg = None, customer_id: CustomerIdArg = None, scope: ScopeArg = None
    ) -> str:
        return await _run(
            knowledge_describe(user_id=user_id, customer_id=customer_id, scope=scope),
            "describing what is known", document=False, flag=KNOWLEDGE_FLAG,
        )

    @mcp.tool(description=QUERY_DESC)
    async def synap_query(
        target: Annotated[str, Field(description="An entity or fact type from synap_describe, "
                                                 "e.g. \"Obligation\".")],
        where: Annotated[Optional[dict[str, Any]], Field(description=(
            "Property to a value, or to {eq|ne|in|before|after|between|gt|lt|contains|exists: "
            "value}, e.g. {\"status\": \"open\", \"due_at\": {\"before\": \"2027-03-01\"}}."))] = None,
        related: Annotated[Optional[dict[str, Any]], Field(description=(
            "Relation to a filter on the node at the other end; nesting is a further hop "
            "(at most 2), e.g. {\"COUNTERPARTY\": {\"type\": \"Organization\", "
            "\"name\": \"Acme\"}}."))] = None,
        as_of: Annotated[Optional[str], Field(description=(
            "An ISO date or datetime; current versions only when unset."))] = None,
        document_ids: Annotated[Optional[list[str]], Field(description=(
            "Only facts from these documents."))] = None,
        aggregate: Annotated[Optional[dict[str, Any]], Field(description=(
            "{\"count\": true} or {\"group_by\": \"<property>\"}."))] = None,
        sort: Annotated[Optional[list[dict[str, str]]], Field(description=(
            "[{\"property\": ..., \"order\": \"asc\"|\"desc\"}]."))] = None,
        limit: Annotated[Optional[int], Field(description=(
            "Rows per page, at most 500 (default 100)."))] = None,
        cursor: Annotated[Optional[str], Field(description=(
            "The previous answer's next_cursor, to get the next page."))] = None,
        user_id: UserIdArg = None,
        customer_id: CustomerIdArg = None,
        scope: ScopeArg = None,
    ) -> str:
        body: dict[str, Any] = {"target": target}
        for key, value in (
            ("where", where), ("related", related), ("as_of", as_of),
            ("document_ids", document_ids), ("aggregate", aggregate), ("sort", sort),
            ("limit", limit), ("cursor", cursor), ("user_id", user_id),
            ("customer_id", customer_id), ("scope", scope),
        ):
            if value is not None:
                body[key] = value
        return await _run(knowledge_query(body), "running the query", document=False,
                          flag=KNOWLEDGE_FLAG)


def register_navigation(mcp) -> None:
    """Attach synap_document_index and synap_document_read (behind
    SYNAP_DOCS_NAVIGATION)."""

    @mcp.tool(description=INDEX_DESC)
    async def synap_document_index(
        document_key: Annotated[str, Field(description=(
            "The document's key, from a citation or a query result."))],
        version: Annotated[Optional[int], Field(description=(
            "A version other than the current one."))] = None,
        user_id: UserIdArg = None,
        customer_id: CustomerIdArg = None,
        scope: ScopeArg = None,
    ) -> str:
        return await _run(
            document_index(document_key, version=version, user_id=user_id,
                           customer_id=customer_id, scope=scope),
            "opening the document index", document=True, flag=NAVIGATION_FLAG,
        )

    @mcp.tool(description=READ_DESC)
    async def synap_document_read(
        document_key: Annotated[str, Field(description=(
            "The document's key, from a citation or a query result."))],
        section_key: Annotated[Optional[str], Field(description=(
            "A section_key from synap_document_index. Exactly one of section_key and "
            "pages."))] = None,
        pages: Annotated[Optional[str], Field(description=(
            "A page range as from-to, e.g. \"3-5\" (at most 20 pages)."))] = None,
        user_id: UserIdArg = None,
        customer_id: CustomerIdArg = None,
        scope: ScopeArg = None,
    ) -> str:
        # Refused here, before any request, so a malformed call is an ERROR the model
        # can fix and costs nothing.
        if (section_key is None) == (pages is None):
            return "ERROR: pass exactly one of section_key or pages."
        if pages is not None:
            m = _PAGES.match(pages.strip())
            if m is None or int(m.group(1)) < 1 or int(m.group(2)) < int(m.group(1)):
                return "ERROR: pages is from-to, whole page numbers from 1, for example 3-5."
        return await _run(
            document_read(document_key, section_key=section_key,
                          pages=None if pages is None else pages.strip(), user_id=user_id,
                          customer_id=customer_id, scope=scope),
            "reading the document", document=True, flag=NAVIGATION_FLAG,
        )
