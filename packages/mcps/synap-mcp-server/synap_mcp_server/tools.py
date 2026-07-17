"""The MCP tools. The descriptions ARE the product UI: the model decides when to call
each tool from the description text alone, which is how we enforce the "forward
everything, Synap decides what matters" model without asking the no-code user to
configure anything.

Error policy:
  - log_exchange   -> hard-fail (return ERROR) so a write failure is visible to the agent.
  - recall_context -> soft-fail (return a benign message) so a slow/failed read never
                      blocks the agent's reply.
"""

import asyncio

from .client import (
    NETWORK_STATUS,
    TIMEOUT_STATUS,
    SynapAPIError,
    create_memory,
    fetch_context,
    get_ingestion_status,
    scope_for,
)
from .config import settings
from .context import MissingTokenError

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

LOG_DESC = (
    "After each user message, send the exchange here so it can be remembered. "
    "You do not need to decide what is important — just forward the user message "
    "(and your reply, if you have one) and Synap will keep what matters. "
    "If your app serves more than one end-user, pass that person's stable id as "
    "user_id (or an organization id as customer_id) so each person's memory stays "
    "separate; if every conversation is the same single user, omit them. "
    "Logging is fire-and-forget by default; set wait_for_processing=true only when "
    "you need to confirm the memory finished extracting before continuing."
)
STATUS_DESC = (
    "Check whether a logged exchange has finished processing. Pass the ingestion_id "
    "returned by log_exchange. Returns the processing status and how many memories "
    "were extracted. Useful to confirm a save completed (extraction is asynchronous)."
)
RECALL_DESC = (
    "Before replying, call this to recall anything already known about this user from "
    "past conversations. Use the user's latest message as the query. If you serve "
    "multiple end-users, pass the same user_id (or customer_id) you log with so you "
    "recall the right person's memory."
)
LIST_DESC = (
    "List recent things remembered about this user. Useful for debugging or to confirm "
    "that memory is working. Pass user_id/customer_id to scope to one person."
)


# Human-readable scope labels for the echo we append to confirmations, so a
# misrouted call (wrong/missing user_id => unexpected scope) is visible to the agent.
_SCOPE_LABELS = {
    "user": "this user",
    "customer": "this customer/org",
    "client": "the shared (account-wide) memory",
}


def _describe_api_error(exc: SynapAPIError, action: str) -> str:
    """Map an upstream status to a clear, actionable tool error. The REST API returns
    distinct codes (401 auth, 402 credits, 429 rate/credit cap, 5xx) — surface them so
    a failure never looks like "memory silently did nothing"."""
    s = exc.status
    if s in (401, 403):
        return f"ERROR: token rejected — check the Bearer token or its scope (status {s})."
    if s == 402:
        return "ERROR: out of credits — top up the workspace to keep using memory."
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
    """Recall is non-blocking, so failures stay benign — but a rate-limit or credit
    stop is worth naming so the builder understands why memory went quiet."""
    if exc.status == 429:
        return "Memory is rate limited right now — replying without it."
    if exc.status == 402:
        return "Memory is paused (out of credits) — replying without it."
    return "No memory available right now."


def _summarize_status(data: dict) -> str:
    status = str((data or {}).get("status", "unknown")).lower()
    created = (data or {}).get("memories_created")
    if status in ("completed", "complete", "success"):
        if created is None:
            return "Processing complete."
        return f"Processing complete — {created} memor{'y' if created == 1 else 'ies'} stored."
    if status == "partial_success":
        return f"Processing finished with partial success — {created or 0} stored."
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
    return "\n".join(lines)


def register(mcp) -> None:
    """Attach the tools to a FastMCP instance."""

    @mcp.tool(description=LOG_DESC)
    async def log_exchange(
        user_message: str,
        assistant_message: str = "",
        conversation_id: str | None = None,
        user_id: str | None = None,
        customer_id: str | None = None,
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
        except SynapAPIError as exc:
            return _describe_api_error(exc, "saving to memory")

        ingestion_id = res.get("ingestion_id")
        scope_note = f"scope: {_SCOPE_LABELS[scope_for(user_id, customer_id)]}"
        if wait_for_processing and ingestion_id:
            try:
                final = await _poll_until_terminal(ingestion_id)
                return (
                    f"Logged to memory (ingestion_id={ingestion_id}, {scope_note}). "
                    f"{_summarize_status(final)}"
                )
            except SynapAPIError:
                # Saving succeeded; only the status check failed — don't hard-fail.
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
        user_id: str | None = None,
        customer_id: str | None = None,
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
        except SynapAPIError as exc:
            return _soft_recall_error(exc)
        scope_label = _SCOPE_LABELS[scope_for(user_id, customer_id)]
        return _format_context(data) or f"Nothing remembered yet for {scope_label}."

    @mcp.tool(description=LIST_DESC)
    async def list_recent_memories(
        max_results: int = 10,
        user_id: str | None = None,
        customer_id: str | None = None,
    ) -> str:
        try:
            data = await fetch_context(
                None, max_results=max_results, user_id=user_id, customer_id=customer_id
            )
        except MissingTokenError as exc:
            return f"ERROR: {exc}"
        except SynapAPIError as exc:
            return _soft_recall_error(exc)
        return _format_context(data) or "No memories yet."
