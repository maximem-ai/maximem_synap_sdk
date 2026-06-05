"""The MCP tools. The descriptions ARE the product UI: the model decides when to call
each tool from the description text alone, which is how we enforce the "forward
everything, Synap decides what matters" model without asking the no-code user to
configure anything.

Error policy:
  - log_exchange   -> hard-fail (return ERROR) so a write failure is visible to the agent.
  - recall_context -> soft-fail (return a benign message) so a slow/failed read never
                      blocks the agent's reply.
"""

from .client import SynapAPIError, create_memory, fetch_context
from .config import settings
from .context import MissingTokenError

LOG_DESC = (
    "After each user message, send the exchange here so it can be remembered. "
    "You do not need to decide what is important — just forward the user message "
    "(and your reply, if you have one) and Synap will keep what matters. "
    "If your app serves more than one end-user, pass that person's stable id as "
    "user_id (or an organization id as customer_id) so each person's memory stays "
    "separate; if every conversation is the same single user, omit them."
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
            return f"ERROR: could not save to memory (status {exc.status})."
        return f"Logged to memory (ingestion_id={res.get('ingestion_id')})."

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
        except SynapAPIError:
            return "No memory available right now."
        return _format_context(data) or "Nothing known about this user yet."

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
        except SynapAPIError:
            return "No memory available right now."
        return _format_context(data) or "No memories yet."
