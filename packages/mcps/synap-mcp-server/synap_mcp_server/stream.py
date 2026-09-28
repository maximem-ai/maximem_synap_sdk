"""Report a logged exchange as a conversation turn, so anticipation sees it.

## What this server can and cannot see

An MCP server is not an agent. It exposes four tools to somebody else's agent
and is handed one call at a time over HTTP with no session; `stateless_http`
is on and the only state is a Bearer token in a ContextVar for the length of
a request. So four of the five things an integration normally reports simply
do not exist here:

- the caller's **tool calls**, **tool results** and **reasoning** never reach
  this process. Their agent decides, calls its own tools and thinks entirely
  on its own side; all we ever learn is that it chose to call one of ours.
  Reporting `recall_context` as a "tool call" would be reporting Synap's own
  plumbing back to Synap, not the agent's work, so it is not done.
- there is no **session lifecycle**: nothing tells a stateless server that a
  conversation ended, so nothing here closes one.

What does exist is the **turn**. `log_exchange` is handed the user's message
and the assistant's reply as arguments — that is the whole turn, stated
explicitly, and the assistant half is the moment anticipation acts on.

## Why HTTP and not gRPC

`report_turn` in `synap_integrations_common` needs `sdk.instance.is_listening`
— a live bidirectional gRPC stream held open by a long-lived
`MaximemSynapSDK`. This server has no SDK, is stateless, and serves many
tenants: every request can carry a different key for a different instance.
Holding a stream per key would mean an unbounded pool of gRPC connections
that nothing ever tells us to close, and opening one per tool call would put
a TLS handshake and an auth round trip on a read that has a ten second
budget.

`POST /v1/events/batch` is the same events, over HTTP, dispatched into the
same listening path. It was built for exactly this shape of caller — one that
cannot keep a socket open. What it cannot do is carry an anticipated bundle
back, which does not matter here: this server's reads go through
`/v1/context/*` like any other REST caller.

## What it does not replace

`log_exchange` still writes its long-range document through
`memories.create`, and that is deliberate. The two are different stores: a
reported turn is written to conversation history and fed to the anticipation
agent, while `memories.create` queues the exchange for long-range extraction
and returns the `ingestion_id` that `check_memory_status` and
`wait_for_processing` are built on. Dropping it would leave both of those
tools with nothing to talk about and would stop the exchange being extracted
at all.

⚠ They are not free of each other. A conversation that grows past the
compaction threshold has its history promoted into memories, so over a long
conversation some of what is extracted from the long-range document is
extracted again from the turns. That is the reason this is off unless a
deployment switches it on (`MCP_STREAM_EVENTS`), rather than something every
caller gets by default.

## Best effort, always

Reporting a turn must never change what `log_exchange` answers. The write has
already succeeded by the time this runs; a failure here means anticipation
missed a turn, which is not something to hand back to the agent as an error.
Everything is swallowed, and a refusal from the server is logged with the
reason it gave rather than dropped, because a silent refusal is how a caller
ends up believing a turn was reported for a week.
"""

import logging
from typing import Optional

from .client import SynapAPIError, send_events
from .config import settings
from .context import MissingTokenError

logger = logging.getLogger("synap-mcp")


def _event(
    *,
    event_type: str,
    role: str,
    content: str,
    conversation_id: str,
    user_id: str,
    customer_id: Optional[str],
    event_id: str,
) -> dict:
    event: dict = {
        "event_type": event_type,
        "role": role,
        "content": content,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "event_id": event_id,
        "metadata": {"source": "mcp-server"},
    }
    # A B2C instance refuses a customer_id outright, so an absent one is left
    # out rather than sent as "".
    if customer_id:
        event["customer_id"] = customer_id
    return event


def _reportable(conversation_id: Optional[str], user_id: Optional[str]) -> bool:
    """Whether the turn can be reported at all.

    The events route requires a ``user_id`` and writes against a
    ``conversation_id``; without either it answers ``rejected``. Both are
    optional on ``log_exchange`` — the documented single-user, no-code shape
    sends neither — so a call without them is skipped here instead of being
    sent to be refused.
    """
    return bool(conversation_id) and bool(user_id)


async def report_exchange(
    *,
    user_message: str,
    assistant_message: str,
    conversation_id: Optional[str],
    user_id: Optional[str],
    customer_id: Optional[str],
    ingestion_id: Optional[str],
) -> bool:
    """Report one exchange as a user turn and an assistant turn.

    Returns whether anything was sent. Never raises: this runs after the write
    the agent is waiting on has already succeeded.
    """
    if not settings.stream_events:
        return False
    if not _reportable(conversation_id, user_id):
        return False

    # Deduped on event_id by the server. Keyed off the ingestion the write
    # just returned so a retry of the same accepted exchange cannot double the
    # turn, and two genuinely repeated messages are still two events.
    base = ingestion_id or f"{conversation_id}:{hash(user_message)}"

    events = []
    if user_message:
        events.append(_event(
            event_type="user_message", role="user", content=user_message,
            conversation_id=str(conversation_id), user_id=str(user_id),
            customer_id=customer_id, event_id=f"mcp:{base}:user",
        ))
    if assistant_message:
        # Last on purpose. `assistant_message` is the event anticipation acts
        # on: it is the moment a turn ends and the next one can be predicted,
        # and it has to arrive after the message it answers.
        events.append(_event(
            event_type="assistant_message", role="assistant",
            content=assistant_message,
            conversation_id=str(conversation_id), user_id=str(user_id),
            customer_id=customer_id, event_id=f"mcp:{base}:assistant",
        ))
    if not events:
        return False

    try:
        result = await send_events(events)
    except (SynapAPIError, MissingTokenError) as exc:
        logger.warning("MCP: could not report the turn to Synap: %s", exc)
        return False
    except Exception:  # noqa: BLE001 — never change what log_exchange answers
        logger.warning("MCP: reporting the turn failed", exc_info=True)
        return False

    _log_refusals(result, conversation_id)
    return bool((result or {}).get("accepted"))


def _log_refusals(result: dict, conversation_id: Optional[str]) -> None:
    """Say why the server refused an event, in its own words.

    A rejection here is a caller-shape problem — no user id, or a missing
    customer id on a B2B instance — and it is the kind that looks like
    "anticipation is just quiet" until somebody goes looking.
    """
    if not (result or {}).get("rejected"):
        return
    reasons = sorted({
        str(item.get("reason", "")).strip()
        for item in (result.get("results") or [])
        if item.get("status") == "rejected" and item.get("reason")
    })
    logger.warning(
        "MCP: Synap refused %s conversation event(s) for conversation %s: %s",
        result.get("rejected"),
        conversation_id,
        "; ".join(reasons) or "no reason given",
    )
