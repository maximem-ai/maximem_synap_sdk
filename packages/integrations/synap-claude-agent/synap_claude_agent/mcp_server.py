"""Synap MCP server for the Claude Agent SDK.

Exposes two in-process MCP tools that let the agent read/write Synap memory
explicitly as tool calls:

- ``synap_search(query, max_results?)`` — searches Synap via ``sdk.fetch``
  and returns the formatted context as plain text. Read failures degrade to
  an explanatory "no context available" result so the agent loop doesn't
  wedge.

- ``synap_remember(content, metadata?)`` — ingests an explicit fact via
  ``sdk.memories.create``. Write failures raise so ingestion outages are
  observable on the agent side as tool errors.

The wrapping tool names in Claude are ``mcp__synap__synap_search`` and
``mcp__synap__synap_remember`` (per the SDK's ``mcp__<server>__<tool>``
naming). Add them to ``ClaudeAgentOptions(allowed_tools=[...])``.

Identifiers
-----------
``user_id`` and ``customer_id`` are bound once at construction and attached to
every tool call. The model never sees them and cannot leave one out, which makes
the ``customer_id`` binding mode-sensitive:

    B2C, ``user_context_isolation = "equals_customer"``
        Send ``user_id`` only. A ``customer_id`` is refused: by the SDK from
        0.4.7, and by the REST API with HTTP 400 since 2026-08-26. There is no
        customer scope on such an instance, because the customer and the user are
        the same node.

    B2B, ``user_context_isolation = "strict"``
        ``customer_id`` is REQUIRED. Unchanged.

So a ``customer_id`` passed to ``create_synap_mcp_server`` on a B2C instance used
to break every search and every write, permanently, with no tool argument able to
fix it. :class:`_CustomerScope` makes the injection conditional instead.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Optional, TypeVar

from claude_agent_sdk import SdkMcpTool, create_sdk_mcp_server, tool
from claude_agent_sdk.types import McpSdkServerConfig
from maximem_synap import MaximemSynapSDK

try:  # SDK >= 0.4.7 raises this client-side on a B2C instance.
    from maximem_synap.scoping import CustomerIdNotAcceptedError as _SdkCustomerIdError
except ImportError:  # pragma: no cover - older SDK, fall back to matching the text
    _SdkCustomerIdError = ()  # type: ignore[assignment]

logger = logging.getLogger(__name__)

B2C_ISOLATION = "equals_customer"

# Substrings that only a B2C identifier rejection produces: the SDK's own message
# quotes the mode, and the API's 400 body carries the error code. Matched as text
# because the failure can arrive as an SDK exception, as an HTTP error from an
# older SDK, or wrapped by a transport, and all three mean the same thing.
_B2C_REJECTION_MARKERS = (B2C_ISOLATION, "customer_id_not_accepted_on_b2c")

T = TypeVar("T")


def _sdk_isolation(sdk: Any) -> Optional[str]:
    """The instance's scoping mode as the SDK understands it, or None.

    ``initialize()`` reads ``GET /api/v1/auth/whoami`` and keeps the answer on the
    instance. It is not part of the SDK's public surface, so it is read through
    ``getattr`` with a public name tried first, and anything that is not a real
    string counts as "not known": a test double answers every attribute, and a
    mock's auto-attribute is not an answer about scoping.

    None means the SDK has not been initialised, or the server does not publish
    the field. Both must mean "do nothing": guessing B2C here would strip a B2B
    caller's MANDATORY customer_id, which is the worse of the two failures.
    """
    for attr in ("user_context_isolation", "_user_context_isolation"):
        try:
            value = getattr(sdk, attr, None)
        except Exception:  # noqa: BLE001 - a property that raises is not an answer
            continue
        if isinstance(value, str) and value:
            return value
    return None


def _is_b2c_rejection(exc: BaseException) -> bool:
    """Whether this failure is the instance saying "no customer_id here"."""
    if _SdkCustomerIdError and isinstance(exc, _SdkCustomerIdError):
        return True
    if type(exc).__name__ == "CustomerIdNotAcceptedError":
        return True
    text = str(exc)
    return any(marker in text for marker in _B2C_REJECTION_MARKERS)


class _CustomerScope:
    """The bound ``customer_id``, and the single place that decides whether to send it.

    Two things establish that the instance is B2C:

    1. The SDK already knows, because ``initialize()`` asked whoami.
    2. The first rejection says so. Only a B2C instance produces that error, so
       one is proof, and it is the only signal available when the SDK was
       constructed but not yet initialised.

    On (2) the offending call is retried once without the id. Nothing was written
    and nothing was read, in either the client-side or the server-side rejection,
    so the retry is safe, and it is what turns "the first call always fails" into
    "it just works".

    Dropping the id costs nothing on B2C: customer and user are the same node
    there, so there is no separate organisation scope the caller could have meant.
    It is logged at WARNING once, because a constructor argument that has no
    effect should be visible to whoever wrote it.
    """

    def __init__(self, sdk: Any, customer_id: str):
        self._configured = customer_id or ""
        self._suppressed = False
        if self._configured and _sdk_isolation(sdk) == B2C_ISOLATION:
            self._suppress("this instance is B2C (user_context_isolation="
                           f"{B2C_ISOLATION!r})")

    def _suppress(self, reason: str) -> None:
        if self._suppressed:
            return
        self._suppressed = True
        logger.warning(
            "synap: ignoring customer_id=%r because %s. On a B2C instance the "
            "user_id is the whole identity and a customer_id is rejected, so it "
            "is not sent. Remove customer_id from create_synap_mcp_server().",
            self._configured, reason,
        )

    @property
    def suppressed(self) -> bool:
        return self._suppressed

    def value(self) -> Optional[str]:
        """The id to attach to the next call, or None."""
        if self._suppressed:
            return None
        return self._configured or None

    def note_rejection(self, exc: BaseException) -> bool:
        """Learn from a failure. True when it was the B2C rejection and the id has
        now been dropped, which means the caller may retry."""
        if self._suppressed or not self._configured:
            return False
        if not _is_b2c_rejection(exc):
            return False
        self._suppress("the instance rejected it as B2C-incompatible")
        return True


async def _without_refused_customer_id(
    scope: _CustomerScope, call: Callable[[Optional[str]], Awaitable[T]]
) -> T:
    """Run ``call(customer_id)``, and retry once without the id if the instance
    refuses it. Any other failure propagates untouched."""
    try:
        return await call(scope.value())
    except Exception as exc:  # noqa: BLE001 - re-raised unless it is the contract
        if not scope.note_rejection(exc):
            raise
    return await call(scope.value())


def _build_synap_tools(
    sdk: MaximemSynapSDK,
    user_id: str,
    customer_id: str,
    conversation_id: Optional[str],
    mode: str,
) -> list[SdkMcpTool]:
    """Build the underlying :class:`SdkMcpTool` list.

    Kept separate from :func:`create_synap_mcp_server` so smoke tests and
    downstream integrations can invoke the tool handlers directly without
    spinning up the MCP server machinery.

    ``customer_id`` keeps its type: it is REQUIRED on a B2B instance, so it
    cannot simply go away. What changes is that it is routed through
    :class:`_CustomerScope` rather than pasted into both calls, so a B2C instance
    stops receiving a field it refuses. Pass "" on B2C.
    """
    scope = _CustomerScope(sdk, customer_id)

    @tool(
        "synap_search",
        "Search the user's Synap memory (facts, preferences, episodes, "
        "emotions, temporal events) for context relevant to a query. Use "
        "this when you need background about the user that isn't in the "
        "current conversation.",
        {"query": str, "max_results": int},
    )
    async def synap_search(args: dict[str, Any]) -> dict[str, Any]:
        query = args.get("query", "")
        max_results = int(args.get("max_results") or 10)
        if not query:
            return {
                "content": [
                    {"type": "text", "text": "synap_search: missing `query` argument."}
                ],
                "isError": True,
            }

        async def _fetch(scoped_customer_id):
            return await sdk.fetch(
                conversation_id=conversation_id,
                user_id=user_id,
                customer_id=scoped_customer_id,
                search_query=[query],
                max_results=max_results,
                mode=mode,
                include_conversation_context=False,
            )

        try:
            response = await _without_refused_customer_id(scope, _fetch)
        except Exception as exc:  # noqa: BLE001 — tool errors surface as text
            logger.error(
                "synap_search: sdk.fetch failed user_id=%s error=%s",
                user_id, exc, exc_info=True,
            )
            if _is_b2c_rejection(exc):
                # Never "no context available". A refused request shape reported
                # as an empty memory is the exact failure this contract removes:
                # one instance served thousands of consecutive empty reads and
                # nobody could see anything wrong.
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "synap_search: this Synap instance is B2C "
                                f"(user_context_isolation={B2C_ISOLATION!r}) and refuses a "
                                "customer_id. Remove customer_id from "
                                f"create_synap_mcp_server(). ({exc})"
                            ),
                        }
                    ],
                    "isError": True,
                }
            return {
                "content": [
                    {"type": "text", "text": f"synap_search: no context available ({exc.__class__.__name__})."}
                ],
                "isError": False,
            }

        text = (getattr(response, "formatted_context", None) or "").strip()
        if not text:
            text = "synap_search: no relevant context."
        return {"content": [{"type": "text", "text": text}]}

    @tool(
        "synap_remember",
        "Persist an explicit fact, preference, or note to the user's "
        "Synap memory for future recall. Call this when the user shares "
        "something worth remembering across sessions.",
        {"content": str, "metadata": dict},
    )
    async def synap_remember(args: dict[str, Any]) -> dict[str, Any]:
        content = args.get("content", "")
        if not content or not str(content).strip():
            return {
                "content": [
                    {"type": "text", "text": "synap_remember: missing `content` argument."}
                ],
                "isError": True,
            }
        metadata = args.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.setdefault("source", "claude_agent_sdk")

        async def _create(scoped_customer_id):
            return await sdk.memories.create(
                document=str(content),
                user_id=user_id,
                customer_id=scoped_customer_id,
                metadata=metadata,
            )

        try:
            result = await _without_refused_customer_id(scope, _create)
        except Exception as exc:  # noqa: BLE001 — surface as tool error
            logger.error(
                "synap_remember: sdk.memories.create failed user_id=%s error=%s",
                user_id, exc, exc_info=True,
            )
            if _is_b2c_rejection(exc):
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "synap_remember: this Synap instance is B2C "
                                f"(user_context_isolation={B2C_ISOLATION!r}) and refuses a "
                                "customer_id. Remove customer_id from "
                                f"create_synap_mcp_server(). ({exc})"
                            ),
                        }
                    ],
                    "isError": True,
                }
            return {
                "content": [
                    {"type": "text", "text": f"synap_remember: ingestion failed ({exc})."}
                ],
                "isError": True,
            }

        ingestion_id = getattr(result, "ingestion_id", None) or ""
        return {
            "content": [
                {"type": "text", "text": f"synap_remember: recorded (ingestion_id={ingestion_id})."}
            ]
        }

    return [synap_search, synap_remember]


def create_synap_mcp_server(
    sdk: MaximemSynapSDK,
    user_id: str,
    customer_id: str = "",
    conversation_id: Optional[str] = None,
    *,
    name: str = "synap",
    version: str = "0.1.0",
    mode: str = "accurate",
) -> McpSdkServerConfig:
    """Build an in-process MCP server exposing Synap as Claude tools.

    Args:
        sdk: A configured ``MaximemSynapSDK``. Call ``await sdk.initialize()``
            first where you can: that is when the SDK learns the instance's
            scoping mode, and knowing it up front is what lets a wrongly-set
            ``customer_id`` be dropped before the first call rather than after it.
        user_id: The end-user this server reads and writes for. Required in both
            modes, and on a B2C instance it is the WHOLE identity.
        customer_id: B2B ONLY. Required on an instance with
            ``user_context_isolation="strict"``, where a ``user_id`` on its own is
            an error. On a B2C instance (``equals_customer``) leave it "": the
            value is refused by the SDK and by the API with HTTP 400, and there is
            no customer scope to address. If it is set anyway on a B2C instance it
            is dropped with a WARNING rather than sent, so memory keeps working.
        conversation_id: Optional static conversation id for the fetch calls.
        name: Server name; sets the ``mcp__<name>__<tool>`` prefix.
        version: Server version string.
        mode: Synap fetch mode, "accurate" (default) or "fast".
    """
    if sdk is None:
        raise ValueError("create_synap_mcp_server requires a non-None sdk")
    if not user_id:
        raise ValueError("create_synap_mcp_server requires a non-empty user_id")

    tools = _build_synap_tools(
        sdk=sdk,
        user_id=user_id,
        customer_id=customer_id,
        conversation_id=conversation_id,
        mode=mode,
    )
    return create_sdk_mcp_server(name=name, version=version, tools=tools)
