"""Context controllers for different scopes.

These controllers provide the facade-layer API for context operations.
They delegate to the HTTP transport for cloud communication and to
the response handler for parsing.

Note: The primary SDK interface uses the *Interface classes in sdk.py
directly. These controllers exist for users who prefer the facade pattern
or for use in custom orchestration flows.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.context import ContextBundle
    from ..models.enums import ContextType, CompactionLevel
    from ..orchestration.request_builder import RequestBuilder
    from ..orchestration.response_handler import ResponseHandler
    from ..transport.http_client import HTTPTransport
    from ..auth.models import AuthContext


class ConversationContextController:
    """Controller for conversation-scoped context operations."""

    def __init__(
        self,
        orchestrator: "RequestBuilder",
        validator: "ResponseHandler",
        transport: Optional["HTTPTransport"] = None,
        auth_provider=None,
    ):
        """Initialize conversation context controller.

        Args:
            orchestrator: Request builder
            validator: Response handler
            transport: HTTP transport for API calls
            auth_provider: Async callable returning AuthContext
        """
        self.orchestrator = orchestrator
        self.validator = validator
        self._transport = transport
        self._auth_provider = auth_provider

    def _ensure_transport(self):
        if not self._transport:
            raise RuntimeError("Transport not initialized. Call SDK.initialize() first.")

    async def fetch(
        self,
        conversation_id: str,
        search_query: Optional[List[str]] = None,
        max_results: int = 10,
        types: Optional[List["ContextType"]] = None,
        mode: str = "fast",
        precision_level: str = "high",
    ):
        """Fetch context for a conversation.

        Args:
            conversation_id: Conversation identifier
            search_query: Optional list of search queries
            max_results: Maximum number of results
            types: Optional list of context types to retrieve
            mode: "fast" or "accurate" retrieval mode

        Returns:
            Parsed context response
        """
        self._ensure_transport()

        auth_context = await self._auth_provider() if self._auth_provider else None

        payload = {
            "conversation_id": conversation_id,
            "search_query": search_query or [],
            "max_results": max_results,
            "types": [t.value for t in types] if types else ["all"],
            "mode": mode,
            **({"precision_level": precision_level} if precision_level != "high" else {}),
        }

        result = await self._transport.post(
            "/v1/context/conversation/fetch",
            auth_context=auth_context,
            json=payload,
        )

        return self.validator.parse_context_response(result)

    async def compact(
        self,
        conversation_id: str,
        strategy: Optional[str] = None,
        compaction_level: Optional[str] = None,
        target_tokens: Optional[int] = None,
        force: bool = False,
    ):
        """Trigger conversation compaction.

        Args:
            conversation_id: Conversation identifier
            strategy: Compaction strategy ("aggressive", "balanced", "conservative", "adaptive")
            compaction_level: Backward-compatible alias for strategy
            target_tokens: Target token count after compaction
            force: Compact even if under threshold

        Returns:
            Compaction response with compaction_id and status
        """
        self._ensure_transport()

        auth_context = await self._auth_provider() if self._auth_provider else None

        resolved_strategy = strategy or compaction_level

        payload = {
            "conversation_id": conversation_id,
            "force": force,
        }
        if resolved_strategy:
            payload["strategy"] = resolved_strategy
        if target_tokens:
            payload["target_tokens"] = target_tokens

        result = await self._transport.post(
            "/v1/conversations/compact",
            auth_context=auth_context,
            json=payload,
        )

        return self.validator.parse_compaction_response(result)

    async def get_compacted(
        self,
        conversation_id: str,
        version: Optional[int] = None,
        format: str = "structured",
    ):
        """Get compacted context for a conversation.

        Args:
            conversation_id: Conversation identifier
            version: Specific version (optional, defaults to latest)
            format: Output format ("structured", "narrative", "bullet_points")

        Returns:
            Compacted context response or None if not found
        """
        self._ensure_transport()

        auth_context = await self._auth_provider() if self._auth_provider else None

        params = {"format": format}
        if version is not None:
            params["version"] = version

        try:
            result = await self._transport.get(
                f"/v1/conversations/{conversation_id}/compacted",
                auth_context=auth_context,
                params=params,
            )
            return self.validator.parse_compacted_context_response(result)
        except Exception as e:
            if "404" in str(e):
                return None
            raise

    async def compaction_status(
        self,
        conversation_id: str,
    ):
        """Get compaction status for a conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Compaction status response
        """
        self._ensure_transport()

        auth_context = await self._auth_provider() if self._auth_provider else None

        result = await self._transport.get(
            f"/v1/conversations/{conversation_id}/compaction/status",
            auth_context=auth_context,
        )

        return self.validator.parse_compaction_status_response(result)


class UserContextController:
    """Controller for user-scoped context operations."""

    def __init__(
        self,
        orchestrator: "RequestBuilder",
        validator: "ResponseHandler",
        transport: Optional["HTTPTransport"] = None,
        auth_provider=None,
    ):
        self.orchestrator = orchestrator
        self.validator = validator
        self._transport = transport
        self._auth_provider = auth_provider

    def _ensure_transport(self):
        if not self._transport:
            raise RuntimeError("Transport not initialized. Call SDK.initialize() first.")

    async def fetch(
        self,
        user_id: str,
        conversation_id: Optional[str] = None,
        search_query: Optional[List[str]] = None,
        max_results: int = 10,
        types: Optional[List["ContextType"]] = None,
        mode: str = "fast",
        precision_level: str = "high",
    ):
        """Fetch context for a user.

        Args:
            user_id: User identifier
            conversation_id: Optional conversation scope
            search_query: Optional list of search queries
            max_results: Maximum number of results
            types: Optional list of context types to retrieve
            mode: "fast" or "accurate" retrieval mode

        Returns:
            Parsed context response
        """
        self._ensure_transport()

        auth_context = await self._auth_provider() if self._auth_provider else None

        payload = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "search_query": search_query or [],
            "max_results": max_results,
            "types": [t.value for t in types] if types else ["all"],
            "mode": mode,
            **({"precision_level": precision_level} if precision_level != "high" else {}),
        }

        result = await self._transport.post(
            "/v1/context/user/fetch",
            auth_context=auth_context,
            json=payload,
        )

        return self.validator.parse_context_response(result)


class CustomerContextController:
    """Controller for customer-scoped context operations."""

    def __init__(
        self,
        orchestrator: "RequestBuilder",
        validator: "ResponseHandler",
        transport: Optional["HTTPTransport"] = None,
        auth_provider=None,
    ):
        self.orchestrator = orchestrator
        self.validator = validator
        self._transport = transport
        self._auth_provider = auth_provider

    def _ensure_transport(self):
        if not self._transport:
            raise RuntimeError("Transport not initialized. Call SDK.initialize() first.")

    async def fetch(
        self,
        customer_id: str,
        conversation_id: Optional[str] = None,
        search_query: Optional[List[str]] = None,
        max_results: int = 10,
        types: Optional[List["ContextType"]] = None,
        mode: str = "fast",
        precision_level: str = "high",
    ):
        """Fetch context for a customer.

        Args:
            customer_id: Customer identifier
            conversation_id: Optional conversation scope
            search_query: Optional list of search queries
            max_results: Maximum number of results
            types: Optional list of context types to retrieve
            mode: "fast" or "accurate" retrieval mode

        Returns:
            Parsed context response
        """
        self._ensure_transport()

        auth_context = await self._auth_provider() if self._auth_provider else None

        payload = {
            "customer_id": customer_id,
            "conversation_id": conversation_id,
            "search_query": search_query or [],
            "max_results": max_results,
            "types": [t.value for t in types] if types else ["all"],
            "mode": mode,
            **({"precision_level": precision_level} if precision_level != "high" else {}),
        }

        result = await self._transport.post(
            "/v1/context/customer/fetch",
            auth_context=auth_context,
            json=payload,
        )

        return self.validator.parse_context_response(result)


class ClientContextController:
    """Controller for client-scoped context operations."""

    def __init__(
        self,
        orchestrator: "RequestBuilder",
        validator: "ResponseHandler",
        transport: Optional["HTTPTransport"] = None,
        auth_provider=None,
    ):
        self.orchestrator = orchestrator
        self.validator = validator
        self._transport = transport
        self._auth_provider = auth_provider

    def _ensure_transport(self):
        if not self._transport:
            raise RuntimeError("Transport not initialized. Call SDK.initialize() first.")

    async def fetch(
        self,
        conversation_id: Optional[str] = None,
        search_query: Optional[List[str]] = None,
        max_results: int = 10,
        types: Optional[List["ContextType"]] = None,
        mode: str = "fast",
        precision_level: str = "high",
    ):
        """Fetch context for a client (org-level).

        Args:
            conversation_id: Optional conversation scope
            search_query: Optional list of search queries
            max_results: Maximum number of results
            types: Optional list of context types to retrieve
            mode: "fast" or "accurate" retrieval mode

        Returns:
            Parsed context response
        """
        self._ensure_transport()

        auth_context = await self._auth_provider() if self._auth_provider else None

        payload = {
            "conversation_id": conversation_id,
            "search_query": search_query or [],
            "max_results": max_results,
            "types": [t.value for t in types] if types else ["all"],
            "mode": mode,
            **({"precision_level": precision_level} if precision_level != "high" else {}),
        }

        result = await self._transport.post(
            "/v1/context/client/fetch",
            auth_context=auth_context,
            json=payload,
        )

        return self.validator.parse_context_response(result)
