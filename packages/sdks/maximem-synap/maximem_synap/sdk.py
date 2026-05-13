"""Main SDK class - the developer-facing interface."""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .config_utils import configure_logging, get_default_storage_path, merge_config
from .registry import SDKRegistry
from .models.config import SDKConfig, TimeoutConfig, RetryPolicy
from .models.context import (
    ContextResponse,
    CompactionResponse,
    CompactionStatusResponse,
    CompactionTriggerResponse,
    ContextForPromptResponse,
    ResponseMetadata,
    UnifiedContextResponse,
)
from .models.enums import CompactionLevel, ContextScope
from .models.errors import (
    SynapError,
    AuthenticationError,
    InvalidInputError,
    ContextNotFoundError,
)
from .auth.manager import CredentialManager
from .cache.manager import CacheManager, CacheScope
from .transport.http_client import HTTPTransport
from .transport.grpc_client import GRPCTransport
from .telemetry.collector import TelemetryCollector, emit_fetch_event
from .telemetry.transport import TelemetryTransport
from .utils.correlation import generate_correlation_id
from ._version import __version__
from .memories.interface import MemoriesInterface
from .facade.instance import InstanceController
from .facade.conversation import ConversationController


logger = logging.getLogger("synap.sdk")


def _build_anticipation_response(
    bundle: Dict,
    correlation_id: str,
    start_time: datetime,
    scope: str,
    mode: str,
    telemetry_collector,
) -> ContextResponse:
    """Convert an anticipated bundle to a ContextResponse.

    Shared helper used by all scope fetch methods when the anticipation
    cache returns a hit.
    """
    items_by_type = bundle.get("items_by_type", {})
    context_data = {ctx_type: items for ctx_type, items in items_by_type.items()}
    if bundle.get("conversation_context"):
        context_data["conversation_context"] = bundle["conversation_context"]
    metadata = ResponseMetadata(
        correlation_id=correlation_id,
        ttl_seconds=0,
        source="anticipation",
        retrieved_at=datetime.now(timezone.utc),
    )
    latency_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
    emit_fetch_event(
        telemetry_collector,
        scope=scope,
        correlation_id=correlation_id,
        latency_ms=latency_ms,
        cache_hit=True,
        mode=mode,
        cache_origin="anticipation_cache",
    )
    return ContextResponse.from_cloud_response(context_data, metadata)


async def _emit_context_fetch_event(
    sdk: "MaximemSynapSDK",
    scope: str,
    search_query: Optional[List[str]],
    types: Optional[List[str]],
    mode: str,
    source: str,
    items_count: int,
    conversation_id: str = "",
    user_id: str = "",
    customer_id: str = "",
) -> None:
    """Emit a context_fetch event over the gRPC stream (fire-and-forget)."""
    try:
        if not sdk.instance.is_listening:
            return
        await sdk.instance._controller._transport.send({
            "event_type": "context_fetch",
            "content": "",
            "role": "system",
            "conversation_id": conversation_id,
            "user_id": user_id,
            "customer_id": customer_id,
            "session_id": "",
            "search_queries": search_query or [],
            "context_types": types or [],
            "metadata": {
                "source": source,
                "items_count": str(items_count),
                "mode": mode,
                "scope": scope,
            },
        })
    except Exception as e:
        logger.debug("context_fetch emit failed (non-fatal): %s", e)


async def _emit_context_used_event(
    sdk,
    *,
    bundle_id: str,
    served_item_ids: List[str],
    scope: str,
    conversation_id: str = "",
    user_id: str = "",
    customer_id: str = "",
    source_bundle_ids: Optional[List[str]] = None,
) -> None:
    """Emit a context_used event over the Listen stream (fire-and-forget).

    Fired by each fetch() path after a successful anticipation cache hit.
    Drives the server's learning loop: per-prefetch outcome scoring,
    per-pattern hit rates, per-query-family priors. Privacy: no raw user
    prompt content is sent — only ids and scope.
    """
    try:
        if not sdk.instance.is_listening:
            return
        await sdk.instance._controller._transport.send_context_used(
            bundle_id=bundle_id,
            conversation_id=conversation_id,
            user_id=user_id,
            customer_id=customer_id,
            served_item_ids=served_item_ids,
            scope=scope,
            source_bundle_ids=source_bundle_ids or [],
        )
    except Exception as e:
        logger.debug("context_used emit failed (non-fatal): %s", e)


def _extract_served_item_ids(anticipated: Dict) -> List[str]:
    """Pull item ids out of an anticipation cache lookup result."""
    ids: List[str] = []
    seen = set()
    for item in anticipated.get("items") or []:
        iid = item.get("item_id") or item.get("id") or ""
        if iid and iid not in seen:
            seen.add(iid)
            ids.append(iid)
    return ids


def _default_tool_description(scope: str) -> str:
    """Default LLM-facing description for the ``as_tool`` helper. Phrased to
    prime the agent to call it for context retrieval."""
    base = (
        "Retrieve stored context (facts, preferences, recent episodes, "
        "emotions, temporal events) from Synap memory."
    )
    if scope == "conversation":
        return (
            base + " Scoped to a specific conversation. Call this when you "
            "need facts already established in the current conversation, "
            "or to reload context after a topic shift."
        )
    if scope == "user":
        return (
            base + " Scoped to the user's long-term memory. Call this when "
            "you need to recall who the user is, their preferences, or "
            "their history across conversations."
        )
    if scope == "customer":
        return (
            base + " Scoped to the customer/organization. Call this when "
            "you need facts about the customer org rather than an individual."
        )
    if scope == "client":
        return (
            base + " Scoped to the integrating client/product. Call this for "
            "product-level knowledge (e.g. policies, documentation)."
        )
    return (
        base + " Cross-scope: merges conversation, user, customer, and "
        "client memory in one call. Call this at the start of a turn when "
        "you need general grounding before responding."
    )


def _tool_input_schema(scope: str, *, has_conversation_id: bool) -> Dict[str, Any]:
    """JSON-Schema for the LLM-callable arguments. Scope ids closed over by
    ``as_tool`` are NOT in the schema — the LLM only chooses runtime args
    (query, types, limits)."""
    props: Dict[str, Any] = {
        "search_query": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Optional list of search queries describing what context "
                "you need. Free-form natural language is fine. Omit to "
                "retrieve the most relevant recent context."
            ),
        },
        "max_results": {
            "type": "integer",
            "minimum": 1,
            "maximum": 50,
            "description": "Maximum items to return (default 10).",
        },
        "types": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["facts", "preferences", "episodes", "emotions", "temporal_events"],
            },
            "description": (
                "Filter to specific memory categories. Omit to retrieve all."
            ),
        },
        "mode": {
            "type": "string",
            "enum": ["fast", "accurate"],
            "description": (
                "Retrieval mode. 'fast' = low-latency (~50ms); 'accurate' = "
                "LLM-decomposed multi-query (~200-500ms). Default 'fast'."
            ),
        },
    }
    if scope == "conversation" and not has_conversation_id:
        # Closed-over conversation_id wasn't provided; LLM must supply per call.
        props["conversation_id"] = {
            "type": "string",
            "description": "The conversation id to fetch context for.",
        }
        required = ["conversation_id"]
    else:
        required = []
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


async def _invoke_scope_fetch(
    *,
    sdk: "MaximemSynapSDK",
    scope: str,
    user_id: Optional[str],
    customer_id: Optional[str],
    conversation_id: Optional[str],
    call_args: Dict[str, Any],
) -> Dict[str, Any]:
    """Dispatch into the right scope's fetch with closed-over ids merged in.

    Returns a plain dict (not a Pydantic model) so the host runtime can
    JSON-serialize directly into the LLM's tool-result message.
    """
    search_query = call_args.get("search_query")
    max_results = int(call_args.get("max_results") or 10)
    types = call_args.get("types")
    mode = str(call_args.get("mode") or "fast")
    call_conv_id = call_args.get("conversation_id") or conversation_id

    if scope == "unified":
        response = await sdk.fetch(
            conversation_id=call_conv_id,
            user_id=user_id,
            customer_id=customer_id,
            search_query=search_query,
            max_results=max_results,
            types=types,
            mode=mode,
        )
        return {
            "formatted_context": response.formatted_context,
            "scopes_queried": response.scopes_queried,
            "total_items": response.total_items,
        }

    if scope == "conversation":
        if not call_conv_id:
            return {"error": "conversation_id is required", "items": []}
        response = await sdk.conversation.context.fetch(
            conversation_id=call_conv_id,
            search_query=search_query,
            max_results=max_results,
            types=types,
            mode=mode,
            user_id=user_id,
        )
    elif scope == "user":
        response = await sdk.user.context.fetch(
            user_id=user_id,
            conversation_id=call_conv_id,
            search_query=search_query,
            max_results=max_results,
            types=types,
            mode=mode,
            customer_id=customer_id,
        )
    elif scope == "customer":
        response = await sdk.customer.context.fetch(
            customer_id=customer_id,
            conversation_id=call_conv_id,
            search_query=search_query,
            max_results=max_results,
            types=types,
            mode=mode,
        )
    else:  # client
        response = await sdk.client.context.fetch(
            conversation_id=call_conv_id,
            search_query=search_query,
            max_results=max_results,
            types=types,
            mode=mode,
        )

    return {
        "facts": [f.model_dump() if hasattr(f, "model_dump") else f for f in response.facts],
        "preferences": [
            p.model_dump() if hasattr(p, "model_dump") else p for p in response.preferences
        ],
        "episodes": [e.model_dump() if hasattr(e, "model_dump") else e for e in response.episodes],
        "emotions": [
            em.model_dump() if hasattr(em, "model_dump") else em for em in response.emotions
        ],
        "temporal_events": [
            te.model_dump() if hasattr(te, "model_dump") else te
            for te in response.temporal_events
        ],
    }


def _merge_user_summary_into_response(
    response: ContextResponse,
    summary_bundle: Dict,
    max_summary_items: int = 3,
) -> ContextResponse:
    """Merge items from a user summary bundle into an existing response."""
    _TYPE_TO_ATTR = {
        "facts": "facts",
        "preferences": "preferences",
        "episodes": "episodes",
        "emotions": "emotions",
        "temporal_events": "temporal_events",
    }
    summary_items = summary_bundle.get("items_by_type", {})
    for ctx_type, items in summary_items.items():
        attr_name = _TYPE_TO_ATTR.get(ctx_type)
        if attr_name is None or not isinstance(items, list):
            continue
        existing = getattr(response, attr_name)
        existing_ids = {item.id for item in existing}
        new_raw = [
            item for item in items
            if isinstance(item, dict) and item.get("item_id") not in existing_ids
        ][:max_summary_items]
        if not new_raw:
            continue
        normalized = [
            {
                "id": item.get("item_id", item.get("id", "")),
                "content": item.get("content", ""),
                "confidence": item.get("confidence", 0.0),
                "source": item.get("source", ""),
                "category": item.get("context_type", item.get("category", "")),
                "emotion_type": item.get("context_type", item.get("emotion_type", "")),
                "strength": item.get("strength", item.get("confidence", 0.0)),
                "summary": item.get("summary", item.get("content", "")),
                "significance": item.get("significance", item.get("confidence", 0.0)),
                "intensity": item.get("intensity", item.get("confidence", 0.0)),
                "context": item.get("context", item.get("content", "")),
                "participants": item.get("participants", []),
                "extracted_at": item.get("extracted_at") or item.get("created_at"),
                "occurred_at": item.get("occurred_at") or item.get("extracted_at") or item.get("created_at"),
                "detected_at": item.get("detected_at") or item.get("extracted_at") or item.get("created_at"),
                "metadata": item.get("metadata", {}),
                "event_date": item.get("event_date"),
                "valid_until": item.get("valid_until"),
                "temporal_category": item.get("temporal_category"),
                "temporal_confidence": item.get("temporal_confidence", 0.0),
            }
            for item in new_raw
        ]
        typed_chunk = ContextResponse.from_cloud_response(
            {ctx_type: normalized}, response.metadata
        )
        existing.extend(getattr(typed_chunk, attr_name))
    return response


def _count_context_items(response: ContextResponse) -> int:
    return (
        len(response.facts)
        + len(response.preferences)
        + len(response.episodes)
        + len(response.emotions)
        + len(response.temporal_events)
    )


class MaximemSynapSDK:
    """Synap SDK - Agentic Context Management.

    Usage:
        sdk = MaximemSynapSDK()  # reads SYNAP_API_KEY from env
        await sdk.initialize()

        # Fetch context
        ctx = await sdk.conversation.context.fetch(conversation_id="...")

        # Compact context
        compacted = await sdk.conversation.context.compact(conversation_id="...")

        # Listen to agent activity
        await sdk.instance.listen()
    """

    def __init__(
        self,
        instance_id: str = "",
        api_key: Optional[str] = None,
        config: Optional[SDKConfig] = None,
        _force_new: bool = False,
    ):
        """Create or get SDK instance.

        Args:
            instance_id: Optional instance ID. If omitted, resolved from the
                API key on initialize() via GET /api/v1/auth/whoami.
            api_key: Synap API key. If omitted, read from SYNAP_API_KEY env var.
            config: Optional configuration overrides.
            _force_new: Force create new instance (for testing).
        """
        # Check singleton registry
        if not _force_new:
            existing = SDKRegistry.get(instance_id)
            if existing is not None:
                # Return existing instance - copy its state
                self.__dict__ = existing.__dict__
                return

        self.instance_id = instance_id or os.environ.get("SYNAP_INSTANCE_ID", "")
        self._api_key = api_key
        self._config = config or SDKConfig()
        self._initialized = False

        # Components (initialized lazily or in initialize())
        self._credential_manager: Optional[CredentialManager] = None
        self._cache_manager: Optional[CacheManager] = None
        self._http_transport: Optional[HTTPTransport] = None
        self._grpc_transport: Optional[GRPCTransport] = None
        self._telemetry_collector: Optional[TelemetryCollector] = None
        self._telemetry_transport: Optional[TelemetryTransport] = None
        self._client_id: Optional[str] = None

        from .cache.anticipation_cache import AnticipationCache
        self._anticipation_cache = AnticipationCache()

        self._turn_counters: Dict[str, int] = {}
        self._user_summary_interval: int = 5

        # Sub-interfaces
        self.conversation = ConversationInterface(self)
        self.user = UserInterface(self)
        self.customer = CustomerInterface(self)
        self.client = ClientInterface(self)
        self.instance = InstanceInterface(self)
        self.cache = CacheInterface(self)
        self.memories = MemoriesInterface(self)

        # Credits is attached lazily so circular imports don't bite
        # (credits.py imports MaximemSynapSDK via TYPE_CHECKING only).
        from .credits import CreditsInterface
        self.credits = CreditsInterface(self)

        # Configure logging
        configure_logging(self._config.log_level)

        # Register in singleton registry
        if not _force_new:
            SDKRegistry.register(instance_id, self)

    def configure(self, **kwargs) -> None:
        """Update SDK configuration.

        Args:
            storage_path: Override default cache storage path
            cache_backend: "sqlite" or None
            session_timeout_minutes: Session timeout (5-1440)
            timeouts: TimeoutConfig or dict
            retry_policy: RetryPolicy, dict, or None to disable
            log_level: "DEBUG", "INFO", "WARNING", "ERROR"
            logger: Custom logger instance
        """
        if self._initialized:
            raise InvalidInputError("Cannot reconfigure after initialization")

        # Handle custom logger
        if "logger" in kwargs:
            custom_logger = kwargs.pop("logger")
            # Replace the SDK logger
            global logger
            logger = custom_logger

        self._config = merge_config(self._config, kwargs)
        configure_logging(self._config.log_level)

    def _increment_turn(self, conversation_id: Optional[str]) -> int:
        """Increment and return the turn count for a conversation."""
        key = conversation_id or "_global"
        self._turn_counters[key] = self._turn_counters.get(key, 0) + 1
        return self._turn_counters[key]

    def _should_inject_user_summary(self, conversation_id: Optional[str]) -> bool:
        """Check if user summary should be injected this turn."""
        key = conversation_id or "_global"
        count = self._turn_counters.get(key, 0)
        return count > 0 and count % self._user_summary_interval == 0

    async def initialize(self) -> None:
        """Initialize the SDK.

        Must be called before any context operations. Reads the Synap API key
        from the api_key= kwarg (highest priority) or the SYNAP_API_KEY env var.
        """
        if self._initialized:
            return

        self._credential_manager = CredentialManager(instance_id=self.instance_id)

        try:
            credentials = self._credential_manager.load(api_key=self._api_key)
            self._client_id = credentials.client_id
            if credentials.instance_id:
                self.instance_id = credentials.instance_id
        except Exception as e:
            raise AuthenticationError(f"SDK initialization failed: {e}") from e

        # Initialize cache
        if self._config.cache_backend:
            self._cache_manager = CacheManager(
                client_id=self._client_id,
                storage_path=self._config.storage_path,
                enabled=True,
            )
        else:
            self._cache_manager = CacheManager(
                client_id=self._client_id,
                enabled=False,
            )

        # Initialize main HTTP transport
        self._http_transport = HTTPTransport(
            instance_id=self.instance_id,
            base_url=self._config.api_base_url,
            timeouts=self._config.timeouts,
            retry_policy=self._config.retry_policy,
            telemetry_callback=self._on_telemetry_event,
        )

        # Initialize telemetry transport
        self._telemetry_transport = TelemetryTransport(
            base_url=self._http_transport.base_url,
            get_auth_context=self._get_auth_context_for_telemetry,
        )

        # Initialize telemetry collector
        self._telemetry_collector = TelemetryCollector(
            instance_id=self.instance_id,
            client_id=self._client_id,
            sdk_version=__version__,
            transport_callback=self._telemetry_transport.send,
            enabled=True,
        )
        await self._telemetry_collector.start()

        # Emit init event
        from .telemetry.models import TelemetryEventType
        self._telemetry_collector.emit(
            event_type=TelemetryEventType.SDK_INIT,
            status="success",
        )

        self._initialized = True
        logger.info(f"SDK initialized for instance {self.instance_id}")

    def anticipation_cache_snapshot(self) -> Dict[str, Any]:
        """Return a read-only summary of the in-process anticipation cache.

        Diagnostic-only view — counts, scope breakdown per bundle, recent
        item content previews, and the BM25 corpus vocabulary. Useful for
        verifying the gRPC anticipation pipeline is delivering bundles
        with the right shape and that lookup tokens are present.

        The cache itself remains private; this method projects a stable
        public shape that won't change with internal refactors.
        """
        if self._anticipation_cache is None:
            return {
                "total_entries": 0,
                "total_item_records": 0,
                "scope_breakdown_overall": {},
                "corpus_vocab_size": 0,
                "corpus_vocab_sample": [],
                "item_records": [],
                "bundles": [],
            }

        from collections import Counter

        cache = self._anticipation_cache
        overall_scope: Counter = Counter()
        bundles: List[Dict[str, Any]] = []
        for bundle_id, entry in cache._entries.items():
            per_bundle: Counter = Counter()
            item_previews: List[Dict[str, Any]] = []
            items_by_type = entry.bundle.get("items_by_type", {})
            for t, items in items_by_type.items():
                for it in items or []:
                    sc = it.get("scope")
                    overall_scope[sc] += 1
                    per_bundle[sc] += 1
                    item_previews.append({
                        "type": t,
                        "scope": sc,
                        "content": (it.get("content") or "")[:140],
                    })
            bundles.append({
                "bundle_id": bundle_id,
                "entity_id": entry.entity_id,
                "conversation_id": entry.conversation_id,
                "bundle_type": entry.bundle_type,
                "search_queries": entry.search_queries,
                "scope_counts": dict(per_bundle),
                "total_items": sum(per_bundle.values()),
                "items": item_previews,
            })

        corpus_vocab_sample = sorted(getattr(cache, "_corpus_vocab", set()))[:80]
        item_record_previews = [
            {
                "bundle_id": rec.bundle_id,
                "item_type": rec.item_type,
                "tokens": rec.tokens[:25],
                "content": (rec.content or "")[:140],
            }
            for rec in getattr(cache, "_items", [])[:20]
        ]
        return {
            "total_entries": len(cache._entries),
            "total_item_records": len(cache._items),
            "scope_breakdown_overall": dict(overall_scope),
            "corpus_vocab_size": len(getattr(cache, "_corpus_vocab", set())),
            "corpus_vocab_sample": corpus_vocab_sample,
            "item_records": item_record_previews,
            "bundles": bundles,
        }

    async def shutdown(self) -> None:
        """Gracefully shutdown the SDK.

        Flushes telemetry, closes connections, and releases resources.
        """
        logger.info("Shutting down SDK")

        # Emit shutdown event
        if self._telemetry_collector:
            from .telemetry.models import TelemetryEventType
            self._telemetry_collector.emit(
                event_type=TelemetryEventType.SDK_SHUTDOWN,
                status="success",
            )
            await self._telemetry_collector.stop()

        # Close telemetry transport
        if self._telemetry_transport:
            await self._telemetry_transport.close()

        # Close transports
        if self._http_transport:
            await self._http_transport.close()

        if self._grpc_transport:
            await self._grpc_transport.close()

        # Close cache
        if self._cache_manager:
            self._cache_manager.close()

        # Unregister from singleton
        SDKRegistry.unregister(self.instance_id)

        self._initialized = False
        logger.info("SDK shutdown complete")

    async def fetch(
        self,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        search_query: Optional[List[str]] = None,
        max_results: int = 20,
        types: Optional[List[str]] = None,
        mode: str = "fast",
        include_conversation_context: bool = True,
        scopes: Optional[List[str]] = None,
        include_scope_labels: bool = False,
    ) -> UnifiedContextResponse:
        """Fetch and merge context across all relevant scopes in a single call.

        This is the recommended method for framework integrations. It:
        1. Fetches context from each scope where an identifier is provided
        2. Runs all scope fetches in parallel via asyncio.gather
        3. Merges results, deduplicating by item ID (first scope wins)
        4. Attributes each item to its source scope in scope_map
        5. Optionally includes conversation context
        6. Returns a formatted_context string ready for LLM prompt injection

        Args:
            conversation_id: Conversation scope identifier (optional)
            user_id: User scope identifier (optional)
            customer_id: Customer scope identifier (optional, required for B2B)
            search_query: Search queries applied to all scopes
            max_results: Max results per scope (total may be higher)
            types: Memory types to include (default: all)
            mode: Retrieval mode - "fast" (default) or "accurate"
            include_conversation_context: Include compacted history + recent messages
            scopes: Explicitly limit which scopes to query (e.g. ["user", "customer"]).
                    Default: all scopes for which an identifier is provided.
            include_scope_labels: If True, annotate each item with its source scope
                                  in the formatted output.

        Returns:
            UnifiedContextResponse with merged items, scope attribution, and
            formatted_context ready for LLM injection.

        Examples:
            # Fetch everything for a conversation
            ctx = await sdk.fetch(
                conversation_id="conv-123",
                user_id="user-456",
                customer_id="cust-789",
                search_query=["user preferences"],
            )
            print(ctx.formatted_context)  # Ready for LLM prompt

            # Fetch only user + customer context (skip conversation scope)
            ctx = await sdk.fetch(
                user_id="user-456",
                customer_id="cust-789",
                scopes=["user", "customer"],
            )
        """
        self._ensure_initialized()

        # Build parallel fetch tasks based on provided identifiers and scope filter
        tasks = []
        scope_labels = []

        if conversation_id and (not scopes or "conversation" in scopes):
            # Section 15: thread user_id into the conversation-scope sub-fetch
            # so the SDK's anticipation cache can apply its strict per-user
            # scope filter. Without this the conversation lookup falls back to
            # broad matching, losing the cross-user privacy guarantee.
            tasks.append(self.conversation.context.fetch(
                conversation_id=conversation_id,
                search_query=search_query,
                max_results=max_results,
                types=types,
                mode=mode,
                user_id=user_id,
            ))
            scope_labels.append("conversation")

        if user_id and (not scopes or "user" in scopes):
            tasks.append(self.user.context.fetch(
                user_id=user_id,
                conversation_id=conversation_id,
                search_query=search_query,
                max_results=max_results,
                types=types,
                mode=mode,
                customer_id=customer_id,
            ))
            scope_labels.append("user")

        if customer_id and (not scopes or "customer" in scopes):
            tasks.append(self.customer.context.fetch(
                customer_id=customer_id,
                conversation_id=conversation_id,
                search_query=search_query,
                max_results=max_results,
                types=types,
                mode=mode,
            ))
            scope_labels.append("customer")

        if not scopes or "client" in scopes:
            # Client scope doesn't need an external ID — it's inferred from auth
            if scopes and "client" in scopes:
                tasks.append(self.client.context.fetch(
                    conversation_id=conversation_id,
                    search_query=search_query,
                    max_results=max_results,
                    types=types,
                    mode=mode,
                ))
                scope_labels.append("client")

        if not tasks:
            # No scopes to query — return empty response
            return UnifiedContextResponse(
                scopes_queried=[],
                formatted_context="",
            )

        # Fetch all scopes in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failed scopes (log but don't raise)
        successful_results = []
        for label, result in zip(scope_labels, results):
            if isinstance(result, Exception):
                logger.warning(
                    "Scope '%s' fetch failed in unified fetch (non-fatal): %s",
                    label, result,
                )
            else:
                successful_results.append((label, result))

        # Merge and deduplicate across scopes
        merged = UnifiedContextResponse.merge(successful_results)

        if include_conversation_context and conversation_id:
            try:
                prompt_ctx = await self.conversation.context.get_context_for_prompt(
                    conversation_id=conversation_id,
                )
                merged.conversation_context = prompt_ctx
            except Exception as e:
                logger.debug(
                    "get_context_for_prompt failed in unified fetch (non-fatal): %s", e
                )

        # Generate formatted context string
        merged.formatted_context = merged.format_for_prompt(
            include_scope=include_scope_labels,
            include_conversation_context=include_conversation_context,
        )

        return merged

    def _ensure_initialized(self) -> None:
        """Ensure SDK is initialized before operations."""
        if not self._initialized:
            raise AuthenticationError(
                "SDK not initialized. Call await sdk.initialize() first."
            )

    async def _get_auth_context(self, correlation_id: Optional[str] = None):
        """Get auth context for requests."""
        self._ensure_initialized()
        return await self._credential_manager.get_auth_context(correlation_id)

    async def _get_auth_context_for_telemetry(self):
        """Get auth context for telemetry transport."""
        return await self._credential_manager.get_auth_context()

    def _on_telemetry_event(self, event: Dict[str, Any]) -> None:
        """Callback for transport telemetry events."""
        if self._telemetry_collector:
            self._telemetry_collector.emit_dict(event)

    def as_tool(
        self,
        *,
        scope: str = "user",
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        style: str = "openai",
    ) -> Dict[str, Any]:
        """Return an LLM-ready tool definition for fetching Synap context.

        Closes over the scope identifiers so the developer (and the LLM
        agent) can't accidentally drop the per-user privacy filter the
        Section 15 anticipation cache relies on. The returned dict carries
        the schema + a bound ``handler`` coroutine the host runtime can
        invoke when the LLM calls the tool.

        Anticipation-friendliness note: tool wrapping works, but the
        ``sdk.fetch(...)`` pre-fetch path is the default-recommended
        integration. Use this helper only when the LLM truly needs agency
        over when context is fetched mid-reasoning. See
        ``docs/architecture/anticipation_feature_overview.md`` for the
        full discussion.

        Args:
            scope: Which scope to fetch from. One of "conversation", "user",
                   "customer", "client", "unified" (cross-scope).
            user_id: Closed-over user id; required for "user" and
                     recommended for "conversation"/"unified" (Section 15).
            customer_id: Closed-over customer id; required for "customer".
            conversation_id: Optional closed-over conversation id. When
                             provided, the tool fetches for this conversation;
                             when omitted, the LLM supplies it per call.
            name: Override the tool name. Defaults to
                  ``synap_fetch_{scope}_context``.
            description: Override the tool description. Defaults to a
                         scope-specific blurb that primes the LLM to call
                         it for context retrieval.
            style: "openai" or "anthropic". Controls the dict shape:
                   - "openai": ``{"type":"function","function":{...}}``
                   - "anthropic": ``{"name","description","input_schema"}``

        Returns:
            Tool definition dict. Always carries an ``async handler`` key
            with the bound coroutine; runtimes that don't use it can ignore.

        Examples:
            >>> tool = sdk.as_tool(scope="user", user_id="user-1")
            >>> # In an OpenAI tool-call loop:
            >>> response = await llm.chat(tools=[tool])
            >>> # In an Anthropic agent loop:
            >>> tool = sdk.as_tool(scope="unified", user_id="u", style="anthropic")
        """
        scope = scope.lower()
        valid = {"conversation", "user", "customer", "client", "unified"}
        if scope not in valid:
            raise InvalidInputError(
                f"scope must be one of {sorted(valid)}, got {scope!r}"
            )

        if scope == "user" and not user_id:
            raise InvalidInputError("scope='user' requires user_id")
        if scope == "customer" and not customer_id:
            raise InvalidInputError("scope='customer' requires customer_id")
        if scope == "conversation" and not user_id:
            # Not a hard error — anticipation cache will refuse cross-user
            # matches anyway — but warn the caller that they're leaving
            # privacy on the floor.
            logger.warning(
                "as_tool(scope='conversation') without user_id: the SDK "
                "anticipation cache cannot apply per-user filtering. Pass "
                "user_id to enable the Section 15 privacy guarantee."
            )

        tool_name = name or f"synap_fetch_{scope}_context"
        tool_desc = description or _default_tool_description(scope)
        schema = _tool_input_schema(scope, has_conversation_id=conversation_id is not None)

        # Closed-over handler. The LLM passes the call-time params; the
        # closed-over scope ids are merged in here so the LLM can't drop
        # them (and the privacy filter holds).
        async def handler(**call_args: Any) -> Dict[str, Any]:
            return await _invoke_scope_fetch(
                sdk=self,
                scope=scope,
                user_id=user_id,
                customer_id=customer_id,
                conversation_id=conversation_id,
                call_args=call_args,
            )

        if style == "openai":
            return {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_desc,
                    "parameters": schema,
                },
                "handler": handler,
            }
        if style == "anthropic":
            return {
                "name": tool_name,
                "description": tool_desc,
                "input_schema": schema,
                "handler": handler,
            }
        raise InvalidInputError(f"style must be 'openai' or 'anthropic', got {style!r}")


# Sub-interfaces for domain-oriented API

class ConversationInterface:
    """Interface for conversation-scoped operations."""

    def __init__(self, sdk: MaximemSynapSDK):
        self._sdk = sdk
        self.context = ConversationContextInterface(sdk)
        self._controller: Optional[ConversationController] = None

    def _ensure_controller(self) -> ConversationController:
        """Lazily create ConversationController after SDK is initialized."""
        if self._controller is None:
            self._sdk._ensure_initialized()
            self._controller = ConversationController(
                transport=self._sdk._http_transport,
                auth_provider=self._sdk._get_auth_context,
            )
        return self._controller

    async def record_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        user_id: str,
        customer_id: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a single conversation message.

        Args:
            conversation_id: Unique conversation identifier
            role: Message role ("user" or "assistant")
            content: Message content text
            user_id: User identifier (required, must be external ID)
            customer_id: Customer identifier (required, must be external ID)
            session_id: Session identifier (optional, auto-generated if not provided)
            metadata: Additional metadata (optional)

        Returns:
            Dict with message_id, conversation_id, session_id, recorded_at
        """
        controller = self._ensure_controller()
        correlation_id = generate_correlation_id(self._sdk.instance_id)
        return await controller.record_message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            user_id=user_id,
            customer_id=customer_id,
            session_id=session_id,
            metadata=metadata,
            correlation_id=correlation_id,
        )

    async def record_messages_batch(
        self,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Record multiple conversation messages in a batch.

        Args:
            messages: List of message dicts with conversation_id, role, content, etc.

        Returns:
            Dict with total, succeeded, failed, results[]
        """
        controller = self._ensure_controller()
        correlation_id = generate_correlation_id(self._sdk.instance_id)
        return await controller.record_messages_batch(
            messages=messages,
            correlation_id=correlation_id,
        )


class ConversationContextInterface:
    """Conversation context operations."""

    def __init__(self, sdk: MaximemSynapSDK):
        self._sdk = sdk

    async def fetch(
        self,
        conversation_id: str,
        search_query: Optional[List[str]] = None,
        max_results: int = 10,
        types: Optional[List[str]] = None,
        mode: str = "fast",
        user_id: Optional[str] = None,
    ) -> ContextResponse:
        """Fetch context for a conversation.

        Args:
            conversation_id: The conversation to fetch context for
            search_query: Optional search queries
            max_results: Maximum results to return (default 10)
            types: Context types to include (default all)
            mode: Retrieval mode - "fast" (default) or "accurate"
                  - "fast": Direct query, low latency (~50-100ms)
                  - "accurate": LLM-enhanced queries, higher quality (~200-500ms)
            user_id: Optional external user id. Section 15 — passing this
                     scopes the in-process anticipation cache lookup to that
                     user, preventing a bundle pushed for User A from being
                     served on User B's conversation lookup. Strongly
                     recommended; will become required in a future release.

        Returns:
            ContextResponse with facts, preferences, episodes, etc.
        """
        self._sdk._ensure_initialized()

        # Validate mode
        valid_modes = ("fast", "accurate")
        if mode not in valid_modes:
            raise InvalidInputError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")

        correlation_id = generate_correlation_id(self._sdk.instance_id)
        start_time = datetime.now(timezone.utc)
        cache_hit = False

        # Build cache key params
        cache_params = {
            "search_query": search_query,
            "max_results": max_results,
            "types": types,
            "mode": mode,
        }

        # Check anticipation cache first (bundles pre-fetched via gRPC stream).
        # Section 15: thread user_id through so the per-user filter can do
        # its job. Without this, the lookup matched conversation_id alone and
        # could return bundles whose entity_id was a *different* user.
        anticipated = self._sdk._anticipation_cache.lookup(
            search_query=search_query,
            entity_id=user_id,
            conversation_id=conversation_id,
        )
        if anticipated:
            response = _build_anticipation_response(
                anticipated, correlation_id, start_time,
                scope="conversation", mode=mode,
                telemetry_collector=self._sdk._telemetry_collector,
            )
            asyncio.ensure_future(_emit_context_used_event(
                self._sdk,
                bundle_id=anticipated.get("bundle_id", ""),
                served_item_ids=_extract_served_item_ids(anticipated),
                scope="conversation",
                conversation_id=conversation_id,
                source_bundle_ids=anticipated.get("source_bundle_ids", []),
            ))
        else:
            # Check local cache
            cached = self._sdk._cache_manager.get(
                scope=CacheScope.CONVERSATION,
                entity_id=conversation_id,
                context_type="context",
                query=cache_params,
            )

            if cached:
                cache_hit = True
                data = json.loads(cached)
                metadata = ResponseMetadata(
                    correlation_id=correlation_id,
                    ttl_seconds=0,  # Already in cache
                    source="cache",
                    retrieved_at=datetime.now(timezone.utc),
                )
                response = ContextResponse.from_cloud_response(data, metadata)
            else:
                # Fetch from cloud
                auth_context = await self._sdk._get_auth_context(correlation_id)

                result = await self._sdk._http_transport.post(
                    "/v1/context/conversation/fetch",
                    auth_context=auth_context,
                    json={
                        "conversation_id": conversation_id,
                        "search_query": search_query,
                        "max_results": max_results,
                        "types": types or ["all"],
                        "mode": mode,
                    },
                    correlation_id=correlation_id,
                )

                metadata = ResponseMetadata(
                    correlation_id=correlation_id,
                    ttl_seconds=result.get("ttl_seconds", 300),
                    source="cloud",
                    retrieved_at=datetime.now(timezone.utc),
                )
                context_data = result.get("context", {})
                if "conversation_context" in result:
                    context_data["conversation_context"] = result["conversation_context"]
                response = ContextResponse.from_cloud_response(context_data, metadata)

                # Cache the result (skip empty responses to avoid caching transient failures)
                if any(context_data.get(k) for k in ("facts", "preferences", "episodes", "emotions", "temporal_events", "conversation_context")):
                    self._sdk._cache_manager.set(
                        scope=CacheScope.CONVERSATION,
                        entity_id=conversation_id,
                        context_type="context",
                        value=json.dumps(context_data).encode(),
                        ttl_seconds=metadata.ttl_seconds,
                        query=cache_params,
                    )

            # Emit telemetry
            latency_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            cache_origin = "local_http_cache" if cache_hit else "cloud_fetch"
            emit_fetch_event(
                self._sdk._telemetry_collector,
                scope="conversation",
                correlation_id=correlation_id,
                latency_ms=latency_ms,
                cache_hit=cache_hit,
                mode=mode,
                cache_origin=cache_origin,
            )

        # Auto-emit context_fetch for server-side retrieval history
        items_count = len(response.facts) + len(response.preferences) + len(response.episodes) + len(response.emotions)
        asyncio.ensure_future(_emit_context_fetch_event(
            sdk=self._sdk,
            scope="conversation",
            search_query=search_query,
            types=types,
            mode=mode,
            source=response.metadata.source,
            items_count=items_count,
            conversation_id=conversation_id,
        ))

        # Periodic user summary injection. Section 15 — only inject when the
        # caller passed a user_id; without it we cannot safely scope the
        # summary lookup, so we skip rather than risk cross-user splice.
        turn = self._sdk._increment_turn(conversation_id)
        if user_id and self._sdk._should_inject_user_summary(conversation_id):
            summary = self._sdk._anticipation_cache.lookup_user_summary(entity_id=user_id)
            if summary:
                response = _merge_user_summary_into_response(response, summary)
                logger.debug("Injected user summary at turn %d for conversation %s", turn, conversation_id)

        return response

    async def compact(
        self,
        conversation_id: str,
        strategy: Optional[str] = None,
        compaction_level: Optional[str] = None,
        target_tokens: Optional[int] = None,
        force: bool = False,
    ) -> CompactionTriggerResponse:
        """Trigger async conversation compaction.

        Compaction runs asynchronously on the server. Use
        get_compaction_status() to poll for completion, then
        get_compacted() to retrieve the result.

        Args:
            conversation_id: The conversation to compact
            strategy: Override strategy (aggressive, balanced, conservative, adaptive)
            compaction_level: Backward-compatible alias for strategy
            target_tokens: Override target token count
            force: Compact even if under threshold

        Returns:
            CompactionTriggerResponse with compaction_id and status
        """
        self._sdk._ensure_initialized()

        correlation_id = generate_correlation_id(self._sdk.instance_id)
        start_time = datetime.now(timezone.utc)

        auth_context = await self._sdk._get_auth_context(correlation_id)

        resolved_strategy = strategy or compaction_level

        result = await self._sdk._http_transport.post(
            "/v1/conversations/compact",
            auth_context=auth_context,
            json={
                "conversation_id": conversation_id,
                "strategy": resolved_strategy,
                "target_tokens": target_tokens,
                "force": force,
            },
            correlation_id=correlation_id,
        )

        # Emit telemetry
        latency_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
        from .telemetry.models import TelemetryEventType
        self._sdk._telemetry_collector.emit(
            event_type=TelemetryEventType.COMPACT_CONTEXT,
            correlation_id=correlation_id,
            latency_ms=latency_ms,
            scope="conversation",
            status="success",
        )

        response = CompactionTriggerResponse(
            compaction_id=result.get("compaction_id", "pending"),
            conversation_id=result.get("conversation_id", conversation_id),
            status=result.get("status", "in_progress"),
            trigger_type=result.get("trigger_type", "manual_api"),
            initiated_at=result.get("initiated_at", datetime.now(timezone.utc)),
            estimated_completion_seconds=result.get("estimated_completion_seconds", 60),
            previous_context=result.get("previous_context"),
            previous_context_age_seconds=result.get("previous_context_age_seconds"),
            previous_compaction_id=result.get("previous_compaction_id"),
        )

        # Cache previous_context locally so get_compacted() can serve it
        if response.previous_context:
            try:
                self._sdk._cache_manager.set(
                    scope=CacheScope.CONVERSATION,
                    entity_id=conversation_id,
                    context_type="compacted_full",
                    value=json.dumps(response.previous_context).encode(),
                    ttl_seconds=300,
                    query={"format": "structured"},
                )
            except Exception:
                pass  # Non-critical

        return response

    async def get_compacted(
        self,
        conversation_id: str,
        version: Optional[int] = None,
        format: str = "structured",
    ) -> Optional[CompactionResponse]:
        """Get existing compacted context without triggering new compaction.

        Checks local SDK cache first (unless a specific version is requested).
        On cache miss, fetches from cloud and caches the result for 5 minutes.

        Args:
            conversation_id: ID of the conversation
            version: Specific version number (latest if not specified).
                     When specified, always fetches from cloud (skip cache).
            format: Output format (structured, narrative, injection)

        Returns:
            CompactionResponse if exists, None otherwise
        """
        self._sdk._ensure_initialized()

        correlation_id = generate_correlation_id(self._sdk.instance_id)

        # Check local cache first (skip when version is pinned)
        if version is None:
            try:
                cached = self._sdk._cache_manager.get(
                    scope=CacheScope.CONVERSATION,
                    entity_id=conversation_id,
                    context_type="compacted_full",
                    query={"format": format},
                )
                if cached:
                    return CompactionResponse(**json.loads(cached))
            except Exception:
                pass  # Cache miss or error — fall through to cloud fetch

        try:
            auth_context = await self._sdk._get_auth_context(correlation_id)

            params = {"format": format}
            if version is not None:
                params["version"] = version

            result = await self._sdk._http_transport.get(
                f"/v1/conversations/{conversation_id}/compacted",
                auth_context=auth_context,
                params=params,
                correlation_id=correlation_id,
            )

            metadata = ResponseMetadata(
                correlation_id=correlation_id,
                ttl_seconds=result.get("ttl_seconds", 300),
                source="cloud",
                retrieved_at=datetime.now(timezone.utc),
            )

            response = CompactionResponse(
                compacted_context=result.get("formatted_context", ""),
                original_token_count=result.get("original_token_count", 0),
                compacted_token_count=result.get("compacted_token_count", 0),
                compression_ratio=result.get("compression_ratio", 0.0),
                level_applied=CompactionLevel(result.get("strategy_used", "adaptive")),
                metadata=metadata,
                compaction_id=result.get("compaction_id"),
                strategy_used=result.get("strategy_used"),
                validation_score=result.get("validation_score"),
                validation_passed=result.get("validation_passed"),
                facts=result.get("facts", []),
                decisions=result.get("decisions", []),
                preferences=result.get("preferences", []),
                current_state=result.get("current_state"),
                quality_warning=result.get("quality_warning"),
            )

            # Cache the response locally (5 min TTL)
            try:
                self._sdk._cache_manager.set(
                    scope=CacheScope.CONVERSATION,
                    entity_id=conversation_id,
                    context_type="compacted_full",
                    value=response.model_dump_json().encode(),
                    ttl_seconds=300,
                    query={"format": format},
                )
            except Exception:
                pass  # Non-critical — caching is best-effort

            return response

        except ContextNotFoundError:
            return None

    async def get_compaction_status(
        self,
        conversation_id: str,
    ) -> CompactionStatusResponse:
        """Get compaction status for a conversation.

        Checks the anticipation cache first — if a ``compaction_update``
        bundle was received via gRPC for this conversation, returns
        ``status="completed"`` immediately without a cloud round-trip.

        Returns info about:
        - Whether compacted context exists
        - Whether it's stale
        - Whether compaction is in progress

        Args:
            conversation_id: ID of the conversation

        Returns:
            CompactionStatusResponse with status information
        """
        self._sdk._ensure_initialized()

        # Check anticipation cache for a compaction_update bundle
        try:
            for entry in self._sdk._anticipation_cache._entries.values():
                if (
                    entry.bundle_type == "compaction_update"
                    and entry.conversation_id == conversation_id
                ):
                    ctx = entry.bundle.get("conversation_context", {})
                    return CompactionStatusResponse(
                        conversation_id=conversation_id,
                        status="completed",
                        compaction_id=ctx.get("compaction_id"),
                        completed_at=ctx.get("compacted_at"),
                    )
        except Exception:
            pass  # Fall through to cloud fetch

        correlation_id = generate_correlation_id(self._sdk.instance_id)
        auth_context = await self._sdk._get_auth_context(correlation_id)

        result = await self._sdk._http_transport.get(
            f"/v1/conversations/{conversation_id}/compaction/status",
            auth_context=auth_context,
            correlation_id=correlation_id,
        )

        return CompactionStatusResponse(**result)

    async def get_context_for_prompt(
        self,
        conversation_id: str,
        style: str = "structured",
    ) -> ContextForPromptResponse:
        """Get compacted context + recent un-compacted messages for LLM prompt injection.

        Returns a single ``formatted_context`` string that combines the compacted
        history with any messages that arrived after the last compaction cutoff,
        ready to inject into an LLM prompt. Also provides raw ``recent_messages``
        for custom formatting.

        If no compaction exists yet, all conversation messages are returned as
        recent messages so the method is useful even before the first compaction.

        Args:
            conversation_id: ID of the conversation
            style: Formatting style - "structured", "narrative", or "bullet_points"

        Returns:
            ContextForPromptResponse with formatted_context, recent_messages, and metadata
        """
        self._sdk._ensure_initialized()

        # Check local cache first (Tier 2 optimization)
        try:
            cached = self._sdk._cache_manager.get(
                scope=CacheScope.CONVERSATION,
                entity_id=conversation_id,
                context_type="compacted_context",
                query={"style": style},
            )
            if cached:
                import json as _json
                return ContextForPromptResponse(**_json.loads(cached))
        except Exception:
            pass  # Cache miss or error — fall through to cloud fetch

        correlation_id = generate_correlation_id(self._sdk.instance_id)
        auth_context = await self._sdk._get_auth_context(correlation_id)

        result = await self._sdk._http_transport.get(
            f"/v1/conversations/{conversation_id}/context-for-prompt",
            params={"style": style},
            auth_context=auth_context,
            correlation_id=correlation_id,
        )

        response = ContextForPromptResponse(
            formatted_context=result.get("formatted_context") or None,
            available=result.get("available", False),
            is_stale=result.get("is_stale", False),
            compression_ratio=result.get("compression_ratio"),
            validation_score=result.get("validation_score"),
            compaction_age_seconds=result.get("compaction_age_seconds"),
            quality_warning=result.get("quality_warning", False),
            recent_messages=result.get("recent_messages", []),
            recent_message_count=result.get("recent_message_count", 0),
            compacted_message_count=result.get("compacted_message_count", 0),
            total_message_count=result.get("total_message_count", 0),
        )

        # Cache the response locally (5 min TTL)
        try:
            self._sdk._cache_manager.set(
                scope=CacheScope.CONVERSATION,
                entity_id=conversation_id,
                context_type="compacted_context",
                value=response.model_dump_json().encode(),
                ttl_seconds=300,
                query={"style": style},
            )
        except Exception:
            pass  # Non-critical — caching is best-effort

        return response


class UserInterface:
    """Interface for user-scoped operations."""

    def __init__(self, sdk: MaximemSynapSDK):
        self._sdk = sdk
        self.context = UserContextInterface(sdk)


class UserContextInterface:
    """User context operations."""

    def __init__(self, sdk: MaximemSynapSDK):
        self._sdk = sdk

    async def fetch(
        self,
        user_id: str,
        conversation_id: Optional[str] = None,
        search_query: Optional[List[str]] = None,
        max_results: int = 10,
        types: Optional[List[str]] = None,
        mode: str = "fast",
        customer_id: Optional[str] = None,
    ) -> ContextResponse:
        """Fetch context for a user.

        Args:
            user_id: The user to fetch context for
            conversation_id: Optional conversation for relevance
            search_query: Optional search queries
            max_results: Maximum results to return
            types: Context types to include
            mode: Retrieval mode - "fast" (default) or "accurate"
            customer_id: Optional customer ID. Required for B2B instances.
                For B2C instances, this is auto-resolved from user_id.

        Returns:
            ContextResponse with user facts, preferences, etc.
        """
        self._sdk._ensure_initialized()

        # Validate mode
        valid_modes = ("fast", "accurate")
        if mode not in valid_modes:
            raise InvalidInputError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")

        correlation_id = generate_correlation_id(self._sdk.instance_id)
        start_time = datetime.now(timezone.utc)
        cache_hit = False

        cache_params = {
            "conversation_id": conversation_id,
            "search_query": search_query,
            "max_results": max_results,
            "types": types,
            "mode": mode,
        }

        # Check anticipation cache first (bundles pre-fetched via gRPC stream)
        anticipated = self._sdk._anticipation_cache.lookup(
            search_query=search_query,
            entity_id=user_id,
        )
        if anticipated:
            response = _build_anticipation_response(
                anticipated, correlation_id, start_time,
                scope="user", mode=mode,
                telemetry_collector=self._sdk._telemetry_collector,
            )
            asyncio.ensure_future(_emit_context_used_event(
                self._sdk,
                bundle_id=anticipated.get("bundle_id", ""),
                served_item_ids=_extract_served_item_ids(anticipated),
                scope="user",
                user_id=user_id,
                customer_id=customer_id or "",
                conversation_id=conversation_id or "",
                source_bundle_ids=anticipated.get("source_bundle_ids", []),
            ))
        else:
            # Check local cache
            cached = self._sdk._cache_manager.get(
                scope=CacheScope.USER,
                entity_id=user_id,
                context_type="context",
                query=cache_params,
            )

            if cached:
                cache_hit = True
                data = json.loads(cached)
                metadata = ResponseMetadata(
                    correlation_id=correlation_id,
                    ttl_seconds=0,
                    source="cache",
                    retrieved_at=datetime.now(timezone.utc),
                )
                response = ContextResponse.from_cloud_response(data, metadata)
            else:
                auth_context = await self._sdk._get_auth_context(correlation_id)

                result = await self._sdk._http_transport.post(
                    "/v1/context/user/fetch",
                    auth_context=auth_context,
                    json={
                        "user_id": user_id,
                        "customer_id": customer_id,
                        "conversation_id": conversation_id,
                        "search_query": search_query,
                        "max_results": max_results,
                        "types": types or ["all"],
                        "mode": mode,
                    },
                    correlation_id=correlation_id,
                )

                metadata = ResponseMetadata(
                    correlation_id=correlation_id,
                    ttl_seconds=result.get("ttl_seconds", 300),
                    source="cloud",
                    retrieved_at=datetime.now(timezone.utc),
                )
                context_data = result.get("context", {})
                if "conversation_context" in result:
                    context_data["conversation_context"] = result["conversation_context"]
                response = ContextResponse.from_cloud_response(context_data, metadata)

                # Cache (skip empty responses to avoid caching transient failures)
                if any(context_data.get(k) for k in ("facts", "preferences", "episodes", "emotions", "temporal_events", "conversation_context")):
                    self._sdk._cache_manager.set(
                        scope=CacheScope.USER,
                        entity_id=user_id,
                        context_type="context",
                        value=json.dumps(context_data).encode(),
                        ttl_seconds=metadata.ttl_seconds,
                        query=cache_params,
                    )

            latency_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            cache_origin = "local_http_cache" if cache_hit else "cloud_fetch"
            emit_fetch_event(
                self._sdk._telemetry_collector,
                scope="user",
                correlation_id=correlation_id,
                latency_ms=latency_ms,
                cache_hit=cache_hit,
                mode=mode,
                cache_origin=cache_origin,
            )

        # Auto-emit context_fetch for server-side retrieval history
        items_count = len(response.facts) + len(response.preferences) + len(response.episodes) + len(response.emotions)
        asyncio.ensure_future(_emit_context_fetch_event(
            sdk=self._sdk,
            scope="user",
            search_query=search_query,
            types=types,
            mode=mode,
            source=response.metadata.source,
            items_count=items_count,
            conversation_id=conversation_id or "",
            user_id=user_id,
        ))

        # Periodic user summary injection
        turn = self._sdk._increment_turn(conversation_id)
        if self._sdk._should_inject_user_summary(conversation_id):
            summary = self._sdk._anticipation_cache.lookup_user_summary(
                entity_id=user_id,
            )
            if summary:
                response = _merge_user_summary_into_response(response, summary)
                logger.debug("Injected user summary at turn %d for user %s", turn, user_id)

        return response


class CustomerInterface:
    """Interface for customer-scoped operations."""

    def __init__(self, sdk: MaximemSynapSDK):
        self._sdk = sdk
        self.context = CustomerContextInterface(sdk)


class CustomerContextInterface:
    """Customer context operations."""

    def __init__(self, sdk: MaximemSynapSDK):
        self._sdk = sdk

    async def fetch(
        self,
        customer_id: str,
        conversation_id: Optional[str] = None,
        search_query: Optional[List[str]] = None,
        max_results: int = 10,
        types: Optional[List[str]] = None,
        mode: str = "fast",
    ) -> ContextResponse:
        """Fetch context for a customer (B2B)."""
        self._sdk._ensure_initialized()

        # Validate mode
        valid_modes = ("fast", "accurate")
        if mode not in valid_modes:
            raise InvalidInputError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")

        correlation_id = generate_correlation_id(self._sdk.instance_id)
        start_time = datetime.now(timezone.utc)
        cache_hit = False

        cache_params = {
            "conversation_id": conversation_id,
            "search_query": search_query,
            "max_results": max_results,
            "types": types,
            "mode": mode,
        }

        # Check anticipation cache first (bundles pre-fetched via gRPC stream)
        anticipated = self._sdk._anticipation_cache.lookup(
            search_query=search_query,
            entity_id=customer_id,
        )
        if anticipated:
            response = _build_anticipation_response(
                anticipated, correlation_id, start_time,
                scope="customer", mode=mode,
                telemetry_collector=self._sdk._telemetry_collector,
            )
            asyncio.ensure_future(_emit_context_used_event(
                self._sdk,
                bundle_id=anticipated.get("bundle_id", ""),
                served_item_ids=_extract_served_item_ids(anticipated),
                scope="customer",
                customer_id=customer_id,
                conversation_id=conversation_id or "",
                source_bundle_ids=anticipated.get("source_bundle_ids", []),
            ))
        else:
            cached = self._sdk._cache_manager.get(
                scope=CacheScope.CUSTOMER,
                entity_id=customer_id,
                context_type="context",
                query=cache_params,
            )

            if cached:
                cache_hit = True
                data = json.loads(cached)
                metadata = ResponseMetadata(
                    correlation_id=correlation_id,
                    ttl_seconds=0,
                    source="cache",
                    retrieved_at=datetime.now(timezone.utc),
                )
                response = ContextResponse.from_cloud_response(data, metadata)
            else:
                auth_context = await self._sdk._get_auth_context(correlation_id)

                result = await self._sdk._http_transport.post(
                    "/v1/context/customer/fetch",
                    auth_context=auth_context,
                    json={
                        "customer_id": customer_id,
                        "conversation_id": conversation_id,
                        "search_query": search_query,
                        "max_results": max_results,
                        "types": types or ["all"],
                        "mode": mode,
                    },
                    correlation_id=correlation_id,
                )

                metadata = ResponseMetadata(
                    correlation_id=correlation_id,
                    ttl_seconds=result.get("ttl_seconds", 300),
                    source="cloud",
                    retrieved_at=datetime.now(timezone.utc),
                )
                context_data = result.get("context", {})
                if "conversation_context" in result:
                    context_data["conversation_context"] = result["conversation_context"]
                response = ContextResponse.from_cloud_response(context_data, metadata)

                # Skip caching empty responses to avoid caching transient failures
                if any(context_data.get(k) for k in ("facts", "preferences", "episodes", "emotions", "temporal_events", "conversation_context")):
                    self._sdk._cache_manager.set(
                        scope=CacheScope.CUSTOMER,
                        entity_id=customer_id,
                        context_type="context",
                        value=json.dumps(context_data).encode(),
                        ttl_seconds=metadata.ttl_seconds,
                        query=cache_params,
                    )

            latency_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            cache_origin = "local_http_cache" if cache_hit else "cloud_fetch"
            emit_fetch_event(
                self._sdk._telemetry_collector,
                scope="customer",
                correlation_id=correlation_id,
                latency_ms=latency_ms,
                cache_hit=cache_hit,
                mode=mode,
                cache_origin=cache_origin,
            )

        # Auto-emit context_fetch for server-side retrieval history
        items_count = len(response.facts) + len(response.preferences) + len(response.episodes) + len(response.emotions)
        asyncio.ensure_future(_emit_context_fetch_event(
            sdk=self._sdk,
            scope="customer",
            search_query=search_query,
            types=types,
            mode=mode,
            source=response.metadata.source,
            items_count=items_count,
            conversation_id=conversation_id or "",
            customer_id=customer_id,
        ))

        # Periodic user summary injection
        turn = self._sdk._increment_turn(conversation_id)
        if self._sdk._should_inject_user_summary(conversation_id):
            summary = self._sdk._anticipation_cache.lookup_user_summary(
                entity_id=customer_id,
            )
            if summary:
                response = _merge_user_summary_into_response(response, summary)
                logger.debug("Injected user summary at turn %d for customer %s", turn, customer_id)

        return response


class ClientInterface:
    """Interface for client (org) scoped operations."""

    def __init__(self, sdk: MaximemSynapSDK):
        self._sdk = sdk
        self.context = ClientContextInterface(sdk)


class ClientContextInterface:
    """Client/org context operations."""

    def __init__(self, sdk: MaximemSynapSDK):
        self._sdk = sdk

    async def fetch(
        self,
        conversation_id: Optional[str] = None,
        search_query: Optional[List[str]] = None,
        max_results: int = 10,
        types: Optional[List[str]] = None,
        mode: str = "fast",
    ) -> ContextResponse:
        """Fetch organizational context."""
        self._sdk._ensure_initialized()

        # Validate mode
        valid_modes = ("fast", "accurate")
        if mode not in valid_modes:
            raise InvalidInputError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")

        correlation_id = generate_correlation_id(self._sdk.instance_id)
        start_time = datetime.now(timezone.utc)
        cache_hit = False

        cache_params = {
            "conversation_id": conversation_id,
            "search_query": search_query,
            "max_results": max_results,
            "types": types,
            "mode": mode,
        }

        # Check anticipation cache first (bundles pre-fetched via gRPC stream)
        anticipated = self._sdk._anticipation_cache.lookup(
            search_query=search_query,
            entity_id="_client",  # sentinel — prevents matching user/customer scoped bundles
        )
        if anticipated:
            response = _build_anticipation_response(
                anticipated, correlation_id, start_time,
                scope="client", mode=mode,
                telemetry_collector=self._sdk._telemetry_collector,
            )
            asyncio.ensure_future(_emit_context_used_event(
                self._sdk,
                bundle_id=anticipated.get("bundle_id", ""),
                served_item_ids=_extract_served_item_ids(anticipated),
                scope="client",
                conversation_id=conversation_id or "",
                source_bundle_ids=anticipated.get("source_bundle_ids", []),
            ))
        else:
            cached = self._sdk._cache_manager.get(
                scope=CacheScope.CLIENT,
                entity_id=self._sdk._client_id,
                context_type="context",
                query=cache_params,
            )

            if cached:
                cache_hit = True
                data = json.loads(cached)
                metadata = ResponseMetadata(
                    correlation_id=correlation_id,
                    ttl_seconds=0,
                    source="cache",
                    retrieved_at=datetime.now(timezone.utc),
                )
                response = ContextResponse.from_cloud_response(data, metadata)
            else:
                auth_context = await self._sdk._get_auth_context(correlation_id)

                result = await self._sdk._http_transport.post(
                    "/v1/context/client/fetch",
                    auth_context=auth_context,
                    json={
                        "conversation_id": conversation_id,
                        "search_query": search_query,
                        "max_results": max_results,
                        "types": types or ["all"],
                        "mode": mode,
                    },
                    correlation_id=correlation_id,
                )

                metadata = ResponseMetadata(
                    correlation_id=correlation_id,
                    ttl_seconds=result.get("ttl_seconds", 1800),  # 30 min for client
                    source="cloud",
                    retrieved_at=datetime.now(timezone.utc),
                )
                context_data = result.get("context", {})
                if "conversation_context" in result:
                    context_data["conversation_context"] = result["conversation_context"]
                response = ContextResponse.from_cloud_response(context_data, metadata)

                # Skip caching empty responses to avoid caching transient failures
                if any(context_data.get(k) for k in ("facts", "preferences", "episodes", "emotions", "temporal_events", "conversation_context")):
                    self._sdk._cache_manager.set(
                        scope=CacheScope.CLIENT,
                        entity_id=self._sdk._client_id,
                        context_type="context",
                        value=json.dumps(context_data).encode(),
                        ttl_seconds=metadata.ttl_seconds,
                        query=cache_params,
                    )

            latency_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            cache_origin = "local_http_cache" if cache_hit else "cloud_fetch"
            emit_fetch_event(
                self._sdk._telemetry_collector,
                scope="client",
                correlation_id=correlation_id,
                latency_ms=latency_ms,
                cache_hit=cache_hit,
                mode=mode,
                cache_origin=cache_origin,
            )

        # Auto-emit context_fetch for server-side retrieval history
        items_count = len(response.facts) + len(response.preferences) + len(response.episodes) + len(response.emotions)
        asyncio.ensure_future(_emit_context_fetch_event(
            sdk=self._sdk,
            scope="client",
            search_query=search_query,
            types=types,
            mode=mode,
            source=response.metadata.source,
            items_count=items_count,
            conversation_id=conversation_id or "",
        ))

        # Periodic user summary injection
        turn = self._sdk._increment_turn(conversation_id)
        if self._sdk._should_inject_user_summary(conversation_id):
            summary = self._sdk._anticipation_cache.lookup_user_summary(
                entity_id="_client",
            )
            if summary:
                response = _merge_user_summary_into_response(response, summary)
                logger.debug("Injected user summary at turn %d for client scope", turn)

        return response


class InstanceInterface:
    """Interface for instance-level operations (listening).

    Wires SDK components into InstanceController and delegates.
    """

    def __init__(self, sdk: MaximemSynapSDK):
        self._sdk = sdk
        self._controller = InstanceController(
            transport_factory=self._create_transport,
            auth_provider=self._sdk._get_auth_context,
        )

    def _create_transport(self, **kwargs) -> GRPCTransport:
        """Factory that creates a GRPCTransport with SDK configuration."""
        transport = GRPCTransport(
            instance_id=self._sdk.instance_id,
            host=self._sdk._config.grpc_host,
            port=self._sdk._config.grpc_port,
            use_tls=self._sdk._config.grpc_use_tls,
            timeouts=self._sdk._config.timeouts,
            telemetry_callback=self._sdk._on_telemetry_event,
            **kwargs,
        )
        self._sdk._grpc_transport = transport
        return transport

    async def listen(
        self,
        on_reconnect: Optional[Callable[[int], None]] = None,
        on_disconnect: Optional[Callable[[str], None]] = None,
        on_context: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Start listening to agent activity.

        Establishes a bidirectional gRPC stream for real-time
        context anticipation.

        Args:
            on_reconnect: Callback when stream reconnects (receives attempt count)
            on_disconnect: Callback when stream disconnects (receives reason)
            on_context: Callback when an anticipated context bundle arrives.
                The bundle dict contains items_by_type, retrieval_mode, etc.
                Bundles are also stored in the SDK's anticipation cache
                automatically, so fetch() can find them without a round-trip.
        """
        self._sdk._ensure_initialized()
        self._on_context_callback = on_context
        await self._controller.listen(
            on_reconnect=on_reconnect,
            on_disconnect=on_disconnect,
            on_message=self._handle_anticipated_bundle,
        )

    def _handle_anticipated_bundle(self, bundle_dict: Dict[str, Any]) -> None:
        """Handle a context bundle received over the gRPC stream."""
        bundle_type = bundle_dict.get("_bundle_type", "anticipation")

        if bundle_type != "reactive":
            self._sdk._anticipation_cache.store(bundle_dict)
        else:
            logger.debug(
                "Skipping reactive bundle: %s",
                bundle_dict.get("bundle_id"),
            )

        if bundle_type == "compaction_update":
            conv_id = bundle_dict.get("_anticipation_conversation_id")
            if conv_id:
                try:
                    self._sdk._cache_manager.delete(
                        scope=CacheScope.CONVERSATION,
                        entity_id=conv_id,
                        context_type="compacted_full",
                    )
                    self._sdk._cache_manager.delete(
                        scope=CacheScope.CONVERSATION,
                        entity_id=conv_id,
                        context_type="compacted_context",
                    )
                    logger.debug(
                        "Invalidated local compaction cache for conversation %s",
                        conv_id,
                    )
                except Exception:
                    pass  # Non-critical

        # Always invoke user callback regardless of type
        if hasattr(self, "_on_context_callback") and self._on_context_callback:
            try:
                if asyncio.iscoroutinefunction(self._on_context_callback):
                    asyncio.create_task(self._on_context_callback(bundle_dict))
                else:
                    self._on_context_callback(bundle_dict)
            except Exception as e:
                logger.warning(f"on_context callback error: {e}")

    async def stop_listening(self) -> None:
        """Stop listening to agent activity."""
        await self._controller.stop()
        self._sdk._grpc_transport = None

    async def send_message(
        self,
        content: str,
        role: str = "user",
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        session_id: Optional[str] = None,
        event_type: str = "user_message",
        metadata: Optional[Dict[str, str]] = None,
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        search_queries: Optional[List[str]] = None,
        context_types: Optional[List[str]] = None,
    ) -> None:
        """Send a conversation message over the active gRPC stream.

        Args:
            content: Message content
            role: "user" or "assistant"
            conversation_id: Conversation identifier
            user_id: User identifier
            customer_id: Customer identifier
            session_id: Session identifier
            event_type: Event type (user_message, assistant_message, tool_call, etc.)
            metadata: Additional string key-value metadata
            tool_name: For tool_call events — the tool the agent is invoking.
                The listening agent reads this when classifying TOOL_CALL signals
                and uses it to anticipate the agent's next data needs.
            tool_args: For tool_call events — JSON-encodable arguments dict.
                Serialized into the tool_args_json proto field.
            search_queries: For tool_call / context_request events — the
                retrieval queries the agent plans to run. Listening agent
                uses these as direct anticipation hints.
            context_types: For tool_call / context_request events — the
                memory categories the agent plans to fetch.

        Raises:
            ListeningNotActiveError: If listen() has not been called.
        """
        from .models.errors import ListeningNotActiveError

        if not self.is_listening:
            raise ListeningNotActiveError()

        payload = {
            "event_type": event_type,
            "content": content,
            "role": role,
            "conversation_id": conversation_id or "",
            "user_id": user_id or "",
            "customer_id": customer_id or "",
            "session_id": session_id or "",
            "metadata": metadata or {},
        }
        if tool_name:
            payload["tool_name"] = tool_name
        if tool_args is not None:
            payload["tool_args_json"] = json.dumps(tool_args)
        if search_queries:
            payload["search_queries"] = list(search_queries)
        if context_types:
            payload["context_types"] = list(context_types)

        await self._controller._transport.send(payload)

    @property
    def is_listening(self) -> bool:
        """Check if currently listening."""
        return self._controller.is_listening


class CacheInterface:
    """Interface for cache management."""

    def __init__(self, sdk: MaximemSynapSDK):
        self._sdk = sdk

    def clear(self) -> None:
        """Clear all cached data."""
        if self._sdk._cache_manager:
            self._sdk._cache_manager.clear_all()

    def clear_user(self, user_id: str) -> None:
        """Clear cached data for a specific user (GDPR)."""
        if self._sdk._cache_manager:
            self._sdk._cache_manager.clear_user(user_id)

    def clear_customer(self, customer_id: str) -> None:
        """Clear cached data for a specific customer."""
        if self._sdk._cache_manager:
            self._sdk._cache_manager.clear_customer(customer_id)

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self._sdk._cache_manager:
            return self._sdk._cache_manager.stats()
        return {"enabled": False}
