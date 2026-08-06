"""Cache manager - handles scoped caching with separate files per user/customer."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from ..models.enums import ContextScope
from .backend import CacheBackend, NullCacheBackend
from .sqlite_backend import SQLiteCacheBackend


logger = logging.getLogger("synap.sdk.cache")

# Alias ContextScope as CacheScope for cache layer usage
CacheScope = ContextScope


class CacheManager:
    """Manages scoped caching with separate files per user/customer.

    File structure:
    ~/.synap/{client_id}/
    ├── client_cache.db              # Org-level context
    ├── customers/
    │   └── {customer_id}.db         # Customer-level context
    └── users/
        └── {user_id}.db             # User-level context (incl. conversations)
    """

    # Default TTLs by scope (seconds)
    DEFAULT_TTLS = {
        CacheScope.CLIENT: 1800,      # 30 minutes
        CacheScope.CUSTOMER: 300,     # 5 minutes
        CacheScope.USER: 300,         # 5 minutes
        CacheScope.CONVERSATION: 300, # 5 minutes
    }

    def __init__(
        self,
        client_id: str,
        storage_path: Optional[str] = None,
        enabled: bool = True,
        instance_id: str = "",
    ):
        self.client_id = client_id
        # The producing/reading Instance. Customer- and client-scoped cache files
        # are shared by every Instance under the same client+customer, so the
        # cache KEY must include instance_id — otherwise two Instances serving the
        # same customer read each other's entries, silently bypassing server-side
        # cross-instance visibility (an Instance would see another Instance's
        # filtered memories straight from the local cache).
        self.instance_id = instance_id or ""
        self.enabled = enabled

        if storage_path:
            self.base_path = Path(storage_path) / client_id
        else:
            self.base_path = Path.home() / ".synap" / client_id

        # Cache of open backends by scope key
        self._backends: Dict[str, CacheBackend] = {}

    def _get_backend(self, scope: CacheScope, entity_id: str) -> CacheBackend:
        """Get or create cache backend for scope/entity combination."""
        if not self.enabled:
            return NullCacheBackend()

        # Determine file path based on scope
        if scope == CacheScope.CLIENT:
            db_path = self.base_path / "client_cache.db"
            backend_key = "client"
        elif scope == CacheScope.CUSTOMER:
            db_path = self.base_path / "customers" / f"{entity_id}.db"
            backend_key = f"customer:{entity_id}"
        elif scope in (CacheScope.USER, CacheScope.CONVERSATION):
            # Conversations are stored in user's cache file
            db_path = self.base_path / "users" / f"{entity_id}.db"
            backend_key = f"user:{entity_id}"
        else:
            return NullCacheBackend()

        # Return existing backend if available
        if backend_key in self._backends:
            return self._backends[backend_key]

        # Create new backend
        backend = SQLiteCacheBackend(db_path)
        self._backends[backend_key] = backend
        return backend

    def _build_key(
        self,
        scope: CacheScope,
        entity_id: str,
        context_type: str,
        query_hash: Optional[str] = None,
    ) -> str:
        """Build cache key.

        Format: {client_id}:{instance_id}:{scope}:{entity_id}:{context_type}:{query_hash}

        ``instance_id`` is part of the key so two Instances under the same
        client+customer (which share the same on-disk cache file) never read
        each other's entries — preserving cross-instance visibility at the
        client cache layer.
        """
        parts = [self.client_id, self.instance_id or "_", scope.value, entity_id, context_type]
        if query_hash:
            parts.append(query_hash)
        return ":".join(parts)

    def _hash_query(self, query: Any) -> str:
        """Create deterministic hash of query parameters."""
        if query is None:
            return "none"
        serialized = json.dumps(query, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:12]

    def get(
        self,
        scope: CacheScope,
        entity_id: str,
        context_type: str,
        query: Any = None,
    ) -> Optional[bytes]:
        """Get cached value.

        Args:
            scope: Cache scope (client, customer, user, conversation)
            entity_id: ID of the entity (client_id, customer_id, user_id, conversation_id)
            context_type: Type of context (facts, preferences, etc.)
            query: Optional query parameters (for cache key differentiation)

        Returns:
            Cached bytes or None if not found
        """
        # For conversation scope, use user_id as the file key
        file_entity_id = entity_id
        if scope == CacheScope.CONVERSATION:
            # Caller must provide user_id in entity_id for conversation scope
            # The actual conversation_id goes into the key
            pass

        backend = self._get_backend(scope, file_entity_id)
        query_hash = self._hash_query(query)
        key = self._build_key(scope, entity_id, context_type, query_hash)

        result = backend.get(key)
        if result:
            logger.debug(f"Cache hit: {key}")
        else:
            logger.debug(f"Cache miss: {key}")
        return result

    def set(
        self,
        scope: CacheScope,
        entity_id: str,
        context_type: str,
        value: bytes,
        ttl_seconds: Optional[int] = None,
        query: Any = None,
    ) -> None:
        """Set cached value.

        Args:
            scope: Cache scope
            entity_id: ID of the entity
            context_type: Type of context
            value: Bytes to cache
            ttl_seconds: TTL in seconds (uses default if not provided)
            query: Optional query parameters
        """
        backend = self._get_backend(scope, entity_id)
        query_hash = self._hash_query(query)
        key = self._build_key(scope, entity_id, context_type, query_hash)
        ttl = ttl_seconds or self.DEFAULT_TTLS.get(scope, 300)

        backend.set(key, value, ttl)
        logger.debug(f"Cache set: {key} (ttl={ttl}s)")

    def delete(
        self,
        scope: CacheScope,
        entity_id: str,
        context_type: Optional[str] = None,
        query: Any = None,
    ) -> None:
        """Delete cached value(s).

        If context_type is None, deletes all entries for entity.
        """
        backend = self._get_backend(scope, entity_id)

        if context_type:
            query_hash = self._hash_query(query)
            key = self._build_key(scope, entity_id, context_type, query_hash)
            backend.delete(key)
        else:
            # Delete all entries for this entity (scoped to THIS instance, to
            # match the instance-namespaced key built above).
            prefix = f"{self.client_id}:{self.instance_id or '_'}:{scope.value}:{entity_id}:"
            backend.clear_scope(prefix)

    def clear_user(self, user_id: str) -> None:
        """Clear all cached data for a user (GDPR deletion)."""
        # Delete the user's cache file entirely
        user_db = self.base_path / "users" / f"{user_id}.db"
        if user_db.exists():
            user_db.unlink()
            logger.info(f"Deleted cache for user {user_id}")

        # Remove from backends cache
        backend_key = f"user:{user_id}"
        if backend_key in self._backends:
            self._backends[backend_key].close()
            del self._backends[backend_key]

    def clear_customer(self, customer_id: str) -> None:
        """Clear all cached data for a customer."""
        customer_db = self.base_path / "customers" / f"{customer_id}.db"
        if customer_db.exists():
            customer_db.unlink()
            logger.info(f"Deleted cache for customer {customer_id}")

        backend_key = f"customer:{customer_id}"
        if backend_key in self._backends:
            self._backends[backend_key].close()
            del self._backends[backend_key]

    def clear_all(self) -> None:
        """Clear all cached data for this client."""
        # Close all backends
        for backend in self._backends.values():
            backend.close()
        self._backends.clear()

        # Delete all cache files
        import shutil
        if self.base_path.exists():
            for item in self.base_path.iterdir():
                if item.suffix == ".db" or item.is_dir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()

        logger.info(f"Cleared all cache for client {self.client_id}")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics across all backends."""
        total_entries = 0
        total_bytes = 0
        backend_stats = []

        for key, backend in self._backends.items():
            s = backend.stats()
            total_entries += s.get("entry_count", 0)
            total_bytes += s.get("total_bytes", 0)
            backend_stats.append({
                "key": key,
                **s
            })

        return {
            "enabled": self.enabled,
            "client_id": self.client_id,
            "base_path": str(self.base_path),
            "total_entries": total_entries,
            "total_bytes": total_bytes,
            "backends": backend_stats,
        }

    def close(self) -> None:
        """Close all cache backends."""
        for backend in self._backends.values():
            backend.close()
        self._backends.clear()
