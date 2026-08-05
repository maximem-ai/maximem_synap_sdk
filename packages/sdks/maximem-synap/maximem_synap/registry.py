"""SDK instance registry for singleton pattern."""

import hashlib
import os
import threading
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .sdk import MaximemSynapSDK


def build_registry_key(instance_id: str, api_key: Optional[str]) -> str:
    """Derive the registry slot an SDK belongs in.

    ``instance_id`` keys the singleton, but it is optional and normally empty
    at construction time — it is resolved from the API key during
    ``initialize()``. Keying on it directly therefore put every
    ``MaximemSynapSDK(api_key=...)`` in one process onto the single ``""``
    slot, and the second one adopted the first's state, credentials included.

    So when no ``instance_id`` is supplied, fall back to the credential that
    *will* resolve the identity. It is stored as a truncated SHA-256 digest,
    never in plaintext, so a registry dump or a log line cannot leak it.

    ``SYNAP_INSTANCE_ID`` is deliberately not consulted. The SDK applies that
    fallback after the registry lookup, and honouring it here would put two
    different credentials back onto one slot whenever the env var happens to
    be set.
    """
    if instance_id:
        return instance_id
    key = api_key or os.environ.get("SYNAP_API_KEY", "")
    if not key:
        # Nothing to key on. The SDK cannot initialize without a credential
        # anyway, so preserve the historical empty-string slot.
        return ""
    return f"apikey:{hashlib.sha256(key.encode()).hexdigest()[:32]}"


class SDKRegistry:
    """Registry for SDK instances (singleton per registry key)."""

    _instances: Dict[str, "MaximemSynapSDK"] = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, instance_id: str) -> Optional["MaximemSynapSDK"]:
        """Get existing SDK instance for instance_id."""
        with cls._lock:
            return cls._instances.get(instance_id)

    @classmethod
    def register(cls, instance_id: str, sdk: "MaximemSynapSDK") -> None:
        """Register an SDK instance."""
        with cls._lock:
            cls._instances[instance_id] = sdk

    @classmethod
    def register_if_absent(
        cls, key: str, sdk: "MaximemSynapSDK"
    ) -> Optional["MaximemSynapSDK"]:
        """Claim ``key`` for ``sdk``, or return whoever already holds it.

        ``get()`` then ``register()`` each take the lock independently, so two
        threads constructing the first SDK for one identity can both miss the
        lookup and the second then overwrites the first — leaving the loser's
        caller holding an SDK the registry no longer knows about, with its own
        caches and its own gRPC stream. The whole of ``__init__`` sits inside
        that window, so it is wide enough to lose at stock switch intervals.

        Returns ``None`` when ``sdk`` won the slot, or the incumbent when it
        did not — the loser is expected to adopt the winner's state.
        """
        with cls._lock:
            existing = cls._instances.get(key)
            if existing is not None:
                return existing
            cls._instances[key] = sdk
            return None

    @classmethod
    def alias_if_absent(cls, key: str, sdk: "MaximemSynapSDK") -> bool:
        """Point ``key`` at ``sdk`` as well, unless the slot is already taken.

        Used once ``initialize()`` resolves the real ``instance_id``: the SDK
        was keyed on its credential because the id did not exist yet, so a
        later ``MaximemSynapSDK(instance_id=...)`` for that same instance
        missed and built a second SDK — two anticipation caches, two
        short-term stores and two Listen streams for one instance.

        The credential slot is kept rather than moved, so constructing by API
        key keeps returning the same SDK too. An occupied target is never
        clobbered: evicting a live SDK is worse than the duplication.

        Returns ``True`` if the alias was installed.
        """
        with cls._lock:
            if not key or key in cls._instances:
                return False
            cls._instances[key] = sdk
            return True

    @classmethod
    def unregister(cls, instance_id: str) -> None:
        """Unregister an SDK instance."""
        with cls._lock:
            cls._instances.pop(instance_id, None)

    @classmethod
    def unregister_if_owner(cls, key: str, sdk: "MaximemSynapSDK") -> None:
        """Drop ``key`` only if it still maps to ``sdk``'s state.

        A shutting-down SDK must not evict whoever currently holds its old
        slot. The comparison is on ``__dict__`` identity, not object identity:
        a second SDK constructed for the same key aliases the registered one's
        ``__dict__``, so shutting either of them down closes the one set of
        transports both share — and the entry has to go with them.
        """
        with cls._lock:
            existing = cls._instances.get(key)
            if existing is not None and existing.__dict__ is sdk.__dict__:
                del cls._instances[key]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered instances (for testing)."""
        with cls._lock:
            cls._instances.clear()
