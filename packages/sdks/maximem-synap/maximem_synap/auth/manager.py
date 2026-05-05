"""Credential manager for API key authentication.

The SDK reads a Synap API key from one of two places, in priority order:
    1. api_key= kwarg passed to MaximemSynapSDK(...)
    2. SYNAP_API_KEY environment variable

That's it. No file-based cache, no bootstrap exchange. Generate a key in
the dashboard, paste it into your .env, done.
"""

import logging
import os
from typing import Optional

from .models import Credentials, AuthContext
from .storage import EnvironmentCredentialStorage
from ..models.errors import AuthenticationError

logger = logging.getLogger("synap.sdk.auth")


class CredentialManager:
    def __init__(self, instance_id: str = ""):
        self.instance_id = instance_id
        self._credentials: Optional[Credentials] = None
        self._storage = EnvironmentCredentialStorage(instance_id)

    def load(self, api_key: Optional[str] = None) -> Credentials:
        """Resolve credentials from the api_key kwarg or SYNAP_API_KEY."""
        key = api_key or os.environ.get("SYNAP_API_KEY")
        if not key:
            raise AuthenticationError(
                "No Synap API key found. Set SYNAP_API_KEY in your environment "
                "or pass api_key= to MaximemSynapSDK(). Generate a key at the "
                "Synap dashboard: Instances -> (your instance) -> Generate API Key."
            )

        instance_id = self.instance_id or os.environ.get("SYNAP_INSTANCE_ID", "")
        credentials = Credentials(
            api_key=key,
            instance_id=instance_id,
            client_id=os.environ.get("SYNAP_CLIENT_ID", ""),
        )
        self._credentials = credentials
        return credentials

    async def get_auth_context(
        self, correlation_id: Optional[str] = None
    ) -> AuthContext:
        if not self._credentials:
            self._credentials = self._storage.load()
        if not self._credentials:
            raise AuthenticationError(
                "No credentials loaded. Call load() or pass api_key= first."
            )
        return AuthContext(
            client_id=self._credentials.client_id,
            instance_id=self._credentials.instance_id,
            api_key=self._credentials.api_key,
            correlation_id=correlation_id,
        )
