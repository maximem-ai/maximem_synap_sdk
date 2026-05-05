"""Credential storage: environment variables only."""

import logging
import os
from typing import Optional

from .models import Credentials

logger = logging.getLogger("synap.sdk.auth.storage")


class EnvironmentCredentialStorage:
    """Read credentials from environment variables.

    Required: SYNAP_API_KEY
    Optional: SYNAP_INSTANCE_ID, SYNAP_CLIENT_ID (resolved server-side via whoami)
    """

    def __init__(self, instance_id: str = ""):
        self.instance_id = instance_id

    def load(self) -> Optional[Credentials]:
        api_key = os.environ.get("SYNAP_API_KEY")
        if not api_key:
            return None
        return Credentials(
            api_key=api_key,
            instance_id=self.instance_id or os.environ.get("SYNAP_INSTANCE_ID", ""),
            client_id=os.environ.get("SYNAP_CLIENT_ID", ""),
        )

    def exists(self) -> bool:
        return "SYNAP_API_KEY" in os.environ
